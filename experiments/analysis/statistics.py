import random
from typing import Any, Dict, List


def analyze_results(
    results: Dict[str, Any],
    bootstrap_samples: int = 5000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Summarize outcomes and estimate a paired D-versus-B task-success effect."""
    conditions = results.get("conditions", {})
    summaries = {}
    by_task = {}

    for condition, condition_data in conditions.items():
        evaluation_rows = [
            row
            for run in condition_data.get("runs", [])
            for row in run.get("evaluation", [])
        ]
        successful = sum(bool(row.get("success")) for row in evaluation_rows)
        opportunities = sum(
            bool(row.get("cross_user_knowledge_available")) for row in evaluation_rows
        )
        agent_reported = sum(
            bool(row.get("cross_user_transfer_reported")) for row in evaluation_rows
        )
        tool_counts = [row.get("tool_call_count", 0) for row in evaluation_rows]
        latencies = [row.get("latency_seconds", 0.0) for row in evaluation_rows]
        summaries[condition] = {
            "successful_tasks": successful,
            "total_tasks": len(evaluation_rows),
            "task_completion_rate": _rate(successful, len(evaluation_rows)),
            "cross_user_knowledge_opportunities": opportunities,
            "agent_reported_cross_user_transfers": agent_reported,
            "agent_reported_transfer_rate": _rate(agent_reported, opportunities),
            "mean_tool_calls": _mean(tool_counts),
            "mean_latency_seconds": _mean(latencies),
        }

        task_values = {}
        for row in evaluation_rows:
            task_values.setdefault(row["task_id"], []).append(float(bool(row.get("success"))))
        by_task[condition] = {
            task_id: _mean(values) for task_id, values in task_values.items()
        }

    comparisons = {}
    control = by_task.get("shared_memory", {})
    treatment = by_task.get("central_learning", {})
    paired_task_ids = sorted(set(control) & set(treatment))
    paired_differences = [treatment[task_id] - control[task_id] for task_id in paired_task_ids]
    if paired_differences:
        lower, upper = _bootstrap_mean_interval(
            paired_differences,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        comparisons["central_learning_vs_shared_memory_tcr"] = {
            "paired_task_count": len(paired_differences),
            "mean_paired_difference": _mean(paired_differences),
            "confidence_interval_95": [lower, upper],
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
        }
    else:
        comparisons["central_learning_vs_shared_memory_tcr"] = {
            "paired_task_count": 0,
            "mean_paired_difference": None,
            "confidence_interval_95": None,
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
        }

    return {"conditions": summaries, "comparisons": comparisons}


def _bootstrap_mean_interval(values: List[float], bootstrap_samples: int, seed: int):
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be at least one.")
    generator = random.Random(seed)
    estimates = [
        _mean(generator.choices(values, k=len(values)))
        for _ in range(bootstrap_samples)
    ]
    estimates.sort()
    lower_index = int(0.025 * (bootstrap_samples - 1))
    upper_index = int(0.975 * (bootstrap_samples - 1))
    return estimates[lower_index], estimates[upper_index]


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0