from experiments.analysis.statistics import analyze_results


def test_analysis_reports_paired_task_effect_and_bootstrap_interval():
    results = {
        "conditions": {
            "shared_memory": {
                "runs": [
                    {"evaluation": [
                        {"task_id": "task_a", "success": False},
                        {"task_id": "task_b", "success": True},
                    ]},
                    {"evaluation": [
                        {"task_id": "task_a", "success": False},
                        {"task_id": "task_b", "success": False},
                    ]},
                ]
            },
            "central_learning": {
                "runs": [
                    {"evaluation": [
                        {"task_id": "task_a", "success": True},
                        {"task_id": "task_b", "success": True},
                    ]},
                    {"evaluation": [
                        {"task_id": "task_a", "success": False},
                        {"task_id": "task_b", "success": True},
                    ]},
                ]
            },
        }
    }

    report = analyze_results(results, bootstrap_samples=500, seed=11)
    effect = report["comparisons"]["central_learning_vs_shared_memory_tcr"]

    assert effect["paired_task_count"] == 2
    assert effect["mean_paired_difference"] == 0.5
    assert effect["confidence_interval_95"][0] <= 0.5
    assert effect["confidence_interval_95"][1] >= 0.5