import csv
import ast
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import yaml

from cls.graph.retrieval import KnowledgeRetrievalAPI
from cls.graph.store import GraphStore
from cls.ingestion.schemas import Trajectory, TrajectoryStep
from cls.service.worker import LearningWorker
from experiments.analysis.statistics import analyze_results


CONDITIONS = ("independent", "shared_memory", "static_kg", "central_learning")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate a YAML experiment definition."""
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("experiment"), dict):
        raise ValueError("The config must contain an 'experiment' mapping.")
    validate_config(config["experiment"])
    config["_config_path"] = str(path)
    return config


def validate_config(experiment: Dict[str, Any]) -> None:
    required = ("name", "project_id", "repository_path", "repository_commit", "tasks", "agent")
    missing = [key for key in required if key not in experiment]
    if missing:
        raise ValueError(f"Missing experiment settings: {', '.join(missing)}")

    agent = experiment["agent"]
    if not isinstance(agent.get("command"), list) or not agent["command"]:
        raise ValueError("agent.command must be a non-empty argument list.")
    purpose = experiment.get("purpose", "research_experiment")
    if purpose != "smoke_test_only_not_research_evidence":
        agent_image = agent.get("container_image", "")
        if agent.get("isolation") != "container" or "@sha256:" not in agent_image:
            raise ValueError(
                "Research runs require agent isolation and a digest-pinned container_image."
            )
        oracle = experiment.get("oracle", {})
        oracle_image = oracle.get("container_image", "")
        if "@sha256:" not in oracle_image:
            raise ValueError("Research runs require a digest-pinned oracle.container_image.")
        if not isinstance(oracle.get("command"), list) or not oracle["command"]:
            raise ValueError("Research runs require an oracle.command argument list.")
        if not oracle.get("bundle_path"):
            raise ValueError("Research runs require an immutable oracle.bundle_path.")
        if not re.fullmatch(r"[a-f0-9]{64}", oracle.get("bundle_sha256", "")):
            raise ValueError("Research runs require a SHA-256 oracle.bundle_sha256.")
    if not isinstance(experiment["tasks"], list) or not experiment["tasks"]:
        raise ValueError("At least one task is required.")

    task_ids = set()
    producer_users = set()
    has_evaluation = False
    for task in experiment["tasks"]:
        required_task_fields = ["task_id", "split", "user_id", "prompt"]
        if purpose == "smoke_test_only_not_research_evidence":
            required_task_fields.append("test_command")
        for key in required_task_fields:
            if key not in task:
                raise ValueError(f"Task is missing required field {key!r}.")
        if task["task_id"] in task_ids:
            raise ValueError(f"Duplicate task ID: {task['task_id']}")
        task_ids.add(task["task_id"])
        if task["split"] not in ("producer", "evaluation"):
            raise ValueError(f"Task {task['task_id']} has an unknown split.")
        if (
            purpose == "smoke_test_only_not_research_evidence"
            and (not isinstance(task["test_command"], list) or not task["test_command"])
        ):
            raise ValueError(f"Task {task['task_id']} needs an objective test_command.")
        if task["split"] == "producer":
            producer_users.add(task["user_id"])
        else:
            has_evaluation = True

    if len(producer_users) < 2:
        raise ValueError("At least two independent producer users are required.")
    if not has_evaluation:
        raise ValueError("At least one held-out evaluation task is required.")
    overlapping_users = producer_users & {
        task["user_id"] for task in experiment["tasks"] if task["split"] == "evaluation"
    }
    if overlapping_users:
        raise ValueError("Evaluation users must be distinct from every producer user.")
    if int(experiment.get("repetitions", 1)) < 1:
        raise ValueError("repetitions must be at least one.")


def run_experiment(config: Dict[str, Any], config_base: Optional[Path] = None) -> Path:
    """Run all memory conditions and write a self-contained result directory."""
    experiment = config.get("experiment", config)
    validate_config(experiment)
    config_base = Path(config_base or Path(config.get("_config_path", ".")).parent).resolve()
    repository = Path(experiment["repository_path"])
    if not repository.is_absolute():
        repository = (config_base / repository).resolve()
    if not repository.is_dir():
        raise ValueError(f"Repository path does not exist: {repository}")

    purpose = experiment.get("purpose", "research_experiment")
    oracle_bundle_path = None
    oracle_bundle_hash = None
    if purpose != "smoke_test_only_not_research_evidence":
        oracle_bundle_path = Path(experiment["oracle"]["bundle_path"])
        if not oracle_bundle_path.is_absolute():
            oracle_bundle_path = (config_base / oracle_bundle_path).resolve()
        if not oracle_bundle_path.is_dir():
            raise ValueError(f"Oracle bundle directory does not exist: {oracle_bundle_path}")
        oracle_bundle_hash = _directory_hash(oracle_bundle_path)
        if oracle_bundle_hash != experiment["oracle"]["bundle_sha256"]:
            raise ValueError("Oracle bundle content does not match oracle.bundle_sha256.")

    commit = _git(repository, "rev-parse", "--verify", f"{experiment['repository_commit']}^{{commit}}").strip()
    repetitions = int(experiment.get("repetitions", 1))
    output_root = Path(experiment.get("output_dir", "experiments/runs"))
    if not output_root.is_absolute():
        output_root = (config_base / output_root).resolve()
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", experiment["name"]).strip("._") or "experiment"
    run_dir = output_root / f"{safe_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
    _create_run_layout(run_dir)

    sanitized_config = _redact(
        {key: value for key, value in config.items() if key != "_config_path"}
    )
    serialized_experiment = sanitized_config.get("experiment", sanitized_config)
    repository_path_value = serialized_experiment.get("repository_path")
    if repository_path_value and Path(repository_path_value).is_absolute():
        serialized_experiment["repository_path"] = "<local-repository>"
    oracle_data = serialized_experiment.get("oracle", {})
    if oracle_data.get("bundle_path") and Path(oracle_data["bundle_path"]).is_absolute():
        oracle_data["bundle_path"] = "<oracle-bundle>"
    config_payload = json.dumps(sanitized_config, sort_keys=True, default=str).encode("utf-8")
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(sanitized_config, sort_keys=False), encoding="utf-8"
    )
    tasks = experiment["tasks"]
    (run_dir / "task_manifest.json").write_text(
        json.dumps(tasks, indent=2, sort_keys=True), encoding="utf-8"
    )
    protocol_version = experiment.get("protocol_version", "unfrozen-draft")
    (run_dir / "protocol_version.txt").write_text(
        f"{protocol_version}\n", encoding="utf-8"
    )
    dependencies = {}
    for distribution in (
        "fastapi",
        "networkx",
        "pydantic",
        "pytest",
        "sqlalchemy",
        "PyYAML",
    ):
        try:
            dependencies[distribution] = version(distribution)
        except PackageNotFoundError:
            dependencies[distribution] = None
    manifest = {
        "run_id": run_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": hashlib.sha256(config_payload).hexdigest(),
        "repository_id": experiment.get("repository_id", experiment["project_id"]),
        "repository_commit": commit,
        "code_revision": _git(Path(__file__).resolve().parents[1], "rev-parse", "HEAD").strip(),
        "harness_source_sha256": _tree_hash(Path(__file__).resolve().parents[1]),
        "code_worktree_dirty": bool(
            _git(Path(__file__).resolve().parents[1], "status", "--porcelain").strip()
        ),
        "python_version": sys.version,
        "platform": platform.platform(),
        "agent_model": experiment["agent"].get("model", "unspecified"),
        "purpose": experiment.get("purpose", "research_experiment"),
        "oracle_bundle_sha256": oracle_bundle_hash,
        "protocol_version": protocol_version,
        "dependencies": dependencies,
        "seed": experiment.get("seed"),
        "repetitions": repetitions,
        "conditions": list(CONDITIONS),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    results = {"manifest": manifest, "conditions": {}}
    for condition in CONDITIONS:
        condition_result = _run_condition(
            condition=condition,
            experiment=experiment,
            tasks=tasks,
            repository=repository,
            commit=commit,
            run_dir=run_dir,
            repetitions=repetitions,
            oracle_bundle_path=oracle_bundle_path,
        )
        results["conditions"][condition] = condition_result

    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_metrics_csv(run_dir / "metrics.csv", results["conditions"])
    _write_json(
        run_dir / "analysis.json",
        analyze_results(
            results,
            bootstrap_samples=int(experiment.get("bootstrap_samples", 5000)),
            seed=int(experiment.get("analysis_seed", experiment.get("seed", 0))),
        ),
    )
    manifest["artifact_sha256"] = _artifact_hashes(run_dir, exclude={"manifest.json"})
    _write_json(run_dir / "manifest.json", manifest)
    return run_dir


def _run_condition(
    condition,
    experiment,
    tasks,
    repository,
    commit,
    run_dir,
    repetitions,
    oracle_bundle_path,
):
    condition_dir = run_dir / "conditions" / condition
    condition_dir.mkdir(parents=True, exist_ok=True)
    condition_results = {"producer": [], "evaluation": [], "runs": []}

    ordered_tasks = sorted(tasks, key=lambda task: task["split"] != "producer")
    for repetition in range(repetitions):
        graph_store = GraphStore()
        retrieval = KnowledgeRetrievalAPI(graph_store)
        worker = LearningWorker(graph_store)
        memory_entries = []
        run_results = {"producer": [], "evaluation": []}

        for task in ordered_tasks:
            context = _build_context(
                condition=condition,
                task=task,
                memory_entries=memory_entries,
                retrieval=retrieval,
                static_graph=_build_static_graph(repository, commit)
                if condition == "static_kg"
                else None,
                project_id=experiment["project_id"],
                commit=commit,
                repetition=repetition,
                run_id=run_dir.name,
            )
            task_result = _run_task(
                task={
                    **task,
                    "project_id": experiment["project_id"],
                    "repository": experiment.get("repository_id", experiment["project_id"]),
                    "seed": int(experiment.get("seed", 0)) + repetition,
                },
                context=context,
                condition=condition,
                repetition=repetition,
                agent=experiment["agent"],
                oracle=experiment.get("oracle", {}),
                purpose=experiment.get("purpose", "research_experiment"),
                oracle_bundle_path=oracle_bundle_path,
                repository=repository,
                commit=commit,
                run_dir=run_dir,
            )
            run_results[task["split"]].append(task_result["result"])
            _write_json(
                condition_dir / f"rep_{repetition}" / f"{task['task_id']}.json",
                task_result["artifact"],
            )
            _write_json(
                run_dir
                / "trajectories"
                / condition
                / f"rep_{repetition}"
                / f"{task['task_id']}.json",
                task_result["trajectory"].model_dump(mode="json"),
            )
            _write_json(
                run_dir
                / "outcomes"
                / condition
                / f"rep_{repetition}"
                / f"{task['task_id']}.json",
                task_result["result"],
            )

            if task["split"] == "producer":
                trajectory = task_result["trajectory"]
                memory_entries.append(_redacted_history_entry(trajectory))
                if condition == "central_learning":
                    import asyncio

                    candidates = asyncio.run(worker.process_trajectory(trajectory))
                    _write_json(
                        run_dir
                        / "candidate_knowledge"
                        / condition
                        / f"rep_{repetition}"
                        / f"{task['task_id']}.json",
                        [node.model_dump(mode="json") for node in candidates],
                    )

        run_results["graph"] = [node.model_dump(mode="json") for node in graph_store.get_all_nodes()]
        run_results["retrieval_events"] = graph_store.get_retrieval_events()
        _write_json(
            condition_dir / f"rep_{repetition}" / "learned_graph.json",
            run_results["graph"],
        )
        _write_json(
            condition_dir / f"rep_{repetition}" / "retrieval_events.json",
            run_results["retrieval_events"],
        )
        _write_json(
            run_dir / "learned_graph" / condition / f"rep_{repetition}.json",
            run_results["graph"],
        )
        _write_json(
            run_dir / "retrieval_events" / condition / f"rep_{repetition}.json",
            run_results["retrieval_events"],
        )
        condition_results["runs"].append(run_results)
        condition_results["producer"].extend(run_results["producer"])
        condition_results["evaluation"].extend(run_results["evaluation"])

    return condition_results


def _build_context(
    condition,
    task,
    memory_entries,
    retrieval,
    static_graph,
    project_id,
    commit,
    repetition,
    run_id,
):
    memory = {"kind": condition, "entries": []}
    if condition == "shared_memory":
        memory["entries"] = list(memory_entries)
    elif condition == "static_kg":
        memory["entries"] = static_graph
    elif condition == "central_learning" and task["split"] == "evaluation":
        procedures = retrieval.retrieve_procedures(
            project_id=project_id,
            task_description=task["prompt"],
            relevant_files=task.get("relevant_files"),
            repository_version=commit,
            consumer_user_id=task["user_id"],
            task_id=task["task_id"],
            experiment_run_id=f"{run_id}:rep_{repetition}",
        )
        memory["entries"] = [procedure.model_dump(mode="json") for procedure in procedures]
    return {"condition": condition, "memory": memory}


def _run_task(
    task,
    context,
    condition,
    repetition,
    agent,
    oracle,
    purpose,
    oracle_bundle_path,
    repository,
    commit,
    run_dir,
):
    worktree = run_dir / "worktrees" / condition / f"rep_{repetition}" / task["task_id"]
    worktree.parent.mkdir(parents=True, exist_ok=True)
    _git(repository, "worktree", "add", "--detach", str(worktree), commit)
    task_dir = run_dir / "conditions" / condition / f"rep_{repetition}"
    log_dir = run_dir / "logs" / condition / f"rep_{repetition}"
    task_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    agent_input = {
        "task": {"task_id": task["task_id"], "prompt": task["prompt"]},
        "context": context,
        "seed": task["seed"],
    }
    started = time.monotonic()
    timed_out = False
    process = None
    response = {}
    error = None

    try:
        command = [
            _format_argument(part, worktree, task["task_id"])
            for part in agent["command"]
        ]
        if agent.get("isolation") == "container":
            command = _container_command(command, agent, worktree)
        try:
            process = subprocess.run(
                command,
                cwd=worktree,
                input=json.dumps(agent_input),
                text=True,
                capture_output=True,
                env=_subprocess_env(agent.get("pass_env", [])),
                timeout=float(agent.get("timeout_seconds", 300)),
                check=False,
            )
            (log_dir / f"{task['task_id']}.stdout.log").write_text(process.stdout, encoding="utf-8")
            (log_dir / f"{task['task_id']}.stderr.log").write_text(process.stderr, encoding="utf-8")
            if process.returncode != 0:
                error = f"Agent exited with status {process.returncode}"
            else:
                response = json.loads(process.stdout)
                if not isinstance(response, dict):
                    raise ValueError("Agent output must be a JSON object.")
        except subprocess.TimeoutExpired:
            timed_out = True
            error = "Agent timed out"
        except OSError as exception:
            error = f"Could not start agent command: {exception}"
        except (json.JSONDecodeError, ValueError) as exception:
            error = f"Invalid agent response: {exception}"

        patch = response.get("patch", "") if error is None else ""
        patch_applied = False
        if patch:
            patch_process = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", "-"],
                cwd=worktree,
                input=patch,
                text=True,
                capture_output=True,
                check=False,
            )
            patch_applied = patch_process.returncode == 0
            if not patch_applied:
                error = f"Patch failed to apply: {patch_process.stderr.strip()}"

        test_result = None
        if error is None:
            if purpose != "smoke_test_only_not_research_evidence":
                test_command = [
                    _format_argument(
                        part,
                        worktree,
                        task["task_id"],
                        oracle_root="/oracle",
                    )
                    for part in oracle["command"]
                ]
                test_command = _oracle_command(
                    test_command,
                    oracle,
                    worktree,
                    task["task_id"],
                    oracle_bundle_path,
                )
            else:
                test_command = [
                    _format_argument(part, worktree, task["task_id"])
                    for part in task["test_command"]
                ]
            try:
                test_result = subprocess.run(
                    test_command,
                    cwd=worktree,
                    text=True,
                    capture_output=True,
                    timeout=float(task.get("test_timeout_seconds", 300)),
                    check=False,
                )
                (log_dir / f"{task['task_id']}.oracle.stdout.log").write_text(
                    test_result.stdout, encoding="utf-8"
                )
                (log_dir / f"{task['task_id']}.oracle.stderr.log").write_text(
                    test_result.stderr, encoding="utf-8"
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                error = "Objective test command timed out"
            except OSError as exception:
                error = f"Could not start objective test command: {exception}"

        successful = error is None and test_result is not None and test_result.returncode == 0
        elapsed = time.monotonic() - started
        trajectory_id = hashlib.sha256(
            f"{condition}:{repetition}:{task['task_id']}".encode("utf-8")
        ).hexdigest()
        trajectory = _make_trajectory(
            task,
            response,
            commit,
            successful,
            elapsed,
            error,
            trajectory_id,
        )
        procedure_ids = [entry.get("procedure_id") for entry in context["memory"]["entries"] if isinstance(entry, dict) and entry.get("procedure_id")]
        used_ids = set(response.get("used_procedure_ids", []))
        cross_user_available = any(
            entry.get("procedure_id") in procedure_ids
            and task["user_id"] not in entry.get("created_from_users", [])
            for entry in context["memory"]["entries"]
            if isinstance(entry, dict)
        )
        result = {
            "task_id": task["task_id"],
            "user_id": task["user_id"],
            "condition": condition,
            "repetition": repetition,
            "success": successful,
            "agent_error": error,
            "timed_out": timed_out,
            "oracle_timed_out": error == "Objective test command timed out",
            "patch_applied": patch_applied,
            "oracle_exit_code": test_result.returncode if test_result else None,
            "tool_call_count": int(response.get("tool_call_count", 0)),
            "token_usage": response.get("token_usage", {}),
            "latency_seconds": elapsed,
            "retrieved_procedure_ids": procedure_ids,
            "used_procedure_ids": sorted(used_ids & set(procedure_ids)),
            "cross_user_knowledge_available": cross_user_available,
            "cross_user_transfer_reported": bool(
                successful and cross_user_available and used_ids & set(procedure_ids)
            ),
        }
        artifact = {
            "task": {key: task[key] for key in ("task_id", "split", "user_id")},
            "input_context": context,
            "agent_response": response,
            "result": result,
            "trajectory": trajectory.model_dump(mode="json"),
        }
        return {"result": result, "artifact": artifact, "trajectory": trajectory}
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=repository,
            text=True,
            capture_output=True,
            check=False,
        )


def _make_trajectory(task, response, commit, successful, elapsed, error, trajectory_id):
    steps = []
    for index, step in enumerate(response.get("steps", [])):
        action = step.get("action", "") if isinstance(step, dict) else str(step)
        if action:
            steps.append(TrajectoryStep(step_id=f"{task['task_id']}_{index}", action=action))
    return Trajectory(
        trajectory_id=trajectory_id,
        user_id=task["user_id"],
        agent_id="command_adapter",
        project_id=task.get("project_id", ""),
        task_id=task["task_id"],
        repository=task.get("repository", ""),
        repository_commit=commit,
        messages=response.get("messages", []),
        steps=steps,
        files_touched=response.get("files_touched", []),
        errors=[error] if error else [],
        outcome="success" if successful else "failure",
        token_usage=response.get("token_usage", {}),
        latency=elapsed,
    )


def _redacted_history_entry(trajectory):
    """Expose action/error history without producer prompts, messages, results, or patches."""
    return json.dumps(
        {
            "source_user": trajectory.user_id,
            "outcome": trajectory.outcome,
            "actions": [step.action for step in trajectory.steps],
            "errors": list(trajectory.errors),
        },
        sort_keys=True,
    )


def _build_static_graph(repository, commit):
    files = _git(repository, "ls-tree", "-r", "--name-only", commit).splitlines()
    module_paths = {}
    for path in files:
        if path.endswith(".py"):
            module = path[:-3].replace("/", ".")
            if module.endswith(".__init__"):
                module = module[: -len(".__init__")]
            module_paths[module] = path

    relations = []
    for path in files:
        if not path.endswith(".py"):
            continue
        source = _git(repository, "show", f"{commit}:{path}")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            imported_modules = []
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
                imported_modules.extend(
                    f"{node.module}.{alias.name}"
                    for alias in node.names
                    if alias.name != "*"
                )
            for imported in imported_modules:
                target = module_paths.get(imported)
                if target is None:
                    target = module_paths.get(imported.rpartition(".")[0])
                if target and target != path:
                    relations.append({"from": path, "relation": "imports", "to": target})

    return {
        "files": files,
        "relations": relations,
    }


def _write_metrics_csv(path, conditions):
    fields = [
        "condition",
        "split",
        "successes",
        "total",
        "task_completion_rate",
        "cross_user_opportunities",
        "reported_cross_user_transfers",
        "reported_cross_user_transfer_rate",
        "mean_tool_calls",
        "mean_latency_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition, data in conditions.items():
            evaluation = data["evaluation"]
            opportunities = sum(item["cross_user_knowledge_available"] for item in evaluation)
            transfers = sum(item["cross_user_transfer_reported"] for item in evaluation)
            total = len(evaluation)
            writer.writerow(
                {
                    "condition": condition,
                    "split": "evaluation",
                    "successes": sum(item["success"] for item in evaluation),
                    "total": total,
                    "task_completion_rate": _rate(
                        sum(item["success"] for item in evaluation), total
                    ),
                    "cross_user_opportunities": opportunities,
                    "reported_cross_user_transfers": transfers,
                    "reported_cross_user_transfer_rate": _rate(transfers, opportunities),
                    "mean_tool_calls": _mean(item["tool_call_count"] for item in evaluation),
                    "mean_latency_seconds": _mean(item["latency_seconds"] for item in evaluation),
                }
            )


def _create_run_layout(run_dir):
    for folder in (
        "conditions",
        "logs",
        "worktrees",
        "trajectories",
        "candidate_knowledge",
        "learned_graph",
        "retrieval_events",
        "outcomes",
    ):
        (run_dir / folder).mkdir(parents=True, exist_ok=True)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _tree_hash(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if any(
            part in {"venv", ".venv", ".git", "__pycache__", ".pytest_cache"}
            for part in path.parts
        ):
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _artifact_hashes(root, exclude):
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in exclude:
            hashes[path.relative_to(root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return hashes


def _git(repository, *arguments):
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout


def _redact(value):
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if re.search(r"token|secret|password|api[_-]?key", key, re.I) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _rate(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _format_argument(value, worktree, task_id, oracle_root=""):
    runner_root = Path(__file__).resolve().parents[1]
    return (
        str(value)
        .replace("{workspace}", str(worktree))
        .replace("{task_id}", task_id)
        .replace("{runner_root}", str(runner_root))
        .replace("{runner_python}", sys.executable)
        .replace("{oracle_root}", oracle_root)
    )


def _subprocess_env(pass_env):
    allowed = {"PATH", "HOME", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL"}
    allowed.update(pass_env)
    return {key: value for key, value in os.environ.items() if key in allowed}


def _container_command(command, agent, worktree):
    runtime = agent.get("container_runtime", "docker")
    return [
        runtime,
        "run",
        "--rm",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=256",
        "--tmpfs=/tmp:rw,nosuid,nodev",
        "--mount",
        f"type=bind,source={worktree},target=/workspace",
        "--workdir=/workspace",
        agent["container_image"],
        *command,
    ]


def _oracle_command(command, oracle, worktree, task_id, oracle_bundle_path):
    runtime = oracle.get("container_runtime", "docker")
    return [
        runtime,
        "run",
        "--rm",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=128",
        "--tmpfs=/tmp:rw,nosuid,nodev",
        "--mount",
        f"type=bind,source={worktree},target=/workspace,readonly",
        "--mount",
        f"type=bind,source={oracle_bundle_path},target=/oracle,readonly",
        "--workdir=/workspace",
        oracle["container_image"],
        *[
            str(part)
            .replace("{task_id}", task_id)
            .replace("{workspace}", "/workspace")
            .replace("{oracle_root}", "/oracle")
            for part in command
        ],
    ]


def _directory_hash(root):
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()