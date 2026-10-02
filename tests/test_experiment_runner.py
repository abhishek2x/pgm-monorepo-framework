import json
import os
import subprocess
import sys
import pytest

from experiments.runner import (
    _container_command,
    _directory_hash,
    _oracle_command,
    _tree_hash,
    run_experiment,
    validate_config,
)


def make_git_repository(path):
    path.mkdir()
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull}
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True, env=env)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True, env=env)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True, env=env)
    (path / "README.md").write_text("fixture repository\n", encoding="utf-8")
    (path / "src").mkdir()
    (path / "src/a.py").write_text("from src import b\n", encoding="utf-8")
    (path / "src/b.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "README.md", "src"], check=True, env=env)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "initial"], check=True, capture_output=True, env=env)
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, env=env
    ).strip()


def test_experiment_runner_writes_auditable_four_arm_run(tmp_path):
    repository = tmp_path / "repo"
    commit = make_git_repository(repository)
    agent_script = (
        "import json,sys; json.load(sys.stdin); "
        "print(json.dumps({'steps':[{'action':'inspect module'}], "
        "'messages':[{'role':'assistant','content':'private producer text'}], 'patch':''}))"
    )
    config = {
        "experiment": {
            "name": "fixture_transfer",
            "purpose": "smoke_test_only_not_research_evidence",
            "project_id": "fixture_project",
            "repository_path": str(repository),
            "repository_commit": commit,
            "seed": 7,
            "repetitions": 1,
            "output_dir": str(tmp_path / "runs"),
            "agent": {
                "command": [sys.executable, "-c", agent_script],
                "model": "fixture-only",
                "timeout_seconds": 10,
            },
            "tasks": [
                {
                    "task_id": "evaluation_c",
                    "split": "evaluation",
                    "user_id": "user_c",
                    "prompt": "Inspect the module.",
                    "test_command": [sys.executable, "-c", "pass"],
                },
                {
                    "task_id": "evaluation_timeout",
                    "split": "evaluation",
                    "user_id": "user_c",
                    "prompt": "This objective check should time out.",
                    "test_command": [
                        sys.executable,
                        "-c",
                        "import time; time.sleep(0.1)",
                    ],
                    "test_timeout_seconds": 0.01,
                },
                {
                    "task_id": "producer_a",
                    "split": "producer",
                    "user_id": "user_a",
                    "prompt": "Inspect the module.",
                    "test_command": [sys.executable, "-c", "pass"],
                },
                {
                    "task_id": "producer_b",
                    "split": "producer",
                    "user_id": "user_b",
                    "prompt": "Inspect the module.",
                    "test_command": [sys.executable, "-c", "pass"],
                },
            ],
        }
    }

    run_dir = run_experiment(config, config_base=tmp_path)

    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert set(results["conditions"]) == {
        "independent",
        "shared_memory",
        "static_kg",
        "central_learning",
    }
    assert len(results["conditions"]["central_learning"]["evaluation"]) == 2
    timeout_result = next(
        row
        for row in results["conditions"]["central_learning"]["evaluation"]
        if row["task_id"] == "evaluation_timeout"
    )
    assert timeout_result["success"] is False
    assert timeout_result["oracle_timed_out"] is True

    d_context = json.loads(
        (
            run_dir
            / "conditions/central_learning/rep_0/evaluation_c.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    serialized_context = json.dumps(d_context["input_context"])
    assert "inspect module" in serialized_context
    assert "private producer text" not in serialized_context
    assert "patch" not in d_context["input_context"]["memory"]

    b_context = json.loads(
        (run_dir / "conditions/shared_memory/rep_0/evaluation_c.json").read_text(
            encoding="utf-8"
        )
    )["input_context"]["memory"]["entries"]
    assert b_context
    assert all(isinstance(entry, str) for entry in b_context)
    assert "private producer text" not in json.dumps(b_context)

    c_context = json.loads(
        (run_dir / "conditions/static_kg/rep_0/evaluation_c.json").read_text(
            encoding="utf-8"
        )
    )["input_context"]["memory"]["entries"]
    assert any(edge["relation"] == "imports" for edge in c_context["relations"])

    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "analysis.json").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol_version"] == "unfrozen-draft"
    assert "results.json" in manifest["artifact_sha256"]
    saved_config = (run_dir / "config.yaml").read_text(encoding="utf-8")
    assert str(repository) not in saved_config


def test_research_config_requires_container_isolation():
    with pytest.raises(ValueError, match="container_image"):
        validate_config(
            {
                "name": "confirmatory_run",
                "project_id": "project_a",
                "repository_path": ".",
                "repository_commit": "HEAD",
                "tasks": [],
                "agent": {"command": ["agent"]},
            }
        )


def test_research_config_requires_digest_pinned_objective_oracle():
    with pytest.raises(ValueError, match="oracle.container_image"):
        validate_config(
            {
                "name": "confirmatory_run",
                "project_id": "project_a",
                "repository_path": ".",
                "repository_commit": "HEAD",
                "tasks": [],
                "agent": {
                    "command": ["agent"],
                    "isolation": "container",
                    "container_image": "agent@sha256:abc",
                },
                "oracle": {"container_image": "oracle:latest"},
            }
        )


def test_research_config_requires_digest_pinned_agent_image():
    with pytest.raises(ValueError, match="agent isolation and a digest-pinned"):
        validate_config(
            {
                "name": "confirmatory_run",
                "project_id": "project_a",
                "repository_path": ".",
                "repository_commit": "HEAD",
                "tasks": [],
                "agent": {
                    "command": ["agent"],
                    "isolation": "container",
                    "container_image": "agent:latest",
                },
            }
        )


def test_container_mounts_only_the_task_worktree_without_network():
    command = _container_command(
        ["agent", "--stdin"],
        {"container_image": "agent-image@sha256:abc"},
        "/tmp/worktree",
    )

    assert "--network=none" in command
    assert "--read-only" in command
    assert "type=bind,source=/tmp/worktree,target=/workspace" in command
    assert command[-3:] == ["agent-image@sha256:abc", "agent", "--stdin"]

    oracle = _oracle_command(
        ["pytest", "/oracle/tests/test_{task_id}.py", "--workspace={workspace}"],
        {
            "container_image": "oracle@sha256:def",
            "bundle_path": "/tmp/oracle-bundle",
        },
        "/tmp/worktree",
        "task_a",
        "/tmp/oracle-bundle",
    )
    assert "--network=none" in oracle
    assert "type=bind,source=/tmp/worktree,target=/workspace,readonly" in oracle
    assert "type=bind,source=/tmp/oracle-bundle,target=/oracle,readonly" in oracle
    assert oracle[-3:] == [
        "pytest",
        "/oracle/tests/test_task_a.py",
        "--workspace=/workspace",
    ]


def test_oracle_bundle_hash_is_stable_for_directory_contents(tmp_path):
    bundle = tmp_path / "oracle"
    bundle.mkdir()
    (bundle / "case.py").write_text("assert True\n", encoding="utf-8")
    first_hash = _directory_hash(bundle)
    (bundle / "case.py").write_text("assert False\n", encoding="utf-8")

    assert _directory_hash(bundle) != first_hash


def test_harness_fingerprint_ignores_virtual_environment_sources(tmp_path):
    source = tmp_path / "src.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    environment_source = tmp_path / ".venv/lib/package.py"
    environment_source.parent.mkdir(parents=True)
    environment_source.write_text("THIRD_PARTY = 1\n", encoding="utf-8")
    first_hash = _tree_hash(tmp_path)

    environment_source.write_text("THIRD_PARTY = 2\n", encoding="utf-8")

    assert _tree_hash(tmp_path) == first_hash