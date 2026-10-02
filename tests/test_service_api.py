import time

from fastapi.testclient import TestClient

from cls.ingestion.schemas import Trajectory, TrajectoryStep
from cls.service.api import create_app


def make_trajectory(user_id: str, trajectory_id: str) -> dict:
    trajectory = Trajectory(
        trajectory_id=trajectory_id,
        user_id=user_id,
        agent_id="test_agent",
        project_id="project_a",
        task_id=f"task_{user_id}",
        repository="example/repo",
        repository_commit="commit_a",
        steps=[TrajectoryStep(step_id=f"step_{user_id}", action="inspect module")],
        outcome="success",
        latency=1.0,
    )
    return trajectory.model_dump(mode="json")


def wait_for_status(client, trajectory_id: str, expected: str) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        response = client.get(f"/trajectories/{trajectory_id}")
        if response.status_code == 200 and response.json()["status"] == expected:
            return
        time.sleep(0.01)
    raise AssertionError(f"Trajectory {trajectory_id} did not reach {expected!r}")


def test_api_persists_processes_and_audits_cross_user_retrieval(tmp_path):
    app = create_app(f"sqlite:///{tmp_path / 'api.db'}", worker_poll_interval=0.01)
    with TestClient(app) as client:
        first = make_trajectory("user_a", "trajectory_a")
        second = make_trajectory("user_b", "trajectory_b")

        assert client.post("/trajectories", json=first).json()["status"] == "queued"
        duplicate = client.post("/trajectories", json=first).json()
        assert duplicate["status"] == "already_received"
        assert client.post("/trajectories", json=second).json()["status"] == "queued"

        wait_for_status(client, "trajectory_a", "completed")
        wait_for_status(client, "trajectory_b", "completed")

        response = client.post(
            "/procedures/retrieve",
            json={
                "project_id": "project_a",
                "task_description": "inspect module",
                "repository_version": "commit_a",
                "consumer_user_id": "user_c",
                "task_id": "evaluation_task",
                "experiment_run_id": "run_1",
            },
        )
        procedures = response.json()["procedures"]
        assert len(procedures) == 1
        assert set(procedures[0]["created_from_users"]) == {"user_a", "user_b"}

        events = app.state.graph_store.get_retrieval_events()
        assert len(events) == 1
        assert events[0]["consumer_user_id"] == "user_c"
        assert events[0]["procedure_ids"] == [procedures[0]["procedure_id"]]