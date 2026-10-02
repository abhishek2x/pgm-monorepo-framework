import pytest

from cls.graph.schema import ProcedureNode
from cls.graph.retrieval import KnowledgeRetrievalAPI
from cls.graph.store import GraphStore
from cls.ingestion.schemas import Trajectory
from cls.ingestion.store import TrajectoryStore, get_engine


def make_trajectory(trajectory_id: str) -> Trajectory:
    return Trajectory(
        trajectory_id=trajectory_id,
        user_id="user_a",
        agent_id="test_agent",
        project_id="project_a",
        task_id="task_a",
        repository="example/repo",
        repository_commit="commit_a",
        outcome="success",
        latency=1.0,
    )


def test_trajectory_queue_is_idempotent_and_survives_restart(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path / 'service.db'}")
    first_store = TrajectoryStore(engine)
    trajectory = make_trajectory("trajectory_a")

    assert first_store.enqueue(trajectory) is True
    assert first_store.enqueue(trajectory) is False
    persisted = first_store.get_trajectory(trajectory.trajectory_id)
    assert persisted is not None
    assert persisted.user_id == "user_a"
    assert persisted.repository_commit == "commit_a"

    conflicting = make_trajectory("trajectory_a")
    conflicting.task_id = "different_task"
    with pytest.raises(ValueError, match="already exists with different content"):
        first_store.enqueue(conflicting)

    claimed = first_store.claim_next()
    assert claimed is not None
    assert claimed.trajectory_id == trajectory.trajectory_id
    first_store.mark_completed(trajectory.trajectory_id)

    second_store = TrajectoryStore(engine)
    assert second_store.get_status(trajectory.trajectory_id) == "completed"


def test_failed_jobs_retry_then_stop_at_attempt_limit(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path / 'retries.db'}")
    store = TrajectoryStore(engine)
    trajectory = make_trajectory("retry_me")
    store.enqueue(trajectory)

    assert store.claim_next() is not None
    store.mark_failed(trajectory.trajectory_id, "temporary failure", max_attempts=2)
    assert store.get_status(trajectory.trajectory_id) == "pending"

    assert store.claim_next() is not None
    store.mark_failed(trajectory.trajectory_id, "permanent failure", max_attempts=2)
    assert store.get_status(trajectory.trajectory_id) == "failed"


def test_graph_nodes_survive_store_recreation(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path / 'service.db'}")
    first_store = GraphStore(engine=engine)
    node = ProcedureNode(
        procedure_id="procedure_a",
        action="inspect source then run focused tests",
        scope="project_a",
        repository_version="commit_a",
        created_from_users=["user_a", "user_b"],
        source_trajectory_ids=["trajectory_a", "trajectory_b"],
        evidence_count=2,
        successful_executions=2,
        validation_status="validated",
    )

    first_store.add_node(node)
    promoted = node.model_copy(update={"validation_status": "retired"})
    first_store.update_node(promoted)
    restored_store = GraphStore(engine=engine)

    assert restored_store.get_node("procedure_a") == promoted


def test_retrieval_is_version_scoped_and_audited_even_when_empty(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path / 'audit.db'}")
    graph_store = GraphStore(engine=engine)
    node = ProcedureNode(
        procedure_id="procedure_a",
        action="run focused tests",
        scope="project_a",
        repository_version="commit_a",
        created_from_users=["user_a", "user_b"],
        source_trajectory_ids=["trajectory_a", "trajectory_b"],
        evidence_count=2,
        successful_executions=2,
        validation_status="validated",
    )
    graph_store.add_node(node)
    retrieval = KnowledgeRetrievalAPI(graph_store)

    assert retrieval.retrieve_procedures(
        project_id="project_a",
        task_description="run tests",
        repository_version="commit_b",
        consumer_user_id="user_c",
        task_id="evaluation_task",
        experiment_run_id="run_1",
    ) == []

    events = graph_store.get_retrieval_events()
    assert len(events) == 1
    assert events[0]["consumer_user_id"] == "user_c"
    assert events[0]["experiment_run_id"] == "run_1"
    assert events[0]["procedure_ids"] == []