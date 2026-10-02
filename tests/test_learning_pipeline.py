import asyncio

from cls.graph.retrieval import KnowledgeRetrievalAPI
from cls.graph.store import GraphStore
from cls.ingestion.schemas import Trajectory, TrajectoryStep
from cls.ingestion.store import TrajectoryStore, get_engine
from cls.service.worker import LearningWorker


def make_trajectory(user_id: str, trajectory_id: str) -> Trajectory:
    return Trajectory(
        trajectory_id=trajectory_id,
        user_id=user_id,
        agent_id="test_agent",
        project_id="project_a",
        task_id=f"task_{user_id}",
        repository="example/repo",
        repository_commit="commit_a",
        messages=[{"role": "assistant", "content": "private producer transcript"}],
        steps=[TrajectoryStep(step_id=f"step_{user_id}", action="inspect module")],
        outcome="success",
        latency=1.0,
    )


def make_failed_trajectory(user_id: str, trajectory_id: str) -> Trajectory:
    return Trajectory(
        trajectory_id=trajectory_id,
        user_id=user_id,
        agent_id="test_agent",
        project_id="project_a",
        task_id=f"task_{user_id}",
        repository="example/repo",
        repository_commit="commit_a",
        steps=[TrajectoryStep(step_id=f"step_{user_id}", action="inspect module")],
        errors=["objective check failed"],
        outcome="failure",
        latency=1.0,
    )


def test_durable_queue_learns_from_two_users_without_exposing_transcripts(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path / 'pipeline.db'}")
    trajectory_store = TrajectoryStore(engine)
    graph_store = GraphStore(engine=engine)
    worker = LearningWorker(graph_store, trajectory_store)

    first = make_trajectory("user_a", "trajectory_a")
    second = make_trajectory("user_b", "trajectory_b")
    assert trajectory_store.enqueue(first)
    assert trajectory_store.enqueue(second)

    assert asyncio.run(worker.run_once()) is True
    assert asyncio.run(worker.run_once()) is True
    assert trajectory_store.get_status("trajectory_a") == "completed"
    assert trajectory_store.get_status("trajectory_b") == "completed"

    procedures = KnowledgeRetrievalAPI(graph_store).retrieve_procedures(
        project_id="project_a",
        task_description="inspect module",
        repository_version="commit_a",
    )

    assert len(procedures) == 1
    assert set(procedures[0].created_from_users) == {"user_a", "user_b"}
    assert set(procedures[0].source_trajectory_ids) == {"trajectory_a", "trajectory_b"}
    assert "private producer transcript" not in str(procedures[0].model_dump())


def test_failed_repetition_is_saved_as_counter_evidence(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path / 'counter-evidence.db'}")
    graph_store = GraphStore(engine=engine)
    worker = LearningWorker(graph_store)

    import asyncio

    asyncio.run(worker.process_trajectory(make_trajectory("user_a", "success_a")))
    asyncio.run(worker.process_trajectory(make_trajectory("user_b", "success_b")))
    asyncio.run(
        worker.process_trajectory(make_failed_trajectory("user_c", "failure_c"))
    )

    procedure = graph_store.get_all_nodes()[0]
    assert procedure.successful_executions == 2
    assert procedure.failed_executions == 1
    assert len(procedure.evidence) == 3
    assert procedure.evidence[-1].successful is False
    assert procedure.confidence == 2 / 3
    assert KnowledgeRetrievalAPI(graph_store).retrieve_procedures(
        "project_a", "inspect module", repository_version="commit_a"
    ) == []