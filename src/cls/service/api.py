import asyncio
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from pydantic import BaseModel

from cls.graph.retrieval import KnowledgeRetrievalAPI
from cls.graph.store import GraphStore
from cls.ingestion.schemas import Trajectory
from cls.ingestion.store import TrajectoryStore, get_engine
from cls.service.worker import LearningWorker

router = APIRouter()


class RetrievalRequest(BaseModel):
    project_id: str
    repository_version: str
    task_description: str
    relevant_files: Optional[List[str]] = None
    consumer_user_id: Optional[str] = None
    task_id: Optional[str] = None
    experiment_run_id: Optional[str] = None


def create_app(database_url: Optional[str] = None, worker_poll_interval: float = 1.0) -> FastAPI:
    """Build the API and its durable stores for one service process."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = get_engine(
            database_url or os.getenv("DATABASE_URL", "sqlite:///./procedural_graph_mesh.db")
        )
        trajectory_store = TrajectoryStore(engine)
        trajectory_store.recover_interrupted()
        graph_store = GraphStore(engine=engine)
        worker = LearningWorker(
            graph_store,
            trajectory_store,
            poll_interval=worker_poll_interval,
        )
        stop_event = asyncio.Event()
        worker_task = asyncio.create_task(worker.run_forever(stop_event))

        app.state.engine = engine
        app.state.trajectory_store = trajectory_store
        app.state.graph_store = graph_store
        app.state.retrieval_api = KnowledgeRetrievalAPI(graph_store)
        app.state.worker = worker

        try:
            yield
        finally:
            stop_event.set()
            await worker_task
            engine.dispose()

    application = FastAPI(title="Procedural Graph Mesh for Team Agents", lifespan=lifespan)
    application.include_router(router)
    return application


app = create_app()


@router.post("/trajectories", status_code=status.HTTP_202_ACCEPTED)
async def ingest_trajectory(trajectory: Trajectory, request: Request):
    """Persist a trajectory before acknowledging it for background processing."""
    queued = await asyncio.to_thread(request.app.state.trajectory_store.enqueue, trajectory)
    if not queued:
        existing_status = await asyncio.to_thread(
            request.app.state.trajectory_store.get_status,
            trajectory.trajectory_id,
        )
        return {
            "status": "already_received",
            "trajectory_id": trajectory.trajectory_id,
            "processing_status": existing_status,
        }
    return {"status": "queued", "trajectory_id": trajectory.trajectory_id}


@router.get("/trajectories/{trajectory_id}")
async def get_trajectory_status(trajectory_id: str, request: Request):
    """Return the durable processing state without exposing trajectory contents."""
    processing_status = await asyncio.to_thread(
        request.app.state.trajectory_store.get_status,
        trajectory_id,
    )
    if processing_status is None:
        raise HTTPException(status_code=404, detail="Trajectory not found")
    return {"trajectory_id": trajectory_id, "status": processing_status}


@router.post("/procedures/retrieve")
async def retrieve_procedures(req: RetrievalRequest, request: Request):
    """Return the project-scoped procedure set most relevant to the task."""
    nodes = await asyncio.to_thread(
        request.app.state.retrieval_api.retrieve_procedures,
        req.project_id,
        req.task_description,
        req.relevant_files,
        req.repository_version,
        req.consumer_user_id,
        req.task_id,
        req.experiment_run_id,
    )
    return {"procedures": [node.model_dump() for node in nodes]}


@router.get("/health")
async def health():
    return {"status": "ok"}
