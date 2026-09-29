from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

from cls.graph.retrieval import KnowledgeRetrievalAPI
from cls.graph.store import GraphStore
from cls.ingestion.schemas import Trajectory
from cls.service.worker import LearningWorker

app = FastAPI(title="Central Learning Service")

store = GraphStore()
retrieval_api = KnowledgeRetrievalAPI(store)
worker = LearningWorker(store)


class RetrievalRequest(BaseModel):
    project_id: str
    task_description: str
    relevant_files: Optional[List[str]] = None


@app.post("/trajectories")
async def ingest_trajectory(trajectory: Trajectory, background_tasks: BackgroundTasks):
    """Queue a fresh trajectory so the learning pipeline can process it."""
    background_tasks.add_task(worker.process_trajectory, trajectory)
    return {"status": "accepted", "message": "Trajectory queued for learning."}


@app.post("/procedures/retrieve")
async def retrieve_procedures(req: RetrievalRequest):
    """Return the project-scoped procedure set most relevant to the task."""
    nodes = retrieval_api.retrieve_procedures(
        project_id=req.project_id,
        task_description=req.task_description,
        relevant_files=req.relevant_files,
    )
    return {"procedures": [node.model_dump() for node in nodes]}
