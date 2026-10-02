from datetime import datetime, timezone
from typing import Any, Dict

from cls.agents.base import BaseAgentAdapter
from cls.ingestion.schemas import Trajectory


class CodexAdapter(BaseAgentAdapter):
    """Demo adapter for a Codex-like agent interface.

    This is intentionally lightweight and meant to show the contract expected by the
    learning pipeline. It does not attempt to talk to the real Codex backend or to
    ingest live telemetry from a production agent.
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.agent_id = "codex_demo"
        self.active_tasks = {}

    def start_task(self, project_id: str, task_id: str) -> str:
        self.active_tasks[task_id] = {
            "project_id": project_id,
            "start_time": datetime.now(timezone.utc),
            "messages": [],
            "steps": [],
            "files_touched": [],
            "errors": [],
        }
        return task_id

    def record_message(self, task_id: str, role: str, content: str) -> None:
        if task_id in self.active_tasks:
            self.active_tasks[task_id]["messages"].append({"role": role, "content": content})

    def record_tool_call(self, task_id: str, tool_name: str, args: Dict[str, Any]) -> None:
        """Demo hook: capture the call shape without pretending to talk to Codex."""

    def record_tool_result(self, task_id: str, result: Any, is_error: bool) -> None:
        """Demo hook: accept tool output and keep the adapter compatible with the pipeline."""

    def finish_task(self, task_id: str, outcome: str) -> Trajectory:
        """Return a trajectory built from the in-memory demo session."""
        if task_id not in self.active_tasks:
            raise ValueError(f"Task {task_id} not found.")

        data = self.active_tasks[task_id]
        latency = (datetime.now(timezone.utc) - data["start_time"]).total_seconds()

        trajectory = Trajectory(
            user_id=self.user_id,
            agent_id=self.agent_id,
            project_id=data["project_id"],
            task_id=task_id,
            repository="demo/repo",
            repository_commit="demo-head",
            messages=data["messages"],
            steps=data["steps"],
            files_touched=data["files_touched"],
            errors=data["errors"],
            outcome=outcome,
            token_usage={},
            latency=latency,
        )
        del self.active_tasks[task_id]
        return trajectory
