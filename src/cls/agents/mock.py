import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from cls.agents.base import BaseAgentAdapter
from cls.ingestion.schemas import ToolCall, Trajectory, TrajectoryStep


class MockAgentAdapter(BaseAgentAdapter):
    """Simple deterministic adapter used in tests and local experiments."""

    def __init__(self, user_id: str = "test_user"):
        self.user_id = user_id
        self.agent_id = "mock_agent_1.0"
        self.active_tasks: Dict[str, dict] = {}

    def start_task(self, project_id: str, task_id: str) -> str:
        self.active_tasks[task_id] = {
            "project_id": project_id,
            "start_time": datetime.now(timezone.utc),
            "messages": [],
            "steps": [],
            "files_touched": [],
            "errors": [],
            "current_step": None,
        }
        return task_id

    def record_message(self, task_id: str, role: str, content: str) -> None:
        if task_id in self.active_tasks:
            self.active_tasks[task_id]["messages"].append({"role": role, "content": content})

    def record_tool_call(self, task_id: str, tool_name: str, args: Dict[str, Any]) -> None:
        if task_id in self.active_tasks:
            step = TrajectoryStep(
                step_id=str(uuid.uuid4()),
                action=f"Call {tool_name}",
                tool_calls=[],
            )
            self.active_tasks[task_id]["current_step"] = {
                "step": step,
                "tool_name": tool_name,
                "args": args,
            }

    def record_tool_result(self, task_id: str, result: Any, is_error: bool) -> None:
        if task_id in self.active_tasks and self.active_tasks[task_id].get("current_step"):
            current = self.active_tasks[task_id]["current_step"]
            tool_call = ToolCall(
                tool_name=current["tool_name"],
                arguments=current["args"],
                result=result,
                is_error=is_error,
            )
            if is_error:
                self.active_tasks[task_id]["errors"].append(str(result))

            step = current["step"]
            step.tool_calls.append(tool_call)
            self.active_tasks[task_id]["steps"].append(step)
            self.active_tasks[task_id]["current_step"] = None

    def finish_task(self, task_id: str, outcome: str) -> Trajectory:
        if task_id not in self.active_tasks:
            raise ValueError(f"Task {task_id} not found.")

        data = self.active_tasks[task_id]
        latency = (datetime.now(timezone.utc) - data["start_time"]).total_seconds()

        trajectory = Trajectory(
            user_id=self.user_id,
            agent_id=self.agent_id,
            project_id=data["project_id"],
            task_id=task_id,
            repository="mock/repo",
            repository_commit="abc1234",
            messages=data["messages"],
            steps=data["steps"],
            files_touched=data["files_touched"],
            errors=data["errors"],
            outcome=outcome,
            token_usage={"prompt": 100, "completion": 50},
            latency=latency,
        )
        del self.active_tasks[task_id]
        return trajectory
