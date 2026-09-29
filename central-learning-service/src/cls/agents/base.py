from abc import ABC, abstractmethod
from typing import Any, Dict

from cls.ingestion.schemas import Trajectory


class BaseAgentAdapter(ABC):
    """Common interface for all agent implementations."""

    @abstractmethod
    def start_task(self, project_id: str, task_id: str) -> str:
        """Begin tracking a new task and return its task identifier."""

    @abstractmethod
    def record_tool_call(self, task_id: str, tool_name: str, args: Dict[str, Any]) -> None:
        """Record a tool invocation made by the agent."""

    @abstractmethod
    def record_tool_result(self, task_id: str, result: Any, is_error: bool) -> None:
        """Record the result of a tool call, including whether it failed."""

    @abstractmethod
    def record_message(self, task_id: str, role: str, content: str) -> None:
        """Store a message such as user input, tool output, or model output."""

    @abstractmethod
    def finish_task(self, task_id: str, outcome: str) -> Trajectory:
        """Finalize the task and return the serialized trajectory."""
