from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Single tool invocation inside a task step."""

    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    is_error: bool


class TrajectoryStep(BaseModel):
    """A single step in an agent's recorded execution path."""

    step_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    action: str
    tool_calls: List[ToolCall] = Field(default_factory=list)


class Trajectory(BaseModel):
    """Full record of an agent's work for a single task."""

    user_id: str
    agent_id: str
    project_id: str
    task_id: str
    repository: str
    repository_commit: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    steps: List[TrajectoryStep] = Field(default_factory=list)
    files_touched: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    outcome: str
    token_usage: Dict[str, int] = Field(default_factory=dict)
    latency: float
