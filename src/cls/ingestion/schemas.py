from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import uuid4

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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    action: str
    tool_calls: List[ToolCall] = Field(default_factory=list)


class Trajectory(BaseModel):
    """Full record of an agent's work for a single task."""

    schema_version: str = "1.0"
    trajectory_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    agent_id: str
    project_id: str
    task_id: str
    repository: str
    repository_commit: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    steps: List[TrajectoryStep] = Field(default_factory=list)
    files_touched: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    outcome: str
    token_usage: Dict[str, int] = Field(default_factory=dict)
    latency: float
