from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ProcedureNode(BaseModel):
    """A reusable procedure distilled from successful agent work."""

    procedure_id: str
    action: str
    scope: str
    preconditions: List[str] = Field(default_factory=list)
    guidance: List[str] = Field(default_factory=list)
    evidence_count: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    confidence: float = 0.0
    repository_version: str
    created_from_users: List[str] = Field(default_factory=list)
    source_trajectory_ids: List[str] = Field(default_factory=list)
    evidence: List["ProcedureEvidence"] = Field(default_factory=list)
    validation_status: Literal["candidate", "validated", "rejected", "stale", "retired"] = "candidate"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ProcedureEvidence(BaseModel):
    """One independently attributable observation supporting a procedure."""

    trajectory_id: str
    user_id: str
    repository_version: str
    successful: bool
    step_ids: List[str] = Field(default_factory=list)


class ProcedureEdge(BaseModel):
    """Relationship between two procedures in the graph."""

    source_id: str
    target_id: str
    relation: str
    confidence: float = 0.0
    precondition: Optional[str] = None
    evidence_count: int = 0
