from datetime import datetime
from typing import List, Optional

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
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ProcedureEdge(BaseModel):
    """Relationship between two procedures in the graph."""

    source_id: str
    target_id: str
    relation: str
    confidence: float = 0.0
    precondition: Optional[str] = None
    evidence_count: int = 0
