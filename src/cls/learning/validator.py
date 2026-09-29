from typing import List, Optional

from cls.graph.schema import ProcedureNode
from cls.graph.store import GraphStore


class KnowledgeValidator:
    """Merge candidates without letting one user create team-level certainty."""

    def __init__(self, store: GraphStore):
        self.store = store

    def validate_and_merge(self, candidate: ProcedureNode) -> ProcedureNode:
        """Check whether a candidate is a duplicate and merge evidence."""
        existing_nodes = self.store.get_subgraph_by_scope(candidate.scope)
        match = self._find_semantic_match(candidate, existing_nodes)

        if match:
            match.evidence_count += candidate.evidence_count
            match.successful_executions += candidate.successful_executions
            match.failed_executions += candidate.failed_executions

            for user in candidate.created_from_users:
                if user not in match.created_from_users:
                    match.created_from_users.append(user)

            match.confidence = min(
                0.99,
                match.confidence + 0.1 * len(candidate.created_from_users),
            )
            return match

        candidate.confidence = 0.5
        return candidate

    def _find_semantic_match(
        self,
        candidate: ProcedureNode,
        existing: List[ProcedureNode],
    ) -> Optional[ProcedureNode]:
        """For the prototype, duplicate detection is intentionally simple."""
        for node in existing:
            if node.action == candidate.action:
                return node
        return None
