from typing import List, Optional
from datetime import datetime, timezone

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
            existing_trajectory_ids = set(match.source_trajectory_ids)
            candidate_trajectory_ids = set(candidate.source_trajectory_ids)
            if existing_trajectory_ids & candidate_trajectory_ids:
                return match

            known_evidence_ids = {evidence.trajectory_id for evidence in match.evidence}
            new_evidence = [
                evidence
                for evidence in candidate.evidence
                if evidence.trajectory_id not in known_evidence_ids
            ]
            match.evidence.extend(new_evidence)
            match.evidence_count += candidate.evidence_count
            match.successful_executions += candidate.successful_executions
            match.failed_executions += candidate.failed_executions

            for user in candidate.created_from_users:
                if user not in match.created_from_users:
                    match.created_from_users.append(user)

            match.source_trajectory_ids = list(
                dict.fromkeys(match.source_trajectory_ids + candidate.source_trajectory_ids)
            )
            match.updated_at = datetime.now(timezone.utc)
            match.validation_status = self._validation_status(match)
            total_executions = match.successful_executions + match.failed_executions
            match.confidence = (
                match.successful_executions / total_executions if total_executions else 0.0
            )
            return match

        total_executions = candidate.successful_executions + candidate.failed_executions
        candidate.confidence = (
            candidate.successful_executions / total_executions if total_executions else 0.0
        )
        candidate.validation_status = self._validation_status(candidate)
        return candidate

    @staticmethod
    def _validation_status(candidate: ProcedureNode) -> str:
        """Require independent users and distinct trajectory evidence to promote."""
        successful_evidence = [evidence for evidence in candidate.evidence if evidence.successful]
        if successful_evidence:
            users = {evidence.user_id for evidence in successful_evidence}
            trajectory_ids = {evidence.trajectory_id for evidence in successful_evidence}
        else:
            users = set(candidate.created_from_users)
            trajectory_ids = set(candidate.source_trajectory_ids)
        if len(users) >= 2 and len(trajectory_ids) >= 2:
            return "validated"
        return "candidate"

    def _find_semantic_match(
        self,
        candidate: ProcedureNode,
        existing: List[ProcedureNode],
    ) -> Optional[ProcedureNode]:
        """For the prototype, duplicate detection is intentionally simple."""
        for node in existing:
            if (
                node.action == candidate.action
                and node.repository_version == candidate.repository_version
            ):
                return node
        return None
