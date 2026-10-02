import uuid
from typing import Any, Dict, List

from cls.graph.schema import ProcedureEvidence, ProcedureNode


class KnowledgeExtractor:
    """Turn a successful episode into a candidate procedure."""

    def extract_candidates(self, episode: Dict[str, Any]) -> List[ProcedureNode]:
        """Create a candidate or counter-evidence from an observed episode."""
        successful = episode.get("outcome") == "success"
        if successful:
            observed_actions = episode.get("successful_actions") or []
        else:
            observed_actions = (
                episode.get("failed_actions")
                or episode.get("execution_sequence")
                or []
            )
        if not observed_actions:
            return []

        observed_steps = [
            step.action if hasattr(step, "action") else str(step)
            for step in observed_actions
        ]
        observed_steps = [action.strip() for action in observed_steps if action.strip()]
        if not observed_steps:
            return []

        node = ProcedureNode(
            procedure_id=f"PROC_{uuid.uuid4().hex[:8]}",
            action=" -> ".join(observed_steps),
            scope=episode["project_id"],
            preconditions=[],
            guidance=observed_steps,
            evidence_count=1,
            successful_executions=1 if successful else 0,
            failed_executions=0 if successful else 1,
            confidence=1.0 if successful else 0.0,
            repository_version=episode["repository_commit"],
            created_from_users=[episode["user_id"]],
            source_trajectory_ids=[episode["trajectory_id"]],
            evidence=[
                ProcedureEvidence(
                    trajectory_id=episode["trajectory_id"],
                    user_id=episode["user_id"],
                    repository_version=episode["repository_commit"],
                    successful=successful,
                    step_ids=[
                        step.step_id
                        for step in observed_actions
                        if hasattr(step, "step_id")
                    ],
                )
            ],
        )
        return [node]
