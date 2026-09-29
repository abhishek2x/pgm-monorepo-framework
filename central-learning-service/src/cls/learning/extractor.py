import uuid
from typing import Any, Dict, List

from cls.graph.schema import ProcedureNode


class KnowledgeExtractor:
    """Turn a successful episode into a candidate procedure."""

    def extract_candidates(self, episode: Dict[str, Any]) -> List[ProcedureNode]:
        """Create a procedure node when the episode ended in a success."""
        if episode.get("outcome") != "success" or not episode.get("successful_actions"):
            return []

        files_touched = ", ".join(episode.get("files_touched") or ["repository"])
        successful_step_count = len(episode.get("successful_actions", []))

        node = ProcedureNode(
            procedure_id=f"PROC_{uuid.uuid4().hex[:8]}",
            action=f"complete task on {files_touched}",
            scope=episode["project_id"],
            preconditions=[],
            guidance=[
                f"Succeeded after {successful_step_count} successful step(s).",
                f"Task ID: {episode.get('task_id', 'unknown')}",
            ],
            evidence_count=1,
            successful_executions=1,
            failed_executions=0,
            confidence=0.5,
            repository_version=episode["repository_commit"],
            created_from_users=[episode["user_id"]],
        )
        return [node]
