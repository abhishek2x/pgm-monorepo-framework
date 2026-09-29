from typing import Any, Dict

from cls.ingestion.schemas import Trajectory


class EpisodeAnalyzer:
    """Convert a raw trajectory into a compact episode for extraction."""

    def analyze(self, trajectory: Trajectory) -> Dict[str, Any]:
        """Split steps into successful and failing work before extraction."""
        failed_steps = []
        successful_steps = []

        for step in trajectory.steps:
            has_error = any(tool_call.is_error for tool_call in step.tool_calls)
            if has_error:
                failed_steps.append(step)
            else:
                successful_steps.append(step)

        return {
            "task_id": trajectory.task_id,
            "project_id": trajectory.project_id,
            "user_id": trajectory.user_id,
            "repository": trajectory.repository,
            "repository_commit": trajectory.repository_commit,
            "outcome": trajectory.outcome,
            "files_touched": trajectory.files_touched,
            "errors": trajectory.errors,
            "failed_actions": failed_steps,
            "successful_actions": successful_steps,
            "execution_sequence": trajectory.steps,
        }
