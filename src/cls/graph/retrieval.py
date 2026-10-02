import re
from typing import List, Optional, Set, Tuple

from cls.graph.schema import ProcedureNode
from cls.graph.store import GraphStore


class KnowledgeRetrievalAPI:
    """Return project-scoped procedures that matter for the current task."""

    def __init__(self, store: GraphStore):
        self.store = store

    def retrieve_procedures(
        self,
        project_id: str,
        task_description: str,
        relevant_files: Optional[List[str]] = None,
        repository_version: Optional[str] = None,
        consumer_user_id: Optional[str] = None,
        task_id: Optional[str] = None,
        experiment_run_id: Optional[str] = None,
    ) -> List[ProcedureNode]:
        """Return the best matching procedures for a project and task."""
        project_nodes = (
            self.store.get_subgraph_by_scope(scope=project_id)
            if project_id and repository_version
            else []
        )

        query_terms = self._tokenize(task_description or "")
        file_terms = {
            term
            for file_path in (relevant_files or [])
            for term in self._tokenize(file_path)
        }

        scored_nodes: List[Tuple[ProcedureNode, float]] = []
        for node in project_nodes:
            if (
                node.validation_status != "validated"
                or node.evidence_count < 2
                or node.confidence < 0.7
                or len(
                    {
                        evidence.user_id
                        for evidence in node.evidence
                        if evidence.successful
                        and evidence.repository_version == node.repository_version
                    }
                ) < 2
                or len(
                    {
                        evidence.trajectory_id
                        for evidence in node.evidence
                        if evidence.successful
                        and evidence.repository_version == node.repository_version
                    }
                ) < 2
            ):
                continue
            if node.repository_version != repository_version:
                continue

            node_terms = self._tokenize(
                " ".join([node.action, *node.guidance, *node.preconditions])
            )
            overlap = len(query_terms & node_terms)
            file_overlap = len(file_terms & node_terms)
            confidence_bonus = node.successful_executions / max(1, node.evidence_count)
            score = overlap + (file_overlap * 2) + confidence_bonus

            if score > 0:
                scored_nodes.append((node, score))

        scored_nodes.sort(
            key=lambda item: (
                item[1],
                item[0].confidence,
                item[0].evidence_count,
                item[0].successful_executions,
            ),
            reverse=True,
        )

        self._log_retrieval(
            project_id,
            task_description,
            [node for node, _ in scored_nodes],
            repository_version,
            consumer_user_id,
            task_id,
            experiment_run_id,
        )
        return [node for node, _ in scored_nodes]

    def _log_retrieval(
        self,
        project_id: str,
        task_description: str,
        retrieved_nodes: List[ProcedureNode],
        repository_version: Optional[str],
        consumer_user_id: Optional[str],
        task_id: Optional[str],
        experiment_run_id: Optional[str],
    ) -> None:
        """Record exposed procedure IDs without persisting the task prompt."""
        self.store.record_retrieval_event(
            project_id=project_id,
            task_description=task_description,
            procedure_ids=[node.procedure_id for node in retrieved_nodes],
            repository_version=repository_version,
            consumer_user_id=consumer_user_id,
            task_id=task_id,
            experiment_run_id=experiment_run_id,
        )

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        """Normalize text into simple searchable tokens."""
        stop_words = {
            "a",
            "an",
            "and",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "in",
            "into",
            "is",
            "it",
            "of",
            "on",
            "or",
            "the",
            "to",
            "with",
        }
        tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
        return {token for token in tokens if token not in stop_words and len(token) > 1}
