import pytest
from cls.learning.validator import KnowledgeValidator
from cls.graph.store import GraphStore
from cls.graph.schema import ProcedureEvidence, ProcedureNode
from cls.graph.retrieval import KnowledgeRetrievalAPI

def test_validation_requires_multiple_users():
    store = GraphStore()
    validator = KnowledgeValidator(store)
    
    # First user's candidate
    candidate_1 = ProcedureNode(
        procedure_id="p1", action="do_thing", scope="proj_a", repository_version="v1",
        created_from_users=["user_a"], evidence_count=1, successful_executions=1, confidence=0.5
        , source_trajectory_ids=["trajectory_a"]
    )
    
    validated_1 = validator.validate_and_merge(candidate_1)
    store.update_node(validated_1)
    assert validated_1.confidence == 1.0
    assert validated_1.validation_status == "candidate"
    
    # Second user's identical candidate
    candidate_2 = ProcedureNode(
        procedure_id="p2", action="do_thing", scope="proj_a", repository_version="v1",
        created_from_users=["user_b"], evidence_count=1, successful_executions=1, confidence=0.5
        , source_trajectory_ids=["trajectory_b"]
    )
    
    validated_2 = validator.validate_and_merge(candidate_2)
    # Independent evidence promotes the candidate.
    assert validated_2.validation_status == "validated"
    assert "user_b" in validated_2.created_from_users
    assert validated_2.evidence_count == 2
    assert validated_2.validation_status == "validated"


def test_repeated_evidence_from_one_user_never_becomes_retrievable():
    store = GraphStore()
    validator = KnowledgeValidator(store)

    for index in range(4):
        candidate = ProcedureNode(
            procedure_id=f"same_user_{index}",
            action="run focused tests before the full suite",
            scope="project_a",
            repository_version="commit_a",
            created_from_users=["user_a"],
            source_trajectory_ids=[f"trajectory_{index}"],
            evidence=[
                ProcedureEvidence(
                    trajectory_id=f"trajectory_{index}",
                    user_id="user_a",
                    repository_version="commit_a",
                    successful=True,
                )
            ],
            evidence_count=1,
            successful_executions=1,
        )
        store.update_node(validator.validate_and_merge(candidate))

    retrieved = KnowledgeRetrievalAPI(store).retrieve_procedures(
        project_id="project_a",
        task_description="run tests",
        repository_version="commit_a",
    )

    assert retrieved == []
