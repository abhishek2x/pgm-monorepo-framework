import pytest
from cls.learning.validator import KnowledgeValidator
from cls.graph.store import GraphStore
from cls.graph.schema import ProcedureNode

def test_validation_requires_multiple_users():
    store = GraphStore()
    validator = KnowledgeValidator(store)
    
    # First user's candidate
    candidate_1 = ProcedureNode(
        procedure_id="p1", action="do_thing", scope="proj_a", repository_version="v1",
        created_from_users=["user_a"], evidence_count=1, successful_executions=1, confidence=0.5
    )
    
    validated_1 = validator.validate_and_merge(candidate_1)
    store.update_node(validated_1)
    assert validated_1.confidence == 0.5  # Stays low
    
    # Second user's identical candidate
    candidate_2 = ProcedureNode(
        procedure_id="p2", action="do_thing", scope="proj_a", repository_version="v1",
        created_from_users=["user_b"], evidence_count=1, successful_executions=1, confidence=0.5
    )
    
    validated_2 = validator.validate_and_merge(candidate_2)
    # Should merge with existing p1, bump confidence, add user_b
    assert validated_2.confidence > 0.5
    assert "user_b" in validated_2.created_from_users
    assert validated_2.evidence_count == 2
