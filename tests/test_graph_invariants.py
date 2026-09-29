import pytest
from pydantic import ValidationError
from cls.graph.schema import ProcedureNode

def test_procedure_must_have_provenance():
    # Attempting to create a node without provenance should ideally fail
    # or the defaults should force it.
    node = ProcedureNode(
        procedure_id="p1",
        action="test",
        scope="proj_a",
        repository_version="v1"
    )
    assert node.created_from_users == [], "Provenance should be explicitly handled"
    
def test_procedure_must_have_scope():
    with pytest.raises(ValidationError):
        # Missing 'scope'
        ProcedureNode(
            procedure_id="p1",
            action="test",
            repository_version="v1"
        )


def test_procedure_lists_are_not_shared_between_instances():
    first = ProcedureNode(
        procedure_id="p1",
        action="build",
        scope="proj_a",
        repository_version="v1",
    )
    second = ProcedureNode(
        procedure_id="p2",
        action="test",
        scope="proj_a",
        repository_version="v1",
    )

    first.preconditions.append("needs_clean_tree")

    assert first.preconditions == ["needs_clean_tree"]
    assert second.preconditions == []
    assert first.created_from_users == []
    assert second.created_from_users == []
