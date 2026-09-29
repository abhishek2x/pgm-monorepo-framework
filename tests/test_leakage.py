import pytest
from cls.graph.store import GraphStore
from cls.graph.retrieval import KnowledgeRetrievalAPI
from cls.graph.schema import ProcedureNode

def test_strict_namespace_isolation():
    store = GraphStore()
    api = KnowledgeRetrievalAPI(store)
    
    # Add high-confidence knowledge in Project A
    node_a = ProcedureNode(
        procedure_id="p_a", action="build_a", scope="project_a", repository_version="v1",
        confidence=0.9, evidence_count=5
    )
    store.add_node(node_a)
    
    # Agent working on Project B asks for procedures
    retrieved = api.retrieve_procedures(
        project_id="project_b", 
        task_description="build"
    )
    
    # Must be completely isolated
    assert len(retrieved) == 0, "Data leakage from Project A to Project B!"
    
    # Agent working on Project A asks
    retrieved_a = api.retrieve_procedures(
        project_id="project_a", 
        task_description="build"
    )
    
    assert len(retrieved_a) == 1
    assert retrieved_a[0].procedure_id == "p_a"


def test_retrieval_prioritizes_task_relevant_procedures():
    store = GraphStore()
    api = KnowledgeRetrievalAPI(store)

    build_node = ProcedureNode(
        procedure_id="p_build",
        action="run project build and test target",
        scope="project_alpha",
        repository_version="v1",
        confidence=0.95,
        evidence_count=4,
    )
    deploy_node = ProcedureNode(
        procedure_id="p_deploy",
        action="bump deployment version and release",
        scope="project_alpha",
        repository_version="v1",
        confidence=0.95,
        evidence_count=4,
    )
    store.add_node(build_node)
    store.add_node(deploy_node)

    results = api.retrieve_procedures(
        project_id="project_alpha",
        task_description="build the app and run the relevant unit tests",
    )

    assert results[0].procedure_id == "p_build"
    assert all(node.procedure_id != "p_deploy" for node in results)
