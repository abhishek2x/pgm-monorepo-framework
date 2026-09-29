import pytest
from cls.learning.extractor import KnowledgeExtractor
from cls.graph.schema import ProcedureNode

def test_knowledge_extractor_proposes_procedure():
    extractor = KnowledgeExtractor()
    
    # Mock episode from successful task
    episode = {
        "task_id": "task_1",
        "project_id": "proj_a",
        "user_id": "user_a",
        "repository_commit": "HEAD",
        "outcome": "success",
        "successful_actions": ["action_1", "action_2"]
    }
    
    candidates = extractor.extract_candidates(episode)
    
    assert len(candidates) == 1
    assert isinstance(candidates[0], ProcedureNode)
    assert candidates[0].scope == "proj_a"
    assert candidates[0].created_from_users == ["user_a"]
    assert candidates[0].confidence == 0.5  # Low initial confidence
