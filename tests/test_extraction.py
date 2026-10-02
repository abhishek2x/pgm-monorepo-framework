import pytest
from cls.learning.extractor import KnowledgeExtractor
from cls.graph.schema import ProcedureNode

def test_knowledge_extractor_proposes_procedure():
    extractor = KnowledgeExtractor()
    
    # Mock episode from successful task
    episode = {
        "task_id": "task_1",
        "trajectory_id": "trajectory_1",
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
    assert candidates[0].source_trajectory_ids == ["trajectory_1"]
    assert candidates[0].guidance == ["action_1", "action_2"]
    assert candidates[0].evidence[0].trajectory_id == "trajectory_1"
    assert candidates[0].evidence[0].user_id == "user_a"
    assert candidates[0].evidence[0].step_ids == []
    assert candidates[0].confidence == 1.0


def test_failed_episode_does_not_create_a_procedure():
    episode = {
        "trajectory_id": "trajectory_failed",
        "project_id": "proj_a",
        "user_id": "user_a",
        "repository_commit": "commit_a",
        "outcome": "failure",
        "successful_actions": ["inspect files"],
    }

    assert KnowledgeExtractor().extract_candidates(episode) == []
