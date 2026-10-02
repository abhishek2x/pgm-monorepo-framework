import pytest
from cls.ingestion.schemas import Trajectory, TrajectoryStep, ToolCall
from datetime import datetime

def test_trajectory_schema_validation():
    # Valid Trajectory
    step = TrajectoryStep(
        step_id="step_1",
        action="Run build",
        tool_calls=[ToolCall(tool_name="bash", arguments={"cmd": "make"}, result="success", is_error=False)]
    )
    
    trajectory = Trajectory(
        user_id="user_1",
        agent_id="agent_v1",
        project_id="proj_xyz",
        task_id="task_1",
        repository="org/repo",
        repository_commit="main",
        messages=[],
        steps=[step],
        files_touched=["Makefile"],
        errors=[],
        outcome="success",
        token_usage={"total": 100},
        latency=1.5
    )
    
    assert trajectory.user_id == "user_1"
    assert trajectory.schema_version == "1.0"
    assert trajectory.trajectory_id
    assert len(trajectory.steps) == 1
    assert trajectory.steps[0].tool_calls[0].tool_name == "bash"
