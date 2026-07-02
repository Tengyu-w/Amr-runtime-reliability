"""Tests for runtime risk scoring."""

from src.amr_agent import AMRAgent
from src.environment import WarehouseEnvironment
from src.reliability_supervisor import ReliabilitySupervisor


def test_risk_score_increases_under_uncertainty():
    """Risk should rise when uncertainty, low confidence, and failures increase."""

    env = WarehouseEnvironment()
    agent = AMRAgent(position=env.start, target=env.target)
    supervisor = ReliabilitySupervisor()

    low_risk = supervisor.evaluate(env, agent).risk_score
    agent.localization_uncertainty = 0.8
    agent.sensor_confidence = 0.25
    agent.replanning_failure_count = 4
    agent.deviated_from_path = True
    high_risk = supervisor.evaluate(env, agent).risk_score

    assert 0.0 <= low_risk <= 1.0
    assert 0.0 <= high_risk <= 1.0
    assert high_risk > low_risk
