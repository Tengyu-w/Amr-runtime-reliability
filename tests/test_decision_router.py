"""Tests for reliability-aware decision routing."""

from src.decision_router import (
    DecisionRouter,
    FailureMechanism,
    MechanismAwareDecisionRouter,
    RouterDecision,
    ScalarRiskRouter,
    diagnose_failure_mechanism,
)
from src.reliability_supervisor import ReliabilityMetrics


def _metrics(**overrides):
    data = {
        "localization_uncertainty": 0.05,
        "sensor_confidence": 0.95,
        "path_blocked_score": 0.0,
        "obstacle_proximity": 0.0,
        "trajectory_deviation": 0.0,
        "replanning_failure_count": 0,
        "task_progress_stagnation": 0.0,
        "risk_score": 0.05,
    }
    data.update(overrides)
    return ReliabilityMetrics(**data)


def test_router_normal_navigation_for_low_risk():
    """Low risk should route to normal navigation."""

    assert DecisionRouter().route(_metrics()) == RouterDecision.NORMAL_NAVIGATION


def test_router_requests_relocalization_for_high_localization_uncertainty():
    """High localization uncertainty should route to relocalization."""

    decision = DecisionRouter().route(
        _metrics(localization_uncertainty=0.8, risk_score=0.55)
    )
    assert decision == RouterDecision.RELOCALIZE


def test_router_safe_stops_after_repeated_replanning_failures():
    """Repeated recovery failure should trigger safe stop."""

    decision = DecisionRouter().route(
        _metrics(replanning_failure_count=4, path_blocked_score=0.6, risk_score=0.75)
    )
    assert decision == RouterDecision.SAFE_STOP


def test_router_keeps_recovering_when_replan_failures_are_not_high_risk():
    """Repeated replan failures below the high-risk stop gate should keep recovering."""

    decision = DecisionRouter().route(
        _metrics(replanning_failure_count=4, path_blocked_score=0.6, risk_score=0.65)
    )
    assert decision == RouterDecision.REPLAN


def test_router_does_not_safe_stop_on_stale_replanning_failure_count():
    """Historical replan failures should not stop the robot after blockage clears."""

    decision = DecisionRouter().route(
        _metrics(replanning_failure_count=4, path_blocked_score=0.0, risk_score=0.45)
    )
    assert decision == RouterDecision.CAUTIOUS_MODE


def test_router_does_not_enter_cautious_mode_for_proximity_alone():
    """Obstacle proximity alone should not make the whole run non-nominal."""

    decision = DecisionRouter().route(_metrics(obstacle_proximity=0.8, risk_score=0.2))
    assert decision == RouterDecision.NORMAL_NAVIGATION


def test_mechanism_router_maps_localization_to_relocalize():
    """Localization uncertainty should use the localization-specific route."""

    metrics = _metrics(localization_uncertainty=0.8, risk_score=0.55)

    assert diagnose_failure_mechanism(metrics) == FailureMechanism.LOCALIZATION_UNCERTAINTY
    assert MechanismAwareDecisionRouter().route(metrics) == RouterDecision.RELOCALIZE


def test_mechanism_router_maps_perception_to_human_review():
    """Low sensor confidence should use the perception-specific route."""

    metrics = _metrics(sensor_confidence=0.3, risk_score=0.5)

    assert diagnose_failure_mechanism(metrics) == FailureMechanism.PERCEPTION_DEGRADED
    assert MechanismAwareDecisionRouter().route(metrics) == RouterDecision.HUMAN_REVIEW


def test_mechanism_router_maps_path_blockage_to_replan():
    """Path blockage should use the path-specific recovery route."""

    metrics = _metrics(path_blocked_score=0.8, risk_score=0.45)

    assert diagnose_failure_mechanism(metrics) == FailureMechanism.PATH_BLOCKED
    assert MechanismAwareDecisionRouter().route(metrics) == RouterDecision.REPLAN


def test_scalar_risk_router_does_not_pick_path_specific_replan():
    """The scalar baseline should not know which mechanism-specific route to use."""

    metrics = _metrics(path_blocked_score=0.8, risk_score=0.45)

    assert ScalarRiskRouter().route(metrics) == RouterDecision.CAUTIOUS_MODE
