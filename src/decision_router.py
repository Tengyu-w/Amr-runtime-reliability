"""Decision router for AMR runtime recovery actions."""

from __future__ import annotations

from enum import Enum

from src.reliability_supervisor import ReliabilityMetrics


class RouterDecision(str, Enum):
    """Possible runtime decisions for the AMR."""

    NORMAL_NAVIGATION = "NORMAL_NAVIGATION"
    CAUTIOUS_MODE = "CAUTIOUS_MODE"
    REPLAN = "REPLAN"
    RELOCALIZE = "RELOCALIZE"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    SAFE_STOP = "SAFE_STOP"


class FailureMechanism(str, Enum):
    """Dominant runtime failure mechanism used for mechanism-aware routing."""

    NOMINAL = "nominal"
    PATH_BLOCKED = "path_blocked"
    LOCALIZATION_UNCERTAINTY = "localization_uncertainty"
    PERCEPTION_DEGRADED = "perception_degraded"
    TRAJECTORY_DEVIATION = "trajectory_deviation"
    PROGRESS_STAGNATION = "progress_stagnation"
    REPEATED_REPLAN_FAILURE = "repeated_replan_failure"
    HIDDEN_COMPOUND_RISK = "hidden_compound_risk"
    ELEVATED_RUNTIME_RISK = "elevated_runtime_risk"


def diagnose_failure_mechanism(metrics: ReliabilityMetrics) -> FailureMechanism:
    """Return the dominant failure mechanism before choosing a route."""

    if metrics.replanning_failure_count >= 4 and metrics.path_blocked_score >= 0.45:
        return FailureMechanism.REPEATED_REPLAN_FAILURE
    if metrics.sensor_confidence <= 0.38:
        return FailureMechanism.PERCEPTION_DEGRADED
    if metrics.localization_uncertainty >= 0.72:
        return FailureMechanism.LOCALIZATION_UNCERTAINTY
    if metrics.trajectory_deviation >= 0.8:
        return FailureMechanism.TRAJECTORY_DEVIATION
    if metrics.path_blocked_score >= 0.45:
        return FailureMechanism.PATH_BLOCKED
    if metrics.task_progress_stagnation >= 0.75:
        return FailureMechanism.PROGRESS_STAGNATION
    if metrics.risk_score >= 0.45:
        return FailureMechanism.HIDDEN_COMPOUND_RISK
    if metrics.risk_score >= 0.35 or (
        metrics.obstacle_proximity >= 0.65 and metrics.path_blocked_score >= 0.25
    ):
        return FailureMechanism.ELEVATED_RUNTIME_RISK
    return FailureMechanism.NOMINAL


class DecisionRouter:
    """Map reliability metrics to explainable runtime actions."""

    def route(self, metrics: ReliabilityMetrics) -> RouterDecision:
        """Choose the next robot action from reliability and risk indicators."""

        if metrics.risk_score >= 0.82 or (
            metrics.replanning_failure_count >= 4
            and metrics.path_blocked_score >= 0.45
            and metrics.risk_score >= 0.70
        ):
            return RouterDecision.SAFE_STOP
        if metrics.sensor_confidence <= 0.38:
            return RouterDecision.HUMAN_REVIEW
        if metrics.localization_uncertainty >= 0.72:
            return RouterDecision.RELOCALIZE
        if metrics.path_blocked_score >= 0.45 and metrics.localization_uncertainty < 0.65:
            return RouterDecision.REPLAN
        if metrics.trajectory_deviation >= 0.8:
            return RouterDecision.REPLAN
        if metrics.task_progress_stagnation >= 0.75:
            return RouterDecision.REPLAN
        if metrics.risk_score >= 0.35 or (
            metrics.obstacle_proximity >= 0.65 and metrics.path_blocked_score >= 0.25
        ):
            return RouterDecision.CAUTIOUS_MODE
        return RouterDecision.NORMAL_NAVIGATION


class ScalarRiskRouter:
    """Route from the aggregate risk score without mechanism-specific actions."""

    def route(self, metrics: ReliabilityMetrics) -> RouterDecision:
        """Choose a conservative action from only the scalar risk score."""

        if metrics.risk_score >= 0.82:
            return RouterDecision.SAFE_STOP
        if metrics.risk_score >= 0.60:
            return RouterDecision.HUMAN_REVIEW
        if metrics.risk_score >= 0.35:
            return RouterDecision.CAUTIOUS_MODE
        return RouterDecision.NORMAL_NAVIGATION


class MechanismAwareDecisionRouter:
    """Route AMR actions through explicit failure mechanisms."""

    def route(self, metrics: ReliabilityMetrics) -> RouterDecision:
        """Choose the next action from the dominant failure mechanism."""

        mechanism = diagnose_failure_mechanism(metrics)
        if mechanism == FailureMechanism.NOMINAL:
            return RouterDecision.NORMAL_NAVIGATION
        if mechanism == FailureMechanism.REPEATED_REPLAN_FAILURE:
            if metrics.risk_score >= 0.70:
                return RouterDecision.SAFE_STOP
            return RouterDecision.REPLAN
        if mechanism == FailureMechanism.PERCEPTION_DEGRADED:
            return RouterDecision.HUMAN_REVIEW
        if mechanism == FailureMechanism.LOCALIZATION_UNCERTAINTY:
            return RouterDecision.RELOCALIZE
        if mechanism in {
            FailureMechanism.PATH_BLOCKED,
            FailureMechanism.TRAJECTORY_DEVIATION,
            FailureMechanism.PROGRESS_STAGNATION,
        }:
            return RouterDecision.REPLAN
        if mechanism == FailureMechanism.HIDDEN_COMPOUND_RISK:
            return RouterDecision.HUMAN_REVIEW
        return RouterDecision.CAUTIOUS_MODE
