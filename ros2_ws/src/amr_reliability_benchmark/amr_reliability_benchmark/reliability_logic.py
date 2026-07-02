from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class RouterDecision(str, Enum):
    NORMAL_NAVIGATION = "NORMAL_NAVIGATION"
    CAUTIOUS_MODE = "CAUTIOUS_MODE"
    REPLAN = "REPLAN"
    RELOCALIZE = "RELOCALIZE"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    SAFE_STOP = "SAFE_STOP"


class FailureMechanism(str, Enum):
    NOMINAL = "nominal"
    PATH_BLOCKED = "path_blocked"
    LOCALIZATION_UNCERTAINTY = "localization_uncertainty"
    PERCEPTION_DEGRADED = "perception_degraded"
    TRAJECTORY_DEVIATION = "trajectory_deviation"
    PROGRESS_STAGNATION = "progress_stagnation"
    REPEATED_REPLAN_FAILURE = "repeated_replan_failure"
    HIDDEN_COMPOUND_RISK = "hidden_compound_risk"
    ELEVATED_RUNTIME_RISK = "elevated_runtime_risk"


@dataclass(frozen=True)
class ReliabilityMetrics:
    time_step: int
    robot_x: float
    robot_y: float
    target_x: float
    target_y: float
    localization_uncertainty: float
    sensor_confidence: float
    path_blocked_score: float
    obstacle_proximity: float
    trajectory_deviation: float
    replanning_failure_count: int
    task_progress_stagnation: float
    risk_score: float

    def to_row(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class TelemetrySnapshot:
    time_step: int
    robot_x: float = 0.0
    robot_y: float = 0.0
    target_x: float = 0.0
    target_y: float = 0.0
    localization_covariance_trace: float = 0.0
    sensor_confidence: float = 0.95
    path_blocked_score: float = 0.0
    obstacle_proximity: float = 0.0
    trajectory_deviation: float = 0.0
    replanning_failure_count: int = 0
    task_progress_stagnation: float = 0.0


WEIGHTS = {
    "localization": 0.18,
    "sensor": 0.16,
    "path_blocked": 0.18,
    "obstacle": 0.12,
    "deviation": 0.14,
    "replan_failures": 0.12,
    "stagnation": 0.10,
}


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compute_risk_score(
    localization_uncertainty: float,
    sensor_confidence: float,
    path_blocked_score: float,
    obstacle_proximity: float,
    trajectory_deviation: float,
    replanning_failure_count: int,
    task_progress_stagnation: float,
) -> float:
    replan_failures = clamp(replanning_failure_count / 4.0)
    risk = (
        WEIGHTS["localization"] * clamp(localization_uncertainty)
        + WEIGHTS["sensor"] * (1.0 - clamp(sensor_confidence))
        + WEIGHTS["path_blocked"] * clamp(path_blocked_score)
        + WEIGHTS["obstacle"] * clamp(obstacle_proximity)
        + WEIGHTS["deviation"] * clamp(trajectory_deviation)
        + WEIGHTS["replan_failures"] * replan_failures
        + WEIGHTS["stagnation"] * clamp(task_progress_stagnation)
    )
    return round(clamp(risk), 4)


def localization_uncertainty_from_covariance(covariance_trace: float, normalizer: float = 1.0) -> float:
    return round(clamp(covariance_trace / max(normalizer, 1e-6)), 4)


def metrics_from_telemetry(snapshot: TelemetrySnapshot) -> ReliabilityMetrics:
    localization_uncertainty = localization_uncertainty_from_covariance(
        snapshot.localization_covariance_trace
    )
    risk = compute_risk_score(
        localization_uncertainty=localization_uncertainty,
        sensor_confidence=snapshot.sensor_confidence,
        path_blocked_score=snapshot.path_blocked_score,
        obstacle_proximity=snapshot.obstacle_proximity,
        trajectory_deviation=snapshot.trajectory_deviation,
        replanning_failure_count=snapshot.replanning_failure_count,
        task_progress_stagnation=snapshot.task_progress_stagnation,
    )
    return ReliabilityMetrics(
        time_step=snapshot.time_step,
        robot_x=round(float(snapshot.robot_x), 4),
        robot_y=round(float(snapshot.robot_y), 4),
        target_x=round(float(snapshot.target_x), 4),
        target_y=round(float(snapshot.target_y), 4),
        localization_uncertainty=localization_uncertainty,
        sensor_confidence=round(clamp(snapshot.sensor_confidence), 4),
        path_blocked_score=round(clamp(snapshot.path_blocked_score), 4),
        obstacle_proximity=round(clamp(snapshot.obstacle_proximity), 4),
        trajectory_deviation=round(clamp(snapshot.trajectory_deviation), 4),
        replanning_failure_count=int(snapshot.replanning_failure_count),
        task_progress_stagnation=round(clamp(snapshot.task_progress_stagnation), 4),
        risk_score=risk,
    )


def diagnose_failure_mechanism(metrics: ReliabilityMetrics) -> FailureMechanism:
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


def route_metrics(metrics: ReliabilityMetrics) -> RouterDecision:
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


def _scenario_signal(scenario_id: str, time_step: int, total_steps: int) -> dict[str, float | int]:
    phase = time_step / max(total_steps - 1, 1)
    signal: dict[str, float | int] = {
        "localization_uncertainty": 0.08 + 0.03 * phase,
        "sensor_confidence": 0.92 - 0.02 * phase,
        "path_blocked_score": 0.05,
        "obstacle_proximity": 0.12,
        "trajectory_deviation": 0.03,
        "replanning_failure_count": 0,
        "task_progress_stagnation": 0.0,
    }
    if scenario_id == "external_path_blockage":
        signal.update(path_blocked_score=0.72, obstacle_proximity=0.78)
    elif scenario_id == "localization_drift":
        signal.update(localization_uncertainty=0.28 + 0.58 * phase)
    elif scenario_id == "perception_degradation":
        signal.update(sensor_confidence=0.88 - 0.62 * phase)
    elif scenario_id == "task_goal_shift_ood_style":
        signal.update(path_blocked_score=0.38 + 0.18 * phase, task_progress_stagnation=0.52 + 0.25 * phase)
    elif scenario_id == "execution_deviation":
        signal.update(trajectory_deviation=0.18 + 0.78 * phase)
    elif scenario_id == "progress_blockage":
        signal.update(path_blocked_score=0.42 + 0.18 * phase, task_progress_stagnation=0.45 + 0.45 * phase)
    elif scenario_id == "planner_backend_failure":
        signal.update(
            localization_uncertainty=0.70,
            sensor_confidence=0.35,
            path_blocked_score=0.95,
            obstacle_proximity=0.85,
            trajectory_deviation=0.50,
            replanning_failure_count=min(5, 1 + time_step // 2),
            task_progress_stagnation=0.90,
        )
    elif scenario_id == "compound_shift_and_degradation":
        signal.update(
            localization_uncertainty=0.22 + 0.56 * phase,
            sensor_confidence=0.82 - 0.55 * phase,
            path_blocked_score=0.32 + 0.28 * phase,
            task_progress_stagnation=0.30 + 0.46 * phase,
        )
    elif scenario_id == "mixed_blockage_and_perception":
        signal.update(
            sensor_confidence=0.82 - 0.58 * phase,
            path_blocked_score=0.52 + 0.22 * phase,
            obstacle_proximity=0.70 + 0.12 * phase,
            task_progress_stagnation=0.28 + 0.35 * phase,
        )
    elif scenario_id == "mixed_drift_and_execution":
        signal.update(
            localization_uncertainty=0.24 + 0.60 * phase,
            trajectory_deviation=0.16 + 0.66 * phase,
            task_progress_stagnation=0.18 + 0.24 * phase,
        )
    elif scenario_id == "boundary_weak_blockage":
        signal.update(
            path_blocked_score=0.25 + 0.18 * phase,
            obstacle_proximity=0.48 + 0.18 * phase,
            task_progress_stagnation=0.16 + 0.36 * phase,
        )
    return signal


def replay_metrics_for_scenario(scenario_id: str, time_step: int, total_steps: int = 12) -> ReliabilityMetrics:
    signal = _scenario_signal(scenario_id, time_step, total_steps)
    risk = compute_risk_score(
        localization_uncertainty=float(signal["localization_uncertainty"]),
        sensor_confidence=float(signal["sensor_confidence"]),
        path_blocked_score=float(signal["path_blocked_score"]),
        obstacle_proximity=float(signal["obstacle_proximity"]),
        trajectory_deviation=float(signal["trajectory_deviation"]),
        replanning_failure_count=int(signal["replanning_failure_count"]),
        task_progress_stagnation=float(signal["task_progress_stagnation"]),
    )
    return ReliabilityMetrics(
        time_step=time_step,
        robot_x=round(1.0 + 0.35 * time_step, 3),
        robot_y=1.0,
        target_x=6.0,
        target_y=4.0,
        localization_uncertainty=round(clamp(float(signal["localization_uncertainty"])), 4),
        sensor_confidence=round(clamp(float(signal["sensor_confidence"])), 4),
        path_blocked_score=round(clamp(float(signal["path_blocked_score"])), 4),
        obstacle_proximity=round(clamp(float(signal["obstacle_proximity"])), 4),
        trajectory_deviation=round(clamp(float(signal["trajectory_deviation"])), 4),
        replanning_failure_count=int(signal["replanning_failure_count"]),
        task_progress_stagnation=round(clamp(float(signal["task_progress_stagnation"])), 4),
        risk_score=risk,
    )
