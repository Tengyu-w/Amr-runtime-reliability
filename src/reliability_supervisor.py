"""Runtime reliability supervisor for AMR risk scoring."""

from __future__ import annotations

from dataclasses import dataclass

from src.amr_agent import AMRAgent
from src.environment import WarehouseEnvironment
from src.utils import clamp, manhattan


@dataclass
class ReliabilityMetrics:
    """Risk-related runtime indicators used by the decision router."""

    localization_uncertainty: float
    sensor_confidence: float
    path_blocked_score: float
    obstacle_proximity: float
    trajectory_deviation: float
    replanning_failure_count: int
    task_progress_stagnation: float
    risk_score: float


class ReliabilitySupervisor:
    """Compute a normalized risk score from AMR runtime reliability indicators."""

    weights = {
        "localization": 0.18,
        "sensor": 0.16,
        "path_blocked": 0.18,
        "obstacle": 0.12,
        "deviation": 0.14,
        "replan_failures": 0.12,
        "stagnation": 0.10,
    }

    def evaluate(self, environment: WarehouseEnvironment, agent: AMRAgent) -> ReliabilityMetrics:
        """Return reliability metrics and aggregate risk score for current state."""

        localization = clamp(agent.localization_uncertainty)
        sensor_confidence = clamp(agent.sensor_confidence)
        path_blocked = self._path_blocked_score(environment, agent)
        obstacle = self._obstacle_proximity(environment, agent)
        deviation = self._trajectory_deviation(agent)
        replan_failures = clamp(agent.replanning_failure_count / 4.0)
        stagnation = clamp(agent.stagnant_steps / 6.0)

        risk = (
            self.weights["localization"] * localization
            + self.weights["sensor"] * (1.0 - sensor_confidence)
            + self.weights["path_blocked"] * path_blocked
            + self.weights["obstacle"] * obstacle
            + self.weights["deviation"] * deviation
            + self.weights["replan_failures"] * replan_failures
            + self.weights["stagnation"] * stagnation
        )

        return ReliabilityMetrics(
            localization_uncertainty=round(localization, 4),
            sensor_confidence=round(sensor_confidence, 4),
            path_blocked_score=round(path_blocked, 4),
            obstacle_proximity=round(obstacle, 4),
            trajectory_deviation=round(deviation, 4),
            replanning_failure_count=agent.replanning_failure_count,
            task_progress_stagnation=round(stagnation, 4),
            risk_score=round(clamp(risk), 4),
        )

    def _path_blocked_score(self, environment: WarehouseEnvironment, agent: AMRAgent) -> float:
        """Estimate how much the current path is blocked or stale."""

        if agent.needs_replan:
            return 0.85
        if not agent.path:
            return 1.0
        if agent.path[-1] != agent.target:
            return 0.85

        try:
            idx = agent.path.index(agent.position)
        except ValueError:
            return 0.75

        upcoming = agent.path[idx + 1 : idx + 5]
        if not upcoming:
            return 0.0
        blocked = sum(1 for cell in upcoming if environment.is_blocked(cell))
        return clamp(blocked / max(1, len(upcoming)))

    def _obstacle_proximity(self, environment: WarehouseEnvironment, agent: AMRAgent) -> float:
        """Convert nearest-obstacle distance into a normalized proximity risk."""

        distance = environment.obstacle_distance(agent.position)
        if distance <= 1:
            return 1.0
        if distance >= 5:
            return 0.0
        return clamp((5 - distance) / 4.0)

    def _trajectory_deviation(self, agent: AMRAgent) -> float:
        """Estimate deviation from the current planned path."""

        if agent.deviated_from_path:
            return 1.0
        if not agent.path:
            return 0.6
        if agent.position in agent.path:
            return 0.0
        nearest = min(manhattan(agent.position, cell) for cell in agent.path)
        return clamp(nearest / 4.0)
