"""Deterministic failure and uncertainty injection for AMR experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from src.amr_agent import AMRAgent
from src.environment import WarehouseEnvironment
from src.utils import GridPosition, clamp


EVENT_METADATA = {
    "dynamic_obstacle_blocks_path": {
        "fault_origin": "external_disturbance",
        "fault_family": "path_blockage",
        "ood_status": "in_distribution_fault",
    },
    "localization_drift_increasing": {
        "fault_origin": "state_estimation_drift",
        "fault_family": "localization",
        "ood_status": "in_distribution_fault",
    },
    "sensor_confidence_drop": {
        "fault_origin": "perception_degradation",
        "fault_family": "sensor_quality",
        "ood_status": "in_distribution_fault",
    },
    "target_changed": {
        "fault_origin": "task_or_goal_shift",
        "fault_family": "task_reassignment",
        "ood_status": "ood_style_shift",
    },
    "replanning_backend_unstable": {
        "fault_origin": "planner_internal_failure",
        "fault_family": "planning_backend",
        "ood_status": "in_distribution_fault",
    },
    "trajectory_deviation": {
        "fault_origin": "execution_error",
        "fault_family": "control_tracking",
        "ood_status": "in_distribution_fault",
    },
    "progress_stagnation_blocker": {
        "fault_origin": "external_disturbance",
        "fault_family": "progress_blockage",
        "ood_status": "in_distribution_fault",
    },
    "dynamic_obstacles_cleared": {
        "fault_origin": "environment_recovery",
        "fault_family": "disturbance_cleared",
        "ood_status": "recovery_event",
    },
}

PRIMARY_EVENT_PRIORITY = [
    "target_changed",
    "replanning_backend_unstable",
    "dynamic_obstacle_blocks_path",
    "progress_stagnation_blocker",
    "trajectory_deviation",
    "sensor_confidence_drop",
    "localization_drift_increasing",
    "dynamic_obstacles_cleared",
]


def summarize_event_metadata(events: list[str]) -> dict[str, str | bool]:
    """Summarize event provenance for CSV logging."""

    if not events:
        return {
            "fault_origin": "none",
            "fault_family": "none",
            "ood_status": "none",
            "primary_fault_event": "none",
            "primary_fault_origin": "none",
            "primary_fault_family": "none",
            "primary_ood_status": "none",
            "has_ood_style_shift": False,
        }

    origins = sorted({EVENT_METADATA[event]["fault_origin"] for event in events if event in EVENT_METADATA})
    families = sorted({EVENT_METADATA[event]["fault_family"] for event in events if event in EVENT_METADATA})
    statuses = sorted({EVENT_METADATA[event]["ood_status"] for event in events if event in EVENT_METADATA})
    primary_event = next((event for event in PRIMARY_EVENT_PRIORITY if event in events), events[0])
    primary = EVENT_METADATA.get(
        primary_event,
        {
            "fault_origin": "unknown",
            "fault_family": "unknown",
            "ood_status": "unknown",
        },
    )
    has_ood_style_shift = "ood_style_shift" in statuses
    return {
        "fault_origin": "|".join(origins) if origins else "unknown",
        "fault_family": "|".join(families) if families else "unknown",
        "ood_status": "|".join(statuses) if statuses else "unknown",
        "primary_fault_event": primary_event,
        "primary_fault_origin": primary["fault_origin"],
        "primary_fault_family": primary["fault_family"],
        "primary_ood_status": primary["ood_status"],
        "has_ood_style_shift": has_ood_style_shift,
    }


@dataclass
class FailureInjector:
    """Inject controlled runtime failures into the warehouse simulation."""

    seed: int = 7
    enabled_events: set[str] | None = None
    forced_replan_failure_steps: set[int] | None = None
    dynamic_obstacle_step: int | None = None
    localization_drift_window: tuple[int, int] | None = None
    sensor_drop_window: tuple[int, int] | None = None
    target_change_step: int | None = None
    deviation_step: int | None = None
    stagnation_window: tuple[int, int] | None = None
    clear_obstacles_step: int | None = None

    def __post_init__(self) -> None:
        """Create a deterministic random generator for repeatable perturbations."""

        self.rng = Random(self.seed)
        if self.dynamic_obstacle_step is None:
            self.dynamic_obstacle_step = 6 + self.rng.randint(-1, 1)
        if self.localization_drift_window is None:
            start = 8 + self.rng.randint(-1, 1)
            self.localization_drift_window = (start, start + 14)
        if self.sensor_drop_window is None:
            start = 12 + self.rng.randint(-2, 2)
            self.sensor_drop_window = (start, start + 10)
        if self.target_change_step is None:
            self.target_change_step = 16 + self.rng.randint(-2, 2)
        if self.forced_replan_failure_steps is None:
            start = 22 + self.rng.randint(-3, 3)
            length = self.rng.choice([3, 4, 5])
            self.forced_replan_failure_steps = set(range(start, start + length))
        if self.deviation_step is None:
            self.deviation_step = 27 + self.rng.randint(-3, 3)
        if self.stagnation_window is None:
            start = 31 + self.rng.randint(-3, 3)
            self.stagnation_window = (start, start + 5)
        if self.clear_obstacles_step is None:
            self.clear_obstacles_step = 38 + self.rng.randint(-3, 3)

    def _enabled(self, event: str) -> bool:
        return self.enabled_events is None or event in self.enabled_events

    def apply(self, time_step: int, environment: WarehouseEnvironment, agent: AMRAgent) -> list[str]:
        """Apply scheduled failures and return human-readable event names."""

        events: list[str] = []

        if self._enabled("dynamic_obstacle_blocks_path") and time_step == self.dynamic_obstacle_step:
            next_cell = agent.next_path_cell()
            if next_cell and environment.add_dynamic_obstacle(next_cell):
                events.append("dynamic_obstacle_blocks_path")

        drift_start, drift_end = self.localization_drift_window
        if self._enabled("localization_drift_increasing") and drift_start <= time_step <= drift_end:
            agent.localization_uncertainty = clamp(agent.localization_uncertainty + 0.035)
            events.append("localization_drift_increasing")

        sensor_start, sensor_end = self.sensor_drop_window
        if self._enabled("sensor_confidence_drop") and sensor_start <= time_step <= sensor_end:
            agent.sensor_confidence = clamp(agent.sensor_confidence - 0.045)
            events.append("sensor_confidence_drop")

        if self._enabled("target_changed") and time_step == self.target_change_step:
            new_target = environment.alternate_targets[0]
            environment.change_target(new_target)
            agent.update_target(new_target)
            events.append("target_changed")

        if self._enabled("replanning_backend_unstable") and time_step in self.forced_replan_failure_steps:
            events.append("replanning_backend_unstable")

        if self._enabled("trajectory_deviation") and time_step == self.deviation_step and agent.force_deviation(environment):
            events.append("trajectory_deviation")

        stagnation_start, stagnation_end = self.stagnation_window
        if self._enabled("progress_stagnation_blocker") and stagnation_start <= time_step <= stagnation_end:
            obstacle = agent.next_path_cell()
            if obstacle and environment.add_dynamic_obstacle(obstacle):
                events.append("progress_stagnation_blocker")

        if self._enabled("dynamic_obstacles_cleared") and time_step == self.clear_obstacles_step:
            environment.clear_dynamic_obstacles()
            events.append("dynamic_obstacles_cleared")

        return events

    def should_force_replan_failure(self, time_step: int) -> bool:
        """Return True when the planner should simulate a repeated backend failure."""

        return time_step in self.forced_replan_failure_steps

    def random_free_neighbor(
        self,
        environment: WarehouseEnvironment,
        position: GridPosition,
    ) -> GridPosition | None:
        """Return a shuffled free neighbor for optional future perturbations."""

        neighbors = environment.neighbors(position)
        self.rng.shuffle(neighbors)
        return neighbors[0] if neighbors else None
