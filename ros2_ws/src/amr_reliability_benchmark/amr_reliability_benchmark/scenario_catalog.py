from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    enabled_events: tuple[str, ...]
    primary_fault_origin: str
    primary_fault_family: str
    primary_ood_status: str
    split_group: str
    expected_recovery: str
    description: str


SCENARIOS = [
    ScenarioSpec(
        scenario_id="nominal",
        enabled_events=(),
        primary_fault_origin="none",
        primary_fault_family="none",
        primary_ood_status="none",
        split_group="nominal",
        expected_recovery="NORMAL_NAVIGATION",
        description="No injected fault; robot should navigate normally.",
    ),
    ScenarioSpec(
        scenario_id="external_path_blockage",
        enabled_events=("dynamic_obstacle_blocks_path",),
        primary_fault_origin="external_disturbance",
        primary_fault_family="path_blockage",
        primary_ood_status="in_distribution_fault",
        split_group="external_disturbance",
        expected_recovery="REPLAN",
        description="A dynamic obstacle blocks the planned route.",
    ),
    ScenarioSpec(
        scenario_id="localization_drift",
        enabled_events=("localization_drift_increasing",),
        primary_fault_origin="state_estimation_drift",
        primary_fault_family="localization",
        primary_ood_status="in_distribution_fault",
        split_group="state_estimation",
        expected_recovery="RELOCALIZE",
        description="Odometry/localization quality degrades during navigation.",
    ),
    ScenarioSpec(
        scenario_id="perception_degradation",
        enabled_events=("sensor_confidence_drop",),
        primary_fault_origin="perception_degradation",
        primary_fault_family="sensor_quality",
        primary_ood_status="in_distribution_fault",
        split_group="perception",
        expected_recovery="HUMAN_REVIEW",
        description="LiDAR/depth quality degrades; robot should slow or verify.",
    ),
    ScenarioSpec(
        scenario_id="task_goal_shift_ood_style",
        enabled_events=("target_changed",),
        primary_fault_origin="task_or_goal_shift",
        primary_fault_family="task_reassignment",
        primary_ood_status="ood_style_shift",
        split_group="task_shift",
        expected_recovery="HUMAN_REVIEW",
        description="The target goal changes during execution; treat as an OOD-style task shift.",
    ),
    ScenarioSpec(
        scenario_id="execution_deviation",
        enabled_events=("trajectory_deviation",),
        primary_fault_origin="execution_error",
        primary_fault_family="control_tracking",
        primary_ood_status="in_distribution_fault",
        split_group="execution",
        expected_recovery="REPLAN",
        description="Robot deviates from the commanded trajectory.",
    ),
    ScenarioSpec(
        scenario_id="progress_blockage",
        enabled_events=("progress_stagnation_blocker",),
        primary_fault_origin="external_disturbance",
        primary_fault_family="progress_blockage",
        primary_ood_status="in_distribution_fault",
        split_group="external_disturbance",
        expected_recovery="REPLAN",
        description="External blocker causes progress stagnation.",
    ),
    ScenarioSpec(
        scenario_id="planner_backend_failure",
        enabled_events=("target_changed", "replanning_backend_unstable"),
        primary_fault_origin="planner_internal_failure",
        primary_fault_family="planning_backend",
        primary_ood_status="in_distribution_fault",
        split_group="planner_backend",
        expected_recovery="SAFE_STOP",
        description="Target changes and the replanning backend becomes unstable.",
    ),
    ScenarioSpec(
        scenario_id="compound_shift_and_degradation",
        enabled_events=("target_changed", "sensor_confidence_drop", "localization_drift_increasing"),
        primary_fault_origin="task_or_goal_shift",
        primary_fault_family="task_reassignment",
        primary_ood_status="ood_style_shift",
        split_group="compound",
        expected_recovery="HUMAN_REVIEW",
        description="Task shift, perception degradation, and localization drift co-occur.",
    ),
    ScenarioSpec(
        scenario_id="mixed_blockage_and_perception",
        enabled_events=("dynamic_obstacle_blocks_path", "sensor_confidence_drop"),
        primary_fault_origin="mixed_external_perception",
        primary_fault_family="path_blockage+sensor_quality",
        primary_ood_status="mixed_fault",
        split_group="mixed_boundary",
        expected_recovery="HUMAN_REVIEW",
        description="External path blockage and sensor degradation co-occur.",
    ),
    ScenarioSpec(
        scenario_id="mixed_drift_and_execution",
        enabled_events=("localization_drift_increasing", "trajectory_deviation"),
        primary_fault_origin="mixed_state_execution",
        primary_fault_family="localization+control_tracking",
        primary_ood_status="mixed_fault",
        split_group="mixed_boundary",
        expected_recovery="RELOCALIZE",
        description="Localization drift and trajectory deviation co-occur.",
    ),
    ScenarioSpec(
        scenario_id="boundary_weak_blockage",
        enabled_events=("weak_dynamic_obstacle_blocks_path",),
        primary_fault_origin="external_disturbance",
        primary_fault_family="boundary_path_blockage",
        primary_ood_status="boundary_case",
        split_group="mixed_boundary",
        expected_recovery="CAUTIOUS_MODE",
        description="A weak intermittent blockage sits near the route-threshold boundary.",
    ),
]


def scenario_rows() -> list[dict[str, object]]:
    return [asdict(scenario) for scenario in SCENARIOS]


def scenario_by_id(scenario_id: str) -> dict[str, object]:
    for scenario in scenario_rows():
        if scenario["scenario_id"] == scenario_id:
            return scenario
    valid_ids = ", ".join(scenario.scenario_id for scenario in SCENARIOS)
    raise ValueError(f"Unknown scenario_id '{scenario_id}'. Valid ids: {valid_ids}")


def main() -> None:
    print(json.dumps(scenario_rows(), indent=2))


if __name__ == "__main__":
    main()
