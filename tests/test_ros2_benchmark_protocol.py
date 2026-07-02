from __future__ import annotations

import sys
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd

DEMO_ROOT = Path(__file__).resolve().parents[1]
ROS_PACKAGE = DEMO_ROOT / "ros2_ws" / "src" / "amr_reliability_benchmark"
if str(ROS_PACKAGE) not in sys.path:
    sys.path.insert(0, str(ROS_PACKAGE))

from amr_reliability_benchmark.reliability_logic import (
    TelemetrySnapshot,
    metrics_from_telemetry,
    replay_metrics_for_scenario,
    route_metrics,
)
from amr_reliability_benchmark.scenario_catalog import SCENARIOS
from experiments.replay_ros2_benchmark_protocol import replay_protocol


def test_ros2_scenario_catalog_matches_replay_protocol() -> None:
    ids = [scenario.scenario_id for scenario in SCENARIOS]
    assert len(ids) == 12
    assert "planner_backend_failure" in ids
    assert "compound_shift_and_degradation" in ids
    assert "mixed_blockage_and_perception" in ids
    assert "mixed_drift_and_execution" in ids
    assert "boundary_weak_blockage" in ids


def test_nav2_bringup_resources_are_present() -> None:
    urdf = ROS_PACKAGE / "urdf" / "reliability_amr.urdf"
    map_yaml = ROS_PACKAGE / "maps" / "reliability_room.yaml"
    map_pgm = ROS_PACKAGE / "maps" / "reliability_room.pgm"
    bringup_launch = ROS_PACKAGE / "launch" / "amr_nav2_bringup.launch.py"
    robot = ElementTree.parse(urdf).getroot()
    assert robot.attrib["name"] == "reliability_amr"
    assert "resolution: 0.60" in map_yaml.read_text(encoding="utf-8")
    assert map_pgm.read_text(encoding="utf-8").startswith("P2")
    assert bringup_launch.exists()


def test_recovery_executor_is_registered_and_launchable() -> None:
    setup_py = ROS_PACKAGE / "setup.py"
    launch_py = ROS_PACKAGE / "launch" / "nav2_runtime_pipeline.launch.py"
    executor_py = ROS_PACKAGE / "amr_reliability_benchmark" / "recovery_executor.py"

    assert executor_py.exists()
    assert "recovery_executor = amr_reliability_benchmark.recovery_executor:main" in setup_py.read_text(
        encoding="utf-8"
    )
    launch_text = launch_py.read_text(encoding="utf-8")
    assert "enable_recovery_executor" in launch_text
    assert "nav2_runtime_recovery_execution.csv" in launch_text
    assert 'executable="recovery_executor"' in launch_text


def test_replay_metrics_drive_distinct_recovery_routes() -> None:
    assert route_metrics(replay_metrics_for_scenario("nominal", 0)).value == "NORMAL_NAVIGATION"
    assert route_metrics(replay_metrics_for_scenario("localization_drift", 11)).value == "RELOCALIZE"
    assert route_metrics(replay_metrics_for_scenario("perception_degradation", 11)).value == "HUMAN_REVIEW"
    assert route_metrics(replay_metrics_for_scenario("planner_backend_failure", 11)).value == "SAFE_STOP"


def test_nav2_telemetry_snapshot_maps_to_runtime_metrics() -> None:
    metrics = metrics_from_telemetry(
        TelemetrySnapshot(
            time_step=4,
            robot_x=1.2,
            robot_y=2.4,
            target_x=6.0,
            target_y=4.0,
            localization_covariance_trace=0.82,
            sensor_confidence=0.91,
            path_blocked_score=0.05,
            obstacle_proximity=0.10,
            trajectory_deviation=0.02,
            replanning_failure_count=0,
            task_progress_stagnation=0.0,
        )
    )
    assert metrics.time_step == 4
    assert metrics.localization_uncertainty == 0.82
    assert route_metrics(metrics).value == "RELOCALIZE"


def test_offline_ros2_protocol_replay_writes_routed_rows(tmp_path: Path) -> None:
    path = replay_protocol(tmp_path, steps=12)
    df = pd.read_csv(path)
    assert len(df) == 144
    assert set(df["scenario_id"]) == {scenario.scenario_id for scenario in SCENARIOS}
    assert {"NORMAL_NAVIGATION", "REPLAN", "RELOCALIZE", "HUMAN_REVIEW", "SAFE_STOP"}.issubset(
        set(df["router_decision"])
    )
