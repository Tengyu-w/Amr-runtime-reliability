"""Replay the ROS 2 reliability protocol offline and write routed episode rows."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parents[1]
ROS_PACKAGE = DEMO_ROOT / "ros2_ws" / "src" / "amr_reliability_benchmark"
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))
if str(ROS_PACKAGE) not in sys.path:
    sys.path.insert(0, str(ROS_PACKAGE))

from amr_reliability_benchmark.reliability_logic import (  # noqa: E402
    diagnose_failure_mechanism,
    replay_metrics_for_scenario,
    route_metrics,
)
from amr_reliability_benchmark.scenario_catalog import SCENARIOS  # noqa: E402
from src.utils import ensure_output_dir  # noqa: E402


CSV_COLUMNS = [
    "episode_id",
    "scenario_id",
    "time_step",
    "source",
    "robot_x",
    "robot_y",
    "target_x",
    "target_y",
    "risk_score",
    "localization_uncertainty",
    "sensor_confidence",
    "path_blocked_score",
    "obstacle_proximity",
    "trajectory_deviation",
    "replanning_failure_count",
    "task_progress_stagnation",
    "failure_mechanism",
    "router_decision",
    "router_mode",
]


def replay_protocol(out_dir: str | Path, steps: int = 12) -> Path:
    output_dir = ensure_output_dir(out_dir)
    path = output_dir / "ros2_protocol_replay.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for scenario in SCENARIOS:
            for time_step in range(steps):
                metrics = replay_metrics_for_scenario(scenario.scenario_id, time_step, steps)
                row = {
                    **metrics.to_row(),
                    "episode_id": f"{scenario.scenario_id}_replay",
                    "scenario_id": scenario.scenario_id,
                    "source": "offline_ros2_protocol_replay",
                    "failure_mechanism": diagnose_failure_mechanism(metrics).value,
                    "router_decision": route_metrics(metrics).value,
                    "router_mode": "mechanism_router",
                }
                writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the ROS 2 reliability protocol offline.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/ros2_protocol_replay"))
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    path = replay_protocol(args.out_dir, steps=args.steps)
    print(f"ROS2 protocol replay: {path}")


if __name__ == "__main__":
    main()
