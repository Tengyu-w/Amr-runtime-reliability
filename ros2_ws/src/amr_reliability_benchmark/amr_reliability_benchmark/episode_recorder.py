from __future__ import annotations

import csv
import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


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


class EpisodeRecorder(Node):
    """Record routed runtime decisions to a CSV episode log."""

    def __init__(self) -> None:
        super().__init__("episode_recorder")
        self.declare_parameter("output_path", "outputs/ros2_episode_logs/episode.csv")
        output_path = self.get_parameter("output_path").get_parameter_value().string_value
        self._path = Path(output_path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS)
        self._writer.writeheader()
        self._subscription = self.create_subscription(
            String,
            "/amr_reliability/router_decision",
            self._on_decision,
            10,
        )
        self.get_logger().info("Recording routed episode rows to %s" % self._path)

    def _on_decision(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning("Skipping malformed decision payload: %s" % exc)
            return
        row = {column: payload.get(column, "") for column in CSV_COLUMNS}
        self._writer.writerow(row)
        self._file.flush()

    def destroy_node(self) -> bool:
        if not self._file.closed:
            self._file.close()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = EpisodeRecorder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
