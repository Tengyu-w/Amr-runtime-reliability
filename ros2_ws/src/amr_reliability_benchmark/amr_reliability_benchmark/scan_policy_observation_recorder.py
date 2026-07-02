from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


BASE_COLUMNS = [
    "episode_id",
    "scenario_id",
    "time_step",
    "robot_x",
    "robot_y",
    "target_x",
    "target_y",
    "goal_dx",
    "goal_dy",
    "goal_distance_l1",
    "risk_score",
    "localization_uncertainty",
    "sensor_confidence",
    "path_blocked_score",
    "obstacle_proximity",
    "trajectory_deviation",
    "replanning_failure_count",
    "task_progress_stagnation",
    "expert_proxy_action",
    "expert_source",
    "policy_evaluable",
    "policy_pred_action",
    "policy_correct",
    "policy_entropy",
    "policy_margin",
    "policy_max_prob",
    "policy_error_mechanism",
    "policy_recovery_route",
    "scan_stamp_sec",
    "scan_age_sec",
    "scan_n_ranges",
    "scan_valid_fraction",
    "scan_min_range",
    "scan_mean_range",
    "scan_front_min_range",
]


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class ScanPolicyObservationRecorder(Node):
    """Align Gazebo lidar observations with Nav2-plan expert action labels."""

    def __init__(self) -> None:
        super().__init__("scan_policy_observation_recorder")
        self.declare_parameter("output_path", "outputs/ros2_episode_logs/scan_policy_observations.csv")
        self.declare_parameter("scan_bins", 72)
        self.declare_parameter("require_evaluable", True)
        self.declare_parameter("require_nav2_plan", True)
        self.declare_parameter("max_scan_age_sec", 1.0)
        output_path = self.get_parameter("output_path").get_parameter_value().string_value
        self._scan_bins = max(8, int(self.get_parameter("scan_bins").value))
        self._require_evaluable = bool(self.get_parameter("require_evaluable").value)
        self._require_nav2_plan = bool(self.get_parameter("require_nav2_plan").value)
        self._max_scan_age_sec = max(0.0, float(self.get_parameter("max_scan_age_sec").value))
        self._scan_columns = [f"scan_bin_{idx:03d}" for idx in range(self._scan_bins)]

        self._path = Path(output_path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=BASE_COLUMNS + self._scan_columns)
        self._writer.writeheader()

        self._latest_scan: dict[str, object] | None = None
        self.create_subscription(LaserScan, "/scan", self._on_scan, 10)
        self.create_subscription(String, "/amr_reliability/policy_decision", self._on_decision, 10)
        self.get_logger().info("Recording scan-policy observations to %s" % self._path)

    def _now_sec(self) -> float:
        now = self.get_clock().now().to_msg()
        return float(now.sec) + float(now.nanosec) / 1e9

    def _on_scan(self, msg: LaserScan) -> None:
        ranges = list(msg.ranges)
        finite_ranges = [
            float(value)
            for value in ranges
            if math.isfinite(float(value)) and float(value) >= float(msg.range_min)
        ]
        max_range = max(float(msg.range_max), 1e-6)
        bins = self._downsample_ranges(ranges, float(msg.range_min), max_range)
        stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9
        front_values = self._front_ranges(ranges, float(msg.range_min), max_range)
        self._latest_scan = {
            "stamp_sec": stamp_sec,
            "received_sec": self._now_sec(),
            "n_ranges": len(ranges),
            "valid_fraction": len(finite_ranges) / max(len(ranges), 1),
            "min_range": min(finite_ranges) if finite_ranges else max_range,
            "mean_range": sum(finite_ranges) / max(len(finite_ranges), 1) if finite_ranges else max_range,
            "front_min_range": min(front_values) if front_values else max_range,
            "bins": bins,
        }

    def _downsample_ranges(self, ranges: list[float], range_min: float, range_max: float) -> list[float]:
        if not ranges:
            return [1.0] * self._scan_bins
        output = []
        for idx in range(self._scan_bins):
            start = int(idx * len(ranges) / self._scan_bins)
            end = int((idx + 1) * len(ranges) / self._scan_bins)
            segment = ranges[start : max(start + 1, end)]
            cleaned = [
                min(max(float(value), range_min), range_max) / range_max
                for value in segment
                if math.isfinite(float(value)) and float(value) >= range_min
            ]
            output.append(min(cleaned) if cleaned else 1.0)
        return output

    @staticmethod
    def _front_ranges(ranges: list[float], range_min: float, range_max: float) -> list[float]:
        if not ranges:
            return []
        width = max(1, len(ranges) // 12)
        indices = list(range(0, width)) + list(range(max(0, len(ranges) - width), len(ranges)))
        return [
            min(max(float(ranges[idx]), range_min), range_max)
            for idx in indices
            if math.isfinite(float(ranges[idx])) and float(ranges[idx]) >= range_min
        ]

    def _on_decision(self, msg: String) -> None:
        if self._latest_scan is None:
            return
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning("Skipping malformed policy payload: %s" % exc)
            return
        if self._require_evaluable and not _as_bool(payload.get("policy_evaluable", False)):
            return
        if self._require_nav2_plan and str(payload.get("expert_source", "")) != "nav2_plan":
            return
        scan_age = self._now_sec() - float(self._latest_scan["received_sec"])
        if self._max_scan_age_sec and scan_age > self._max_scan_age_sec:
            return

        robot_x = _safe_float(payload.get("robot_x"))
        robot_y = _safe_float(payload.get("robot_y"))
        target_x = _safe_float(payload.get("target_x"))
        target_y = _safe_float(payload.get("target_y"))
        goal_dx = target_x - robot_x
        goal_dy = target_y - robot_y
        row = {column: payload.get(column, "") for column in BASE_COLUMNS}
        row.update(
            {
                "goal_dx": round(goal_dx, 6),
                "goal_dy": round(goal_dy, 6),
                "goal_distance_l1": round(abs(goal_dx) + abs(goal_dy), 6),
                "scan_stamp_sec": round(float(self._latest_scan["stamp_sec"]), 6),
                "scan_age_sec": round(scan_age, 6),
                "scan_n_ranges": int(self._latest_scan["n_ranges"]),
                "scan_valid_fraction": round(float(self._latest_scan["valid_fraction"]), 6),
                "scan_min_range": round(float(self._latest_scan["min_range"]), 6),
                "scan_mean_range": round(float(self._latest_scan["mean_range"]), 6),
                "scan_front_min_range": round(float(self._latest_scan["front_min_range"]), 6),
            }
        )
        for column, value in zip(self._scan_columns, self._latest_scan["bins"]):
            row[column] = round(float(value), 6)
        self._writer.writerow(row)
        self._file.flush()

    def destroy_node(self) -> bool:
        if not self._file.closed:
            self._file.close()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = ScanPolicyObservationRecorder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
