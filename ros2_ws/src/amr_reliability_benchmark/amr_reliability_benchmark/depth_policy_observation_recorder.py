from __future__ import annotations

import csv
import json
import math
import struct
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
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
    "depth_stamp_sec",
    "depth_age_sec",
    "depth_width",
    "depth_height",
    "depth_encoding",
    "depth_valid_fraction",
    "depth_min_m",
    "depth_mean_m",
    "depth_center_min_m",
]


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class DepthPolicyObservationRecorder(Node):
    """Align Gazebo depth images with Nav2-plan expert action labels."""

    def __init__(self) -> None:
        super().__init__("depth_policy_observation_recorder")
        self.declare_parameter("output_path", "outputs/ros2_episode_logs/depth_policy_observations.csv")
        self.declare_parameter("topic", "/depth_image")
        self.declare_parameter("grid_rows", 8)
        self.declare_parameter("grid_cols", 12)
        self.declare_parameter("max_depth_m", 8.0)
        self.declare_parameter("require_evaluable", True)
        self.declare_parameter("require_nav2_plan", True)
        self.declare_parameter("max_depth_age_sec", 1.0)

        output_path = self.get_parameter("output_path").get_parameter_value().string_value
        topic = self.get_parameter("topic").get_parameter_value().string_value
        self._grid_rows = max(2, int(self.get_parameter("grid_rows").value))
        self._grid_cols = max(2, int(self.get_parameter("grid_cols").value))
        self._max_depth_m = max(0.1, float(self.get_parameter("max_depth_m").value))
        self._require_evaluable = bool(self.get_parameter("require_evaluable").value)
        self._require_nav2_plan = bool(self.get_parameter("require_nav2_plan").value)
        self._max_depth_age_sec = max(0.0, float(self.get_parameter("max_depth_age_sec").value))
        self._depth_columns = [
            f"depth_cell_r{row:02d}_c{col:02d}"
            for row in range(self._grid_rows)
            for col in range(self._grid_cols)
        ]

        self._path = Path(output_path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=BASE_COLUMNS + self._depth_columns)
        self._writer.writeheader()

        self._latest_depth: dict[str, object] | None = None
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Image, topic, self._on_depth, sensor_qos)
        self.create_subscription(String, "/amr_reliability/policy_decision", self._on_decision, 10)
        self.get_logger().info("Recording depth-policy observations from %s to %s" % (topic, self._path))

    def _now_sec(self) -> float:
        now = self.get_clock().now().to_msg()
        return float(now.sec) + float(now.nanosec) / 1e9

    def _on_depth(self, msg: Image) -> None:
        grid, values = self._depth_grid(msg)
        if not grid:
            return
        stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9
        center_values = self._center_values(msg)
        self._latest_depth = {
            "stamp_sec": stamp_sec,
            "received_sec": self._now_sec(),
            "width": int(msg.width),
            "height": int(msg.height),
            "encoding": str(msg.encoding),
            "valid_fraction": len(values) / max(int(msg.width) * int(msg.height), 1),
            "min_m": min(values) if values else self._max_depth_m,
            "mean_m": sum(values) / max(len(values), 1) if values else self._max_depth_m,
            "center_min_m": min(center_values) if center_values else self._max_depth_m,
            "grid": grid,
        }

    def _pixel_depth(self, msg: Image, row: int, col: int) -> float | None:
        if row < 0 or col < 0 or row >= int(msg.height) or col >= int(msg.width):
            return None
        encoding = str(msg.encoding).upper()
        if encoding in {"32FC1", "R_FLOAT32"}:
            offset = int(row) * int(msg.step) + int(col) * 4
            if offset + 4 > len(msg.data):
                return None
            value = struct.unpack_from("<f", bytes(msg.data), offset)[0]
        elif encoding in {"16UC1", "MONO16"}:
            offset = int(row) * int(msg.step) + int(col) * 2
            if offset + 2 > len(msg.data):
                return None
            value = float(struct.unpack_from("<H", bytes(msg.data), offset)[0]) / 1000.0
        else:
            return None
        if not math.isfinite(value) or value <= 0.0:
            return None
        return min(max(float(value), 0.0), self._max_depth_m)

    def _depth_grid(self, msg: Image) -> tuple[list[float], list[float]]:
        if int(msg.width) <= 0 or int(msg.height) <= 0:
            return [], []
        grid = []
        all_values = []
        for grid_row in range(self._grid_rows):
            row_start = int(grid_row * int(msg.height) / self._grid_rows)
            row_end = int((grid_row + 1) * int(msg.height) / self._grid_rows)
            for grid_col in range(self._grid_cols):
                col_start = int(grid_col * int(msg.width) / self._grid_cols)
                col_end = int((grid_col + 1) * int(msg.width) / self._grid_cols)
                samples = []
                for row_frac in (0.25, 0.5, 0.75):
                    row = row_start + int(max(row_end - row_start - 1, 0) * row_frac)
                    for col_frac in (0.25, 0.5, 0.75):
                        col = col_start + int(max(col_end - col_start - 1, 0) * col_frac)
                        value = self._pixel_depth(msg, row, col)
                        if value is not None:
                            samples.append(value)
                            all_values.append(value)
                cell_depth = min(samples) if samples else self._max_depth_m
                grid.append(round(cell_depth / self._max_depth_m, 6))
        return grid, all_values

    def _center_values(self, msg: Image) -> list[float]:
        row0 = int(msg.height * 0.35)
        row1 = int(msg.height * 0.65)
        col0 = int(msg.width * 0.35)
        col1 = int(msg.width * 0.65)
        values = []
        for row_frac in (0.2, 0.5, 0.8):
            row = row0 + int(max(row1 - row0 - 1, 0) * row_frac)
            for col_frac in (0.2, 0.5, 0.8):
                col = col0 + int(max(col1 - col0 - 1, 0) * col_frac)
                value = self._pixel_depth(msg, row, col)
                if value is not None:
                    values.append(value)
        return values

    def _on_decision(self, msg: String) -> None:
        if self._latest_depth is None:
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
        depth_age = self._now_sec() - float(self._latest_depth["received_sec"])
        if self._max_depth_age_sec and depth_age > self._max_depth_age_sec:
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
                "depth_stamp_sec": round(float(self._latest_depth["stamp_sec"]), 6),
                "depth_age_sec": round(depth_age, 6),
                "depth_width": int(self._latest_depth["width"]),
                "depth_height": int(self._latest_depth["height"]),
                "depth_encoding": str(self._latest_depth["encoding"]),
                "depth_valid_fraction": round(float(self._latest_depth["valid_fraction"]), 6),
                "depth_min_m": round(float(self._latest_depth["min_m"]), 6),
                "depth_mean_m": round(float(self._latest_depth["mean_m"]), 6),
                "depth_center_min_m": round(float(self._latest_depth["center_min_m"]), 6),
            }
        )
        for column, value in zip(self._depth_columns, self._latest_depth["grid"]):
            row[column] = value
        self._writer.writerow(row)
        self._file.flush()

    def destroy_node(self) -> bool:
        if not self._file.closed:
            self._file.close()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = DepthPolicyObservationRecorder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
