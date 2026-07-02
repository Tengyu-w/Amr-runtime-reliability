from __future__ import annotations

import json

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32, Int32, String

from amr_reliability_benchmark.reliability_logic import TelemetrySnapshot, metrics_from_telemetry


class Nav2TelemetryAdapter(Node):
    """Convert Nav2/Gazebo telemetry topics into the benchmark metrics schema."""

    def __init__(self) -> None:
        super().__init__("nav2_telemetry_adapter")
        self.declare_parameter("episode_id", "nav2_episode")
        self.declare_parameter("publish_period_sec", 0.25)
        self.declare_parameter("amcl_covariance_normalizer", 100.0)

        self._episode_id = str(self.get_parameter("episode_id").value)
        period = float(self.get_parameter("publish_period_sec").value)
        self._amcl_covariance_normalizer = max(
            float(self.get_parameter("amcl_covariance_normalizer").value),
            1e-6,
        )
        self._scenario_id = "unknown"
        self._step = 0
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._target_x = 0.0
        self._target_y = 0.0
        self._localization_covariance_trace = 0.0
        self._fault_localization_uncertainty = 0.0
        self._sensor_confidence = 0.95
        self._path_blocked_score = 0.0
        self._obstacle_proximity = 0.0
        self._trajectory_deviation = 0.0
        self._replanning_failure_count = 0
        self._task_progress_stagnation = 0.0

        self._publisher = self.create_publisher(String, "/amr_reliability/runtime_metrics", 10)
        self.create_subscription(String, "/amr_reliability/scenario", self._on_scenario, 10)
        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self._on_amcl_pose, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal_pose, 10)
        self.create_subscription(Float32, "/amr_reliability/localization_uncertainty", self._on_localization_uncertainty, 10)
        self.create_subscription(Float32, "/amr_reliability/sensor_confidence", self._on_sensor_confidence, 10)
        self.create_subscription(Float32, "/amr_reliability/path_blocked_score", self._on_path_blocked_score, 10)
        self.create_subscription(Float32, "/amr_reliability/obstacle_proximity", self._on_obstacle_proximity, 10)
        self.create_subscription(Float32, "/amr_reliability/trajectory_deviation", self._on_trajectory_deviation, 10)
        self.create_subscription(Float32, "/amr_reliability/task_progress_stagnation", self._on_stagnation, 10)
        self.create_subscription(Int32, "/amr_reliability/replanning_failure_count", self._on_replan_failures, 10)
        self._timer = self.create_timer(max(period, 0.05), self._publish_metrics)
        self.get_logger().info("Nav2 telemetry adapter publishing benchmark runtime metrics")

    def _on_scenario(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self._scenario_id = str(payload.get("scenario_id", "unknown"))

    def _on_odom(self, msg: Odometry) -> None:
        self._robot_x = float(msg.pose.pose.position.x)
        self._robot_y = float(msg.pose.pose.position.y)

    def _on_amcl_pose(self, msg: PoseWithCovarianceStamped) -> None:
        cov = msg.pose.covariance
        self._localization_covariance_trace = float(cov[0] + cov[7] + cov[35])

    def _on_localization_uncertainty(self, msg: Float32) -> None:
        self._fault_localization_uncertainty = float(msg.data)

    def _on_goal_pose(self, msg: PoseStamped) -> None:
        self._target_x = float(msg.pose.position.x)
        self._target_y = float(msg.pose.position.y)

    def _on_sensor_confidence(self, msg: Float32) -> None:
        self._sensor_confidence = float(msg.data)

    def _on_path_blocked_score(self, msg: Float32) -> None:
        self._path_blocked_score = float(msg.data)

    def _on_obstacle_proximity(self, msg: Float32) -> None:
        self._obstacle_proximity = float(msg.data)

    def _on_trajectory_deviation(self, msg: Float32) -> None:
        self._trajectory_deviation = float(msg.data)

    def _on_stagnation(self, msg: Float32) -> None:
        self._task_progress_stagnation = float(msg.data)

    def _on_replan_failures(self, msg: Int32) -> None:
        self._replanning_failure_count = int(msg.data)

    def _publish_metrics(self) -> None:
        amcl_localization_uncertainty = max(
            0.0,
            min(1.0, self._localization_covariance_trace / self._amcl_covariance_normalizer),
        )
        snapshot = TelemetrySnapshot(
            time_step=self._step,
            robot_x=self._robot_x,
            robot_y=self._robot_y,
            target_x=self._target_x,
            target_y=self._target_y,
            localization_covariance_trace=max(
                amcl_localization_uncertainty,
                self._fault_localization_uncertainty,
            ),
            sensor_confidence=self._sensor_confidence,
            path_blocked_score=self._path_blocked_score,
            obstacle_proximity=self._obstacle_proximity,
            trajectory_deviation=self._trajectory_deviation,
            replanning_failure_count=self._replanning_failure_count,
            task_progress_stagnation=self._task_progress_stagnation,
        )
        row = metrics_from_telemetry(snapshot).to_row()
        row.update(
            {
                "episode_id": self._episode_id,
                "scenario_id": self._scenario_id,
                "source": "nav2_telemetry_adapter",
            }
        )
        msg = String()
        msg.data = json.dumps(row, sort_keys=True)
        self._publisher.publish(msg)
        self._step += 1


def main() -> None:
    rclpy.init()
    node = Nav2TelemetryAdapter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
