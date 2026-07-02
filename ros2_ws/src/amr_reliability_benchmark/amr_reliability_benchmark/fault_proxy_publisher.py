from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32, String

from amr_reliability_benchmark.reliability_logic import replay_metrics_for_scenario


class FaultProxyPublisher(Node):
    """Publish scenario-driven reliability proxy topics for Nav2/Gazebo experiments."""

    def __init__(self) -> None:
        super().__init__("fault_proxy_publisher")
        self.declare_parameter("steps", 12)
        self.declare_parameter("publish_period_sec", 0.5)
        self._scenario_id = "nominal"
        self._step = 0
        self._steps = max(1, int(self.get_parameter("steps").value))
        period = float(self.get_parameter("publish_period_sec").value)

        self.create_subscription(String, "/amr_reliability/scenario", self._on_scenario, 10)
        self._localization_uncertainty = self.create_publisher(Float32, "/amr_reliability/localization_uncertainty", 10)
        self._sensor_confidence = self.create_publisher(Float32, "/amr_reliability/sensor_confidence", 10)
        self._path_blocked_score = self.create_publisher(Float32, "/amr_reliability/path_blocked_score", 10)
        self._obstacle_proximity = self.create_publisher(Float32, "/amr_reliability/obstacle_proximity", 10)
        self._trajectory_deviation = self.create_publisher(Float32, "/amr_reliability/trajectory_deviation", 10)
        self._stagnation = self.create_publisher(Float32, "/amr_reliability/task_progress_stagnation", 10)
        self._replan_failures = self.create_publisher(Int32, "/amr_reliability/replanning_failure_count", 10)
        self._timer = self.create_timer(max(period, 0.05), self._publish_proxy_values)
        self.get_logger().info("Fault proxy publisher active")

    def _on_scenario(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        scenario_id = str(payload.get("scenario_id", "nominal"))
        if scenario_id != self._scenario_id:
            self._step = 0
        self._scenario_id = scenario_id

    def _publish_float(self, publisher, value: float) -> None:
        msg = Float32()
        msg.data = float(value)
        publisher.publish(msg)

    def _publish_proxy_values(self) -> None:
        time_step = min(self._step, self._steps - 1)
        metrics = replay_metrics_for_scenario(self._scenario_id, time_step, self._steps)
        self._publish_float(self._localization_uncertainty, metrics.localization_uncertainty)
        self._publish_float(self._sensor_confidence, metrics.sensor_confidence)
        self._publish_float(self._path_blocked_score, metrics.path_blocked_score)
        self._publish_float(self._obstacle_proximity, metrics.obstacle_proximity)
        self._publish_float(self._trajectory_deviation, metrics.trajectory_deviation)
        self._publish_float(self._stagnation, metrics.task_progress_stagnation)
        failures = Int32()
        failures.data = int(metrics.replanning_failure_count)
        self._replan_failures.publish(failures)
        self._step += 1


def main() -> None:
    rclpy.init()
    node = FaultProxyPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
