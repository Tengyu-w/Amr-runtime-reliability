from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from amr_reliability_benchmark.scenario_catalog import scenario_by_id


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class ScenarioRunner(Node):
    """Publish the active benchmark scenario for downstream experiment nodes."""

    def __init__(self) -> None:
        super().__init__("scenario_runner")
        self.declare_parameter("scenario_id", "nominal")
        self.declare_parameter("publish_once", False)
        self.declare_parameter("publish_period_sec", 1.0)

        scenario_id = str(self.get_parameter("scenario_id").value)
        self._publish_once = _as_bool(self.get_parameter("publish_once").value)
        period = float(self.get_parameter("publish_period_sec").value)

        try:
            self._scenario = scenario_by_id(scenario_id)
        except ValueError as exc:
            self.get_logger().error(str(exc))
            raise

        self._publisher = self.create_publisher(String, "/amr_reliability/scenario", 10)
        self.get_logger().info(
            "Loaded scenario %s -> expected recovery %s"
            % (self._scenario["scenario_id"], self._scenario["expected_recovery"])
        )
        self._publish_scenario()

        if self._publish_once:
            return

        self._timer = self.create_timer(max(period, 0.1), self._publish_scenario)

    def _publish_scenario(self) -> None:
        msg = String()
        msg.data = json.dumps(self._scenario, sort_keys=True)
        self._publisher.publish(msg)


def main() -> None:
    rclpy.init()
    node = ScenarioRunner()
    if node._publish_once:
        node.destroy_node()
        rclpy.shutdown()
        return

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
