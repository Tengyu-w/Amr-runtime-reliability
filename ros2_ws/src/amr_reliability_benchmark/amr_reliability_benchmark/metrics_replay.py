from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from amr_reliability_benchmark.reliability_logic import replay_metrics_for_scenario
from amr_reliability_benchmark.scenario_catalog import scenario_by_id


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class MetricsReplay(Node):
    """Publish synthetic runtime metrics with the same schema as future Nav2 logs."""

    def __init__(self) -> None:
        super().__init__("metrics_replay")
        self.declare_parameter("scenario_id", "nominal")
        self.declare_parameter("episode_id", "smoke_episode")
        self.declare_parameter("steps", 12)
        self.declare_parameter("publish_period_sec", 0.25)
        self.declare_parameter("publish_once", False)

        self._scenario_id = str(self.get_parameter("scenario_id").value)
        self._episode_id = str(self.get_parameter("episode_id").value)
        self._steps = max(1, int(self.get_parameter("steps").value))
        self._publish_once = _as_bool(self.get_parameter("publish_once").value)
        period = float(self.get_parameter("publish_period_sec").value)
        scenario_by_id(self._scenario_id)

        self._publisher = self.create_publisher(String, "/amr_reliability/runtime_metrics", 10)
        self._step = 0
        self.get_logger().info(
            "Replaying metrics for scenario %s with %d steps" % (self._scenario_id, self._steps)
        )
        if self._publish_once:
            self._publish_step()
            return
        self._timer = self.create_timer(max(period, 0.05), self._publish_step)

    def _publish_step(self) -> None:
        if self._step >= self._steps:
            return
        metrics = replay_metrics_for_scenario(self._scenario_id, self._step, self._steps).to_row()
        metrics.update(
            {
                "episode_id": self._episode_id,
                "scenario_id": self._scenario_id,
                "source": "metrics_replay",
            }
        )
        msg = String()
        msg.data = json.dumps(metrics, sort_keys=True)
        self._publisher.publish(msg)
        self._step += 1


def main() -> None:
    rclpy.init()
    node = MetricsReplay()
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
