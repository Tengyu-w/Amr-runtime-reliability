from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from amr_reliability_benchmark.reliability_logic import (
    ReliabilityMetrics,
    diagnose_failure_mechanism,
    route_metrics,
)


def _metrics_from_payload(payload: dict[str, object]) -> ReliabilityMetrics:
    return ReliabilityMetrics(
        time_step=int(payload.get("time_step", 0)),
        robot_x=float(payload.get("robot_x", 0.0)),
        robot_y=float(payload.get("robot_y", 0.0)),
        target_x=float(payload.get("target_x", 0.0)),
        target_y=float(payload.get("target_y", 0.0)),
        localization_uncertainty=float(payload.get("localization_uncertainty", 0.0)),
        sensor_confidence=float(payload.get("sensor_confidence", 1.0)),
        path_blocked_score=float(payload.get("path_blocked_score", 0.0)),
        obstacle_proximity=float(payload.get("obstacle_proximity", 0.0)),
        trajectory_deviation=float(payload.get("trajectory_deviation", 0.0)),
        replanning_failure_count=int(payload.get("replanning_failure_count", 0)),
        task_progress_stagnation=float(payload.get("task_progress_stagnation", 0.0)),
        risk_score=float(payload.get("risk_score", 0.0)),
    )


class RuntimeRouter(Node):
    """Route runtime metrics to recovery actions and publish structured decisions."""

    def __init__(self) -> None:
        super().__init__("runtime_router")
        self._publisher = self.create_publisher(String, "/amr_reliability/router_decision", 10)
        self._subscription = self.create_subscription(
            String,
            "/amr_reliability/runtime_metrics",
            self._on_metrics,
            10,
        )
        self.get_logger().info("Runtime router listening on /amr_reliability/runtime_metrics")

    def _on_metrics(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            metrics = _metrics_from_payload(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warning("Skipping malformed metrics payload: %s" % exc)
            return

        mechanism = diagnose_failure_mechanism(metrics).value
        decision = route_metrics(metrics).value
        routed = dict(payload)
        routed.update(
            {
                "failure_mechanism": mechanism,
                "router_decision": decision,
                "router_mode": "mechanism_router",
            }
        )
        out = String()
        out.data = json.dumps(routed, sort_keys=True)
        self._publisher.publish(out)


def main() -> None:
    rclpy.init()
    node = RuntimeRouter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
