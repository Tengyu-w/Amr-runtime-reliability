from __future__ import annotations

import json
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String


SHIFT_SCENARIOS = {
    "task_goal_shift_ood_style",
    "planner_backend_failure",
    "compound_shift_and_degradation",
}


class ScenarioGoalPublisher(Node):
    """Publish benchmark navigation goals for Nav2-compatible experiments."""

    def __init__(self) -> None:
        super().__init__("scenario_goal_publisher")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("goal_x", 4.5)
        self.declare_parameter("goal_y", 3.0)
        self.declare_parameter("alternate_goal_x", -4.5)
        self.declare_parameter("alternate_goal_y", 3.0)
        self.declare_parameter("goal_shift_step", 6)
        self.declare_parameter("publish_period_sec", 1.0)
        self.declare_parameter("initial_publish_delay_sec", 0.0)

        self._frame_id = str(self.get_parameter("frame_id").value)
        self._goal_x = float(self.get_parameter("goal_x").value)
        self._goal_y = float(self.get_parameter("goal_y").value)
        self._alternate_goal_x = float(self.get_parameter("alternate_goal_x").value)
        self._alternate_goal_y = float(self.get_parameter("alternate_goal_y").value)
        self._goal_shift_step = int(self.get_parameter("goal_shift_step").value)
        period = float(self.get_parameter("publish_period_sec").value)
        self._period = max(period, 0.1)
        initial_delay = float(self.get_parameter("initial_publish_delay_sec").value)
        self._scenario_id = "nominal"
        self._step = 0
        self._initial_delay_ticks = max(0, math.ceil(initial_delay / self._period))
        self._delay_ticks_remaining = self._initial_delay_ticks
        self._published_goal_signature: tuple[str, bool] | None = None

        self._publisher = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.create_subscription(String, "/amr_reliability/scenario", self._on_scenario, 10)
        self._timer = self.create_timer(self._period, self._publish_goal)
        self.get_logger().info("Scenario goal publisher active on /goal_pose")

    def _on_scenario(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        scenario_id = str(payload.get("scenario_id", "nominal"))
        if scenario_id != self._scenario_id:
            self._scenario_id = scenario_id
            self._step = 0
            self._delay_ticks_remaining = self._initial_delay_ticks
            self._published_goal_signature = None

    def _publish_goal(self) -> None:
        if self._delay_ticks_remaining > 0:
            self._delay_ticks_remaining -= 1
            return

        use_alternate = self._scenario_id in SHIFT_SCENARIOS and self._step >= self._goal_shift_step
        goal_signature = (self._scenario_id, use_alternate)
        self._step += 1
        if goal_signature == self._published_goal_signature:
            return

        x = self._alternate_goal_x if use_alternate else self._goal_x
        y = self._alternate_goal_y if use_alternate else self._goal_y
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = self._frame_id
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.w = 1.0
        self._publisher.publish(goal)
        self._published_goal_signature = goal_signature


def main() -> None:
    rclpy.init()
    node = ScenarioGoalPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
