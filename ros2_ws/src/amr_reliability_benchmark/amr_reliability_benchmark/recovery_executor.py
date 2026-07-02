from __future__ import annotations

import csv
import json
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from rclpy.node import Node
from std_msgs.msg import String


CSV_COLUMNS = [
    "episode_id",
    "scenario_id",
    "time_step",
    "router_decision",
    "failure_mechanism",
    "executor_action",
    "robot_x",
    "robot_y",
    "goal_x",
    "goal_y",
    "result",
    "note",
]


class RecoveryExecutor(Node):
    """Translate reliability routes into Nav2-facing recovery actions.

    The executor is intentionally conservative. It does not bypass Nav2 or
    publish velocity commands. It only reissues goals or initial pose messages
    that the navigation stack already understands, then records the action.
    """

    def __init__(self) -> None:
        super().__init__("recovery_executor")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("output_path", "outputs/ros2_episode_logs/recovery_execution.csv")
        self.declare_parameter("replan_cooldown_steps", 20)
        self.declare_parameter("relocalize_cooldown_steps", 8)

        self._frame_id = str(self.get_parameter("frame_id").value)
        self._replan_cooldown_steps = max(int(self.get_parameter("replan_cooldown_steps").value), 0)
        self._relocalize_cooldown_steps = max(
            int(self.get_parameter("relocalize_cooldown_steps").value),
            0,
        )
        self._last_replan_step = -10_000
        self._last_relocalize_step = -10_000
        self._latest_goal: tuple[float, float] | None = None

        output_path = Path(str(self.get_parameter("output_path").value)).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = output_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS)
        self._writer.writeheader()

        self._goal_publisher = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self._initial_pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            "/initialpose",
            10,
        )
        self._event_publisher = self.create_publisher(
            String,
            "/amr_reliability/recovery_execution",
            10,
        )
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal_pose, 10)
        self.create_subscription(
            String,
            "/amr_reliability/router_decision",
            self._on_router_decision,
            10,
        )
        self.get_logger().info(
            "Recovery executor active: REPLAN reissues /goal_pose, "
            "RELOCALIZE publishes /initialpose"
        )

    def _on_goal_pose(self, msg: PoseStamped) -> None:
        self._latest_goal = (float(msg.pose.position.x), float(msg.pose.position.y))

    def _on_router_decision(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Skipping malformed router payload: {exc}")
            return

        decision = str(payload.get("router_decision", "NORMAL_NAVIGATION"))
        time_step = int(payload.get("time_step", 0))
        if decision == "REPLAN":
            event = self._execute_replan(payload, time_step)
        elif decision == "RELOCALIZE":
            event = self._execute_relocalize(payload, time_step)
        elif decision in {"CAUTIOUS_MODE", "HUMAN_REVIEW", "SAFE_STOP"}:
            event = self._record_non_nav2_action(payload, decision)
        else:
            return

        self._publish_event(event)
        self._write_event(event)

    def _execute_replan(self, payload: dict[str, object], time_step: int) -> dict[str, object]:
        goal = self._goal_from_payload(payload)
        if goal is None:
            return self._event(
                payload,
                "WAIT_FOR_VALID_GOAL",
                None,
                "skipped",
                "REPLAN received before a valid Nav2 goal was available",
            )
        if time_step - self._last_replan_step < self._replan_cooldown_steps:
            return self._event(
                payload,
                "REISSUE_GOAL_COOLDOWN",
                goal,
                "skipped",
                "recent REPLAN already issued",
            )

        self._last_replan_step = time_step
        self._publish_goal(goal)
        return self._event(
            payload,
            "REISSUE_GOAL_FOR_NAV2_REPLAN",
            goal,
            "published",
            "reissued current goal so Nav2 can replan with updated costmap",
        )

    def _execute_relocalize(self, payload: dict[str, object], time_step: int) -> dict[str, object]:
        robot = (
            float(payload.get("robot_x", 0.0)),
            float(payload.get("robot_y", 0.0)),
        )
        if time_step - self._last_relocalize_step < self._relocalize_cooldown_steps:
            return self._event(
                payload,
                "RELOCALIZE_COOLDOWN",
                self._goal_from_payload(payload),
                "skipped",
                "recent RELOCALIZE already issued",
            )

        self._last_relocalize_step = time_step
        self._publish_initial_pose(robot)
        return self._event(
            payload,
            "PUBLISH_INITIALPOSE_FOR_RELOCALIZE",
            self._goal_from_payload(payload),
            "published",
            "published robot pose estimate to /initialpose",
        )

    def _record_non_nav2_action(
        self,
        payload: dict[str, object],
        decision: str,
    ) -> dict[str, object]:
        action = {
            "CAUTIOUS_MODE": "RECORD_CAUTIOUS_MODE",
            "HUMAN_REVIEW": "RECORD_HUMAN_REVIEW_REQUIRED",
            "SAFE_STOP": "RECORD_SAFE_STOP_REQUIRED",
        }[decision]
        return self._event(
            payload,
            action,
            self._goal_from_payload(payload),
            "recorded",
            "route requires downstream controller or operator handling",
        )

    def _goal_from_payload(self, payload: dict[str, object]) -> tuple[float, float] | None:
        if self._latest_goal is not None:
            return self._latest_goal
        target_x = float(payload.get("target_x", 0.0))
        target_y = float(payload.get("target_y", 0.0))
        if abs(target_x) + abs(target_y) <= 1e-6:
            return None
        return (target_x, target_y)

    def _publish_goal(self, goal: tuple[float, float]) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.pose.position.x = goal[0]
        msg.pose.position.y = goal[1]
        msg.pose.orientation.w = 1.0
        self._goal_publisher.publish(msg)

    def _publish_initial_pose(self, robot: tuple[float, float]) -> None:
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.pose.pose.position.x = robot[0]
        msg.pose.pose.position.y = robot[1]
        msg.pose.pose.orientation.w = 1.0
        msg.pose.covariance[0] = 0.25
        msg.pose.covariance[7] = 0.25
        msg.pose.covariance[35] = 0.0685
        self._initial_pose_publisher.publish(msg)

    def _event(
        self,
        payload: dict[str, object],
        executor_action: str,
        goal: tuple[float, float] | None,
        result: str,
        note: str,
    ) -> dict[str, object]:
        return {
            "episode_id": payload.get("episode_id", ""),
            "scenario_id": payload.get("scenario_id", ""),
            "time_step": payload.get("time_step", 0),
            "router_decision": payload.get("router_decision", ""),
            "failure_mechanism": payload.get("failure_mechanism", ""),
            "executor_action": executor_action,
            "robot_x": payload.get("robot_x", 0.0),
            "robot_y": payload.get("robot_y", 0.0),
            "goal_x": "" if goal is None else goal[0],
            "goal_y": "" if goal is None else goal[1],
            "result": result,
            "note": note,
        }

    def _publish_event(self, event: dict[str, object]) -> None:
        msg = String()
        msg.data = json.dumps(event, sort_keys=True)
        self._event_publisher.publish(msg)

    def _write_event(self, event: dict[str, object]) -> None:
        self._writer.writerow({column: event.get(column, "") for column in CSV_COLUMNS})
        self._file.flush()

    def destroy_node(self) -> bool:
        if not self._file.closed:
            self._file.close()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = RecoveryExecutor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
