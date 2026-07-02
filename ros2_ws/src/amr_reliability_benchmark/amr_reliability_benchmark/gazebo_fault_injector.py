from __future__ import annotations

import copy
import json
import math
import random
import subprocess
from dataclasses import dataclass

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Int32, String


BLOCKAGE_SCENARIOS = {
    "external_path_blockage",
    "progress_blockage",
    "planner_backend_failure",
    "compound_shift_and_degradation",
    "mixed_blockage_and_perception",
    "boundary_weak_blockage",
}
DRIFT_SCENARIOS = {
    "localization_drift",
    "planner_backend_failure",
    "compound_shift_and_degradation",
    "mixed_drift_and_execution",
}
PERCEPTION_SCENARIOS = {
    "perception_degradation",
    "planner_backend_failure",
    "compound_shift_and_degradation",
    "mixed_blockage_and_perception",
}
DEVIATION_SCENARIOS = {"execution_deviation", "mixed_drift_and_execution"}


@dataclass(frozen=True)
class FaultState:
    obstacle_x: float
    obstacle_y: float
    obstacle_active: bool
    scan_drop_rate: float
    scan_noise_std: float
    odom_drift_x: float
    odom_drift_y: float
    localization_uncertainty: float
    sensor_confidence: float
    path_blocked_score: float
    obstacle_proximity: float
    trajectory_deviation: float
    replanning_failure_count: int
    task_progress_stagnation: float


class GazeboFaultInjector(Node):
    """Inject physical and signal-level Gazebo faults for reliability episodes."""

    def __init__(self) -> None:
        super().__init__("gazebo_fault_injector")
        self.declare_parameter("world", "reliability_room")
        self.declare_parameter("obstacle_name", "dynamic_obstacle_placeholder")
        self.declare_parameter("timer_period_sec", 0.5)
        self.declare_parameter("seed", 17)
        self.declare_parameter("enable_gazebo_pose_service", True)

        self._world = str(self.get_parameter("world").value)
        self._obstacle_name = str(self.get_parameter("obstacle_name").value)
        self._enable_pose_service = bool(self.get_parameter("enable_gazebo_pose_service").value)
        period = max(float(self.get_parameter("timer_period_sec").value), 0.1)
        self._rng = random.Random(int(self.get_parameter("seed").value))
        self._scenario_id = "nominal"
        self._step = 0
        self._last_obstacle_pose: tuple[float, float] | None = None

        self.create_subscription(String, "/amr_reliability/scenario", self._on_scenario, 10)
        self.create_subscription(LaserScan, "/gazebo/scan", self._on_scan, 10)
        self.create_subscription(Odometry, "/gazebo/odom", self._on_odom, 10)

        self._scan_pub = self.create_publisher(LaserScan, "/scan", 10)
        self._odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self._event_pub = self.create_publisher(String, "/amr_reliability/simulation_fault_event", 10)
        self._localization_uncertainty = self.create_publisher(Float32, "/amr_reliability/localization_uncertainty", 10)
        self._sensor_confidence = self.create_publisher(Float32, "/amr_reliability/sensor_confidence", 10)
        self._path_blocked_score = self.create_publisher(Float32, "/amr_reliability/path_blocked_score", 10)
        self._obstacle_proximity = self.create_publisher(Float32, "/amr_reliability/obstacle_proximity", 10)
        self._trajectory_deviation = self.create_publisher(Float32, "/amr_reliability/trajectory_deviation", 10)
        self._stagnation = self.create_publisher(Float32, "/amr_reliability/task_progress_stagnation", 10)
        self._replan_failures = self.create_publisher(Int32, "/amr_reliability/replanning_failure_count", 10)
        self._timer = self.create_timer(period, self._on_timer)
        self.get_logger().info("Gazebo fault injector active: /gazebo/scan,/gazebo/odom -> /scan,/odom")

    def _on_scenario(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        scenario_id = str(payload.get("scenario_id", "nominal"))
        if scenario_id != self._scenario_id:
            self._scenario_id = scenario_id
            self._step = 0
            self._last_obstacle_pose = None

    def _fault_state(self) -> FaultState:
        phase = min(self._step / 36.0, 1.0)
        obstacle_active = self._scenario_id in BLOCKAGE_SCENARIOS
        if self._scenario_id == "progress_blockage":
            obstacle_active = 8 <= self._step <= 44
        if self._scenario_id == "boundary_weak_blockage":
            obstacle_active = 10 <= self._step <= 28
        if self._scenario_id == "nominal":
            obstacle_active = False

        if obstacle_active:
            obstacle_x = -0.15 + 0.35 * math.sin(0.16 * self._step)
            obstacle_y = -0.15 + 0.25 * math.cos(0.11 * self._step)
        else:
            obstacle_x = 0.0
            obstacle_y = 3.75

        scan_drop_rate = 0.0
        scan_noise_std = 0.01
        sensor_confidence = 0.92
        if self._scenario_id in PERCEPTION_SCENARIOS:
            scan_drop_rate = 0.12 + 0.48 * phase
            scan_noise_std = 0.04 + 0.12 * phase
            sensor_confidence = max(0.16, 0.88 - 0.72 * phase)

        drift_gain = 0.0
        if self._scenario_id in DRIFT_SCENARIOS:
            drift_gain = 0.92 * phase
        if self._scenario_id in DEVIATION_SCENARIOS:
            drift_gain = 0.40 * phase

        path_score = 0.72 if obstacle_active else 0.05
        obstacle_proximity = 0.82 if obstacle_active else 0.12
        if self._scenario_id == "boundary_weak_blockage":
            path_score = 0.34 if obstacle_active else 0.10
            obstacle_proximity = 0.68 if obstacle_active else 0.18
        trajectory_deviation = 0.12
        if self._scenario_id in DEVIATION_SCENARIOS:
            trajectory_deviation = min(0.95, 0.20 + 0.82 * phase)
        if self._scenario_id == "mixed_drift_and_execution":
            trajectory_deviation = min(0.95, 0.22 + 0.72 * phase)

        replan_failures = 0
        stagnation = 0.0
        if self._scenario_id == "progress_blockage":
            stagnation = 0.82 if obstacle_active else 0.25
        if self._scenario_id == "boundary_weak_blockage":
            stagnation = 0.48 if obstacle_active else 0.10
        if self._scenario_id == "planner_backend_failure":
            replan_failures = min(5, 1 + self._step // 12)
            stagnation = 0.90
            path_score = 0.94
        if self._scenario_id == "mixed_blockage_and_perception":
            stagnation = 0.58 if obstacle_active else 0.20

        localization_uncertainty = min(1.0, 0.10 + drift_gain)
        if self._scenario_id == "mixed_drift_and_execution":
            localization_uncertainty = max(localization_uncertainty, 0.80)
        return FaultState(
            obstacle_x=obstacle_x,
            obstacle_y=obstacle_y,
            obstacle_active=obstacle_active,
            scan_drop_rate=scan_drop_rate,
            scan_noise_std=scan_noise_std,
            odom_drift_x=0.45 * drift_gain,
            odom_drift_y=-0.28 * drift_gain,
            localization_uncertainty=localization_uncertainty,
            sensor_confidence=sensor_confidence,
            path_blocked_score=path_score,
            obstacle_proximity=obstacle_proximity,
            trajectory_deviation=trajectory_deviation,
            replanning_failure_count=replan_failures,
            task_progress_stagnation=stagnation,
        )

    def _on_scan(self, msg: LaserScan) -> None:
        state = self._fault_state()
        out = copy.deepcopy(msg)
        if state.scan_drop_rate > 0.0 or state.scan_noise_std > 0.01:
            degraded = []
            for value in msg.ranges:
                if not math.isfinite(value):
                    degraded.append(value)
                    continue
                if self._rng.random() < state.scan_drop_rate:
                    degraded.append(float("inf"))
                    continue
                noisy = value + self._rng.gauss(0.0, state.scan_noise_std)
                degraded.append(max(msg.range_min, min(msg.range_max, noisy)))
            out.ranges = degraded
        self._scan_pub.publish(out)

    def _on_odom(self, msg: Odometry) -> None:
        state = self._fault_state()
        out = copy.deepcopy(msg)
        out.pose.pose.position.x += state.odom_drift_x
        out.pose.pose.position.y += state.odom_drift_y
        if state.odom_drift_x or state.odom_drift_y:
            out.pose.covariance[0] = max(out.pose.covariance[0], state.localization_uncertainty)
            out.pose.covariance[7] = max(out.pose.covariance[7], state.localization_uncertainty)
            out.pose.covariance[35] = max(out.pose.covariance[35], 0.25 * state.localization_uncertainty)
        self._odom_pub.publish(out)

    def _publish_float(self, publisher, value: float) -> None:
        msg = Float32()
        msg.data = float(value)
        publisher.publish(msg)

    def _publish_fault_state(self, state: FaultState) -> None:
        self._publish_float(self._localization_uncertainty, state.localization_uncertainty)
        self._publish_float(self._sensor_confidence, state.sensor_confidence)
        self._publish_float(self._path_blocked_score, state.path_blocked_score)
        self._publish_float(self._obstacle_proximity, state.obstacle_proximity)
        self._publish_float(self._trajectory_deviation, state.trajectory_deviation)
        self._publish_float(self._stagnation, state.task_progress_stagnation)
        failures = Int32()
        failures.data = int(state.replanning_failure_count)
        self._replan_failures.publish(failures)
        event = String()
        event.data = json.dumps(
            {
                "scenario_id": self._scenario_id,
                "time_step": self._step,
                "obstacle_active": state.obstacle_active,
                "obstacle_x": round(state.obstacle_x, 4),
                "obstacle_y": round(state.obstacle_y, 4),
                "scan_drop_rate": round(state.scan_drop_rate, 4),
                "odom_drift_x": round(state.odom_drift_x, 4),
                "odom_drift_y": round(state.odom_drift_y, 4),
                "localization_uncertainty": round(state.localization_uncertainty, 4),
                "sensor_confidence": round(state.sensor_confidence, 4),
            },
            sort_keys=True,
        )
        self._event_pub.publish(event)

    def _set_obstacle_pose(self, state: FaultState) -> None:
        pose = (round(state.obstacle_x, 3), round(state.obstacle_y, 3))
        if pose == self._last_obstacle_pose or not self._enable_pose_service:
            return
        request = (
            f'name: "{self._obstacle_name}" '
            f'position {{ x: {pose[0]} y: {pose[1]} z: 0.35 }} '
            "orientation { w: 1.0 }"
        )
        try:
            subprocess.run(
                [
                    "gz",
                    "service",
                    "-s",
                    f"/world/{self._world}/set_pose",
                    "--reqtype",
                    "gz.msgs.Pose",
                    "--reptype",
                    "gz.msgs.Boolean",
                    "--timeout",
                    "400",
                    "--req",
                    request,
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=0.7,
            )
            self._last_obstacle_pose = pose
        except (OSError, subprocess.SubprocessError):
            self.get_logger().debug("Gazebo set_pose service unavailable for obstacle injection.")

    def _on_timer(self) -> None:
        state = self._fault_state()
        self._set_obstacle_pose(state)
        self._publish_fault_state(state)
        self._step += 1


def main() -> None:
    rclpy.init()
    node = GazeboFaultInjector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
