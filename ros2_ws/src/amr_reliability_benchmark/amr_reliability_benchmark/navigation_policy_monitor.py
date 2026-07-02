from __future__ import annotations

import json
import math
from pathlib import Path

import rclpy
from nav_msgs.msg import Path as NavPath
from rclpy.node import Node
from std_msgs.msg import String


ACTIONS = ["STAY", "NORTH", "SOUTH", "EAST", "WEST"]
FEATURE_COLUMNS = [
    "time_step_norm",
    "observed_robot_x_norm",
    "observed_robot_y_norm",
    "target_x_norm",
    "target_y_norm",
    "goal_dx_norm",
    "goal_dy_norm",
    "distance_to_goal_norm",
    "local_free_north",
    "local_free_south",
    "local_free_east",
    "local_free_west",
    "sensor_confidence",
    "localization_uncertainty",
    "path_blocked_score",
    "obstacle_proximity",
    "trajectory_deviation",
    "task_progress_stagnation",
    "risk_score",
]


RECOVERY_ROUTES = {
    "perception_misread": "HUMAN_REVIEW",
    "localization_state_error": "RELOCALIZE",
    "blocked_path_misjudgment": "REPLAN",
    "control_tracking_error": "REPLAN",
    "policy_boundary_uncertainty": "CAUTIOUS_MODE",
    "mixed_mechanism_confusion": "HUMAN_REVIEW",
    "geometric_policy_error": "REPLAN",
    "none": "NORMAL_NAVIGATION",
}


def _dot(weight_row: list[float], values: list[float], bias: float) -> float:
    return sum(w * x for w, x in zip(weight_row, values)) + bias


def _softmax(logits: list[float]) -> list[float]:
    maximum = max(logits)
    exps = [math.exp(value - maximum) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


class JsonMlpPolicy:
    def __init__(self, model_path: str | Path) -> None:
        payload = json.loads(Path(model_path).read_text(encoding="utf-8"))
        self.actions = [str(action) for action in payload.get("actions", ACTIONS)]
        self.feature_columns = [str(col) for col in payload.get("feature_columns", FEATURE_COLUMNS)]
        self.scaler_mean = [float(value) for value in payload["scaler_mean"]]
        self.scaler_scale = [max(float(value), 1e-6) for value in payload["scaler_scale"]]
        self.layers = payload["layers"]

    def predict(self, features: dict[str, float]) -> tuple[str, list[float], float, float, float]:
        values = [float(features.get(col, 0.0)) for col in self.feature_columns]
        activations = [
            (value - mean) / scale
            for value, mean, scale in zip(values, self.scaler_mean, self.scaler_scale)
        ]
        for layer in self.layers:
            weight = layer["weight"]
            bias = layer["bias"]
            activations = [
                _dot([float(item) for item in weight_row], activations, float(layer_bias))
                for weight_row, layer_bias in zip(weight, bias)
            ]
            if layer.get("activation") == "relu":
                activations = [max(0.0, value) for value in activations]
        probs = _softmax(activations)
        pred_idx = max(range(len(probs)), key=lambda idx: probs[idx])
        sorted_probs = sorted(probs)
        entropy = -sum(prob * math.log(max(prob, 1e-12)) for prob in probs)
        margin = sorted_probs[-1] - sorted_probs[-2] if len(sorted_probs) >= 2 else 1.0
        return self.actions[pred_idx], probs, entropy, margin, sorted_probs[-1]


class NavigationPolicyMonitor(Node):
    """Run a task-policy probe beside Nav2 and diagnose policy action errors."""

    def __init__(self) -> None:
        super().__init__("navigation_policy_monitor")
        self.declare_parameter("model_path", "")
        self.declare_parameter("room_width", 18.0)
        self.declare_parameter("room_height", 12.0)
        self.declare_parameter("time_normalizer", 280.0)
        model_path = str(self.get_parameter("model_path").value)
        self._policy: JsonMlpPolicy | None = None
        if model_path:
            try:
                self._policy = JsonMlpPolicy(model_path)
                self.get_logger().info(f"Loaded navigation policy model: {model_path}")
            except (OSError, KeyError, json.JSONDecodeError, ValueError) as exc:
                self.get_logger().warning(f"Policy model unavailable; using heuristic probe: {exc}")
        else:
            self.get_logger().warning("No policy model_path provided; using heuristic probe.")

        self._room_width = max(float(self.get_parameter("room_width").value), 1e-6)
        self._room_height = max(float(self.get_parameter("room_height").value), 1e-6)
        self._time_normalizer = max(float(self.get_parameter("time_normalizer").value), 1e-6)
        self._latest_plan: list[tuple[float, float]] = []
        self._publisher = self.create_publisher(String, "/amr_reliability/policy_decision", 10)
        self.create_subscription(String, "/amr_reliability/runtime_metrics", self._on_metrics, 10)
        self.create_subscription(NavPath, "/plan", self._on_plan, 10)

    def _on_plan(self, msg: NavPath) -> None:
        self._latest_plan = [
            (float(pose.pose.position.x), float(pose.pose.position.y))
            for pose in msg.poses
        ]

    def _features_from_metrics(self, row: dict[str, object]) -> dict[str, float]:
        robot_x = float(row.get("robot_x", 0.0))
        robot_y = float(row.get("robot_y", 0.0))
        target_x = float(row.get("target_x", 0.0))
        target_y = float(row.get("target_y", 0.0))
        sensor = float(row.get("sensor_confidence", 0.95))
        path_blocked = float(row.get("path_blocked_score", 0.0))
        obstacle = float(row.get("obstacle_proximity", 0.0))
        dx = target_x - robot_x
        dy = target_y - robot_y
        free_forward = 0.0 if path_blocked >= 0.45 or obstacle >= 0.85 else 1.0
        features = {
            "time_step_norm": float(row.get("time_step", 0.0)) / self._time_normalizer,
            "observed_robot_x_norm": robot_x / self._room_width,
            "observed_robot_y_norm": robot_y / self._room_height,
            "target_x_norm": target_x / self._room_width,
            "target_y_norm": target_y / self._room_height,
            "goal_dx_norm": dx / self._room_width,
            "goal_dy_norm": dy / self._room_height,
            "distance_to_goal_norm": (abs(dx) + abs(dy)) / (self._room_width + self._room_height),
            "local_free_north": 1.0,
            "local_free_south": 1.0,
            "local_free_east": free_forward if dx > abs(dy) else 1.0,
            "local_free_west": free_forward if -dx > abs(dy) else 1.0,
            "sensor_confidence": sensor,
            "localization_uncertainty": float(row.get("localization_uncertainty", 0.0)),
            "path_blocked_score": path_blocked,
            "obstacle_proximity": obstacle,
            "trajectory_deviation": float(row.get("trajectory_deviation", 0.0)),
            "task_progress_stagnation": float(row.get("task_progress_stagnation", 0.0)),
            "risk_score": float(row.get("risk_score", 0.0)),
        }
        return features

    @staticmethod
    def _action_from_vector(dx: float, dy: float) -> str:
        if abs(dx) < 0.015 and abs(dy) < 0.015:
            return "STAY"
        if abs(dx) >= abs(dy):
            return "EAST" if dx >= 0 else "WEST"
        return "SOUTH" if dy >= 0 else "NORTH"

    def _expert_proxy_action(self, features: dict[str, float]) -> tuple[str, str, bool]:
        target_known = abs(features["target_x_norm"]) + abs(features["target_y_norm"]) > 1e-5
        if not target_known:
            return "UNKNOWN", "target_unavailable", False
        if features["path_blocked_score"] >= 0.70 or features["obstacle_proximity"] >= 0.90:
            return "STAY", "blocked_goal_direction_proxy", True
        dx = features["goal_dx_norm"]
        dy = features["goal_dy_norm"]
        return self._action_from_vector(dx, dy), "goal_direction_proxy", True

    def _plan_expert_action(self, robot_x: float, robot_y: float, fallback: tuple[str, str, bool]) -> tuple[str, str, bool]:
        if len(self._latest_plan) < 2:
            return fallback
        nearest_idx = min(
            range(len(self._latest_plan)),
            key=lambda idx: (self._latest_plan[idx][0] - robot_x) ** 2 + (self._latest_plan[idx][1] - robot_y) ** 2,
        )
        current_x, current_y = self._latest_plan[nearest_idx]
        for lookahead_x, lookahead_y in self._latest_plan[nearest_idx + 1 :]:
            dx = lookahead_x - current_x
            dy = lookahead_y - current_y
            if abs(dx) + abs(dy) >= 0.08:
                return self._action_from_vector(dx, dy), "nav2_plan", True
        return fallback

    def _heuristic_policy_action(self, features: dict[str, float]) -> tuple[str, list[float], float, float, float]:
        action, _, _ = self._expert_proxy_action(features)
        if action == "UNKNOWN":
            action = "STAY"
        probs = [0.02] * len(ACTIONS)
        probs[ACTIONS.index(action)] = 0.92
        total = sum(probs)
        probs = [prob / total for prob in probs]
        entropy = -sum(prob * math.log(max(prob, 1e-12)) for prob in probs)
        sorted_probs = sorted(probs)
        return action, probs, entropy, sorted_probs[-1] - sorted_probs[-2], sorted_probs[-1]

    def _diagnose(self, row: dict[str, object], entropy: float, margin: float) -> str:
        channels: list[str] = []
        if float(row.get("sensor_confidence", 1.0)) < 0.45:
            channels.append("perception_misread")
        if float(row.get("localization_uncertainty", 0.0)) >= 0.70:
            channels.append("localization_state_error")
        if float(row.get("path_blocked_score", 0.0)) >= 0.45 or float(row.get("obstacle_proximity", 0.0)) >= 0.75:
            channels.append("blocked_path_misjudgment")
        if float(row.get("trajectory_deviation", 0.0)) >= 0.75:
            channels.append("control_tracking_error")
        if entropy >= 0.65 or margin <= 0.25:
            channels.append("policy_boundary_uncertainty")
        if len(channels) >= 2:
            return "mixed_mechanism_confusion"
        return channels[0] if channels else "geometric_policy_error"

    def _on_metrics(self, msg: String) -> None:
        try:
            row = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        features = self._features_from_metrics(row)
        if self._policy is None:
            pred_action, probs, entropy, margin, max_prob = self._heuristic_policy_action(features)
        else:
            pred_action, probs, entropy, margin, max_prob = self._policy.predict(features)
        expert_action, expert_source, policy_evaluable = self._plan_expert_action(
            float(row.get("robot_x", 0.0)),
            float(row.get("robot_y", 0.0)),
            self._expert_proxy_action(features),
        )
        policy_correct = bool(policy_evaluable and pred_action == expert_action)
        mechanism = "none" if policy_correct or not policy_evaluable else self._diagnose(row, entropy, margin)
        out = dict(row)
        out.update(
            {
                "source": "navigation_policy_monitor",
                "policy_pred_action": pred_action,
                "expert_proxy_action": expert_action,
                "expert_source": expert_source,
                "policy_evaluable": policy_evaluable,
                "policy_correct": policy_correct,
                "policy_entropy": round(entropy, 6),
                "policy_margin": round(margin, 6),
                "policy_max_prob": round(max_prob, 6),
                "policy_error_mechanism": mechanism,
                "policy_recovery_route": RECOVERY_ROUTES[mechanism],
            }
        )
        for action, prob in zip(ACTIONS, probs):
            out[f"policy_prob_{action}"] = round(prob, 6)
        out_msg = String()
        out_msg.data = json.dumps(out, sort_keys=True)
        self._publisher.publish(out_msg)


def main() -> None:
    rclpy.init()
    node = NavigationPolicyMonitor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
