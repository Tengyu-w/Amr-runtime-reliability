"""Train and audit a navigation policy before building recovery routes.

This is the ECG-style pipeline for the AMR demo:

1. Train a task policy that chooses movement actions.
2. Roll the policy out under controlled faults and observe its errors.
3. Diagnose policy error mechanisms from internal uncertainty/embedding evidence.
4. Map diagnosed mechanisms to recovery routes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn

DEMO_ROOT = Path(__file__).resolve().parents[1]
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from experiments.generate_scenario_dataset import SCENARIOS, ScenarioSpec
from src.amr_agent import AMRAgent
from src.decision_router import RouterDecision
from src.environment import WarehouseEnvironment
from src.failure_injection import FailureInjector
from src.planner import AStarPlanner
from src.utils import GridPosition, SimulationConfig, clamp, ensure_output_dir, manhattan


ACTIONS = ["STAY", "NORTH", "SOUTH", "EAST", "WEST"]
ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTIONS)}
ACTION_DELTAS: dict[str, GridPosition] = {
    "STAY": (0, 0),
    "NORTH": (0, -1),
    "SOUTH": (0, 1),
    "EAST": (1, 0),
    "WEST": (-1, 0),
}
DELTA_TO_ACTION = {delta: action for action, delta in ACTION_DELTAS.items()}

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

CORE_TRAIN_SCENARIOS = {
    "nominal",
    "external_path_blockage",
    "localization_drift",
    "perception_degradation",
    "execution_deviation",
    "progress_blockage",
}


class NavigationPolicyNet(nn.Module):
    """Movement policy with an inspectable representation layer."""

    def __init__(self, n_features: int, n_actions: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, n_actions)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x)
        return self.head(embedding), embedding


@dataclass
class TrainedNavigationPolicy:
    model: NavigationPolicyNet
    scaler: StandardScaler
    feature_columns: list[str]


def _split_for_seed(seed: int) -> str:
    bucket = seed % 10
    if bucket <= 5:
        return "train"
    if bucket <= 7:
        return "val"
    return "test"


def _action_from_step(current: GridPosition, next_cell: GridPosition | None) -> str:
    if next_cell is None:
        return "STAY"
    delta = (next_cell[0] - current[0], next_cell[1] - current[1])
    return DELTA_TO_ACTION.get(delta, "STAY")


def _next_cell(position: GridPosition, action: str) -> GridPosition:
    dx, dy = ACTION_DELTAS[action]
    return position[0] + dx, position[1] + dy


def _expert_action(environment: WarehouseEnvironment, position: GridPosition, target: GridPosition) -> str:
    path = AStarPlanner(environment).plan(position, target, include_dynamic=True)
    if len(path) < 2:
        return "STAY"
    return _action_from_step(position, path[1])


def _scenario_by_id(scenario_id: str) -> ScenarioSpec:
    for scenario in SCENARIOS:
        if scenario.scenario_id == scenario_id:
            return scenario
    raise ValueError(f"Unknown scenario: {scenario_id}")


def _scenario_injector(seed: int, scenario: ScenarioSpec) -> FailureInjector:
    events = set(scenario.enabled_events)
    if scenario.scenario_id == "boundary_weak_blockage":
        events.add("dynamic_obstacles_cleared")
    return FailureInjector(seed=seed, enabled_events=events)


def _apply_boundary_weak_blockage(
    scenario_id: str,
    time_step: int,
    environment: WarehouseEnvironment,
    agent: AMRAgent,
) -> list[str]:
    if scenario_id != "boundary_weak_blockage":
        return []
    if 12 <= time_step <= 20 and time_step % 2 == 0:
        next_cell = agent.next_path_cell()
        if next_cell and environment.add_dynamic_obstacle(next_cell):
            return ["weak_dynamic_obstacle_blocks_path"]
    if time_step == 24:
        environment.clear_dynamic_obstacles()
        return ["dynamic_obstacles_cleared"]
    return []


def _risk_score(
    localization_uncertainty: float,
    sensor_confidence: float,
    path_blocked_score: float,
    obstacle_proximity: float,
    trajectory_deviation: float,
    stagnation: float,
) -> float:
    return round(
        clamp(
            0.20 * localization_uncertainty
            + 0.18 * (1.0 - sensor_confidence)
            + 0.22 * path_blocked_score
            + 0.14 * obstacle_proximity
            + 0.16 * trajectory_deviation
            + 0.10 * stagnation
        ),
        4,
    )


def _fault_observation(
    scenario_id: str,
    time_step: int,
    agent: AMRAgent,
    environment: WarehouseEnvironment,
) -> dict[str, float]:
    phase = min(time_step / 45.0, 1.0)
    path_blocked = 1.0 if agent.next_path_cell() in environment.dynamic_obstacles else 0.0
    obstacle_distance = environment.obstacle_distance(agent.position)
    obstacle_proximity = clamp(1.0 - obstacle_distance / 8.0)
    localization = agent.localization_uncertainty
    sensor = agent.sensor_confidence
    deviation = 0.90 if agent.deviated_from_path else 0.08
    stagnation = clamp(agent.stagnant_steps / 8.0)

    if scenario_id == "mixed_blockage_and_perception":
        sensor = min(sensor, max(0.18, 0.82 - 0.58 * phase))
        path_blocked = max(path_blocked, 0.70)
        obstacle_proximity = max(obstacle_proximity, 0.78)
        stagnation = max(stagnation, 0.55)
    elif scenario_id == "mixed_drift_and_execution":
        localization = max(localization, 0.25 + 0.62 * phase)
        deviation = max(deviation, 0.25 + 0.65 * phase)
    elif scenario_id == "boundary_weak_blockage":
        path_blocked = max(path_blocked, 0.32 if 12 <= time_step <= 24 else 0.08)
        obstacle_proximity = max(obstacle_proximity, 0.62 if 12 <= time_step <= 24 else 0.18)
        stagnation = max(stagnation, 0.42 if 12 <= time_step <= 24 else 0.05)

    risk = _risk_score(localization, sensor, path_blocked, obstacle_proximity, deviation, stagnation)
    return {
        "localization_uncertainty": round(clamp(localization), 4),
        "sensor_confidence": round(clamp(sensor), 4),
        "path_blocked_score": round(clamp(path_blocked), 4),
        "obstacle_proximity": round(clamp(obstacle_proximity), 4),
        "trajectory_deviation": round(clamp(deviation), 4),
        "task_progress_stagnation": round(clamp(stagnation), 4),
        "risk_score": risk,
    }


def _observed_position(
    true_position: GridPosition,
    observation: dict[str, float],
    scenario_id: str,
) -> tuple[float, float]:
    drift = observation["localization_uncertainty"]
    if scenario_id in {"localization_drift", "mixed_drift_and_execution", "compound_shift_and_degradation"}:
        return true_position[0] + 1.6 * drift, true_position[1] - 1.1 * drift
    return float(true_position[0]), float(true_position[1])


def _local_free_features(
    environment: WarehouseEnvironment,
    position: GridPosition,
    sensor_confidence: float,
    scenario_id: str,
) -> dict[str, float]:
    features = {}
    hide_dynamic = sensor_confidence < 0.45 or scenario_id == "mixed_blockage_and_perception"
    for action, name in [
        ("NORTH", "local_free_north"),
        ("SOUTH", "local_free_south"),
        ("EAST", "local_free_east"),
        ("WEST", "local_free_west"),
    ]:
        cell = _next_cell(position, action)
        static_blocked = environment.is_blocked(cell, include_dynamic=False)
        dynamic_blocked = cell in environment.dynamic_obstacles
        perceived_blocked = static_blocked or (dynamic_blocked and not hide_dynamic)
        features[name] = 0.0 if perceived_blocked else 1.0
    return features


def _feature_row(
    scenario: ScenarioSpec,
    seed: int,
    time_step: int,
    environment: WarehouseEnvironment,
    agent: AMRAgent,
    observation: dict[str, float],
) -> dict[str, float | int | str | bool]:
    observed_x, observed_y = _observed_position(agent.position, observation, scenario.scenario_id)
    width = max(environment.width - 1, 1)
    height = max(environment.height - 1, 1)
    target = agent.target
    local_free = _local_free_features(
        environment,
        agent.position,
        observation["sensor_confidence"],
        scenario.scenario_id,
    )
    distance = manhattan(agent.position, target)
    return {
        "scenario_id": scenario.scenario_id,
        "seed": seed,
        "split": _split_for_seed(seed),
        "time_step": time_step,
        "time_step_norm": time_step / 70.0,
        "true_robot_x": agent.position[0],
        "true_robot_y": agent.position[1],
        "observed_robot_x": round(observed_x, 4),
        "observed_robot_y": round(observed_y, 4),
        "observed_robot_x_norm": observed_x / width,
        "observed_robot_y_norm": observed_y / height,
        "target_x": target[0],
        "target_y": target[1],
        "target_x_norm": target[0] / width,
        "target_y_norm": target[1] / height,
        "goal_dx_norm": (target[0] - observed_x) / width,
        "goal_dy_norm": (target[1] - observed_y) / height,
        "distance_to_goal_norm": distance / (width + height),
        "scenario_primary_fault_origin": scenario.primary_fault_origin,
        "scenario_primary_fault_family": scenario.primary_fault_family,
        "scenario_primary_ood_status": scenario.primary_ood_status,
        **local_free,
        **observation,
    }


def generate_navigation_policy_demonstrations(
    seeds: list[int],
    scenarios: list[str],
    max_steps: int = 70,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario_id in scenarios:
        scenario = _scenario_by_id(scenario_id)
        for seed in seeds:
            config = SimulationConfig(seed=seed, max_steps=max_steps)
            environment = WarehouseEnvironment(config=config)
            agent = AMRAgent(position=environment.start, target=environment.target)
            planner = AStarPlanner(environment)
            agent.set_path(planner.plan(agent.position, agent.target))
            injector = _scenario_injector(seed, scenario)
            for time_step in range(max_steps):
                events = injector.apply(time_step, environment, agent)
                events.extend(_apply_boundary_weak_blockage(scenario_id, time_step, environment, agent))
                if agent.needs_replan or not agent.path or agent.next_path_cell() is None:
                    agent.set_path(planner.plan(agent.position, agent.target))
                observation = _fault_observation(scenario_id, time_step, agent, environment)
                expert_action = _expert_action(environment, agent.position, agent.target)
                row = _feature_row(scenario, seed, time_step, environment, agent, observation)
                next_cell = _next_cell(agent.position, expert_action)
                row.update(
                    {
                        "episode_id": f"navigation_demo_{scenario_id}__seed_{seed}",
                        "expert_action": expert_action,
                        "expert_action_index": ACTION_TO_INDEX[expert_action],
                        "fault_events": "|".join(events) if events else "none",
                        "completed": agent.completed,
                    }
                )
                rows.append(row)
                if expert_action == "STAY" or environment.is_blocked(next_cell):
                    agent._update_progress()
                else:
                    agent.position = next_cell
                    agent.completed = agent.position == agent.target
                    agent._update_progress()
                if agent.completed:
                    break
    return pd.DataFrame(rows)


def _prepare_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df.copy()
    for column in FEATURE_COLUMNS:
        table[column] = pd.to_numeric(table[column], errors="coerce").fillna(0.0)
    table["expert_action_index"] = table["expert_action"].map(ACTION_TO_INDEX).astype(int)
    return table


def _class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=len(ACTIONS)).astype(np.float32)
    total = max(float(counts.sum()), 1.0)
    return torch.tensor(total / (len(ACTIONS) * np.maximum(counts, 1.0)), dtype=torch.float32)


def train_navigation_policy(
    table: pd.DataFrame,
    train_scenarios: set[str],
    epochs: int = 300,
    lr: float = 0.01,
    seed: int = 23,
) -> TrainedNavigationPolicy:
    torch.manual_seed(seed)
    train = table[table["split"].eq("train") & table["scenario_id"].isin(train_scenarios)]
    if train.empty:
        raise ValueError("No training rows after scenario/split filtering.")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[FEATURE_COLUMNS].to_numpy(float)).astype(np.float32)
    y_train = train["expert_action_index"].to_numpy(int)
    model = NavigationPolicyNet(len(FEATURE_COLUMNS), len(ACTIONS))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=_class_weights(y_train))
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits, _ = model(x_tensor)
        loss = loss_fn(logits, y_tensor)
        loss.backward()
        optimizer.step()
    return TrainedNavigationPolicy(model=model, scaler=scaler, feature_columns=list(FEATURE_COLUMNS))


def _predict(policy: TrainedNavigationPolicy, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = policy.scaler.transform(df[policy.feature_columns].to_numpy(float)).astype(np.float32)
    policy.model.eval()
    with torch.no_grad():
        logits, embedding = policy.model(torch.tensor(x, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs, embedding.cpu().numpy()


def _score_supervised(policy: TrainedNavigationPolicy, table: pd.DataFrame) -> pd.DataFrame:
    scored_parts = []
    train = table[table["split"].eq("train")].copy()
    _, train_emb = _predict(policy, train)
    train_y = train["expert_action_index"].to_numpy(int)
    centroids = {
        action: train_emb[train_y == idx].mean(axis=0)
        for idx, action in enumerate(ACTIONS)
        if np.any(train_y == idx)
    }
    for split, split_df in table.groupby("split"):
        split_df = split_df.copy()
        probs, embedding = _predict(policy, split_df)
        pred_idx = probs.argmax(axis=1)
        sorted_probs = np.sort(probs, axis=1)
        split_df["policy_pred_action"] = [ACTIONS[idx] for idx in pred_idx]
        split_df["policy_correct"] = pred_idx == split_df["expert_action_index"].to_numpy(int)
        split_df["policy_entropy"] = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
        split_df["policy_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
        split_df["policy_max_prob"] = sorted_probs[:, -1]
        split_df["embedding_norm"] = np.linalg.norm(embedding, axis=1)
        nearest = []
        teacher_dist = []
        pred_dist = []
        for i, vector in enumerate(embedding):
            distances = {action: float(np.linalg.norm(vector - centroid)) for action, centroid in centroids.items()}
            nearest.append(min(distances.values()) if distances else math.nan)
            teacher_dist.append(distances.get(str(split_df["expert_action"].iloc[i]), math.nan))
            pred_dist.append(distances.get(str(split_df["policy_pred_action"].iloc[i]), math.nan))
        split_df["embedding_nearest_action_distance"] = nearest
        split_df["embedding_distance_to_expert_action"] = teacher_dist
        split_df["embedding_distance_to_pred_action"] = pred_dist
        for idx, action in enumerate(ACTIONS):
            split_df[f"prob_{action}"] = probs[:, idx]
        scored_parts.append(split_df)
    return pd.concat(scored_parts, ignore_index=True)


def _diagnose_policy_error(row: pd.Series) -> str:
    channels = []
    if float(row.get("sensor_confidence", 1.0)) < 0.45:
        channels.append("perception_misread")
    if float(row.get("localization_uncertainty", 0.0)) >= 0.70:
        channels.append("localization_state_error")
    if float(row.get("path_blocked_score", 0.0)) >= 0.45 or float(row.get("obstacle_proximity", 0.0)) >= 0.75:
        channels.append("blocked_path_misjudgment")
    if float(row.get("trajectory_deviation", 0.0)) >= 0.75:
        channels.append("control_tracking_error")
    if float(row.get("policy_entropy", 0.0)) >= 0.65 or float(row.get("policy_margin", 1.0)) <= 0.25:
        channels.append("policy_boundary_uncertainty")
    if len(channels) >= 2:
        return "mixed_mechanism_confusion"
    return channels[0] if channels else "geometric_policy_error"


def _route_for_policy_error(mechanism: str) -> str:
    mapping = {
        "perception_misread": RouterDecision.HUMAN_REVIEW.value,
        "localization_state_error": RouterDecision.RELOCALIZE.value,
        "blocked_path_misjudgment": RouterDecision.REPLAN.value,
        "control_tracking_error": RouterDecision.REPLAN.value,
        "policy_boundary_uncertainty": RouterDecision.CAUTIOUS_MODE.value,
        "mixed_mechanism_confusion": RouterDecision.HUMAN_REVIEW.value,
        "geometric_policy_error": RouterDecision.REPLAN.value,
    }
    return mapping[mechanism]


def _score_closed_loop_rollouts(
    policy: TrainedNavigationPolicy,
    seeds: list[int],
    scenarios: list[str],
    max_steps: int = 70,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario_id in scenarios:
        scenario = _scenario_by_id(scenario_id)
        for seed in seeds:
            config = SimulationConfig(seed=seed, max_steps=max_steps)
            environment = WarehouseEnvironment(config=config)
            agent = AMRAgent(position=environment.start, target=environment.target)
            planner = AStarPlanner(environment)
            agent.set_path(planner.plan(agent.position, agent.target))
            injector = _scenario_injector(seed, scenario)
            for time_step in range(max_steps):
                events = injector.apply(time_step, environment, agent)
                events.extend(_apply_boundary_weak_blockage(scenario_id, time_step, environment, agent))
                if agent.needs_replan or not agent.path:
                    agent.set_path(planner.plan(agent.position, agent.target))
                observation = _fault_observation(scenario_id, time_step, agent, environment)
                expert_action = _expert_action(environment, agent.position, agent.target)
                row = _feature_row(scenario, seed, time_step, environment, agent, observation)
                table_row = _prepare_table(pd.DataFrame([{**row, "expert_action": expert_action}]))
                probs, embedding = _predict(policy, table_row)
                pred_idx = int(probs[0].argmax())
                pred_action = ACTIONS[pred_idx]
                sorted_probs = np.sort(probs[0])
                pred_cell = _next_cell(agent.position, pred_action)
                attempted_blocked_move = pred_action != "STAY" and environment.is_blocked(pred_cell)
                expert_cell = _next_cell(agent.position, expert_action)
                expert_reduces_distance = manhattan(expert_cell, agent.target) < manhattan(agent.position, agent.target)
                pred_reduces_distance = manhattan(pred_cell, agent.target) < manhattan(agent.position, agent.target)
                policy_error = pred_action != expert_action
                row.update(
                    {
                        "episode_id": f"navigation_rollout_{scenario_id}__seed_{seed}",
                        "expert_action": expert_action,
                        "policy_pred_action": pred_action,
                        "policy_correct": not policy_error,
                        "policy_entropy": float(-np.sum(probs[0] * np.log(np.maximum(probs[0], 1e-12)))),
                        "policy_margin": float(sorted_probs[-1] - sorted_probs[-2]),
                        "policy_max_prob": float(sorted_probs[-1]),
                        "embedding_norm": float(np.linalg.norm(embedding[0])),
                        "attempted_blocked_move": attempted_blocked_move,
                        "progress_wrong_direction": bool(expert_reduces_distance and not pred_reduces_distance),
                        "fault_events": "|".join(events) if events else "none",
                        "completed": agent.completed,
                    }
                )
                if policy_error:
                    mechanism = _diagnose_policy_error(pd.Series(row))
                    row["policy_error_mechanism"] = mechanism
                    row["recovery_route"] = _route_for_policy_error(mechanism)
                else:
                    row["policy_error_mechanism"] = "none"
                    row["recovery_route"] = RouterDecision.NORMAL_NAVIGATION.value
                rows.append(row)
                if not attempted_blocked_move and pred_action != "STAY":
                    agent.position = pred_cell
                    agent.completed = agent.position == agent.target
                agent._update_progress()
                if agent.completed:
                    break
    return pd.DataFrame(rows)


def _metric_rows(scored: pd.DataFrame, pred_col: str = "policy_pred_action") -> list[dict[str, object]]:
    rows = []
    for split, sub in scored.groupby("split"):
        y = sub["expert_action"].map(ACTION_TO_INDEX).to_numpy(int)
        pred = sub[pred_col].map(ACTION_TO_INDEX).to_numpy(int)
        rows.append(
            {
                "group": split,
                "n": int(len(sub)),
                "accuracy": float(accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                "mean_entropy": float(pd.to_numeric(sub["policy_entropy"], errors="coerce").mean()),
                "mean_margin": float(pd.to_numeric(sub["policy_margin"], errors="coerce").mean()),
            }
        )
    for (split, origin), sub in scored.groupby(["split", "scenario_primary_fault_origin"]):
        y = sub["expert_action"].map(ACTION_TO_INDEX).to_numpy(int)
        pred = sub[pred_col].map(ACTION_TO_INDEX).to_numpy(int)
        rows.append(
            {
                "group": f"{split}:{origin}",
                "n": int(len(sub)),
                "accuracy": float(accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                "mean_entropy": float(pd.to_numeric(sub["policy_entropy"], errors="coerce").mean()),
                "mean_margin": float(pd.to_numeric(sub["policy_margin"], errors="coerce").mean()),
            }
        )
    return rows


def _mechanism_tables(rollouts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    errors = rollouts[~rollouts["policy_correct"].astype(bool)].copy()
    rows = []
    for (split, scenario_id, mechanism, route), sub in errors.groupby(
        ["split", "scenario_id", "policy_error_mechanism", "recovery_route"]
    ):
        rows.append(
            {
                "split": split,
                "scenario_id": scenario_id,
                "policy_error_mechanism": mechanism,
                "recovery_route": route,
                "n_errors": int(len(sub)),
                "high_conf_error_rate": float((sub["policy_max_prob"] >= 0.90).mean()),
                "blocked_move_rate": float(sub["attempted_blocked_move"].astype(bool).mean()),
                "wrong_direction_rate": float(sub["progress_wrong_direction"].astype(bool).mean()),
                "mean_policy_entropy": float(sub["policy_entropy"].mean()),
                "mean_policy_max_prob": float(sub["policy_max_prob"].mean()),
                "mean_policy_margin": float(sub["policy_margin"].mean()),
                "mean_risk": float(sub["risk_score"].mean()),
            }
        )
    route_rows = []
    for (split, route), sub in errors.groupby(["split", "recovery_route"]):
        route_rows.append(
            {
                "split": split,
                "recovery_route": route,
                "n_policy_errors": int(len(sub)),
                "dominant_mechanism": sub["policy_error_mechanism"].mode().iloc[0],
                "scenarios": "|".join(sorted(sub["scenario_id"].unique())),
                "high_conf_error_rate": float((sub["policy_max_prob"] >= 0.90).mean()),
                "mean_policy_entropy": float(sub["policy_entropy"].mean()),
                "mean_policy_max_prob": float(sub["policy_max_prob"].mean()),
                "mean_risk": float(sub["risk_score"].mean()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(route_rows)


def run_navigation_policy_mechanism_pipeline(
    out_dir: str | Path,
    seeds: list[int],
    scenarios: list[str],
    train_scenarios: set[str],
    max_steps: int = 70,
    epochs: int = 300,
    ros_model_path: str | Path | None = None,
) -> dict[str, Path]:
    output_dir = ensure_output_dir(out_dir)
    demos = generate_navigation_policy_demonstrations(seeds, scenarios, max_steps=max_steps)
    table = _prepare_table(demos)
    policy = train_navigation_policy(table, train_scenarios=train_scenarios, epochs=epochs)
    supervised_scores = _score_supervised(policy, table)
    rollout_scores = _score_closed_loop_rollouts(policy, seeds, scenarios, max_steps=max_steps)
    supervised_metrics = pd.DataFrame(_metric_rows(supervised_scores))
    rollout_metrics = pd.DataFrame(_metric_rows(rollout_scores))
    mechanism_evidence, recovery_routes = _mechanism_tables(rollout_scores)

    paths = {
        "demonstrations": output_dir / "navigation_policy_demonstrations.csv",
        "supervised_scores": output_dir / "navigation_policy_supervised_scores.csv",
        "supervised_metrics": output_dir / "navigation_policy_supervised_metrics.csv",
        "rollout_scores": output_dir / "navigation_policy_rollout_scores.csv",
        "rollout_metrics": output_dir / "navigation_policy_rollout_metrics.csv",
        "mechanism_evidence": output_dir / "policy_error_mechanism_evidence.csv",
        "recovery_routes": output_dir / "policy_error_recovery_routes.csv",
        "report": output_dir / "navigation_policy_mechanism_report.json",
        "ros_model": output_dir / "navigation_policy_model_export.json",
    }
    demos.to_csv(paths["demonstrations"], index=False)
    supervised_scores.to_csv(paths["supervised_scores"], index=False)
    supervised_metrics.to_csv(paths["supervised_metrics"], index=False)
    rollout_scores.to_csv(paths["rollout_scores"], index=False)
    rollout_metrics.to_csv(paths["rollout_metrics"], index=False)
    mechanism_evidence.to_csv(paths["mechanism_evidence"], index=False)
    recovery_routes.to_csv(paths["recovery_routes"], index=False)
    export_navigation_policy(policy, paths["ros_model"])
    if ros_model_path is not None:
        export_navigation_policy(policy, ros_model_path)
    report = {
        "model": "two-layer MLP navigation policy",
        "task": "predict next grid movement action from observed robot/goal/local obstacle state",
        "actions": ACTIONS,
        "feature_columns": FEATURE_COLUMNS,
        "train_scenarios": sorted(train_scenarios),
        "evaluation_scenarios": scenarios,
        "seeds": seeds,
        "n_demonstration_rows": int(len(demos)),
        "n_rollout_rows": int(len(rollout_scores)),
        "input_exclusion_note": "scenario_id, fault origin/family, and OOD labels are audit columns only, not policy inputs.",
        "recovery_mapping_note": "Recovery routes are assigned after policy errors are observed and diagnosed.",
    }
    paths["report"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    return paths


def export_navigation_policy(policy: TrainedNavigationPolicy, path: str | Path) -> Path:
    """Export the trained MLP to a small JSON file usable from ROS without torch."""

    export_path = Path(path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    state = policy.model.state_dict()
    payload = {
        "model": "two-layer MLP navigation policy",
        "actions": ACTIONS,
        "feature_columns": policy.feature_columns,
        "scaler_mean": policy.scaler.mean_.tolist(),
        "scaler_scale": policy.scaler.scale_.tolist(),
        "layers": [
            {
                "weight": state["encoder.0.weight"].cpu().numpy().tolist(),
                "bias": state["encoder.0.bias"].cpu().numpy().tolist(),
                "activation": "relu",
            },
            {
                "weight": state["encoder.2.weight"].cpu().numpy().tolist(),
                "bias": state["encoder.2.bias"].cpu().numpy().tolist(),
                "activation": "relu",
            },
            {
                "weight": state["head.weight"].cpu().numpy().tolist(),
                "bias": state["head.bias"].cpu().numpy().tolist(),
                "activation": "linear",
            },
        ],
    }
    export_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return export_path


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_seed_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a task navigation policy and analyze policy-error mechanisms.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/navigation_policy_mechanism"))
    parser.add_argument("--seeds", type=str, default="10,11,12,16,17,18,19")
    parser.add_argument(
        "--scenarios",
        type=str,
        default=",".join(
            [
                "nominal",
                "external_path_blockage",
                "localization_drift",
                "perception_degradation",
                "execution_deviation",
                "progress_blockage",
                "mixed_blockage_and_perception",
                "mixed_drift_and_execution",
                "boundary_weak_blockage",
            ]
        ),
    )
    parser.add_argument("--train-scenarios", type=str, default=",".join(sorted(CORE_TRAIN_SCENARIOS)))
    parser.add_argument("--max-steps", type=int, default=70)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--ros-model-path",
        type=Path,
        default=None,
        help="Optional extra export path for the ROS policy monitor JSON model.",
    )
    args = parser.parse_args()

    paths = run_navigation_policy_mechanism_pipeline(
        out_dir=args.out_dir,
        seeds=_parse_seed_list(args.seeds),
        scenarios=_parse_csv_list(args.scenarios),
        train_scenarios=set(_parse_csv_list(args.train_scenarios)),
        max_steps=args.max_steps,
        epochs=args.epochs,
        ros_model_path=args.ros_model_path,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
