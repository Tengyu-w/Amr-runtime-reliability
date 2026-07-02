"""Train Gazebo-native navigation policies and audit model failure modes.

This script starts the corrected research loop:

1. Use Gazebo/Nav2 observer logs as the policy dataset.
2. Train a baseline movement policy from Nav2-plan expert actions.
3. Analyze imbalance, recall, confusion, calibration, and high-confidence errors.
4. Train an upgraded policy with class-weighted focal loss.
5. Compare baseline and upgraded models before any recovery-routing claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch import nn

DEMO_ROOT = Path(__file__).resolve().parents[1]
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from src.utils import ensure_output_dir


ACTIONS = ["STAY", "NORTH", "SOUTH", "EAST", "WEST"]
ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTIONS)}

BASE_FEATURE_COLUMNS = [
    "time_step",
    "robot_x",
    "robot_y",
    "target_x",
    "target_y",
    "risk_score",
    "localization_uncertainty",
    "sensor_confidence",
    "path_blocked_score",
    "obstacle_proximity",
    "trajectory_deviation",
    "replanning_failure_count",
    "task_progress_stagnation",
]

DERIVED_FEATURE_COLUMNS = [
    "goal_dx",
    "goal_dy",
    "goal_distance_l1",
    "abs_goal_dx",
    "abs_goal_dy",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + DERIVED_FEATURE_COLUMNS


class GazeboNativePolicyNet(nn.Module):
    """Small MLP policy with an inspectable representation layer."""

    def __init__(self, n_features: int, n_actions: int, hidden_dim: int = 48) -> None:
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
class TrainedPolicy:
    model: GazeboNativePolicyNet
    scaler: StandardScaler
    feature_columns: list[str]
    model_name: str


def _bool_series(series: pd.Series) -> pd.Series:
    return series.fillna(False).map(lambda value: str(value).strip().lower() in {"1", "true", "yes"})


def prepare_policy_dataset(timesteps: pd.DataFrame) -> pd.DataFrame:
    """Prepare evaluable Gazebo/Nav2 expert-action rows without scenario labels as inputs."""

    df = timesteps.copy()
    if "policy_evaluable" in df:
        df = df[_bool_series(df["policy_evaluable"])].copy()
    if "expert_source" in df:
        df = df[df["expert_source"].astype(str).eq("nav2_plan")].copy()
    df = df[df["expert_proxy_action"].isin(ACTION_TO_INDEX)].copy()
    if df.empty:
        raise ValueError("No evaluable Nav2-plan expert rows found.")

    for column in BASE_FEATURE_COLUMNS:
        if column not in df:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    df["goal_dx"] = df["target_x"] - df["robot_x"]
    df["goal_dy"] = df["target_y"] - df["robot_y"]
    df["goal_distance_l1"] = df["goal_dx"].abs() + df["goal_dy"].abs()
    df["abs_goal_dx"] = df["goal_dx"].abs()
    df["abs_goal_dy"] = df["goal_dy"].abs()
    for column in DERIVED_FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    if "split" not in df:
        df["split"] = "train"
    df["action_index"] = df["expert_proxy_action"].map(ACTION_TO_INDEX).astype(int)
    return df


def _class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=len(ACTIONS)).astype(np.float32)
    total = max(float(counts.sum()), 1.0)
    return torch.tensor(total / (len(ACTIONS) * np.maximum(counts, 1.0)), dtype=torch.float32)


def _focal_loss(logits: torch.Tensor, y: torch.Tensor, class_weights: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    ce = nn.functional.cross_entropy(logits, y, weight=class_weights, reduction="none")
    pt = torch.exp(-ce)
    return (((1.0 - pt) ** gamma) * ce).mean()


def train_policy(
    table: pd.DataFrame,
    model_name: str,
    weighted: bool = False,
    focal: bool = False,
    epochs: int = 260,
    lr: float = 0.01,
    seed: int = 31,
) -> TrainedPolicy:
    torch.manual_seed(seed)
    train = table[table["split"].eq("train")]
    if train.empty:
        raise ValueError("Training split is empty.")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[FEATURE_COLUMNS].to_numpy(float)).astype(np.float32)
    y_train = train["action_index"].to_numpy(int)
    model = GazeboNativePolicyNet(n_features=len(FEATURE_COLUMNS), n_actions=len(ACTIONS))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    weights = _class_weights(y_train) if weighted else torch.ones(len(ACTIONS), dtype=torch.float32)
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits, _ = model(x_tensor)
        if focal:
            loss = _focal_loss(logits, y_tensor, weights)
        else:
            loss = nn.functional.cross_entropy(logits, y_tensor, weight=weights)
        loss.backward()
        optimizer.step()
    return TrainedPolicy(model=model, scaler=scaler, feature_columns=list(FEATURE_COLUMNS), model_name=model_name)


def _predict(policy: TrainedPolicy, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = policy.scaler.transform(df[policy.feature_columns].to_numpy(float)).astype(np.float32)
    policy.model.eval()
    with torch.no_grad():
        logits, embedding = policy.model(torch.tensor(x, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs, logits.cpu().numpy(), embedding.cpu().numpy()


def score_policy(policy: TrainedPolicy, table: pd.DataFrame) -> pd.DataFrame:
    scored_parts = []
    for split, split_df in table.groupby("split"):
        split_df = split_df.copy()
        probs, logits, embedding = _predict(policy, split_df)
        pred_idx = probs.argmax(axis=1)
        sorted_probs = np.sort(probs, axis=1)
        split_df["model_name"] = policy.model_name
        split_df["policy_pred_action_gazebo"] = [ACTIONS[idx] for idx in pred_idx]
        split_df["policy_correct_gazebo"] = pred_idx == split_df["action_index"].to_numpy(int)
        split_df["policy_entropy_gazebo"] = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
        split_df["policy_margin_gazebo"] = sorted_probs[:, -1] - sorted_probs[:, -2]
        split_df["policy_max_prob_gazebo"] = sorted_probs[:, -1]
        split_df["embedding_norm_gazebo"] = np.linalg.norm(embedding, axis=1)
        for idx, action in enumerate(ACTIONS):
            split_df[f"prob_{action}"] = probs[:, idx]
            split_df[f"logit_{action}"] = logits[:, idx]
        scored_parts.append(split_df)
    return pd.concat(scored_parts, ignore_index=True)


def _expected_calibration_error(y: np.ndarray, pred: np.ndarray, conf: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    rows = []
    ece = 0.0
    correct = (y == pred).astype(float)
    for idx in range(bins):
        lo = idx / bins
        hi = (idx + 1) / bins
        mask = (conf >= lo) & (conf < hi if idx < bins - 1 else conf <= hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin": idx, "confidence_low": lo, "confidence_high": hi, "n": 0})
            continue
        acc = float(correct[mask].mean())
        mean_conf = float(conf[mask].mean())
        gap = abs(acc - mean_conf)
        ece += (n / max(len(conf), 1)) * gap
        rows.append(
            {
                "bin": idx,
                "confidence_low": lo,
                "confidence_high": hi,
                "n": n,
                "accuracy": acc,
                "mean_confidence": mean_conf,
                "abs_gap": gap,
            }
        )
    return float(ece), rows


def metric_rows(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric = []
    class_rows = []
    calibration_rows = []
    for (model_name, split), sub in scored.groupby(["model_name", "split"]):
        y = sub["action_index"].to_numpy(int)
        pred = sub["policy_pred_action_gazebo"].map(ACTION_TO_INDEX).to_numpy(int)
        conf = pd.to_numeric(sub["policy_max_prob_gazebo"], errors="coerce").fillna(0.0).to_numpy(float)
        ece, bins = _expected_calibration_error(y, pred, conf)
        metric.append(
            {
                "model_name": model_name,
                "group": split,
                "n": int(len(sub)),
                "accuracy": float(accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                "ece_10bin": ece,
                "mean_entropy": float(sub["policy_entropy_gazebo"].mean()),
                "mean_margin": float(sub["policy_margin_gazebo"].mean()),
                "high_conf_error_count": int(((y != pred) & (conf >= 0.90)).sum()),
            }
        )
        precision, recall, f1, support = precision_recall_fscore_support(
            y,
            pred,
            labels=list(range(len(ACTIONS))),
            zero_division=0,
        )
        for idx, action in enumerate(ACTIONS):
            class_rows.append(
                {
                    "model_name": model_name,
                    "split": split,
                    "action": action,
                    "support": int(support[idx]),
                    "precision": float(precision[idx]),
                    "recall": float(recall[idx]),
                    "f1": float(f1[idx]),
                }
            )
        for row in bins:
            row.update({"model_name": model_name, "split": split})
            calibration_rows.append(row)

    for (model_name, split, scenario), sub in scored.groupby(["model_name", "split", "scenario_id"]):
        y = sub["action_index"].to_numpy(int)
        pred = sub["policy_pred_action_gazebo"].map(ACTION_TO_INDEX).to_numpy(int)
        metric.append(
            {
                "model_name": model_name,
                "group": f"{split}:{scenario}",
                "n": int(len(sub)),
                "accuracy": float(accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                "ece_10bin": np.nan,
                "mean_entropy": float(sub["policy_entropy_gazebo"].mean()),
                "mean_margin": float(sub["policy_margin_gazebo"].mean()),
                "high_conf_error_count": int(
                    ((y != pred) & (pd.to_numeric(sub["policy_max_prob_gazebo"], errors="coerce") >= 0.90)).sum()
                ),
            }
        )
    return pd.DataFrame(metric), pd.DataFrame(class_rows), pd.DataFrame(calibration_rows)


def imbalance_table(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, action), sub in table.groupby(["split", "expert_proxy_action"]):
        rows.append({"split": split, "action": action, "n": int(len(sub))})
    out = pd.DataFrame(rows)
    totals = out.groupby("split")["n"].transform("sum")
    out["fraction"] = out["n"] / totals
    return out.sort_values(["split", "action"])


def confusion_table(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_name, split), sub in scored.groupby(["model_name", "split"]):
        y = sub["expert_proxy_action"].astype(str)
        pred = sub["policy_pred_action_gazebo"].astype(str)
        matrix = confusion_matrix(y, pred, labels=ACTIONS)
        for i, actual in enumerate(ACTIONS):
            for j, predicted in enumerate(ACTIONS):
                rows.append(
                    {
                        "model_name": model_name,
                        "split": split,
                        "actual_action": actual,
                        "predicted_action": predicted,
                        "n": int(matrix[i, j]),
                    }
                )
    return pd.DataFrame(rows)


def high_conf_errors(scored: pd.DataFrame) -> pd.DataFrame:
    return scored[
        (~scored["policy_correct_gazebo"].astype(bool))
        & (pd.to_numeric(scored["policy_max_prob_gazebo"], errors="coerce") >= 0.90)
    ].copy()


def model_upgrade_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline = metrics[metrics["model_name"].eq("baseline")]
    upgraded = metrics[metrics["model_name"].eq("class_weighted_focal")]
    for group in sorted(set(baseline["group"]) & set(upgraded["group"])):
        base = baseline[baseline["group"].eq(group)].iloc[0]
        up = upgraded[upgraded["group"].eq(group)].iloc[0]
        rows.append(
            {
                "group": group,
                "baseline_accuracy": base["accuracy"],
                "upgraded_accuracy": up["accuracy"],
                "accuracy_delta": up["accuracy"] - base["accuracy"],
                "baseline_macro_f1": base["macro_f1"],
                "upgraded_macro_f1": up["macro_f1"],
                "macro_f1_delta": up["macro_f1"] - base["macro_f1"],
                "baseline_ece_10bin": base["ece_10bin"],
                "upgraded_ece_10bin": up["ece_10bin"],
                "ece_delta": up["ece_10bin"] - base["ece_10bin"],
                "baseline_high_conf_error_count": base["high_conf_error_count"],
                "upgraded_high_conf_error_count": up["high_conf_error_count"],
                "high_conf_error_delta": up["high_conf_error_count"] - base["high_conf_error_count"],
            }
        )
    return pd.DataFrame(rows)


def run_gazebo_native_policy_training(
    policy_timesteps_path: str | Path,
    out_dir: str | Path,
    epochs: int = 260,
) -> dict[str, Path]:
    output_dir = ensure_output_dir(out_dir)
    timesteps = pd.read_csv(policy_timesteps_path)
    table = prepare_policy_dataset(timesteps)
    baseline = train_policy(table, "baseline", weighted=False, focal=False, epochs=epochs)
    upgraded = train_policy(table, "class_weighted_focal", weighted=True, focal=True, epochs=epochs)
    scored = pd.concat([score_policy(baseline, table), score_policy(upgraded, table)], ignore_index=True)
    metrics, class_metrics, calibration = metric_rows(scored)
    paths = {
        "train_table": output_dir / "gazebo_native_policy_train_table.csv",
        "scores": output_dir / "gazebo_native_policy_scores.csv",
        "metrics": output_dir / "gazebo_native_policy_metrics.csv",
        "class_metrics": output_dir / "gazebo_native_policy_class_metrics.csv",
        "imbalance": output_dir / "gazebo_native_policy_imbalance.csv",
        "confusion": output_dir / "gazebo_native_policy_confusion.csv",
        "calibration": output_dir / "gazebo_native_policy_calibration_bins.csv",
        "high_conf_errors": output_dir / "gazebo_native_policy_high_conf_errors.csv",
        "upgrade_summary": output_dir / "gazebo_native_policy_upgrade_summary.csv",
        "report": output_dir / "gazebo_native_policy_report.json",
    }
    table.to_csv(paths["train_table"], index=False)
    scored.to_csv(paths["scores"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    class_metrics.to_csv(paths["class_metrics"], index=False)
    imbalance_table(table).to_csv(paths["imbalance"], index=False)
    confusion_table(scored).to_csv(paths["confusion"], index=False)
    calibration.to_csv(paths["calibration"], index=False)
    high_conf_errors(scored).to_csv(paths["high_conf_errors"], index=False)
    model_upgrade_summary(metrics).to_csv(paths["upgrade_summary"], index=False)
    report = {
        "task": "Gazebo/Nav2-native movement policy imitation",
        "label": "expert_proxy_action from Nav2 /plan",
        "actions": ACTIONS,
        "feature_columns": FEATURE_COLUMNS,
        "models": ["baseline", "class_weighted_focal"],
        "n_rows": int(len(table)),
        "n_episodes": int(table["episode_id"].nunique()) if "episode_id" in table else None,
        "input_exclusion_note": "scenario_id, fault origin/family, and OOD labels are audit columns only.",
    }
    paths["report"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and audit Gazebo-native movement policies.")
    parser.add_argument(
        "--policy-timesteps",
        type=Path,
        default=Path("outputs/gazebo_policy_monitor_matrix_v2/policy_evidence/policy_timesteps.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/gazebo_native_policy"))
    parser.add_argument("--epochs", type=int, default=260)
    args = parser.parse_args()
    paths = run_gazebo_native_policy_training(args.policy_timesteps, args.out_dir, epochs=args.epochs)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
