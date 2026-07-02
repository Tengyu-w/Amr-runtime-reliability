"""Train a neural recovery-route model from ROS/Gazebo episode data."""

from __future__ import annotations

import argparse
import json
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

from src.decision_router import RouterDecision
from src.utils import ensure_output_dir


DEFAULT_FEATURE_COLUMNS = [
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

ACTIONS = [decision.value for decision in RouterDecision]
ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTIONS)}


class GazeboRouteNet(nn.Module):
    """Compact MLP route classifier with an inspectable representation layer."""

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
class TrainedGazeboRouteModel:
    model: GazeboRouteNet
    scaler: StandardScaler
    feature_columns: list[str]


def prepare_gazebo_route_table(
    timesteps: pd.DataFrame,
    label_column: str = "target_action",
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Prepare numeric inputs and route labels without scenario/fault-origin inputs."""

    columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    table = timesteps.copy()
    if label_column not in table:
        raise KeyError(f"Missing label column: {label_column}")
    table = table[table[label_column].isin(ACTION_TO_INDEX)].copy()
    if table.empty:
        raise ValueError("No trainable rows after filtering route labels.")
    if "split" not in table:
        table["split"] = "train"
    for col in columns:
        if col not in table:
            table[col] = 0.0
        table[col] = pd.to_numeric(table[col], errors="coerce").fillna(0.0)
    table["action_index"] = table[label_column].map(ACTION_TO_INDEX).astype(int)
    return table


def _class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=len(ACTIONS)).astype(np.float32)
    total = max(float(counts.sum()), 1.0)
    weights = total / (len(ACTIONS) * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def train_route_model(
    table: pd.DataFrame,
    feature_columns: list[str] | None = None,
    epochs: int = 300,
    lr: float = 0.01,
    seed: int = 11,
) -> TrainedGazeboRouteModel:
    columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    torch.manual_seed(seed)
    train = table[table["split"].eq("train")]
    if train.empty:
        train = table
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[columns].to_numpy(float)).astype(np.float32)
    y_train = train["action_index"].to_numpy(int)
    model = GazeboRouteNet(n_features=len(columns), n_actions=len(ACTIONS))
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
    return TrainedGazeboRouteModel(model=model, scaler=scaler, feature_columns=columns)


def _predict(policy: TrainedGazeboRouteModel, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = policy.scaler.transform(df[policy.feature_columns].to_numpy(float)).astype(np.float32)
    policy.model.eval()
    with torch.no_grad():
        logits, embedding = policy.model(torch.tensor(x, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs, logits.cpu().numpy(), embedding.cpu().numpy()


def score_route_model(policy: TrainedGazeboRouteModel, table: pd.DataFrame) -> pd.DataFrame:
    scored_parts = []
    for split, split_df in table.groupby("split"):
        probs, logits, embedding = _predict(policy, split_df)
        pred_idx = probs.argmax(axis=1)
        sorted_probs = np.sort(probs, axis=1)
        entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
        out = split_df.copy()
        out["route_model_pred_action"] = [ACTIONS[idx] for idx in pred_idx]
        out["route_model_correct"] = pred_idx == out["action_index"].to_numpy(int)
        out["route_model_entropy"] = entropy
        out["route_model_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
        out["route_model_max_prob"] = sorted_probs[:, -1]
        out["embedding_norm"] = np.linalg.norm(embedding, axis=1)
        for idx, action in enumerate(ACTIONS):
            out[f"prob_{action}"] = probs[:, idx]
            out[f"logit_{action}"] = logits[:, idx]
        scored_parts.append(out)
    return pd.concat(scored_parts, ignore_index=True)


def metric_rows(scored: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for split, sub in scored.groupby("split"):
        y = sub["action_index"].to_numpy(int)
        pred = np.array([ACTION_TO_INDEX[action] for action in sub["route_model_pred_action"]])
        rows.append(
            {
                "group": split,
                "n": int(len(sub)),
                "accuracy": float(accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                "mean_entropy": float(sub["route_model_entropy"].mean()),
                "mean_margin": float(sub["route_model_margin"].mean()),
            }
        )
    if "scenario_primary_fault_origin" in scored:
        for (split, origin), sub in scored.groupby(["split", "scenario_primary_fault_origin"]):
            y = sub["action_index"].to_numpy(int)
            pred = np.array([ACTION_TO_INDEX[action] for action in sub["route_model_pred_action"]])
            rows.append(
                {
                    "group": f"{split}:{origin}",
                    "n": int(len(sub)),
                    "accuracy": float(accuracy_score(y, pred)),
                    "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                    "mean_entropy": float(sub["route_model_entropy"].mean()),
                    "mean_margin": float(sub["route_model_margin"].mean()),
                }
            )
    return rows


def train_and_audit_gazebo_route_model(
    dataset_dir: str | Path,
    out_dir: str | Path,
    label_column: str = "target_action",
    feature_columns: list[str] | None = None,
    epochs: int = 300,
) -> tuple[Path, Path, Path, Path]:
    output_dir = ensure_output_dir(out_dir)
    timesteps = pd.read_csv(Path(dataset_dir) / "timesteps.csv")
    table = prepare_gazebo_route_table(timesteps, label_column=label_column, feature_columns=feature_columns)
    columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    policy = train_route_model(table, feature_columns=columns, epochs=epochs)
    scored = score_route_model(policy, table)
    metrics = pd.DataFrame(metric_rows(scored))

    scores_path = output_dir / "gazebo_route_scores.csv"
    metrics_path = output_dir / "gazebo_route_metrics.csv"
    report_path = output_dir / "gazebo_route_model_report.json"
    model_path = output_dir / "gazebo_route_model.pt"
    scored.to_csv(scores_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    torch.save(
        {
            "model_state_dict": policy.model.state_dict(),
            "feature_columns": policy.feature_columns,
            "actions": ACTIONS,
            "scaler_mean": policy.scaler.mean_.tolist(),
            "scaler_scale": policy.scaler.scale_.tolist(),
        },
        model_path,
    )
    report = {
        "model": "two-layer PyTorch MLP route model",
        "label_column": label_column,
        "feature_columns": policy.feature_columns,
        "excluded_inputs": ["scenario_id", "fault_origin", "fault_family", "ood_status"],
        "n_rows": int(len(table)),
        "n_episodes": int(table["episode_id"].nunique()) if "episode_id" in table else None,
        "actions": ACTIONS,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return scores_path, metrics_path, report_path, model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a neural route model from Gazebo episode data.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("outputs/gazebo_episode_dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/gazebo_route_model"))
    parser.add_argument("--label-column", type=str, default="target_action")
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    scores_path, metrics_path, report_path, model_path = train_and_audit_gazebo_route_model(
        args.dataset_dir,
        args.out_dir,
        label_column=args.label_column,
        epochs=args.epochs,
    )
    print(f"Route scores: {scores_path}")
    print(f"Route metrics: {metrics_path}")
    print(f"Route report: {report_path}")
    print(f"Route model: {model_path}")


if __name__ == "__main__":
    main()
