"""Train a neural multi-action AMR recovery policy and audit its evidence."""

from __future__ import annotations

import argparse
import json
import re
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

from experiments.generate_scenario_dataset import generate_dataset
from src.decision_router import RouterDecision
from src.utils import ensure_output_dir


FEATURE_COLUMNS = [
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
    "moved_numeric",
]

ACTIONS = [decision.value for decision in RouterDecision]
ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTIONS)}


class PolicyNet(nn.Module):
    """Small neural policy with an explicit hidden representation."""

    def __init__(self, n_features: int, n_actions: int, hidden_dim: int = 32) -> None:
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
        logits = self.head(embedding)
        return logits, embedding


@dataclass
class TrainedPolicy:
    model: PolicyNet
    scaler: StandardScaler


def _parse_position(value: object) -> tuple[float, float]:
    match = re.match(r"\(([-0-9.]+),\s*([-0-9.]+)\)", str(value))
    if not match:
        return 0.0, 0.0
    return float(match.group(1)), float(match.group(2))


def prepare_policy_table(timesteps: pd.DataFrame, teacher_mode: str = "mechanism_router") -> pd.DataFrame:
    """Create action-policy features and labels without fault-source inputs."""

    df = timesteps[timesteps["mode"].eq(teacher_mode)].copy()
    robot_xy = df["robot_position"].map(_parse_position)
    target_xy = df["target_position"].map(_parse_position)
    df["robot_x"] = [xy[0] for xy in robot_xy]
    df["robot_y"] = [xy[1] for xy in robot_xy]
    df["target_x"] = [xy[0] for xy in target_xy]
    df["target_y"] = [xy[1] for xy in target_xy]
    df["moved_numeric"] = df["moved"].astype(str).str.lower().isin({"true", "1"}).astype(float)
    df["action_index"] = df["router_decision"].map(ACTION_TO_INDEX).astype(int)
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def _class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=len(ACTIONS)).astype(np.float32)
    total = max(float(counts.sum()), 1.0)
    weights = total / (len(ACTIONS) * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def train_policy(table: pd.DataFrame, epochs: int = 400, lr: float = 0.01, seed: int = 7) -> TrainedPolicy:
    """Train a neural action policy from mechanism-router action labels."""

    torch.manual_seed(seed)
    train = table[table["split"].eq("train")]
    if train.empty:
        raise ValueError("Training split is empty. Generate dataset with train seeds first.")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[FEATURE_COLUMNS].to_numpy(float)).astype(np.float32)
    y_train = train["action_index"].to_numpy(int)
    model = PolicyNet(n_features=len(FEATURE_COLUMNS), n_actions=len(ACTIONS))
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
    return TrainedPolicy(model=model, scaler=scaler)


def _predict(policy: TrainedPolicy, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = policy.scaler.transform(df[FEATURE_COLUMNS].to_numpy(float)).astype(np.float32)
    policy.model.eval()
    with torch.no_grad():
        logits, embedding = policy.model(torch.tensor(x, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs, logits.cpu().numpy(), embedding.cpu().numpy()


def _add_embedding_margins(train_embeddings: np.ndarray, train_y: np.ndarray, emb: np.ndarray, out: pd.DataFrame) -> None:
    centroids = {}
    for action_idx, action in enumerate(ACTIONS):
        mask = train_y == action_idx
        if mask.any():
            centroids[action] = train_embeddings[mask].mean(axis=0)
    pred_actions = out["policy_pred_action"].astype(str).tolist()
    teacher_actions = out["router_decision"].astype(str).tolist()
    pred_dist = []
    teacher_dist = []
    nearest_dist = []
    for i, vector in enumerate(emb):
        distances = {action: float(np.linalg.norm(vector - centroid)) for action, centroid in centroids.items()}
        nearest_dist.append(min(distances.values()) if distances else np.nan)
        pred_dist.append(distances.get(pred_actions[i], np.nan))
        teacher_dist.append(distances.get(teacher_actions[i], np.nan))
    out["embedding_distance_to_pred_action"] = pred_dist
    out["embedding_distance_to_teacher_action"] = teacher_dist
    out["embedding_nearest_action_distance"] = nearest_dist
    out["embedding_teacher_minus_pred_distance"] = out["embedding_distance_to_teacher_action"] - out[
        "embedding_distance_to_pred_action"
    ]


def score_policy(policy: TrainedPolicy, table: pd.DataFrame) -> pd.DataFrame:
    """Score every split with action probabilities and hidden representations."""

    train = table[table["split"].eq("train")]
    train_probs, _, train_emb = _predict(policy, train)
    del train_probs
    train_y = train["action_index"].to_numpy(int)

    scored_parts = []
    for split in ["train", "val", "test"]:
        split_df = table[table["split"].eq(split)].copy()
        if split_df.empty:
            continue
        probs, logits, emb = _predict(policy, split_df)
        pred_idx = probs.argmax(axis=1)
        sorted_probs = np.sort(probs, axis=1)
        entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
        split_df["policy_pred_action"] = [ACTIONS[idx] for idx in pred_idx]
        split_df["policy_correct"] = pred_idx == split_df["action_index"].to_numpy(int)
        split_df["policy_entropy"] = entropy
        split_df["policy_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
        split_df["policy_max_prob"] = sorted_probs[:, -1]
        for idx, action in enumerate(ACTIONS):
            split_df[f"prob_{action}"] = probs[:, idx]
            split_df[f"logit_{action}"] = logits[:, idx]
        _add_embedding_margins(train_emb, train_y, emb, split_df)
        scored_parts.append(split_df)
    return pd.concat(scored_parts, ignore_index=True)


def _metric_rows(scored: pd.DataFrame) -> list[dict]:
    rows = []
    for split, sub in scored.groupby("split"):
        y = sub["action_index"].to_numpy(int)
        pred = np.array([ACTION_TO_INDEX[action] for action in sub["policy_pred_action"]])
        rows.append(
            {
                "group": split,
                "n": len(sub),
                "accuracy": float(accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                "mean_entropy": float(sub["policy_entropy"].mean()),
                "mean_margin": float(sub["policy_margin"].mean()),
            }
        )
    for (split, origin), sub in scored.groupby(["split", "scenario_primary_fault_origin"]):
        y = sub["action_index"].to_numpy(int)
        pred = np.array([ACTION_TO_INDEX[action] for action in sub["policy_pred_action"]])
        rows.append(
            {
                "group": f"{split}:{origin}",
                "n": len(sub),
                "accuracy": float(accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                "mean_entropy": float(sub["policy_entropy"].mean()),
                "mean_margin": float(sub["policy_margin"].mean()),
            }
        )
    return rows


def train_and_audit_policy(
    dataset_dir: str | Path,
    out_dir: str | Path,
    teacher_mode: str = "mechanism_router",
) -> tuple[Path, Path, Path]:
    dataset_path = Path(dataset_dir)
    output_dir = ensure_output_dir(out_dir)
    timesteps = pd.read_csv(dataset_path / "timesteps.csv")
    table = prepare_policy_table(timesteps, teacher_mode=teacher_mode)
    policy = train_policy(table)
    scored = score_policy(policy, table)
    metrics = pd.DataFrame(_metric_rows(scored))

    scores_path = output_dir / "policy_scores.csv"
    metrics_path = output_dir / "policy_metrics.csv"
    report_path = output_dir / "policy_report.json"
    scored.to_csv(scores_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    report = {
        "model": "two-layer PyTorch MLP policy",
        "teacher_mode": teacher_mode,
        "actions": ACTIONS,
        "feature_columns": FEATURE_COLUMNS,
        "input_exclusion_note": "fault_origin, fault_family, scenario_id, and OOD labels are not model inputs.",
        "evidence_columns": [
            "policy_entropy",
            "policy_margin",
            "policy_max_prob",
            "embedding_distance_to_pred_action",
            "embedding_distance_to_teacher_action",
            "embedding_nearest_action_distance",
        ],
        "n_rows": int(len(table)),
        "n_train": int(table["split"].eq("train").sum()),
        "n_val": int(table["split"].eq("val").sum()),
        "n_test": int(table["split"].eq("test").sum()),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return scores_path, metrics_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a neural multi-action AMR recovery policy.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("outputs/scenario_dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/policy_model"))
    parser.add_argument("--teacher-mode", type=str, default="mechanism_router")
    parser.add_argument("--generate-if-missing", action="store_true")
    args = parser.parse_args()

    if args.generate_if_missing and not (args.dataset_dir / "timesteps.csv").exists():
        generate_dataset(
            seeds=[10, 11, 12, 16, 17, 18, 19],
            out_dir=args.dataset_dir,
            modes=["baseline", "risk_router", "mechanism_router"],
        )
    scores_path, metrics_path, report_path = train_and_audit_policy(
        args.dataset_dir,
        args.out_dir,
        teacher_mode=args.teacher_mode,
    )
    print(f"Policy scores: {scores_path}")
    print(f"Policy metrics: {metrics_path}")
    print(f"Policy report: {report_path}")


if __name__ == "__main__":
    main()
