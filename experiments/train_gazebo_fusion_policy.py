"""Train a Gazebo scan+depth fusion policy from Nav2-plan expert labels."""

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

from experiments.train_gazebo_depth_policy import CONTEXT_COLUMNS as DEPTH_CONTEXT_COLUMNS
from experiments.train_gazebo_depth_policy import prepare_depth_policy_table, load_depth_policy_rows
from experiments.train_gazebo_scan_policy import CONTEXT_COLUMNS as SCAN_CONTEXT_COLUMNS
from experiments.train_gazebo_scan_policy import prepare_scan_policy_table, load_scan_policy_rows
from src.utils import ensure_output_dir


ACTIONS = ["STAY", "NORTH", "SOUTH", "EAST", "WEST"]
ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTIONS)}
MERGE_KEYS = ["episode_id", "scenario_id", "time_step", "expert_proxy_action", "split"]
FUSION_CONTEXT_COLUMNS = [
    "robot_x",
    "robot_y",
    "target_x",
    "target_y",
    "goal_dx",
    "goal_dy",
    "goal_distance_l1",
    "risk_score",
    "localization_uncertainty",
    "sensor_confidence",
    "path_blocked_score",
    "obstacle_proximity",
    "trajectory_deviation",
    "replanning_failure_count",
    "task_progress_stagnation",
    "scan_valid_fraction",
    "scan_min_range",
    "scan_mean_range",
    "scan_front_min_range",
    "depth_valid_fraction",
    "depth_min_m",
    "depth_mean_m",
    "depth_center_min_m",
]


class FusionPolicyNet(nn.Module):
    def __init__(self, n_scan: int, n_depth: int, n_context: int, n_actions: int) -> None:
        super().__init__()
        self.scan_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.depth_encoder = nn.Sequential(
            nn.Linear(n_depth, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
        )
        self.context_encoder = nn.Sequential(nn.Linear(n_context, 32), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(32 * 8 + 64 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, scan: torch.Tensor, depth: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scan_embedding = self.scan_encoder(scan.unsqueeze(1)).flatten(start_dim=1)
        depth_embedding = self.depth_encoder(depth)
        context_embedding = self.context_encoder(context)
        embedding = torch.cat([scan_embedding, depth_embedding, context_embedding], dim=1)
        return self.head(embedding), embedding


@dataclass
class TrainedFusionPolicy:
    model: FusionPolicyNet
    context_scaler: StandardScaler
    scan_columns: list[str]
    depth_columns: list[str]
    model_name: str


def _class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=len(ACTIONS)).astype(np.float32)
    total = max(float(counts.sum()), 1.0)
    return torch.tensor(total / (len(ACTIONS) * np.maximum(counts, 1.0)), dtype=torch.float32)


def _focal_loss(logits: torch.Tensor, y: torch.Tensor, class_weights: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    ce = nn.functional.cross_entropy(logits, y, weight=class_weights, reduction="none")
    pt = torch.exp(-ce)
    return ((1.0 - pt) ** gamma * ce).mean()


def prepare_fusion_table(scan_log_dir: Path, depth_log_dir: Path) -> pd.DataFrame:
    scan = prepare_scan_policy_table(load_scan_policy_rows(None, scan_log_dir))
    depth = prepare_depth_policy_table(load_depth_policy_rows(None, depth_log_dir))
    scan_columns = sorted([column for column in scan.columns if column.startswith("scan_bin_")])
    depth_columns = sorted([column for column in depth.columns if column.startswith("depth_cell_")])
    scan_keep = MERGE_KEYS + SCAN_CONTEXT_COLUMNS + scan_columns
    depth_keep = MERGE_KEYS + [column for column in DEPTH_CONTEXT_COLUMNS if column.startswith("depth_")] + depth_columns
    merged = pd.merge(
        scan[scan_keep],
        depth[depth_keep],
        on=MERGE_KEYS,
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No aligned scan/depth policy rows found.")
    for column in FUSION_CONTEXT_COLUMNS + scan_columns + depth_columns:
        if column not in merged:
            merged[column] = 0.0
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    merged = merged.copy()
    merged["action_index"] = merged["expert_proxy_action"].astype(str).map(ACTION_TO_INDEX).astype(int)
    return merged.copy()


def train_fusion_policy(
    table: pd.DataFrame,
    model_name: str,
    weighted: bool = False,
    focal: bool = False,
    epochs: int = 220,
) -> TrainedFusionPolicy:
    torch.manual_seed(7)
    np.random.seed(7)
    scan_columns = sorted([column for column in table.columns if column.startswith("scan_bin_")])
    depth_columns = sorted([column for column in table.columns if column.startswith("depth_cell_")])
    train = table[table["split"].eq("train")].copy()
    if train.empty:
        raise ValueError("Training split is empty.")
    scan_train = train[scan_columns].to_numpy(np.float32)
    depth_train = train[depth_columns].to_numpy(np.float32)
    context_scaler = StandardScaler()
    context_train = context_scaler.fit_transform(train[FUSION_CONTEXT_COLUMNS].to_numpy(float)).astype(np.float32)
    y_train = train["action_index"].to_numpy(np.int64)
    model = FusionPolicyNet(len(scan_columns), len(depth_columns), len(FUSION_CONTEXT_COLUMNS), len(ACTIONS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    weights = _class_weights(y_train) if weighted else torch.ones(len(ACTIONS), dtype=torch.float32)
    scan_tensor = torch.from_numpy(scan_train)
    depth_tensor = torch.from_numpy(depth_train)
    context_tensor = torch.from_numpy(context_train)
    y_tensor = torch.from_numpy(y_train)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits, _ = model(scan_tensor, depth_tensor, context_tensor)
        loss = _focal_loss(logits, y_tensor, weights) if focal else nn.functional.cross_entropy(logits, y_tensor, weight=weights)
        loss.backward()
        optimizer.step()
    return TrainedFusionPolicy(model, context_scaler, scan_columns, depth_columns, model_name)


def score_policy(table: pd.DataFrame, trained: TrainedFusionPolicy) -> pd.DataFrame:
    scored = []
    trained.model.eval()
    with torch.no_grad():
        for split, split_df in table.groupby("split"):
            scan = torch.from_numpy(split_df[trained.scan_columns].to_numpy(np.float32))
            depth = torch.from_numpy(split_df[trained.depth_columns].to_numpy(np.float32))
            context = torch.from_numpy(
                trained.context_scaler.transform(split_df[FUSION_CONTEXT_COLUMNS].to_numpy(float)).astype(np.float32)
            )
            logits, embeddings = trained.model(scan, depth, context)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred_idx = probs.argmax(axis=1)
            out = split_df.copy()
            out["model_name"] = trained.model_name
            out["policy_pred_action_fusion"] = [ACTIONS[idx] for idx in pred_idx]
            out["fusion_policy_correct"] = out["policy_pred_action_fusion"].eq(out["expert_proxy_action"].astype(str))
            out["fusion_policy_max_prob"] = probs.max(axis=1)
            sorted_probs = np.sort(probs, axis=1)
            out["fusion_policy_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
            out["fusion_policy_entropy"] = -(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=1)
            out["embedding_norm"] = np.linalg.norm(embeddings.cpu().numpy(), axis=1)
            scored.append(out)
    return pd.concat(scored, ignore_index=True)


def _ece(y_true: np.ndarray, pred: np.ndarray, confidence: np.ndarray, n_bins: int = 10) -> float:
    ece = 0.0
    for bin_idx in range(n_bins):
        lower = bin_idx / n_bins
        upper = (bin_idx + 1) / n_bins
        mask = (confidence > lower) & (confidence <= upper if bin_idx < n_bins - 1 else confidence <= upper)
        if mask.any():
            ece += abs(float((y_true[mask] == pred[mask]).mean()) - float(confidence[mask].mean())) * float(mask.mean())
    return float(ece)


def audit_scores(scores: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    output_dir = ensure_output_dir(out_dir)
    metrics_rows = []
    class_rows = []
    confusion_rows = []
    for (model_name, split), sub in scores.groupby(["model_name", "split"]):
        y = sub["expert_proxy_action"].astype(str).to_numpy()
        pred = sub["policy_pred_action_fusion"].astype(str).to_numpy()
        conf = sub["fusion_policy_max_prob"].to_numpy(float)
        metrics_rows.append(
            {
                "model_name": model_name,
                "split": split,
                "n_rows": int(len(sub)),
                "accuracy": float(accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, labels=ACTIONS, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y, pred, labels=ACTIONS, average="weighted", zero_division=0)),
                "mean_confidence": float(conf.mean()),
                "ece": _ece(y, pred, conf),
                "n_high_conf_errors": int(((y != pred) & (conf >= 0.90)).sum()),
            }
        )
        precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=ACTIONS, zero_division=0)
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
        matrix = confusion_matrix(y, pred, labels=ACTIONS)
        for i, actual in enumerate(ACTIONS):
            for j, predicted in enumerate(ACTIONS):
                confusion_rows.append({"model_name": model_name, "split": split, "actual": actual, "predicted": predicted, "count": int(matrix[i, j])})
    unique_rows = scores.drop_duplicates(["episode_id", "time_step", "expert_proxy_action"])
    imbalance_rows = []
    for (split, action), sub in unique_rows.groupby(["split", "expert_proxy_action"]):
        total = len(unique_rows[unique_rows["split"].eq(split)])
        imbalance_rows.append({"split": split, "action": action, "count": int(len(sub)), "fraction": float(len(sub) / max(total, 1))})
    high_conf = scores[(~scores["fusion_policy_correct"]) & (scores["fusion_policy_max_prob"] >= 0.90)].copy()
    paths = {
        "scores": output_dir / "gazebo_fusion_policy_scores.csv",
        "metrics": output_dir / "gazebo_fusion_policy_metrics.csv",
        "class_metrics": output_dir / "gazebo_fusion_policy_class_metrics.csv",
        "confusion": output_dir / "gazebo_fusion_policy_confusion.csv",
        "imbalance": output_dir / "gazebo_fusion_policy_imbalance.csv",
        "high_conf_errors": output_dir / "gazebo_fusion_policy_high_conf_errors.csv",
    }
    scores.to_csv(paths["scores"], index=False)
    pd.DataFrame(metrics_rows).to_csv(paths["metrics"], index=False)
    pd.DataFrame(class_rows).to_csv(paths["class_metrics"], index=False)
    pd.DataFrame(confusion_rows).to_csv(paths["confusion"], index=False)
    pd.DataFrame(imbalance_rows).to_csv(paths["imbalance"], index=False)
    high_conf.to_csv(paths["high_conf_errors"], index=False)
    return paths


def train_and_audit_fusion_policy(scan_log_dir: Path, depth_log_dir: Path, out_dir: Path, epochs: int = 220) -> dict[str, Path]:
    output_dir = ensure_output_dir(out_dir)
    table = prepare_fusion_table(scan_log_dir, depth_log_dir)
    table_path = output_dir / "gazebo_fusion_policy_train_table.csv"
    table.to_csv(table_path, index=False)
    baseline = train_fusion_policy(table, "baseline_scan_depth_goal", weighted=False, focal=False, epochs=epochs)
    focal = train_fusion_policy(table, "class_weighted_focal_scan_depth_goal", weighted=True, focal=True, epochs=epochs)
    scores = pd.concat([score_policy(table, baseline), score_policy(table, focal)], ignore_index=True)
    paths = audit_scores(scores, output_dir)
    report = {
        "task": "Gazebo scan+depth+goal fusion policy",
        "label": "expert_proxy_action from Nav2 /plan",
        "actions": ACTIONS,
        "n_rows": int(len(table)),
        "scan_columns": sorted([column for column in table.columns if column.startswith("scan_bin_")]),
        "depth_columns": sorted([column for column in table.columns if column.startswith("depth_cell_")]),
        "context_columns": FUSION_CONTEXT_COLUMNS,
        "models": ["baseline_scan_depth_goal", "class_weighted_focal_scan_depth_goal"],
        "limitation": "Simulation-derived lidar and depth observations; not real-robot validation.",
    }
    report_path = output_dir / "gazebo_fusion_policy_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    paths["train_table"] = table_path
    paths["report"] = report_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Gazebo scan+depth fusion policy from expert labels.")
    parser.add_argument("--scan-log-dir", type=Path, required=True)
    parser.add_argument("--depth-log-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/gazebo_fusion_policy_v1"))
    parser.add_argument("--epochs", type=int, default=220)
    args = parser.parse_args()
    paths = train_and_audit_fusion_policy(args.scan_log_dir, args.depth_log_dir, args.out_dir, epochs=args.epochs)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
