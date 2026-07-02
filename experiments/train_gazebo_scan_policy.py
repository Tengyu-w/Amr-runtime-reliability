"""Train a Gazebo scan-observation policy from Nav2-plan expert labels.

This is the first perception-side policy stage:

1. Use Gazebo /scan observations plus task context as policy input.
2. Use Nav2 /plan-derived expert actions as labels.
3. Audit imbalance, recall, confusion, calibration, and high-confidence errors.

It does not claim real-robot visual validation. The evidence is simulation-derived.
"""

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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch import nn

DEMO_ROOT = Path(__file__).resolve().parents[1]
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from src.utils import ensure_output_dir


ACTIONS = ["STAY", "NORTH", "SOUTH", "EAST", "WEST"]
ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTIONS)}
CONTEXT_COLUMNS = [
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
]


class ScanGoalPolicyNet(nn.Module):
    """1D scan encoder fused with task context for discrete navigation actions."""

    def __init__(self, n_scan_bins: int, n_context_features: int, n_actions: int) -> None:
        super().__init__()
        self.scan_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(n_context_features, 32),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 8 + 32, 96),
            nn.ReLU(),
            nn.Linear(96, n_actions),
        )

    def forward(self, scan: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scan_embedding = self.scan_encoder(scan.unsqueeze(1)).flatten(start_dim=1)
        context_embedding = self.context_encoder(context)
        embedding = torch.cat([scan_embedding, context_embedding], dim=1)
        return self.head(embedding), embedding


@dataclass
class TrainedScanPolicy:
    model: ScanGoalPolicyNet
    context_scaler: StandardScaler
    scan_columns: list[str]
    context_columns: list[str]
    model_name: str


def _bool_series(series: pd.Series) -> pd.Series:
    return series.fillna(False).map(lambda value: str(value).strip().lower() in {"1", "true", "yes"})


def _seed_from_episode_id(episode_id: str) -> int | None:
    match = re.search(r"(?:^|__)seed_([0-9]+)(?:__|$)", episode_id)
    return int(match.group(1)) if match else None


def _split_for_seed(seed: int | None, fallback_index: int) -> str:
    if seed is None:
        bucket = fallback_index % 10
    else:
        bucket = seed % 10
    if bucket <= 5:
        return "train"
    if bucket <= 7:
        return "val"
    return "test"


def load_scan_policy_rows(input_csv: Path | None, log_dir: Path | None) -> pd.DataFrame:
    if input_csv is not None:
        frames = [pd.read_csv(input_csv)]
    elif log_dir is not None:
        paths = sorted(log_dir.glob("*.scan_policy.csv"))
        if not paths:
            raise FileNotFoundError(f"No *.scan_policy.csv files found in {log_dir}")
        frames = []
        for path in paths:
            df = pd.read_csv(path)
            if df.empty:
                continue
            if "episode_id" not in df or df["episode_id"].isna().all():
                df["episode_id"] = path.stem.replace(".scan_policy", "")
            df["scan_policy_log_path"] = str(path)
            frames.append(df)
    else:
        raise ValueError("Either input_csv or log_dir must be provided.")
    if not frames:
        raise FileNotFoundError("No non-empty scan-policy rows found.")
    return pd.concat(frames, ignore_index=True)


def prepare_scan_policy_table(rows: pd.DataFrame, require_nav2_plan: bool = True) -> pd.DataFrame:
    df = rows.copy()
    if "policy_evaluable" in df:
        df = df[_bool_series(df["policy_evaluable"])].copy()
    if require_nav2_plan and "expert_source" in df:
        df = df[df["expert_source"].astype(str).eq("nav2_plan")].copy()
    df = df[df["expert_proxy_action"].astype(str).isin(ACTION_TO_INDEX)].copy()
    scan_columns = sorted([column for column in df.columns if column.startswith("scan_bin_")])
    if not scan_columns:
        raise ValueError("No scan_bin_* columns found. Run scan_policy_observation_recorder first.")
    if df.empty:
        raise ValueError("No evaluable Nav2-plan scan-policy rows found.")

    for column in scan_columns + CONTEXT_COLUMNS:
        if column not in df:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    if "split" not in df:
        splits = []
        for idx, episode_id in enumerate(df.get("episode_id", pd.Series([""] * len(df))).astype(str)):
            splits.append(_split_for_seed(_seed_from_episode_id(episode_id), idx))
        df["split"] = splits
    df["action_index"] = df["expert_proxy_action"].astype(str).map(ACTION_TO_INDEX).astype(int)
    return df


def _class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=len(ACTIONS)).astype(np.float32)
    total = max(float(counts.sum()), 1.0)
    return torch.tensor(total / (len(ACTIONS) * np.maximum(counts, 1.0)), dtype=torch.float32)


def _focal_loss(logits: torch.Tensor, y: torch.Tensor, class_weights: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    ce = nn.functional.cross_entropy(logits, y, weight=class_weights, reduction="none")
    pt = torch.exp(-ce)
    return ((1.0 - pt) ** gamma * ce).mean()


def train_scan_policy(
    table: pd.DataFrame,
    model_name: str,
    weighted: bool = False,
    focal: bool = False,
    epochs: int = 220,
    lr: float = 0.002,
) -> TrainedScanPolicy:
    torch.manual_seed(7)
    np.random.seed(7)
    scan_columns = sorted([column for column in table.columns if column.startswith("scan_bin_")])
    train = table[table["split"].eq("train")].copy()
    if train.empty:
        raise ValueError("Training split is empty.")
    scan_train = train[scan_columns].to_numpy(np.float32)
    context_scaler = StandardScaler()
    context_train = context_scaler.fit_transform(train[CONTEXT_COLUMNS].to_numpy(float)).astype(np.float32)
    y_train = train["action_index"].to_numpy(np.int64)

    model = ScanGoalPolicyNet(
        n_scan_bins=len(scan_columns),
        n_context_features=len(CONTEXT_COLUMNS),
        n_actions=len(ACTIONS),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    weights = _class_weights(y_train) if weighted else torch.ones(len(ACTIONS), dtype=torch.float32)
    scan_tensor = torch.from_numpy(scan_train)
    context_tensor = torch.from_numpy(context_train)
    y_tensor = torch.from_numpy(y_train)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits, _ = model(scan_tensor, context_tensor)
        if focal:
            loss = _focal_loss(logits, y_tensor, weights)
        else:
            loss = nn.functional.cross_entropy(logits, y_tensor, weight=weights)
        loss.backward()
        optimizer.step()
    return TrainedScanPolicy(
        model=model,
        context_scaler=context_scaler,
        scan_columns=scan_columns,
        context_columns=list(CONTEXT_COLUMNS),
        model_name=model_name,
    )


def score_policy(table: pd.DataFrame, trained: TrainedScanPolicy) -> pd.DataFrame:
    scored = []
    trained.model.eval()
    with torch.no_grad():
        for split, split_df in table.groupby("split"):
            scan = torch.from_numpy(split_df[trained.scan_columns].to_numpy(np.float32))
            context = torch.from_numpy(
                trained.context_scaler.transform(split_df[trained.context_columns].to_numpy(float)).astype(np.float32)
            )
            logits, embeddings = trained.model(scan, context)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred_idx = probs.argmax(axis=1)
            out = split_df.copy()
            out["model_name"] = trained.model_name
            out["policy_pred_action_scan"] = [ACTIONS[idx] for idx in pred_idx]
            out["scan_policy_correct"] = out["policy_pred_action_scan"].eq(out["expert_proxy_action"].astype(str))
            out["scan_policy_max_prob"] = probs.max(axis=1)
            sorted_probs = np.sort(probs, axis=1)
            out["scan_policy_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
            out["scan_policy_entropy"] = -(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=1)
            out["embedding_norm"] = np.linalg.norm(embeddings.cpu().numpy(), axis=1)
            for idx, action in enumerate(ACTIONS):
                out[f"scan_policy_prob_{action}"] = probs[:, idx]
            scored.append(out)
    return pd.concat(scored, ignore_index=True)


def _ece(y_true: np.ndarray, pred: np.ndarray, confidence: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict[str, float]]]:
    rows = []
    ece = 0.0
    for bin_idx in range(n_bins):
        lower = bin_idx / n_bins
        upper = (bin_idx + 1) / n_bins
        mask = (confidence > lower) & (confidence <= upper if bin_idx < n_bins - 1 else confidence <= upper)
        if not mask.any():
            continue
        accuracy = float((y_true[mask] == pred[mask]).mean())
        mean_conf = float(confidence[mask].mean())
        weight = float(mask.mean())
        ece += abs(accuracy - mean_conf) * weight
        rows.append(
            {
                "confidence_lower": lower,
                "confidence_upper": upper,
                "n_rows": int(mask.sum()),
                "accuracy": accuracy,
                "mean_confidence": mean_conf,
                "abs_gap": abs(accuracy - mean_conf),
            }
        )
    return float(ece), rows


def audit_scores(scores: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    output_dir = ensure_output_dir(out_dir)
    metrics_rows = []
    class_rows = []
    confusion_rows = []
    calibration_rows = []
    for (model_name, split), sub in scores.groupby(["model_name", "split"]):
        y = sub["expert_proxy_action"].astype(str).to_numpy()
        pred = sub["policy_pred_action_scan"].astype(str).to_numpy()
        conf = sub["scan_policy_max_prob"].to_numpy(float)
        metrics_rows.append(
            {
                "model_name": model_name,
                "split": split,
                "n_rows": int(len(sub)),
                "accuracy": float(accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, labels=ACTIONS, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y, pred, labels=ACTIONS, average="weighted", zero_division=0)),
                "mean_confidence": float(conf.mean()),
                "n_high_conf_errors": int(((y != pred) & (conf >= 0.90)).sum()),
            }
        )
        ece, calibration = _ece(y, pred, conf)
        metrics_rows[-1]["ece"] = ece
        for row in calibration:
            row["model_name"] = model_name
            row["split"] = split
            calibration_rows.append(row)

        precision, recall, f1, support = precision_recall_fscore_support(
            y,
            pred,
            labels=ACTIONS,
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
        matrix = confusion_matrix(y, pred, labels=ACTIONS)
        for i, actual in enumerate(ACTIONS):
            for j, predicted in enumerate(ACTIONS):
                confusion_rows.append(
                    {
                        "model_name": model_name,
                        "split": split,
                        "actual": actual,
                        "predicted": predicted,
                        "count": int(matrix[i, j]),
                    }
                )

    imbalance_rows = []
    unique_label_rows = scores.drop_duplicates(["episode_id", "time_step", "expert_proxy_action"])
    for (split, action), sub in unique_label_rows.groupby(["split", "expert_proxy_action"]):
        split_total = len(unique_label_rows[unique_label_rows["split"].eq(split)])
        imbalance_rows.append(
            {
                "split": split,
                "action": action,
                "count": int(len(sub)),
                "fraction": float(len(sub) / max(split_total, 1)),
            }
        )

    high_conf = scores[
        (~scores["scan_policy_correct"]) & (pd.to_numeric(scores["scan_policy_max_prob"], errors="coerce") >= 0.90)
    ].copy()

    paths = {
        "scores": output_dir / "gazebo_scan_policy_scores.csv",
        "metrics": output_dir / "gazebo_scan_policy_metrics.csv",
        "class_metrics": output_dir / "gazebo_scan_policy_class_metrics.csv",
        "confusion": output_dir / "gazebo_scan_policy_confusion.csv",
        "calibration": output_dir / "gazebo_scan_policy_calibration_bins.csv",
        "imbalance": output_dir / "gazebo_scan_policy_imbalance.csv",
        "high_conf_errors": output_dir / "gazebo_scan_policy_high_conf_errors.csv",
    }
    scores.to_csv(paths["scores"], index=False)
    pd.DataFrame(metrics_rows).to_csv(paths["metrics"], index=False)
    pd.DataFrame(class_rows).to_csv(paths["class_metrics"], index=False)
    pd.DataFrame(confusion_rows).to_csv(paths["confusion"], index=False)
    pd.DataFrame(calibration_rows).to_csv(paths["calibration"], index=False)
    pd.DataFrame(imbalance_rows).to_csv(paths["imbalance"], index=False)
    high_conf.to_csv(paths["high_conf_errors"], index=False)
    return paths


def train_and_audit_scan_policy(
    input_csv: Path | None,
    log_dir: Path | None,
    out_dir: Path,
    epochs: int = 220,
    require_nav2_plan: bool = True,
) -> dict[str, Path]:
    output_dir = ensure_output_dir(out_dir)
    raw = load_scan_policy_rows(input_csv, log_dir)
    table = prepare_scan_policy_table(raw, require_nav2_plan=require_nav2_plan)
    table_path = output_dir / "gazebo_scan_policy_train_table.csv"
    table.to_csv(table_path, index=False)
    baseline = train_scan_policy(table, "baseline_scan_goal", weighted=False, focal=False, epochs=epochs)
    focal = train_scan_policy(table, "class_weighted_focal_scan_goal", weighted=True, focal=True, epochs=epochs)
    scores = pd.concat([score_policy(table, baseline), score_policy(table, focal)], ignore_index=True)
    paths = audit_scores(scores, output_dir)
    report = {
        "task": "Gazebo scan+goal observation policy",
        "input": str(input_csv or log_dir),
        "label": "expert_proxy_action from Nav2 /plan" if require_nav2_plan else "expert_proxy_action from Nav2 /plan plus proxy fallback labels",
        "require_nav2_plan": require_nav2_plan,
        "actions": ACTIONS,
        "n_rows": int(len(table)),
        "scan_columns": sorted([column for column in table.columns if column.startswith("scan_bin_")]),
        "context_columns": CONTEXT_COLUMNS,
        "models": ["baseline_scan_goal", "class_weighted_focal_scan_goal"],
        "limitation": "Simulation-derived lidar observations; not real-robot visual validation.",
    }
    report_path = output_dir / "gazebo_scan_policy_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    paths["train_table"] = table_path
    paths["report"] = report_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Gazebo scan-observation policy from expert labels.")
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/gazebo_scan_policy_v1"))
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument(
        "--allow-proxy-labels",
        action="store_true",
        help="Train with fallback proxy expert labels as well as Nav2 /plan labels.",
    )
    args = parser.parse_args()
    paths = train_and_audit_scan_policy(
        args.input_csv,
        args.log_dir,
        args.out_dir,
        epochs=args.epochs,
        require_nav2_plan=not args.allow_proxy_labels,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
