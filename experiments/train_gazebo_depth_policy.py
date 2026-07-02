"""Train a Gazebo depth-observation policy from Nav2-plan expert labels."""

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
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
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
    "depth_valid_fraction",
    "depth_min_m",
    "depth_mean_m",
    "depth_center_min_m",
]


class DepthGoalPolicyNet(nn.Module):
    def __init__(self, n_depth_cells: int, n_context_features: int, n_actions: int) -> None:
        super().__init__()
        self.depth_encoder = nn.Sequential(
            nn.Linear(n_depth_cells, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(n_context_features, 32),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, depth: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        depth_embedding = self.depth_encoder(depth)
        context_embedding = self.context_encoder(context)
        embedding = torch.cat([depth_embedding, context_embedding], dim=1)
        return self.head(embedding), embedding


@dataclass
class TrainedDepthPolicy:
    model: DepthGoalPolicyNet
    context_scaler: StandardScaler
    depth_columns: list[str]
    model_name: str


def _bool_series(series: pd.Series) -> pd.Series:
    return series.fillna(False).map(lambda value: str(value).strip().lower() in {"1", "true", "yes"})


def _seed_from_episode_id(episode_id: str) -> int | None:
    match = re.search(r"(?:^|__)seed_([0-9]+)(?:__|$)", episode_id)
    return int(match.group(1)) if match else None


def _split_for_seed(seed: int | None, fallback_index: int) -> str:
    bucket = fallback_index % 10 if seed is None else seed % 10
    if bucket <= 5:
        return "train"
    if bucket <= 7:
        return "val"
    return "test"


def load_depth_policy_rows(input_csv: Path | None, log_dir: Path | None) -> pd.DataFrame:
    if input_csv is not None:
        frames = [pd.read_csv(input_csv)]
    elif log_dir is not None:
        paths = sorted(log_dir.glob("*.depth_policy.csv"))
        if not paths:
            raise FileNotFoundError(f"No *.depth_policy.csv files found in {log_dir}")
        frames = []
        for path in paths:
            df = pd.read_csv(path)
            if df.empty:
                continue
            if "episode_id" not in df or df["episode_id"].isna().all():
                df["episode_id"] = path.stem.replace(".depth_policy", "")
            df["depth_policy_log_path"] = str(path)
            frames.append(df)
    else:
        raise ValueError("Either input_csv or log_dir must be provided.")
    if not frames:
        raise FileNotFoundError("No non-empty depth-policy rows found.")
    return pd.concat(frames, ignore_index=True)


def prepare_depth_policy_table(rows: pd.DataFrame, require_nav2_plan: bool = True) -> pd.DataFrame:
    df = rows.copy()
    if "policy_evaluable" in df:
        df = df[_bool_series(df["policy_evaluable"])].copy()
    if require_nav2_plan and "expert_source" in df:
        df = df[df["expert_source"].astype(str).eq("nav2_plan")].copy()
    df = df[df["expert_proxy_action"].astype(str).isin(ACTION_TO_INDEX)].copy()
    depth_columns = sorted([column for column in df.columns if column.startswith("depth_cell_")])
    if not depth_columns:
        raise ValueError("No depth_cell_* columns found. Run depth_policy_observation_recorder first.")
    if df.empty:
        raise ValueError("No evaluable Nav2-plan depth-policy rows found.")
    for column in depth_columns + CONTEXT_COLUMNS:
        if column not in df:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    df = df.copy()
    if "split" not in df:
        df["split"] = [
            _split_for_seed(_seed_from_episode_id(str(episode_id)), idx)
            for idx, episode_id in enumerate(df.get("episode_id", pd.Series([""] * len(df))).astype(str))
        ]
    df["action_index"] = df["expert_proxy_action"].astype(str).map(ACTION_TO_INDEX).astype(int)
    return df


def train_depth_policy(table: pd.DataFrame, model_name: str, epochs: int = 160) -> TrainedDepthPolicy:
    torch.manual_seed(7)
    np.random.seed(7)
    depth_columns = sorted([column for column in table.columns if column.startswith("depth_cell_")])
    train = table[table["split"].eq("train")].copy()
    if train.empty:
        raise ValueError("Training split is empty.")
    depth_train = train[depth_columns].to_numpy(np.float32)
    context_scaler = StandardScaler()
    context_train = context_scaler.fit_transform(train[CONTEXT_COLUMNS].to_numpy(float)).astype(np.float32)
    y_train = train["action_index"].to_numpy(np.int64)
    model = DepthGoalPolicyNet(len(depth_columns), len(CONTEXT_COLUMNS), len(ACTIONS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    depth_tensor = torch.from_numpy(depth_train)
    context_tensor = torch.from_numpy(context_train)
    y_tensor = torch.from_numpy(y_train)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits, _ = model(depth_tensor, context_tensor)
        loss = nn.functional.cross_entropy(logits, y_tensor)
        loss.backward()
        optimizer.step()
    return TrainedDepthPolicy(model=model, context_scaler=context_scaler, depth_columns=depth_columns, model_name=model_name)


def score_policy(table: pd.DataFrame, trained: TrainedDepthPolicy) -> pd.DataFrame:
    scored = []
    trained.model.eval()
    with torch.no_grad():
        for split, split_df in table.groupby("split"):
            depth = torch.from_numpy(split_df[trained.depth_columns].to_numpy(np.float32))
            context = torch.from_numpy(
                trained.context_scaler.transform(split_df[CONTEXT_COLUMNS].to_numpy(float)).astype(np.float32)
            )
            logits, embeddings = trained.model(depth, context)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred_idx = probs.argmax(axis=1)
            out = split_df.copy()
            out["model_name"] = trained.model_name
            out["policy_pred_action_depth"] = [ACTIONS[idx] for idx in pred_idx]
            out["depth_policy_correct"] = out["policy_pred_action_depth"].eq(out["expert_proxy_action"].astype(str))
            out["depth_policy_max_prob"] = probs.max(axis=1)
            sorted_probs = np.sort(probs, axis=1)
            out["depth_policy_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
            out["depth_policy_entropy"] = -(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=1)
            out["embedding_norm"] = np.linalg.norm(embeddings.cpu().numpy(), axis=1)
            scored.append(out)
    return pd.concat(scored, ignore_index=True)


def audit_depth_scores(scores: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    output_dir = ensure_output_dir(out_dir)
    metrics_rows = []
    class_rows = []
    for (model_name, split), sub in scores.groupby(["model_name", "split"]):
        y = sub["expert_proxy_action"].astype(str).to_numpy()
        pred = sub["policy_pred_action_depth"].astype(str).to_numpy()
        conf = sub["depth_policy_max_prob"].to_numpy(float)
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
    unique_rows = scores.drop_duplicates(["episode_id", "time_step", "expert_proxy_action"])
    imbalance = []
    for (split, action), sub in unique_rows.groupby(["split", "expert_proxy_action"]):
        total = len(unique_rows[unique_rows["split"].eq(split)])
        imbalance.append({"split": split, "action": action, "count": int(len(sub)), "fraction": float(len(sub) / max(total, 1))})
    high_conf = scores[(~scores["depth_policy_correct"]) & (scores["depth_policy_max_prob"] >= 0.90)].copy()
    paths = {
        "scores": output_dir / "gazebo_depth_policy_scores.csv",
        "metrics": output_dir / "gazebo_depth_policy_metrics.csv",
        "class_metrics": output_dir / "gazebo_depth_policy_class_metrics.csv",
        "imbalance": output_dir / "gazebo_depth_policy_imbalance.csv",
        "high_conf_errors": output_dir / "gazebo_depth_policy_high_conf_errors.csv",
    }
    scores.to_csv(paths["scores"], index=False)
    pd.DataFrame(metrics_rows).to_csv(paths["metrics"], index=False)
    pd.DataFrame(class_rows).to_csv(paths["class_metrics"], index=False)
    pd.DataFrame(imbalance).to_csv(paths["imbalance"], index=False)
    high_conf.to_csv(paths["high_conf_errors"], index=False)
    return paths


def _ece(y_true: np.ndarray, pred: np.ndarray, confidence: np.ndarray, n_bins: int = 10) -> float:
    ece = 0.0
    for bin_idx in range(n_bins):
        lower = bin_idx / n_bins
        upper = (bin_idx + 1) / n_bins
        mask = (confidence > lower) & (confidence <= upper if bin_idx < n_bins - 1 else confidence <= upper)
        if mask.any():
            accuracy = float((y_true[mask] == pred[mask]).mean())
            mean_conf = float(confidence[mask].mean())
            ece += abs(accuracy - mean_conf) * float(mask.mean())
    return float(ece)


def train_and_audit_depth_policy(input_csv: Path | None, log_dir: Path | None, out_dir: Path, epochs: int = 160) -> dict[str, Path]:
    output_dir = ensure_output_dir(out_dir)
    table = prepare_depth_policy_table(load_depth_policy_rows(input_csv, log_dir))
    table_path = output_dir / "gazebo_depth_policy_train_table.csv"
    table.to_csv(table_path, index=False)
    trained = train_depth_policy(table, "baseline_depth_goal", epochs=epochs)
    scores = score_policy(table, trained)
    paths = audit_depth_scores(scores, output_dir)
    report_path = output_dir / "gazebo_depth_policy_report.json"
    report = {
        "task": "Gazebo depth+goal observation policy",
        "label": "expert_proxy_action from Nav2 /plan",
        "actions": ACTIONS,
        "n_rows": int(len(table)),
        "depth_columns": sorted([column for column in table.columns if column.startswith("depth_cell_")]),
        "context_columns": CONTEXT_COLUMNS,
        "limitation": "Simulation-derived depth images; not real-robot visual validation.",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    paths["train_table"] = table_path
    paths["report"] = report_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Gazebo depth-observation policy from expert labels.")
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/gazebo_depth_policy_v1"))
    parser.add_argument("--epochs", type=int, default=160)
    args = parser.parse_args()
    paths = train_and_audit_depth_policy(args.input_csv, args.log_dir, args.out_dir, epochs=args.epochs)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
