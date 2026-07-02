"""Train and audit a learned AMR recovery-need model."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEMO_ROOT = Path(__file__).resolve().parents[1]
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from experiments.generate_scenario_dataset import generate_dataset
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

HARD_RECOVERY_FAMILIES = {
    "path_blockage",
    "progress_blockage",
    "control_tracking",
    "planning_backend",
    "task_reassignment",
}


def _parse_position(value: object) -> tuple[float, float]:
    match = re.match(r"\(([-0-9.]+),\s*([-0-9.]+)\)", str(value))
    if not match:
        return 0.0, 0.0
    return float(match.group(1)), float(match.group(2))


def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(set(y_true.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def _safe_aupr(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(set(y_true.tolist())) < 2:
        return float("nan")
    return float(average_precision_score(y_true, score))


def prepare_learning_table(timesteps: pd.DataFrame) -> pd.DataFrame:
    """Create model features and labels from timestep logs."""

    df = timesteps[timesteps["mode"].eq("baseline")].copy()
    robot_xy = df["robot_position"].map(_parse_position)
    target_xy = df["target_position"].map(_parse_position)
    df["robot_x"] = [xy[0] for xy in robot_xy]
    df["robot_y"] = [xy[1] for xy in robot_xy]
    df["target_x"] = [xy[0] for xy in target_xy]
    df["target_y"] = [xy[1] for xy in target_xy]
    df["moved_numeric"] = df["moved"].astype(str).str.lower().isin({"true", "1"}).astype(float)

    fault_family = df["scenario_primary_fault_family"].astype(str)
    event_family = df["primary_fault_family"].astype(str)
    df["needs_hard_recovery"] = (
        event_family.isin(HARD_RECOVERY_FAMILIES)
        | (
            df["scenario_primary_fault_family"].astype(str).isin(HARD_RECOVERY_FAMILIES)
            & df["failure_event"].astype(str).ne("none")
        )
    ).astype(int)
    df["has_fault_event"] = df["failure_event"].astype(str).ne("none").astype(int)
    df["is_ood_style"] = df["scenario_primary_ood_status"].astype(str).eq("ood_style_shift").astype(int)
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def _hidden_embedding(model: Pipeline, x: np.ndarray) -> np.ndarray:
    scaler = model.named_steps["scale"]
    mlp = model.named_steps["mlp"]
    x_scaled = scaler.transform(x)
    hidden = x_scaled @ mlp.coefs_[0] + mlp.intercepts_[0]
    return np.maximum(hidden, 0.0)


def _add_embedding_distances(train_emb: np.ndarray, train_y: np.ndarray, emb: np.ndarray, out: pd.DataFrame) -> None:
    if train_y.sum() == 0 or train_y.sum() == len(train_y):
        out["embedding_distance_to_recovery"] = np.nan
        out["embedding_distance_to_nominal"] = np.nan
        out["embedding_recovery_margin"] = np.nan
        return
    pos_centroid = train_emb[train_y == 1].mean(axis=0)
    neg_centroid = train_emb[train_y == 0].mean(axis=0)
    dist_pos = np.linalg.norm(emb - pos_centroid, axis=1)
    dist_neg = np.linalg.norm(emb - neg_centroid, axis=1)
    out["embedding_distance_to_recovery"] = dist_pos
    out["embedding_distance_to_nominal"] = dist_neg
    out["embedding_recovery_margin"] = dist_neg - dist_pos


def _score_split(model: Pipeline, train_emb: np.ndarray, train_y: np.ndarray, df: pd.DataFrame, split: str) -> pd.DataFrame:
    split_df = df[df["split"].eq(split)].copy()
    x = split_df[FEATURE_COLUMNS].to_numpy(float)
    probs = model.predict_proba(x)[:, 1]
    pred = (probs >= 0.5).astype(int)
    entropy = -(
        probs * np.log(np.maximum(probs, 1e-12))
        + (1.0 - probs) * np.log(np.maximum(1.0 - probs, 1e-12))
    )
    margin = np.abs(probs - 0.5) * 2.0
    emb = _hidden_embedding(model, x)
    split_df["recovery_prob"] = probs
    split_df["recovery_pred"] = pred
    split_df["recovery_entropy"] = entropy
    split_df["recovery_margin"] = margin
    split_df["model_error"] = pred != split_df["needs_hard_recovery"].to_numpy(int)
    _add_embedding_distances(train_emb, train_y, emb, split_df)
    return split_df


def _metric_rows(scored: pd.DataFrame) -> list[dict]:
    rows = []
    for split, sub in scored.groupby("split"):
        y = sub["needs_hard_recovery"].to_numpy(int)
        score = sub["recovery_prob"].to_numpy(float)
        pred = sub["recovery_pred"].to_numpy(int)
        rows.append(
            {
                "group": split,
                "n": len(sub),
                "positive_rate": float(y.mean()) if len(y) else 0.0,
                "accuracy": float(accuracy_score(y, pred)) if len(y) else float("nan"),
                "auroc": _safe_auc(y, score),
                "aupr": _safe_aupr(y, score),
                "mean_entropy": float(sub["recovery_entropy"].mean()),
                "mean_margin": float(sub["recovery_margin"].mean()),
            }
        )
    for (split, origin), sub in scored.groupby(["split", "scenario_primary_fault_origin"]):
        y = sub["needs_hard_recovery"].to_numpy(int)
        score = sub["recovery_prob"].to_numpy(float)
        rows.append(
            {
                "group": f"{split}:{origin}",
                "n": len(sub),
                "positive_rate": float(y.mean()) if len(y) else 0.0,
                "accuracy": float(accuracy_score(y, sub["recovery_pred"].to_numpy(int))) if len(y) else float("nan"),
                "auroc": _safe_auc(y, score),
                "aupr": _safe_aupr(y, score),
                "mean_entropy": float(sub["recovery_entropy"].mean()),
                "mean_margin": float(sub["recovery_margin"].mean()),
            }
        )
    return rows


def train_and_audit(dataset_dir: str | Path, out_dir: str | Path) -> tuple[Path, Path, Path]:
    dataset_path = Path(dataset_dir)
    output_dir = ensure_output_dir(out_dir)
    timesteps = pd.read_csv(dataset_path / "timesteps.csv")
    table = prepare_learning_table(timesteps)

    train = table[table["split"].eq("train")]
    if train.empty:
        raise ValueError("Training split is empty. Generate dataset with train seeds first.")
    x_train = train[FEATURE_COLUMNS].to_numpy(float)
    y_train = train["needs_hard_recovery"].to_numpy(int)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(10,),
                    alpha=0.001,
                    random_state=7,
                    max_iter=1000,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    train_emb = _hidden_embedding(model, x_train)

    scored = pd.concat(
        [_score_split(model, train_emb, y_train, table, split) for split in ["train", "val", "test"]],
        ignore_index=True,
    )
    metrics = pd.DataFrame(_metric_rows(scored))

    scored_path = output_dir / "learned_recovery_scores.csv"
    metrics_path = output_dir / "learned_recovery_metrics.csv"
    report_path = output_dir / "learned_recovery_report.json"
    scored.to_csv(scored_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    report = {
        "feature_columns": FEATURE_COLUMNS,
        "label": "needs_hard_recovery",
        "label_note": "Ground-truth fault family is used only as a training/evaluation target, not as an inference feature.",
        "model": "StandardScaler + one-hidden-layer MLPClassifier",
        "hard_recovery_families": sorted(HARD_RECOVERY_FAMILIES),
        "n_rows": int(len(table)),
        "n_train": int(len(train)),
        "n_val": int(table["split"].eq("val").sum()),
        "n_test": int(table["split"].eq("test").sum()),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return scored_path, metrics_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a learned recovery-need model from AMR simulation data.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("outputs/scenario_dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/learned_recovery_model"))
    parser.add_argument(
        "--generate-if-missing",
        action="store_true",
        help="Generate the default scenario dataset before training when timesteps.csv is missing.",
    )
    args = parser.parse_args()

    if args.generate_if_missing and not (args.dataset_dir / "timesteps.csv").exists():
        generate_dataset(
            seeds=[10, 11, 12, 16, 17, 18, 19],
            out_dir=args.dataset_dir,
            modes=["baseline", "risk_router", "mechanism_router"],
        )
    scored_path, metrics_path, report_path = train_and_audit(args.dataset_dir, args.out_dir)
    print(f"Scores: {scored_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
