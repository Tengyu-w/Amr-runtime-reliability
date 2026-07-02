"""Run feature ablations and evidence tables for Gazebo route learning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

DEMO_ROOT = Path(__file__).resolve().parents[1]
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from experiments.train_gazebo_route_model import (
    ACTION_TO_INDEX,
    DEFAULT_FEATURE_COLUMNS,
    prepare_gazebo_route_table,
    score_route_model,
    train_route_model,
)
from src.utils import ensure_output_dir


FEATURE_SETS = {
    "full": DEFAULT_FEATURE_COLUMNS,
    "risk_only": ["risk_score"],
    "no_risk_score": [col for col in DEFAULT_FEATURE_COLUMNS if col != "risk_score"],
    "no_localization": [col for col in DEFAULT_FEATURE_COLUMNS if col != "localization_uncertainty"],
    "no_perception": [col for col in DEFAULT_FEATURE_COLUMNS if col != "sensor_confidence"],
    "no_blockage": [
        col for col in DEFAULT_FEATURE_COLUMNS if col not in {"path_blocked_score", "obstacle_proximity"}
    ],
    "no_execution": [
        col for col in DEFAULT_FEATURE_COLUMNS if col not in {"trajectory_deviation", "task_progress_stagnation"}
    ],
}


def _metrics_for_scored(scored: pd.DataFrame, ablation: str) -> list[dict[str, object]]:
    rows = []
    for split, sub in scored.groupby("split"):
        y = sub["action_index"].to_numpy(int)
        pred = np.array([ACTION_TO_INDEX[action] for action in sub["route_model_pred_action"]])
        rows.append(
            {
                "ablation": ablation,
                "group": split,
                "n": int(len(sub)),
                "accuracy": float(accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            }
        )
    for (split, origin), sub in scored.groupby(["split", "scenario_primary_fault_origin"]):
        y = sub["action_index"].to_numpy(int)
        pred = np.array([ACTION_TO_INDEX[action] for action in sub["route_model_pred_action"]])
        rows.append(
            {
                "ablation": ablation,
                "group": f"{split}:{origin}",
                "n": int(len(sub)),
                "accuracy": float(accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            }
        )
    return rows


def _evidence_by_fault_origin(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, origin, expected), sub in scored.groupby(
        ["split", "scenario_primary_fault_origin", "target_action"]
    ):
        rows.append(
            {
                "split": split,
                "fault_origin": origin,
                "expected_recovery": expected,
                "n": int(len(sub)),
                "predicted_modal_recovery": sub["route_model_pred_action"].mode().iloc[0],
                "expected_capture_rate": float((sub["route_model_pred_action"] == expected).mean()),
                "mean_entropy": float(sub["route_model_entropy"].mean()),
                "mean_risk": float(pd.to_numeric(sub["risk_score"], errors="coerce").fillna(0.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _bool_rate(series: pd.Series) -> float:
    normalized = series.fillna(False).map(
        lambda value: str(value).strip().lower() in {"1", "true", "yes"}
    )
    return float(normalized.mean()) if len(normalized) else float("nan")


def _evidence_by_episode_outcome(scored: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    if "outcome_label" not in episodes:
        return pd.DataFrame()
    episode_cols = [
        "episode_id",
        "outcome_label",
        "expected_route_observed",
        "goal_reached_proxy",
        "collision_risk_proxy",
        "recovery_latency_steps",
    ]
    merged = scored.merge(episodes[episode_cols], on="episode_id", how="left")
    rows = []
    for (split, outcome), sub in merged.groupby(["split", "outcome_label"], dropna=False):
        y = sub["action_index"].to_numpy(int)
        pred = np.array([ACTION_TO_INDEX[action] for action in sub["route_model_pred_action"]])
        rows.append(
            {
                "split": split,
                "outcome_label": outcome,
                "n_rows": int(len(sub)),
                "n_episodes": int(sub["episode_id"].nunique()),
                "accuracy": float(accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                "mean_entropy": float(sub["route_model_entropy"].mean()),
                "mean_risk": float(pd.to_numeric(sub["risk_score"], errors="coerce").fillna(0.0).mean()),
                "expected_route_observed_rate": _bool_rate(sub["expected_route_observed"]),
                "goal_reached_proxy_rate": _bool_rate(sub["goal_reached_proxy"]),
                "collision_risk_proxy_rate": _bool_rate(sub["collision_risk_proxy"]),
                "mean_recovery_latency_steps": float(
                    pd.to_numeric(sub["recovery_latency_steps"], errors="coerce").mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def run_gazebo_route_ablation(
    dataset_dir: str | Path,
    out_dir: str | Path,
    label_column: str = "target_action",
    epochs: int = 220,
) -> tuple[Path, Path, Path, Path]:
    output_dir = ensure_output_dir(out_dir)
    dataset_path = Path(dataset_dir)
    timesteps = pd.read_csv(dataset_path / "timesteps.csv")
    episodes = pd.read_csv(dataset_path / "episodes.csv")
    metric_rows = []
    full_scores = None
    for ablation, features in FEATURE_SETS.items():
        table = prepare_gazebo_route_table(timesteps, label_column=label_column, feature_columns=features)
        policy = train_route_model(table, feature_columns=features, epochs=epochs)
        scored = score_route_model(policy, table)
        metric_rows.extend(_metrics_for_scored(scored, ablation))
        if ablation == "full":
            full_scores = scored

    ablation_path = output_dir / "ablation_metrics.csv"
    evidence_path = output_dir / "evidence_by_fault_origin.csv"
    confusion_path = output_dir / "scenario_route_confusion.csv"
    outcome_path = output_dir / "evidence_by_episode_outcome.csv"
    pd.DataFrame(metric_rows).to_csv(ablation_path, index=False)
    if full_scores is None:
        raise RuntimeError("Full feature model did not run.")
    _evidence_by_fault_origin(full_scores).to_csv(evidence_path, index=False)
    _evidence_by_episode_outcome(full_scores, episodes).to_csv(outcome_path, index=False)
    pd.crosstab(
        [full_scores["scenario_id"], full_scores["target_action"]],
        full_scores["route_model_pred_action"],
        dropna=False,
    ).to_csv(confusion_path)
    return ablation_path, evidence_path, confusion_path, outcome_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gazebo route-model feature ablations.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("outputs/gazebo_episode_dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/gazebo_route_ablation"))
    parser.add_argument("--label-column", type=str, default="target_action")
    parser.add_argument("--epochs", type=int, default=220)
    args = parser.parse_args()
    ablation_path, evidence_path, confusion_path, outcome_path = run_gazebo_route_ablation(
        args.dataset_dir,
        args.out_dir,
        label_column=args.label_column,
        epochs=args.epochs,
    )
    print(f"Ablation metrics: {ablation_path}")
    print(f"Evidence by fault origin: {evidence_path}")
    print(f"Scenario route confusion: {confusion_path}")
    print(f"Evidence by episode outcome: {outcome_path}")


if __name__ == "__main__":
    main()
