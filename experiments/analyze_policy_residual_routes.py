"""Analyze residual policy errors and prototype recovery routing evidence.

This script consumes the deterministic scan/depth/fusion policy outputs and
produces the evidence package for:

1. residual policy mechanism analysis,
2. recovery-route prototype rules,
3. modality/loss ablations,
4. compact evidence tables and figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

DEMO_ROOT = Path(__file__).resolve().parents[1]
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from src.utils import ensure_output_dir


DEFAULT_INPUTS = {
    "scan": {
        "scores": DEMO_ROOT / "outputs/gazebo_scan_policy_depth_matrix_train_v2/gazebo_scan_policy_scores.csv",
        "metrics": DEMO_ROOT / "outputs/gazebo_scan_policy_depth_matrix_train_v2/gazebo_scan_policy_metrics.csv",
        "pred": "policy_pred_action_scan",
        "correct": "scan_policy_correct",
        "confidence": "scan_policy_max_prob",
        "entropy": "scan_policy_entropy",
        "margin": "scan_policy_margin",
    },
    "depth": {
        "scores": DEMO_ROOT / "outputs/gazebo_depth_policy_formal_train_v2/gazebo_depth_policy_scores.csv",
        "metrics": DEMO_ROOT / "outputs/gazebo_depth_policy_formal_train_v2/gazebo_depth_policy_metrics.csv",
        "pred": "policy_pred_action_depth",
        "correct": "depth_policy_correct",
        "confidence": "depth_policy_max_prob",
        "entropy": "depth_policy_entropy",
        "margin": "depth_policy_margin",
    },
    "fusion": {
        "scores": DEMO_ROOT / "outputs/gazebo_fusion_policy_formal_train_v2/gazebo_fusion_policy_scores.csv",
        "metrics": DEMO_ROOT / "outputs/gazebo_fusion_policy_formal_train_v2/gazebo_fusion_policy_metrics.csv",
        "pred": "policy_pred_action_fusion",
        "correct": "fusion_policy_correct",
        "confidence": "fusion_policy_max_prob",
        "entropy": "fusion_policy_entropy",
        "margin": "fusion_policy_margin",
    },
}


def _bool_series(series: pd.Series) -> pd.Series:
    return series.fillna(False).map(lambda value: str(value).strip().lower() in {"1", "true", "yes"})


def load_scores(config: dict[str, dict[str, object]]) -> pd.DataFrame:
    frames = []
    for modality, item in config.items():
        df = pd.read_csv(Path(item["scores"]))
        pred_col = str(item["pred"])
        correct_col = str(item["correct"])
        confidence_col = str(item["confidence"])
        entropy_col = str(item["entropy"])
        margin_col = str(item["margin"])
        out = pd.DataFrame(
            {
                "modality": modality,
                "model_name": df["model_name"].astype(str),
                "split": df["split"].astype(str),
                "episode_id": df["episode_id"].astype(str),
                "scenario_id": df["scenario_id"].astype(str),
                "time_step": pd.to_numeric(df["time_step"], errors="coerce").fillna(-1).astype(int),
                "actual_action": df["expert_proxy_action"].astype(str),
                "predicted_action": df[pred_col].astype(str),
                "policy_correct": _bool_series(df[correct_col]),
                "confidence": pd.to_numeric(df[confidence_col], errors="coerce").fillna(0.0),
                "entropy": pd.to_numeric(df.get(entropy_col, 0.0), errors="coerce").fillna(0.0),
                "margin": pd.to_numeric(df.get(margin_col, 0.0), errors="coerce").fillna(0.0),
            }
        )
        for column in [
            "risk_score",
            "sensor_confidence",
            "path_blocked_score",
            "obstacle_proximity",
            "localization_uncertainty",
            "trajectory_deviation",
            "task_progress_stagnation",
            "goal_dx",
            "goal_dy",
            "scan_front_min_range",
            "depth_center_min_m",
            "depth_min_m",
            "scan_min_range",
        ]:
            out[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0) if column in df else 0.0
        frames.append(out)
    scores = pd.concat(frames, ignore_index=True)
    scores["is_error"] = ~scores["policy_correct"]
    scores["is_high_conf_error"] = scores["is_error"] & (scores["confidence"] >= 0.90)
    scores["confusion_pair"] = scores["actual_action"] + "->" + scores["predicted_action"]
    scores["residual_mechanism"] = scores.apply(_mechanism_label, axis=1)
    scores["recovery_route"] = scores.apply(_recovery_route, axis=1)
    return scores


def load_metrics(config: dict[str, dict[str, object]]) -> pd.DataFrame:
    frames = []
    for modality, item in config.items():
        df = pd.read_csv(Path(item["metrics"]))
        df.insert(0, "modality", modality)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _mechanism_label(row: pd.Series) -> str:
    if not bool(row["is_error"]):
        return "correct"
    scenario = str(row["scenario_id"])
    pair = str(row["confusion_pair"])
    sensor_conf = float(row["sensor_confidence"])
    path_blocked = float(row["path_blocked_score"])
    obstacle = float(row["obstacle_proximity"])
    localization = float(row["localization_uncertainty"])
    if localization >= 0.70:
        return "localization_state_error"
    if "perception" in scenario or sensor_conf < 0.55:
        if pair in {"SOUTH->EAST", "EAST->SOUTH"}:
            return "perception_axis_confusion"
        if pair in {"WEST->NORTH", "NORTH->WEST", "WEST->SOUTH"}:
            return "perception_lateral_depth_confusion"
        return "perception_degradation_confusion"
    if "blockage" in scenario or path_blocked >= 0.45 or obstacle >= 0.70:
        if row["confidence"] >= 0.90:
            return "blocked_path_high_conf_direction_error"
        return "blocked_path_direction_error"
    if pair in {"NORTH->WEST", "WEST->NORTH"}:
        return "boundary_direction_confusion"
    return "geometric_policy_residual"


def _recovery_route(row: pd.Series) -> str:
    mechanism = str(row["residual_mechanism"])
    if mechanism == "correct":
        return "NORMAL_NAVIGATION"
    if mechanism == "localization_state_error":
        return "RELOCALIZE"
    if mechanism in {"perception_axis_confusion", "perception_lateral_depth_confusion", "perception_degradation_confusion"}:
        return "CAUTIOUS_REPLAN"
    if mechanism in {"blocked_path_high_conf_direction_error", "blocked_path_direction_error"}:
        return "REPLAN"
    if mechanism == "boundary_direction_confusion":
        return "CAUTIOUS_MODE"
    return "HUMAN_REVIEW"


def _aggregate_tables(scores: pd.DataFrame, metrics: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    out_dir = ensure_output_dir(out_dir)
    paths["unified_scores"] = out_dir / "unified_policy_scores.csv"
    scores.to_csv(paths["unified_scores"], index=False)

    metrics_path = out_dir / "modality_ablation_metrics.csv"
    metrics.sort_values(["split", "modality", "model_name"]).to_csv(metrics_path, index=False)
    paths["modality_ablation_metrics"] = metrics_path

    test_metrics = metrics[metrics["split"].eq("test")].copy()
    scan_baseline = test_metrics[
        test_metrics["modality"].eq("scan") & test_metrics["model_name"].eq("baseline_scan_goal")
    ]
    if not scan_baseline.empty:
        base = scan_baseline.iloc[0]
        for column in ["accuracy", "macro_f1", "weighted_f1", "ece", "n_high_conf_errors"]:
            if column in test_metrics:
                test_metrics[f"delta_vs_scan_baseline_{column}"] = test_metrics[column] - base[column]
    paths["test_ablation_delta"] = out_dir / "test_ablation_delta_vs_scan_baseline.csv"
    test_metrics.sort_values(["accuracy", "weighted_f1"], ascending=False).to_csv(paths["test_ablation_delta"], index=False)

    errors = scores[scores["is_error"]].copy()
    high_conf = scores[scores["is_high_conf_error"]].copy()

    scenario_summary = (
        scores.groupby(["modality", "model_name", "split", "scenario_id"])
        .agg(
            n_rows=("episode_id", "size"),
            n_errors=("is_error", "sum"),
            n_high_conf_errors=("is_high_conf_error", "sum"),
            mean_confidence=("confidence", "mean"),
            mean_risk=("risk_score", "mean"),
            mean_sensor_confidence=("sensor_confidence", "mean"),
            mean_path_blocked_score=("path_blocked_score", "mean"),
        )
        .reset_index()
    )
    scenario_summary["error_rate"] = scenario_summary["n_errors"] / scenario_summary["n_rows"].clip(lower=1)
    scenario_summary["high_conf_error_rate"] = scenario_summary["n_high_conf_errors"] / scenario_summary["n_rows"].clip(lower=1)
    paths["scenario_error_summary"] = out_dir / "scenario_error_summary.csv"
    scenario_summary.to_csv(paths["scenario_error_summary"], index=False)

    mechanism_summary = (
        errors.groupby(["modality", "model_name", "split", "residual_mechanism", "recovery_route"])
        .agg(
            n_errors=("episode_id", "size"),
            n_high_conf_errors=("is_high_conf_error", "sum"),
            mean_confidence=("confidence", "mean"),
            mean_entropy=("entropy", "mean"),
            scenarios=("scenario_id", lambda value: "|".join(sorted(set(map(str, value))))),
            confusion_pairs=("confusion_pair", lambda value: "|".join(sorted(set(map(str, value))))),
        )
        .reset_index()
        .sort_values(["split", "n_high_conf_errors", "n_errors"], ascending=[True, False, False])
    )
    paths["residual_mechanism_summary"] = out_dir / "residual_mechanism_summary.csv"
    mechanism_summary.to_csv(paths["residual_mechanism_summary"], index=False)

    high_conf_patterns = (
        high_conf.groupby(["modality", "model_name", "split", "scenario_id", "confusion_pair", "residual_mechanism", "recovery_route"])
        .agg(
            n_high_conf_errors=("episode_id", "size"),
            mean_confidence=("confidence", "mean"),
            mean_risk=("risk_score", "mean"),
            mean_sensor_confidence=("sensor_confidence", "mean"),
            mean_path_blocked_score=("path_blocked_score", "mean"),
        )
        .reset_index()
        .sort_values(["split", "n_high_conf_errors"], ascending=[True, False])
    )
    paths["high_conf_error_patterns"] = out_dir / "high_conf_error_patterns.csv"
    high_conf_patterns.to_csv(paths["high_conf_error_patterns"], index=False)

    route_eval = (
        high_conf.groupby(["modality", "model_name", "split", "recovery_route"])
        .agg(
            n_high_conf_errors=("episode_id", "size"),
            mechanisms=("residual_mechanism", lambda value: "|".join(sorted(set(map(str, value))))),
            scenarios=("scenario_id", lambda value: "|".join(sorted(set(map(str, value))))),
            confusion_pairs=("confusion_pair", lambda value: "|".join(sorted(set(map(str, value))))),
        )
        .reset_index()
        .sort_values(["split", "n_high_conf_errors"], ascending=[True, False])
    )
    route_eval["route_role"] = route_eval["recovery_route"].map(
        {
            "CAUTIOUS_REPLAN": "perception-risk recovery",
            "REPLAN": "path-blockage recovery",
            "RELOCALIZE": "localization recovery",
            "CAUTIOUS_MODE": "boundary uncertainty recovery",
            "HUMAN_REVIEW": "unclassified residual review",
        }
    )
    paths["recovery_route_evidence"] = out_dir / "recovery_route_evidence.csv"
    route_eval.to_csv(paths["recovery_route_evidence"], index=False)

    route_coverage = (
        high_conf.groupby(["modality", "model_name", "split"])
        .agg(total_high_conf_errors=("episode_id", "size"))
        .reset_index()
    )
    actionable = high_conf[high_conf["recovery_route"].isin(["CAUTIOUS_REPLAN", "REPLAN", "RELOCALIZE", "CAUTIOUS_MODE"])]
    actionable_counts = (
        actionable.groupby(["modality", "model_name", "split"])
        .agg(actionable_high_conf_errors=("episode_id", "size"))
        .reset_index()
    )
    route_coverage = route_coverage.merge(actionable_counts, how="left").fillna({"actionable_high_conf_errors": 0})
    route_coverage["actionable_fraction"] = (
        route_coverage["actionable_high_conf_errors"] / route_coverage["total_high_conf_errors"].clip(lower=1)
    )
    paths["recovery_route_coverage"] = out_dir / "recovery_route_coverage.csv"
    route_coverage.to_csv(paths["recovery_route_coverage"], index=False)
    return paths


def _write_figures(metrics: pd.DataFrame, scores: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    figure_dir = ensure_output_dir(out_dir / "figures")
    paths: dict[str, Path] = {}
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return paths

    test = metrics[metrics["split"].eq("test")].copy()
    test["label"] = test["modality"] + "\n" + test["model_name"].str.replace("_", " ", regex=False)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(test["label"], test["accuracy"], color="#4d7c8a")
    ax.set_ylim(0.75, 1.0)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Policy Ablation: Test Accuracy")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()
    paths["test_accuracy"] = figure_dir / "test_accuracy_by_modality.png"
    fig.savefig(paths["test_accuracy"], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(test["label"], test["n_high_conf_errors"], color="#b35c44")
    ax.set_ylabel("High-confidence errors")
    ax.set_title("Policy Ablation: Test High-Confidence Errors")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()
    paths["test_high_conf"] = figure_dir / "test_high_conf_errors_by_modality.png"
    fig.savefig(paths["test_high_conf"], dpi=180)
    plt.close(fig)

    high_conf = scores[scores["is_high_conf_error"] & scores["split"].eq("test")].copy()
    if not high_conf.empty:
        route_counts = high_conf.groupby("recovery_route").size().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(route_counts.index, route_counts.values, color="#687b3e")
        ax.set_ylabel("Test high-confidence errors")
        ax.set_title("Prototype Recovery Routes For Residual Errors")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        paths["routes"] = figure_dir / "test_recovery_route_distribution.png"
        fig.savefig(paths["routes"], dpi=180)
        plt.close(fig)

    return paths


def analyze_policy_residual_routes(out_dir: Path) -> dict[str, Path]:
    output_dir = ensure_output_dir(out_dir)
    scores = load_scores(DEFAULT_INPUTS)
    metrics = load_metrics(DEFAULT_INPUTS)
    paths = _aggregate_tables(scores, metrics, output_dir)
    paths.update(_write_figures(metrics, scores, output_dir))
    report = {
        "task": "Residual policy mechanism analysis and recovery-route prototype",
        "inputs": {name: {key: str(value) for key, value in item.items() if key in {"scores", "metrics"}} for name, item in DEFAULT_INPUTS.items()},
        "high_confidence_threshold": 0.90,
        "route_policy": {
            "perception_*": "CAUTIOUS_REPLAN",
            "blocked_path_*": "REPLAN",
            "localization_state_error": "RELOCALIZE",
            "boundary_direction_confusion": "CAUTIOUS_MODE",
            "geometric_policy_residual": "HUMAN_REVIEW",
        },
        "limitations": [
            "Simulation-derived evidence only.",
            "One seed per train/val/test split.",
            "Recovery routes are prototype labels evaluated on residual errors, not deployed robot actions.",
        ],
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    report_path = output_dir / "residual_route_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    paths["report"] = report_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze residual policy mechanisms and recovery routes.")
    parser.add_argument("--out-dir", type=Path, default=DEMO_ROOT / "outputs/gazebo_policy_residual_routes_v1")
    args = parser.parse_args()
    paths = analyze_policy_residual_routes(args.out_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
