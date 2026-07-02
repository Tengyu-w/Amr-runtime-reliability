"""Analyze whether policy uncertainty captures routing errors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

DEMO_ROOT = Path(__file__).resolve().parents[1]
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from src.utils import ensure_output_dir


SIGNALS = {
    "policy_entropy": "higher_is_riskier",
    "policy_margin": "lower_is_riskier",
    "policy_max_prob": "lower_is_riskier",
    "embedding_nearest_action_distance": "higher_is_riskier",
    "embedding_distance_to_teacher_action": "higher_is_riskier",
}


def _risk_values(df: pd.DataFrame, signal: str) -> pd.Series:
    values = pd.to_numeric(df[signal], errors="coerce")
    if SIGNALS[signal] == "lower_is_riskier":
        return -values
    return values


def _safe_auc(y_error: pd.Series, risk_values: pd.Series) -> tuple[float | None, float | None]:
    valid = y_error.notna() & risk_values.notna()
    y = y_error[valid].astype(int)
    scores = risk_values[valid].astype(float)
    if y.nunique() < 2:
        return None, None
    return float(roc_auc_score(y, scores)), float(average_precision_score(y, scores))


def _top_quantile_capture(y_error: pd.Series, risk_values: pd.Series, quantile: float) -> float | None:
    valid = y_error.notna() & risk_values.notna()
    y = y_error[valid].astype(int)
    scores = risk_values[valid].astype(float)
    if y.sum() == 0 or scores.empty:
        return None
    threshold = scores.quantile(1.0 - quantile)
    selected = scores >= threshold
    return float(y[selected].sum() / y.sum())


def analyze_uncertainty_capture(scores_path: str | Path, out_dir: str | Path) -> tuple[Path, Path]:
    scores = pd.read_csv(scores_path)
    scores["policy_error"] = ~scores["policy_correct"].astype(bool)
    output_dir = ensure_output_dir(out_dir)
    rows = []
    for split, split_df in scores.groupby("split"):
        for signal in SIGNALS:
            if signal not in split_df:
                continue
            risk = _risk_values(split_df, signal)
            auroc, auprc = _safe_auc(split_df["policy_error"], risk)
            rows.append(
                {
                    "group": split,
                    "signal": signal,
                    "n": int(len(split_df)),
                    "n_errors": int(split_df["policy_error"].sum()),
                    "auroc_error_detection": auroc,
                    "auprc_error_detection": auprc,
                    "top10pct_error_capture": _top_quantile_capture(split_df["policy_error"], risk, 0.10),
                    "top20pct_error_capture": _top_quantile_capture(split_df["policy_error"], risk, 0.20),
                    "risk_direction": SIGNALS[signal],
                }
            )
    if "scenario_primary_fault_origin" in scores:
        for (split, origin), sub in scores.groupby(["split", "scenario_primary_fault_origin"]):
            for signal in SIGNALS:
                if signal not in sub:
                    continue
                risk = _risk_values(sub, signal)
                auroc, auprc = _safe_auc(sub["policy_error"], risk)
                rows.append(
                    {
                        "group": f"{split}:{origin}",
                        "signal": signal,
                        "n": int(len(sub)),
                        "n_errors": int(sub["policy_error"].sum()),
                        "auroc_error_detection": auroc,
                        "auprc_error_detection": auprc,
                        "top10pct_error_capture": _top_quantile_capture(sub["policy_error"], risk, 0.10),
                        "top20pct_error_capture": _top_quantile_capture(sub["policy_error"], risk, 0.20),
                        "risk_direction": SIGNALS[signal],
                    }
                )

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "uncertainty_error_capture.csv"
    error_cases_path = output_dir / "policy_error_cases.csv"
    summary.to_csv(summary_path, index=False)
    scores[scores["policy_error"]].to_csv(error_cases_path, index=False)
    return summary_path, error_cases_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze whether policy uncertainty captures action errors.")
    parser.add_argument("--scores", type=Path, default=Path("outputs/policy_model/policy_scores.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/policy_uncertainty_capture"))
    args = parser.parse_args()
    summary_path, error_cases_path = analyze_uncertainty_capture(args.scores, args.out_dir)
    print(f"Uncertainty capture summary: {summary_path}")
    print(f"Policy error cases: {error_cases_path}")


if __name__ == "__main__":
    main()
