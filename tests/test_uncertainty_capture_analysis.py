from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.analyze_policy_uncertainty_capture import analyze_uncertainty_capture


def test_uncertainty_capture_analysis_writes_summary(tmp_path: Path) -> None:
    scores = pd.DataFrame(
        [
            {
                "split": "test",
                "policy_correct": True,
                "policy_entropy": 0.1,
                "policy_margin": 0.8,
                "policy_max_prob": 0.9,
                "embedding_nearest_action_distance": 0.2,
                "embedding_distance_to_teacher_action": 0.2,
                "scenario_primary_fault_origin": "none",
            },
            {
                "split": "test",
                "policy_correct": False,
                "policy_entropy": 1.4,
                "policy_margin": 0.1,
                "policy_max_prob": 0.4,
                "embedding_nearest_action_distance": 1.7,
                "embedding_distance_to_teacher_action": 2.1,
                "scenario_primary_fault_origin": "planner_internal_failure",
            },
        ]
    )
    scores_path = tmp_path / "policy_scores.csv"
    scores.to_csv(scores_path, index=False)
    summary_path, error_cases_path = analyze_uncertainty_capture(scores_path, tmp_path / "analysis")
    summary = pd.read_csv(summary_path)
    errors = pd.read_csv(error_cases_path)
    assert not summary.empty
    assert set(summary["signal"]).issuperset({"policy_entropy", "policy_margin"})
    assert len(errors) == 1
