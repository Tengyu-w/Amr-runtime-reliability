import pandas as pd

from experiments.generate_scenario_dataset import generate_dataset
from experiments.train_policy_model import (
    ACTIONS,
    FEATURE_COLUMNS,
    prepare_policy_table,
    train_and_audit_policy,
)


def test_policy_features_do_not_include_ground_truth_fault_labels() -> None:
    forbidden = ["fault", "scenario", "ood", "event"]

    assert all(not any(fragment in feature for fragment in forbidden) for feature in FEATURE_COLUMNS)


def test_prepare_policy_table_uses_teacher_actions(tmp_path) -> None:
    _, _, _, timestep_path = generate_dataset(
        seeds=[10, 18],
        out_dir=tmp_path,
        modes=["mechanism_router"],
    )
    table = prepare_policy_table(pd.read_csv(timestep_path), teacher_mode="mechanism_router")

    assert set(FEATURE_COLUMNS).issubset(table.columns)
    assert table["router_decision"].isin(ACTIONS).all()
    assert table["action_index"].between(0, len(ACTIONS) - 1).all()


def test_train_policy_writes_action_evidence(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    out_dir = tmp_path / "policy"
    generate_dataset(
        seeds=[10, 11, 12, 16, 17, 18, 19],
        out_dir=dataset_dir,
        modes=["baseline", "risk_router", "mechanism_router"],
    )

    scores_path, metrics_path, report_path = train_and_audit_policy(dataset_dir, out_dir)
    scores = pd.read_csv(scores_path)
    metrics = pd.read_csv(metrics_path)

    for column in [
        "policy_pred_action",
        "policy_correct",
        "policy_entropy",
        "policy_margin",
        "policy_max_prob",
        "embedding_distance_to_pred_action",
        "embedding_distance_to_teacher_action",
        "embedding_nearest_action_distance",
    ]:
        assert column in scores.columns
    assert {"train", "val", "test"}.issubset(set(metrics["group"]))
    assert report_path.exists()
