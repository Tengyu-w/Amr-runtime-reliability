import pandas as pd

from experiments.generate_scenario_dataset import generate_dataset
from experiments.train_learned_recovery_model import (
    FEATURE_COLUMNS,
    prepare_learning_table,
    train_and_audit,
)


def test_learning_features_do_not_include_fault_labels() -> None:
    forbidden_fragments = ["fault", "scenario", "ood", "event"]

    assert all(not any(fragment in col for fragment in forbidden_fragments) for col in FEATURE_COLUMNS)


def test_prepare_learning_table_creates_recovery_target(tmp_path) -> None:
    _, _, _, timestep_path = generate_dataset(
        seeds=[10, 18],
        out_dir=tmp_path,
        modes=["baseline"],
    )
    table = prepare_learning_table(pd.read_csv(timestep_path))

    assert set(FEATURE_COLUMNS).issubset(table.columns)
    assert "needs_hard_recovery" in table.columns
    assert table["needs_hard_recovery"].isin([0, 1]).all()
    assert table["split"].isin(["train", "test"]).all()


def test_train_and_audit_writes_model_evidence(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    out_dir = tmp_path / "model"
    generate_dataset(
        seeds=[10, 11, 12, 16, 17, 18, 19],
        out_dir=dataset_dir,
        modes=["baseline"],
    )

    scored_path, metrics_path, report_path = train_and_audit(dataset_dir, out_dir)
    scored = pd.read_csv(scored_path)
    metrics = pd.read_csv(metrics_path)

    for column in [
        "recovery_prob",
        "recovery_pred",
        "recovery_entropy",
        "recovery_margin",
        "embedding_distance_to_recovery",
        "embedding_distance_to_nominal",
        "embedding_recovery_margin",
    ]:
        assert column in scored.columns
    assert {"train", "val", "test"}.issubset(set(metrics["group"]))
    assert report_path.exists()
