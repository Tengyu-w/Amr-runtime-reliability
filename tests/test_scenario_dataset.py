import pandas as pd

from experiments.generate_scenario_dataset import (
    SCENARIOS,
    _scenario_catalog_rows,
    _split_for_seed,
    generate_dataset,
)


def test_scenario_catalog_has_explicit_fault_sources() -> None:
    catalog = {row["scenario_id"]: row for row in _scenario_catalog_rows()}

    assert len(catalog) == len(SCENARIOS)
    assert catalog["external_path_blockage"]["primary_fault_origin"] == "external_disturbance"
    assert catalog["task_goal_shift_ood_style"]["primary_ood_status"] == "ood_style_shift"
    assert catalog["nominal"]["enabled_events"] == "none"


def test_seed_split_is_episode_level() -> None:
    assert _split_for_seed(10) == "train"
    assert _split_for_seed(17) == "val"
    assert _split_for_seed(18) == "test"


def test_generate_dataset_writes_labelled_episode_tables(tmp_path) -> None:
    catalog_path, split_path, episode_path, timestep_path = generate_dataset(
        seeds=[10, 18],
        out_dir=tmp_path,
        modes=["baseline", "mechanism_router"],
    )

    catalog = pd.read_csv(catalog_path)
    splits = pd.read_csv(split_path)
    episodes = pd.read_csv(episode_path)
    timesteps = pd.read_csv(timestep_path)

    assert {"scenario_id", "primary_fault_origin", "primary_ood_status"}.issubset(catalog.columns)
    assert set(splits["split"]) == {"train", "test"}
    assert len(episodes) == len(SCENARIOS) * 2 * 2
    assert {"episode_id", "scenario_primary_fault_origin", "router_decision"}.issubset(timesteps.columns)
    assert "ood_style_shift" in set(episodes["primary_ood_status"])
