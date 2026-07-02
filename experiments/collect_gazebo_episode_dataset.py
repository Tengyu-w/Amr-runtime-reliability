"""Collect ROS/Gazebo routed episode CSVs into a trainable dataset."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

DEMO_ROOT = Path(__file__).resolve().parents[1]
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from experiments.generate_scenario_dataset import SCENARIOS
from src.utils import ensure_output_dir


EXPECTED_RECOVERY = {
    "nominal": "NORMAL_NAVIGATION",
    "external_path_blockage": "REPLAN",
    "localization_drift": "RELOCALIZE",
    "perception_degradation": "HUMAN_REVIEW",
    "task_goal_shift_ood_style": "HUMAN_REVIEW",
    "execution_deviation": "REPLAN",
    "progress_blockage": "REPLAN",
    "planner_backend_failure": "SAFE_STOP",
    "compound_shift_and_degradation": "HUMAN_REVIEW",
    "mixed_blockage_and_perception": "HUMAN_REVIEW",
    "mixed_drift_and_execution": "RELOCALIZE",
    "boundary_weak_blockage": "CAUTIOUS_MODE",
}

SCENARIO_BY_ID = {scenario.scenario_id: scenario for scenario in SCENARIOS}


def _split_for_seed(seed: int) -> str:
    bucket = seed % 10
    if bucket <= 5:
        return "train"
    if bucket <= 7:
        return "val"
    return "test"


def _seed_from_episode_id(episode_id: str) -> int | None:
    match = re.search(r"(?:^|__)seed_([0-9]+)(?:__|$)", episode_id)
    if not match:
        match = re.search(r"(?:^|_)seed_([0-9]+)(?:_|$)", episode_id)
    return int(match.group(1)) if match else None


def _split_for_episode(episode_id: str) -> str:
    seed = _seed_from_episode_id(episode_id)
    if seed is not None:
        return _split_for_seed(seed)
    digest = hashlib.sha1(episode_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10
    if bucket <= 5:
        return "train"
    if bucket <= 7:
        return "val"
    return "test"


def _catalog_rows() -> list[dict[str, object]]:
    rows = []
    for scenario in SCENARIOS:
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "primary_fault_origin": scenario.primary_fault_origin,
                "primary_fault_family": scenario.primary_fault_family,
                "primary_ood_status": scenario.primary_ood_status,
                "split_group": scenario.split_group,
                "enabled_events": "|".join(scenario.enabled_events) if scenario.enabled_events else "none",
                "expected_recovery": EXPECTED_RECOVERY.get(scenario.scenario_id, "HUMAN_REVIEW"),
            }
        )
    return rows


def _read_episode_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    if "episode_id" not in df or df["episode_id"].isna().all():
        df["episode_id"] = path.stem
    df["episode_log_path"] = str(path)
    return df


def _episode_scenario_id(df: pd.DataFrame) -> str:
    if "scenario_id" not in df:
        return "unknown"
    values = df["scenario_id"].dropna().astype(str)
    values = values[~values.isin({"", "unknown"})]
    if values.empty:
        return "unknown"
    return str(values.mode().iloc[0])


def _row_target_actions(df: pd.DataFrame, expected_recovery: str) -> list[str]:
    if "failure_mechanism" not in df:
        return [expected_recovery for _ in range(len(df))]
    mechanisms = df["failure_mechanism"].fillna("nominal").astype(str)
    return [
        "NORMAL_NAVIGATION" if mechanism in {"", "nominal"} else expected_recovery
        for mechanism in mechanisms
    ]


def _numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _episode_outcome(df: pd.DataFrame, expected_recovery: str) -> dict[str, object]:
    mechanisms = (
        df["failure_mechanism"].fillna("nominal").astype(str)
        if "failure_mechanism" in df
        else pd.Series(["nominal"] * len(df), index=df.index)
    )
    decisions = (
        df["router_decision"].fillna("").astype(str)
        if "router_decision" in df
        else pd.Series([""] * len(df), index=df.index)
    )
    risk = _numeric_series(df, "risk_score")
    obstacle = _numeric_series(df, "obstacle_proximity")
    deviation = _numeric_series(df, "trajectory_deviation")
    robot_x = _numeric_series(df, "robot_x")
    robot_y = _numeric_series(df, "robot_y")
    target_x = _numeric_series(df, "target_x")
    target_y = _numeric_series(df, "target_y")

    non_nominal_mask = ~mechanisms.isin(["", "nominal"])
    expected_mask = decisions.eq(expected_recovery)
    first_fault_step = int(non_nominal_mask.idxmax()) if bool(non_nominal_mask.any()) else None
    if first_fault_step is None:
        recovery_latency = ""
    else:
        after_fault = expected_mask.loc[first_fault_step:]
        recovery_latency = int(after_fault.idxmax() - first_fault_step) if bool(after_fault.any()) else ""

    final_distance = float(
        ((robot_x.iloc[-1] - target_x.iloc[-1]) ** 2 + (robot_y.iloc[-1] - target_y.iloc[-1]) ** 2) ** 0.5
    )
    goal_reached_proxy = final_distance <= 0.75
    expected_route_observed = bool(expected_mask.any())
    safe_stop_observed = bool(decisions.eq("SAFE_STOP").any())
    collision_risk_proxy = bool(((obstacle >= 0.85) & (deviation >= 0.75)).any())
    final_mechanism = str(mechanisms.iloc[-1]) if len(mechanisms) else "nominal"
    final_risk = float(risk.iloc[-1]) if len(risk) else 0.0

    if not bool(non_nominal_mask.any()) and final_risk < 0.35:
        outcome_label = "nominal_stable"
    elif safe_stop_observed:
        outcome_label = "safe_stop_observed"
    elif not expected_route_observed:
        outcome_label = "missed_expected_route"
    elif final_mechanism in {"", "nominal"} or final_risk < 0.35:
        outcome_label = "routed_and_recovered_proxy"
    else:
        outcome_label = "routed_but_unresolved_proxy"

    return {
        "first_fault_row": first_fault_step if first_fault_step is not None else "",
        "recovery_latency_steps": recovery_latency,
        "expected_route_observed": expected_route_observed,
        "safe_stop_observed": safe_stop_observed,
        "goal_reached_proxy": goal_reached_proxy,
        "collision_risk_proxy": collision_risk_proxy,
        "final_goal_distance": round(final_distance, 4),
        "final_risk": round(final_risk, 4),
        "final_failure_mechanism": final_mechanism,
        "outcome_label": outcome_label,
    }


def _outcome_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return episodes
    rows = []
    for (split, scenario_id, outcome), sub in episodes.groupby(["split", "scenario_id", "outcome_label"]):
        rows.append(
            {
                "split": split,
                "scenario_id": scenario_id,
                "outcome_label": outcome,
                "n_episodes": int(len(sub)),
                "expected_route_observed_rate": float(sub["expected_route_observed"].astype(bool).mean()),
                "goal_reached_proxy_rate": float(sub["goal_reached_proxy"].astype(bool).mean()),
                "collision_risk_proxy_rate": float(sub["collision_risk_proxy"].astype(bool).mean()),
                "mean_recovery_latency_steps": float(
                    pd.to_numeric(sub["recovery_latency_steps"], errors="coerce").mean()
                ),
                "mean_final_goal_distance": float(
                    pd.to_numeric(sub["final_goal_distance"], errors="coerce").mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def collect_gazebo_dataset(log_dir: str | Path, out_dir: str | Path) -> tuple[Path, Path, Path, Path]:
    """Aggregate per-episode routed CSV logs into catalog, episodes, and timesteps files."""

    log_root = Path(log_dir)
    output_dir = ensure_output_dir(out_dir)
    csv_paths = sorted(path for path in log_root.glob("*.csv") if path.is_file())
    if not csv_paths:
        raise FileNotFoundError(f"No episode CSV logs found in {log_root}")

    timestep_frames = []
    episode_rows = []
    for path in csv_paths:
        df = _read_episode_csv(path)
        if df.empty:
            continue
        episode_id = str(df["episode_id"].dropna().iloc[0])
        scenario_id = _episode_scenario_id(df)
        seed = _seed_from_episode_id(episode_id)
        scenario = SCENARIO_BY_ID.get(scenario_id)
        split = _split_for_episode(episode_id)
        expected = EXPECTED_RECOVERY.get(scenario_id, "HUMAN_REVIEW")
        origin = scenario.primary_fault_origin if scenario else "unknown"
        family = scenario.primary_fault_family if scenario else "unknown"
        ood_status = scenario.primary_ood_status if scenario else "unknown"
        split_group = scenario.split_group if scenario else "unknown"

        df["observed_scenario_id"] = df["scenario_id"] if "scenario_id" in df else "unknown"
        df["scenario_id"] = scenario_id
        df["split"] = split
        df["scenario_primary_fault_origin"] = origin
        df["scenario_primary_fault_family"] = family
        df["scenario_primary_ood_status"] = ood_status
        df["scenario_split_group"] = split_group
        df["expected_recovery"] = expected
        df["target_action"] = _row_target_actions(df, expected)
        df["seed"] = seed if seed is not None else ""
        timestep_frames.append(df)

        decisions = df["router_decision"].astype(str) if "router_decision" in df else pd.Series(dtype=str)
        mechanisms = df["failure_mechanism"].astype(str) if "failure_mechanism" in df else pd.Series(dtype=str)
        outcome = _episode_outcome(df, expected)
        episode_rows.append(
            {
                "episode_id": episode_id,
                "scenario_id": scenario_id,
                "seed": seed if seed is not None else "",
                "split": split,
                "n_steps": int(len(df)),
                "primary_fault_origin": origin,
                "primary_fault_family": family,
                "primary_ood_status": ood_status,
                "split_group": split_group,
                "expected_recovery": expected,
                "final_router_decision": decisions.iloc[-1] if not decisions.empty else "",
                "modal_router_decision": decisions.mode().iloc[0] if not decisions.empty else "",
                "max_risk": float(pd.to_numeric(df.get("risk_score", 0.0), errors="coerce").fillna(0.0).max()),
                "mean_risk": float(pd.to_numeric(df.get("risk_score", 0.0), errors="coerce").fillna(0.0).mean()),
                "n_expected_action_steps": int((decisions == expected).sum()) if not decisions.empty else 0,
                "n_non_nominal_mechanism_steps": int((mechanisms != "nominal").sum()) if not mechanisms.empty else 0,
                "episode_log_path": str(path),
                **outcome,
            }
        )

    if not timestep_frames:
        raise ValueError(f"Episode CSV logs in {log_root} were empty.")

    timesteps = pd.concat(timestep_frames, ignore_index=True)
    episodes = pd.DataFrame(episode_rows)
    catalog = pd.DataFrame(_catalog_rows())
    outcome = _outcome_summary(episodes)

    catalog_path = output_dir / "scenario_catalog.csv"
    episode_path = output_dir / "episodes.csv"
    timestep_path = output_dir / "timesteps.csv"
    outcome_path = output_dir / "episode_outcome_summary.csv"
    catalog.to_csv(catalog_path, index=False)
    episodes.to_csv(episode_path, index=False)
    timesteps.to_csv(timestep_path, index=False)
    outcome.to_csv(outcome_path, index=False)
    return catalog_path, episode_path, timestep_path, outcome_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect ROS/Gazebo episode logs into a trainable dataset.")
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/ros2_episode_logs"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/gazebo_episode_dataset"))
    args = parser.parse_args()

    catalog_path, episode_path, timestep_path, outcome_path = collect_gazebo_dataset(args.log_dir, args.out_dir)
    print(f"Scenario catalog: {catalog_path}")
    print(f"Episodes: {episode_path}")
    print(f"Timesteps: {timestep_path}")
    print(f"Episode outcome summary: {outcome_path}")


if __name__ == "__main__":
    main()
