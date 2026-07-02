"""Analyze Gazebo observer-mode navigation-policy monitor logs."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

DEMO_ROOT = Path(__file__).resolve().parents[1]
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from experiments.generate_scenario_dataset import SCENARIOS
from src.utils import ensure_output_dir


SCENARIO_BY_ID = {scenario.scenario_id: scenario for scenario in SCENARIOS}


def _seed_from_episode_id(episode_id: str) -> int | None:
    match = re.search(r"(?:^|__)seed_([0-9]+)(?:__|$)", episode_id)
    return int(match.group(1)) if match else None


def _split_for_seed(seed: int | None) -> str:
    if seed is None:
        return "unknown"
    bucket = seed % 10
    if bucket <= 5:
        return "train"
    if bucket <= 7:
        return "val"
    return "test"


def _episode_scenario_id(df: pd.DataFrame) -> str:
    if "scenario_id" not in df:
        return "unknown"
    values = df["scenario_id"].dropna().astype(str)
    values = values[~values.isin({"", "unknown"})]
    if values.empty:
        return "unknown"
    return str(values.mode().iloc[0])


def _bool_series(series: pd.Series) -> pd.Series:
    return series.fillna(False).map(lambda value: str(value).strip().lower() in {"1", "true", "yes"})


def _numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _load_policy_logs(log_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(log_dir.glob("*.policy.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        if "episode_id" not in df or df["episode_id"].isna().all():
            df["episode_id"] = path.stem.replace(".policy", "")
        episode_id = str(df["episode_id"].dropna().iloc[0])
        seed = _seed_from_episode_id(episode_id)
        scenario_id = _episode_scenario_id(df)
        scenario = SCENARIO_BY_ID.get(scenario_id)
        df["episode_id"] = episode_id
        df["scenario_id"] = scenario_id
        df["seed"] = seed if seed is not None else ""
        df["split"] = _split_for_seed(seed)
        df["episode_log_path"] = str(path)
        df["scenario_primary_fault_origin"] = scenario.primary_fault_origin if scenario else "unknown"
        df["scenario_primary_fault_family"] = scenario.primary_fault_family if scenario else "unknown"
        df["scenario_primary_ood_status"] = scenario.primary_ood_status if scenario else "unknown"
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No non-empty *.policy.csv logs found in {log_dir}")
    return pd.concat(frames, ignore_index=True)


def _episode_rows(policy_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for episode_id, sub in policy_rows.groupby("episode_id"):
        evaluable = (
            _bool_series(sub["policy_evaluable"])
            if "policy_evaluable" in sub
            else pd.Series([True] * len(sub), index=sub.index)
        )
        sub_eval = sub[evaluable].copy()
        correct = _bool_series(sub_eval["policy_correct"]) if "policy_correct" in sub_eval else pd.Series(dtype=bool)
        errors = ~correct
        max_prob = _numeric(sub_eval, "policy_max_prob")
        mechanisms = sub_eval["policy_error_mechanism"].fillna("unknown").astype(str)
        routes = sub_eval["policy_recovery_route"].fillna("unknown").astype(str)
        rows.append(
            {
                "episode_id": episode_id,
                "scenario_id": str(sub["scenario_id"].iloc[0]),
                "seed": sub["seed"].iloc[0],
                "split": str(sub["split"].iloc[0]),
                "n_rows": int(len(sub)),
                "n_evaluable_rows": int(len(sub_eval)),
                "n_policy_errors": int(errors.sum()),
                "policy_error_rate": float(errors.mean()) if len(errors) else 0.0,
                "n_high_conf_policy_errors": int((errors & (max_prob >= 0.90)).sum()),
                "high_conf_error_rate_among_errors": float(
                    (errors & (max_prob >= 0.90)).sum() / max(int(errors.sum()), 1)
                ),
                "dominant_error_mechanism": mechanisms[errors].mode().iloc[0] if errors.any() else "none",
                "dominant_recovery_route": routes[errors].mode().iloc[0] if errors.any() else "NORMAL_NAVIGATION",
                "mean_policy_entropy": float(_numeric(sub_eval, "policy_entropy").mean()) if len(sub_eval) else 0.0,
                "mean_policy_margin": float(_numeric(sub_eval, "policy_margin").mean()) if len(sub_eval) else 0.0,
                "mean_policy_max_prob": float(max_prob.mean()),
                "mean_risk": float(_numeric(sub_eval, "risk_score").mean()) if len(sub_eval) else 0.0,
                "final_policy_route": str(routes.iloc[-1]) if len(routes) else "",
                "episode_log_path": str(sub["episode_log_path"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _scenario_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, scenario_id), sub in episodes.groupby(["split", "scenario_id"]):
        rows.append(
            {
                "split": split,
                "scenario_id": scenario_id,
                "n_episodes": int(len(sub)),
                "n_rows": int(sub["n_rows"].sum()),
                "n_evaluable_rows": int(sub["n_evaluable_rows"].sum()),
                "n_policy_errors": int(sub["n_policy_errors"].sum()),
                "mean_policy_error_rate": float(sub["policy_error_rate"].mean()),
                "n_high_conf_policy_errors": int(sub["n_high_conf_policy_errors"].sum()),
                "mean_high_conf_error_rate_among_errors": float(sub["high_conf_error_rate_among_errors"].mean()),
                "dominant_error_mechanism": sub["dominant_error_mechanism"].mode().iloc[0],
                "dominant_recovery_route": sub["dominant_recovery_route"].mode().iloc[0],
                "mean_policy_entropy": float(sub["mean_policy_entropy"].mean()),
                "mean_policy_max_prob": float(sub["mean_policy_max_prob"].mean()),
                "mean_risk": float(sub["mean_risk"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _mechanism_summary(policy_rows: pd.DataFrame) -> pd.DataFrame:
    evaluable = (
        _bool_series(policy_rows["policy_evaluable"])
        if "policy_evaluable" in policy_rows
        else pd.Series([True] * len(policy_rows), index=policy_rows.index)
    )
    correct = _bool_series(policy_rows["policy_correct"])
    errors = policy_rows[evaluable & ~correct].copy()
    if errors.empty:
        return pd.DataFrame()
    rows = []
    for (split, scenario_id, mechanism, route), sub in errors.groupby(
        ["split", "scenario_id", "policy_error_mechanism", "policy_recovery_route"]
    ):
        max_prob = _numeric(sub, "policy_max_prob")
        rows.append(
            {
                "split": split,
                "scenario_id": scenario_id,
                "policy_error_mechanism": mechanism,
                "policy_recovery_route": route,
                "n_errors": int(len(sub)),
                "high_conf_error_rate": float((max_prob >= 0.90).mean()),
                "mean_policy_entropy": float(_numeric(sub, "policy_entropy").mean()),
                "mean_policy_margin": float(_numeric(sub, "policy_margin").mean()),
                "mean_policy_max_prob": float(max_prob.mean()),
                "mean_risk": float(_numeric(sub, "risk_score").mean()),
            }
        )
    return pd.DataFrame(rows)


def _route_summary(policy_rows: pd.DataFrame) -> pd.DataFrame:
    evaluable = (
        _bool_series(policy_rows["policy_evaluable"])
        if "policy_evaluable" in policy_rows
        else pd.Series([True] * len(policy_rows), index=policy_rows.index)
    )
    correct = _bool_series(policy_rows["policy_correct"])
    errors = policy_rows[evaluable & ~correct].copy()
    if errors.empty:
        return pd.DataFrame()
    rows = []
    for (split, route), sub in errors.groupby(["split", "policy_recovery_route"]):
        rows.append(
            {
                "split": split,
                "policy_recovery_route": route,
                "n_policy_errors": int(len(sub)),
                "dominant_error_mechanism": sub["policy_error_mechanism"].mode().iloc[0],
                "scenarios": "|".join(sorted(sub["scenario_id"].astype(str).unique())),
                "high_conf_error_rate": float((_numeric(sub, "policy_max_prob") >= 0.90).mean()),
                "mean_policy_entropy": float(_numeric(sub, "policy_entropy").mean()),
                "mean_policy_max_prob": float(_numeric(sub, "policy_max_prob").mean()),
                "mean_risk": float(_numeric(sub, "risk_score").mean()),
            }
        )
    return pd.DataFrame(rows)


def analyze_gazebo_policy_monitor(log_dir: str | Path, out_dir: str | Path) -> dict[str, Path]:
    output_dir = ensure_output_dir(out_dir)
    policy_rows = _load_policy_logs(Path(log_dir))
    episodes = _episode_rows(policy_rows)
    scenario_summary = _scenario_summary(episodes)
    mechanism_summary = _mechanism_summary(policy_rows)
    route_summary = _route_summary(policy_rows)
    high_conf_errors = policy_rows[
        (
            _bool_series(policy_rows["policy_evaluable"])
            if "policy_evaluable" in policy_rows
            else pd.Series([True] * len(policy_rows), index=policy_rows.index)
        )
        & (~_bool_series(policy_rows["policy_correct"]))
        & (_numeric(policy_rows, "policy_max_prob") >= 0.90)
    ].copy()

    paths = {
        "policy_timesteps": output_dir / "policy_timesteps.csv",
        "policy_episodes": output_dir / "policy_episodes.csv",
        "policy_scenario_summary": output_dir / "policy_scenario_summary.csv",
        "policy_error_mechanism_summary": output_dir / "policy_error_mechanism_summary.csv",
        "policy_recovery_route_summary": output_dir / "policy_recovery_route_summary.csv",
        "high_conf_policy_errors": output_dir / "high_conf_policy_errors.csv",
    }
    policy_rows.to_csv(paths["policy_timesteps"], index=False)
    episodes.to_csv(paths["policy_episodes"], index=False)
    scenario_summary.to_csv(paths["policy_scenario_summary"], index=False)
    mechanism_summary.to_csv(paths["policy_error_mechanism_summary"], index=False)
    route_summary.to_csv(paths["policy_recovery_route_summary"], index=False)
    high_conf_errors.to_csv(paths["high_conf_policy_errors"], index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Gazebo navigation-policy monitor logs.")
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/gazebo_validation_matrix/policy_episode_logs"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/gazebo_policy_monitor_evidence"))
    args = parser.parse_args()
    paths = analyze_gazebo_policy_monitor(args.log_dir, args.out_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
