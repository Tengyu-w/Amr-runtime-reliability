"""Collect Gazebo scan-policy data across goal directions.

The standard validation matrix keeps the benchmark scenario set stable. This script is
only for policy-data expansion: it changes Nav2 goals to elicit more expert actions
from the same Gazebo/Nav2 simulator and recorder stack.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

DEMO_ROOT = Path(__file__).resolve().parents[1]
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from experiments.run_gazebo_validation_matrix import (  # noqa: E402
    EpisodeRunResult,
    _parse_csv_list,
    _parse_seed_list,
    _write_manifest,
    run_episode,
)
from src.utils import ensure_output_dir  # noqa: E402


DEFAULT_GOALS = [
    "east_axis:4.5:-3.0",
    "east_south:4.5:3.0",
    "west_near:-5.7:-3.0",
    "north_near:-4.5:-3.8",
    "south_axis:-4.5:3.0",
]


@dataclass
class GoalVariant:
    goal_id: str
    x: float
    y: float


def _parse_goal(raw: str) -> GoalVariant:
    parts = [part.strip() for part in raw.split(":")]
    if len(parts) != 3:
        raise ValueError(f"Goal must use id:x:y format, got {raw!r}")
    return GoalVariant(goal_id=parts[0], x=float(parts[1]), y=float(parts[2]))


def _parse_goals(raw: str) -> list[GoalVariant]:
    return [_parse_goal(item) for item in _parse_csv_list(raw)]


def _action_summary(scan_policy_dir: Path, out_path: Path) -> Path:
    frames = []
    for path in sorted(scan_policy_dir.glob("*.scan_policy.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        if "episode_id" not in df:
            df["episode_id"] = path.stem.replace(".scan_policy", "")
        frames.append(df)
    rows = []
    if frames:
        table = pd.concat(frames, ignore_index=True)
        for (episode_id, scenario_id, action), sub in table.groupby(
            ["episode_id", "scenario_id", "expert_proxy_action"]
        ):
            rows.append(
                {
                    "episode_id": episode_id,
                    "scenario_id": scenario_id,
                    "expert_proxy_action": action,
                    "count": int(len(sub)),
                }
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["episode_id", "scenario_id", "expert_proxy_action", "count"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path


def run_action_diversity_collection(
    goals: list[GoalVariant],
    scenarios: list[str],
    seeds: list[int],
    out_dir: Path,
    timeout_sec: int,
    wsl_distro: str,
    retries: int,
    dry_run: bool,
    scan_policy_require_nav2_plan: bool,
) -> tuple[Path, Path]:
    output_dir = ensure_output_dir(out_dir)
    episode_dir = ensure_output_dir(output_dir / "episode_logs")
    policy_episode_dir = ensure_output_dir(output_dir / "policy_episode_logs")
    scan_policy_episode_dir = ensure_output_dir(output_dir / "scan_policy_episode_logs")
    depth_policy_episode_dir = ensure_output_dir(output_dir / "depth_policy_episode_logs")
    ros_log_dir = ensure_output_dir(output_dir / "ros_logs")
    manifest_rows: list[EpisodeRunResult] = []

    for seed in seeds:
        for goal in goals:
            for scenario_id in scenarios:
                print(f"Running scenario={scenario_id} goal={goal.goal_id} seed={seed}")
                result = None
                for attempt in range(1, max(1, retries) + 2):
                    result = run_episode(
                        scenario_id=scenario_id,
                        seed=seed,
                        episode_dir=episode_dir,
                        policy_episode_dir=policy_episode_dir,
                        scan_policy_episode_dir=scan_policy_episode_dir,
                        depth_policy_episode_dir=depth_policy_episode_dir,
                        ros_log_dir=ros_log_dir,
                        timeout_sec=timeout_sec,
                        wsl_distro=wsl_distro,
                        attempt=attempt,
                        dry_run=dry_run,
                        goal_id=goal.goal_id,
                        goal_x=goal.x,
                        goal_y=goal.y,
                        scan_policy_require_nav2_plan=scan_policy_require_nav2_plan,
                    )
                    if result.status in {"ok", "dry_run"}:
                        break
                    print(f"  attempt={attempt} status={result.status}; retrying")
                assert result is not None
                manifest_rows.append(result)
                print(
                    f"  status={result.status} rows={result.csv_rows} "
                    f"policy_rows={result.policy_csv_rows} scan_policy_rows={result.scan_policy_csv_rows} "
                    f"depth_policy_rows={result.depth_policy_csv_rows}"
                )
                _write_manifest(manifest_rows, output_dir / "run_manifest.csv")

    manifest_path = _write_manifest(manifest_rows, output_dir / "run_manifest.csv")
    action_summary_path = _action_summary(scan_policy_episode_dir, output_dir / "scan_policy_action_counts.csv")
    return manifest_path, action_summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect action-diverse Gazebo scan-policy data.")
    parser.add_argument("--goals", type=str, default=",".join(DEFAULT_GOALS), help="Comma list of id:x:y goals.")
    parser.add_argument("--scenarios", type=str, default="nominal")
    parser.add_argument("--seeds", type=str, default="10")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/gazebo_scan_policy_action_diversity_v1"))
    parser.add_argument("--timeout-sec", type=int, default=90)
    parser.add_argument("--wsl-distro", type=str, default="Ubuntu")
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--scan-policy-allow-proxy-labels",
        action="store_true",
        help="Include fallback proxy expert labels, useful for collecting STAY under blockage.",
    )
    args = parser.parse_args()

    manifest_path, action_summary_path = run_action_diversity_collection(
        goals=_parse_goals(args.goals),
        scenarios=_parse_csv_list(args.scenarios),
        seeds=_parse_seed_list(args.seeds),
        out_dir=args.out_dir,
        timeout_sec=args.timeout_sec,
        wsl_distro=args.wsl_distro,
        retries=args.retries,
        dry_run=args.dry_run,
        scan_policy_require_nav2_plan=not args.scan_policy_allow_proxy_labels,
    )
    print(f"Run manifest: {manifest_path}")
    print(f"Action counts: {action_summary_path}")


if __name__ == "__main__":
    main()
