"""Run a multi-seed ROS/Gazebo/Nav2 reliability validation matrix."""

from __future__ import annotations

import argparse
import csv
import os
import posixpath
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parents[1]
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from experiments.collect_gazebo_episode_dataset import collect_gazebo_dataset
from experiments.run_gazebo_route_ablation import run_gazebo_route_ablation
from experiments.train_gazebo_route_model import train_and_audit_gazebo_route_model
from src.utils import ensure_output_dir


DEFAULT_SCENARIOS = [
    "nominal",
    "external_path_blockage",
    "localization_drift",
    "perception_degradation",
    "execution_deviation",
    "progress_blockage",
    "mixed_blockage_and_perception",
    "mixed_drift_and_execution",
    "boundary_weak_blockage",
]


@dataclass
class EpisodeRunResult:
    episode_id: str
    scenario_id: str
    goal_id: str
    goal_x: float
    goal_y: float
    seed: int
    attempt: int
    csv_path: str
    policy_csv_path: str
    scan_policy_csv_path: str
    depth_policy_csv_path: str
    log_path: str
    return_code: int
    duration_sec: float
    csv_rows: int
    policy_csv_rows: int
    scan_policy_csv_rows: int
    depth_policy_csv_rows: int
    entity_created_count: int
    begin_navigating_count: int
    rejected_goal_count: int
    managed_active_count: int
    final_router_decision: str
    final_failure_mechanism: str
    status: str


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_seed_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    if os.name != "nt":
        return str(resolved)
    drive = resolved.drive.rstrip(":").lower()
    rest = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{rest}"


def _count_in_file(path: Path, needle: str) -> int:
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text.count(needle)


def _csv_summary(path: Path) -> tuple[int, str, str]:
    if not path.exists():
        return 0, "", ""
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0, "", ""
    final = rows[-1]
    return len(rows), str(final.get("router_decision", "")), str(final.get("failure_mechanism", ""))


def _write_manifest(rows: list[EpisodeRunResult], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else list(EpisodeRunResult.__annotations__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return path


def _episode_command(
    scenario_id: str,
    seed: int,
    episode_id: str,
    csv_path: Path,
    policy_csv_path: Path,
    scan_policy_csv_path: Path,
    depth_policy_csv_path: Path,
    log_path: Path,
    timeout_sec: int,
    policy_model_path: Path | None = None,
    goal_x: float = 4.5,
    goal_y: float = 3.0,
    alternate_goal_x: float = -4.5,
    alternate_goal_y: float = 3.0,
    goal_shift_step: int = 6,
    scan_policy_require_nav2_plan: bool = True,
) -> str:
    ws_root = _to_wsl_path(DEMO_ROOT / "ros2_ws")
    csv_wsl = _to_wsl_path(csv_path)
    policy_csv_wsl = _to_wsl_path(policy_csv_path)
    scan_policy_csv_wsl = _to_wsl_path(scan_policy_csv_path)
    depth_policy_csv_wsl = _to_wsl_path(depth_policy_csv_path)
    log_wsl = _to_wsl_path(log_path)
    policy_model_wsl = _to_wsl_path(policy_model_path) if policy_model_path else ""
    launch_args = [
        f"scenario_id:={shlex.quote(scenario_id)}",
        f"episode_id:={shlex.quote(episode_id)}",
        f"fault_seed:={int(seed)}",
        f"output_path:={shlex.quote(csv_wsl)}",
        f"policy_output_path:={shlex.quote(policy_csv_wsl)}",
        f"scan_policy_output_path:={shlex.quote(scan_policy_csv_wsl)}",
        f"depth_policy_output_path:={shlex.quote(depth_policy_csv_wsl)}",
        f"goal_x:={float(goal_x)}",
        f"goal_y:={float(goal_y)}",
        f"alternate_goal_x:={float(alternate_goal_x)}",
        f"alternate_goal_y:={float(alternate_goal_y)}",
        f"goal_shift_step:={int(goal_shift_step)}",
        f"scan_policy_require_nav2_plan:={str(bool(scan_policy_require_nav2_plan)).lower()}",
    ]
    if policy_model_wsl:
        launch_args.append(f"policy_model_path:={shlex.quote(policy_model_wsl)}")
    return " && ".join(
        [
            "source /opt/ros/jazzy/setup.bash",
            f"source {shlex.quote(ws_root + '/install/setup.bash')}",
            (
                f"mkdir -p {shlex.quote(posixpath.dirname(csv_wsl))} "
                f"{shlex.quote(posixpath.dirname(policy_csv_wsl))} "
                f"{shlex.quote(posixpath.dirname(scan_policy_csv_wsl))} "
                f"{shlex.quote(posixpath.dirname(depth_policy_csv_wsl))} "
                f"{shlex.quote(posixpath.dirname(log_wsl))}"
            ),
            (
                f"rm -f {shlex.quote(csv_wsl)} {shlex.quote(policy_csv_wsl)} "
                f"{shlex.quote(scan_policy_csv_wsl)} {shlex.quote(depth_policy_csv_wsl)} "
                f"{shlex.quote(log_wsl)}"
            ),
            (
                f"timeout {int(timeout_sec)}s ros2 launch amr_reliability_benchmark "
                f"gazebo_nav2_benchmark.launch.py "
                f"{' '.join(launch_args)} "
                f"> {shlex.quote(log_wsl)} 2>&1 || true"
            ),
        ]
    )


def _cleanup_wsl_sim(wsl_distro: str) -> None:
    cleanup = "; ".join(
        [
            "pkill -f '[r]os2 launch amr_reliability_benchmark gazebo_nav2_benchmark' 2>/dev/null || true",
            "pkill -f '[g]z sim' 2>/dev/null || true",
            "pkill -f '[p]arameter_bridge' 2>/dev/null || true",
            "pkill -f '[c]omponent_container_isolated' 2>/dev/null || true",
        ]
    )
    subprocess.run(
        ["wsl", "-d", wsl_distro, "--", "bash", "-lc", cleanup],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run_episode(
    scenario_id: str,
    seed: int,
    episode_dir: Path,
    policy_episode_dir: Path,
    scan_policy_episode_dir: Path,
    depth_policy_episode_dir: Path,
    ros_log_dir: Path,
    timeout_sec: int,
    wsl_distro: str,
    attempt: int = 1,
    dry_run: bool = False,
    policy_model_path: Path | None = None,
    goal_id: str = "default",
    goal_x: float = 4.5,
    goal_y: float = 3.0,
    alternate_goal_x: float = -4.5,
    alternate_goal_y: float = 3.0,
    goal_shift_step: int = 6,
    scan_policy_require_nav2_plan: bool = True,
) -> EpisodeRunResult:
    episode_id = f"gazebo_{scenario_id}__goal_{goal_id}__seed_{seed}"
    csv_path = episode_dir / f"{episode_id}.csv"
    policy_csv_path = policy_episode_dir / f"{episode_id}.policy.csv"
    scan_policy_csv_path = scan_policy_episode_dir / f"{episode_id}.scan_policy.csv"
    depth_policy_csv_path = depth_policy_episode_dir / f"{episode_id}.depth_policy.csv"
    log_path = ros_log_dir / f"{episode_id}.attempt_{attempt}.log"
    command = _episode_command(
        scenario_id,
        seed,
        episode_id,
        csv_path,
        policy_csv_path,
        scan_policy_csv_path,
        depth_policy_csv_path,
        log_path,
        timeout_sec,
        policy_model_path=policy_model_path,
        goal_x=goal_x,
        goal_y=goal_y,
        alternate_goal_x=alternate_goal_x,
        alternate_goal_y=alternate_goal_y,
        goal_shift_step=goal_shift_step,
        scan_policy_require_nav2_plan=scan_policy_require_nav2_plan,
    )
    start = time.time()
    return_code = 0
    if dry_run:
        print(f"[dry-run] wsl -d {wsl_distro} -- bash -lc {command}")
    else:
        _cleanup_wsl_sim(wsl_distro)
        proc = subprocess.run(
            ["wsl", "-d", wsl_distro, "--", "bash", "-lc", command],
            check=False,
            text=True,
        )
        return_code = int(proc.returncode)
        _cleanup_wsl_sim(wsl_distro)
    duration = time.time() - start
    csv_rows, final_decision, final_mechanism = _csv_summary(csv_path)
    policy_csv_rows, _, _ = _csv_summary(policy_csv_path)
    scan_policy_csv_rows, _, _ = _csv_summary(scan_policy_csv_path)
    depth_policy_csv_rows, _, _ = _csv_summary(depth_policy_csv_path)
    entity_created = _count_in_file(log_path, "Entity creation successful")
    begin_count = _count_in_file(log_path, "Begin navigating")
    reject_count = _count_in_file(log_path, "Rejecting the goal")
    status = "dry_run"
    if not dry_run:
        if csv_rows <= 0:
            status = "missing_csv"
        elif entity_created <= 0:
            status = "spawn_failed"
        elif begin_count <= 0:
            status = "navigation_not_started"
        elif reject_count > 0:
            status = "goal_rejected"
        else:
            status = "ok"
    return EpisodeRunResult(
        episode_id=episode_id,
        scenario_id=scenario_id,
        goal_id=goal_id,
        goal_x=float(goal_x),
        goal_y=float(goal_y),
        seed=seed,
        attempt=attempt,
        csv_path=str(csv_path),
        policy_csv_path=str(policy_csv_path),
        scan_policy_csv_path=str(scan_policy_csv_path),
        depth_policy_csv_path=str(depth_policy_csv_path),
        log_path=str(log_path),
        return_code=return_code,
        duration_sec=round(duration, 2),
        csv_rows=csv_rows,
        policy_csv_rows=policy_csv_rows,
        scan_policy_csv_rows=scan_policy_csv_rows,
        depth_policy_csv_rows=depth_policy_csv_rows,
        entity_created_count=entity_created,
        begin_navigating_count=begin_count,
        rejected_goal_count=reject_count,
        managed_active_count=_count_in_file(log_path, "Managed nodes are active"),
        final_router_decision=final_decision,
        final_failure_mechanism=final_mechanism,
        status=status,
    )


def run_validation_matrix(
    scenarios: list[str],
    seeds: list[int],
    out_dir: str | Path,
    timeout_sec: int = 45,
    wsl_distro: str = "Ubuntu",
    dry_run: bool = False,
    skip_analysis: bool = False,
    retries: int = 1,
    policy_model_path: str | Path | None = None,
    goal_id: str = "default",
    goal_x: float = 4.5,
    goal_y: float = 3.0,
    alternate_goal_x: float = -4.5,
    alternate_goal_y: float = 3.0,
    goal_shift_step: int = 6,
    scan_policy_require_nav2_plan: bool = True,
) -> tuple[Path, Path | None, Path | None, Path | None]:
    output_dir = ensure_output_dir(out_dir)
    episode_dir = ensure_output_dir(output_dir / "episode_logs")
    policy_episode_dir = ensure_output_dir(output_dir / "policy_episode_logs")
    scan_policy_episode_dir = ensure_output_dir(output_dir / "scan_policy_episode_logs")
    depth_policy_episode_dir = ensure_output_dir(output_dir / "depth_policy_episode_logs")
    ros_log_dir = ensure_output_dir(output_dir / "ros_logs")
    manifest_rows: list[EpisodeRunResult] = []
    for seed in seeds:
        for scenario_id in scenarios:
            print(f"Running scenario={scenario_id} seed={seed}")
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
                    policy_model_path=Path(policy_model_path) if policy_model_path else None,
                    goal_id=goal_id,
                    goal_x=goal_x,
                    goal_y=goal_y,
                    alternate_goal_x=alternate_goal_x,
                    alternate_goal_y=alternate_goal_y,
                    goal_shift_step=goal_shift_step,
                    scan_policy_require_nav2_plan=scan_policy_require_nav2_plan,
                )
                if result.status in {"ok", "dry_run"}:
                    break
                print(f"  attempt={attempt} status={result.status}; retrying")
            assert result is not None
            manifest_rows.append(result)
            print(
                f"  attempt={result.attempt} status={result.status} rows={result.csv_rows} "
                f"policy_rows={result.policy_csv_rows} "
                f"scan_policy_rows={result.scan_policy_csv_rows} "
                f"depth_policy_rows={result.depth_policy_csv_rows} "
                f"spawn={result.entity_created_count} begin={result.begin_navigating_count} "
                f"reject={result.rejected_goal_count} "
                f"final={result.final_router_decision}/{result.final_failure_mechanism}"
            )
            _write_manifest(manifest_rows, output_dir / "run_manifest.csv")

    manifest_path = _write_manifest(manifest_rows, output_dir / "run_manifest.csv")
    if dry_run or skip_analysis:
        return manifest_path, None, None, None

    dataset_dir = output_dir / "dataset"
    route_model_dir = output_dir / "route_model"
    ablation_dir = output_dir / "route_ablation"
    collect_gazebo_dataset(episode_dir, dataset_dir)
    train_and_audit_gazebo_route_model(dataset_dir, route_model_dir, epochs=260)
    run_gazebo_route_ablation(dataset_dir, ablation_dir, epochs=180)
    return manifest_path, dataset_dir, route_model_dir, ablation_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-seed Gazebo/Nav2 reliability validation matrix.")
    parser.add_argument("--scenarios", type=str, default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--seeds", type=str, default="10,11,12,16,17,18,19")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/gazebo_validation_matrix"))
    parser.add_argument("--timeout-sec", type=int, default=45)
    parser.add_argument("--wsl-distro", type=str, default="Ubuntu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument(
        "--policy-model-path",
        type=Path,
        default=None,
        help="Optional JSON navigation-policy model path passed to the Gazebo policy monitor.",
    )
    parser.add_argument("--goal-id", type=str, default="default")
    parser.add_argument("--goal-x", type=float, default=4.5)
    parser.add_argument("--goal-y", type=float, default=3.0)
    parser.add_argument("--alternate-goal-x", type=float, default=-4.5)
    parser.add_argument("--alternate-goal-y", type=float, default=3.0)
    parser.add_argument("--goal-shift-step", type=int, default=6)
    parser.add_argument(
        "--scan-policy-allow-proxy-labels",
        action="store_true",
        help="Include scan-policy rows whose expert label comes from the proxy fallback instead of Nav2 /plan.",
    )
    args = parser.parse_args()

    manifest_path, dataset_dir, route_model_dir, ablation_dir = run_validation_matrix(
        scenarios=_parse_csv_list(args.scenarios),
        seeds=_parse_seed_list(args.seeds),
        out_dir=args.out_dir,
        timeout_sec=args.timeout_sec,
        wsl_distro=args.wsl_distro,
        dry_run=args.dry_run,
        skip_analysis=args.skip_analysis,
        retries=args.retries,
        policy_model_path=args.policy_model_path,
        goal_id=args.goal_id,
        goal_x=args.goal_x,
        goal_y=args.goal_y,
        alternate_goal_x=args.alternate_goal_x,
        alternate_goal_y=args.alternate_goal_y,
        goal_shift_step=args.goal_shift_step,
        scan_policy_require_nav2_plan=not args.scan_policy_allow_proxy_labels,
    )
    print(f"Run manifest: {manifest_path}")
    if dataset_dir:
        print(f"Dataset: {dataset_dir}")
    if route_model_dir:
        print(f"Route model: {route_model_dir}")
    if ablation_dir:
        print(f"Ablation: {ablation_dir}")


if __name__ == "__main__":
    main()
