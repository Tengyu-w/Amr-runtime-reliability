"""Generate labelled AMR reliability simulation episodes."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEMO_ROOT = Path(__file__).resolve().parents[1]
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from experiments.run_baseline import run as run_baseline
from experiments.run_reliability_supervisor import run as run_supervisor
from src.failure_injection import FailureInjector
from src.utils import SimulationConfig, ensure_output_dir, write_csv


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    description: str
    enabled_events: tuple[str, ...]
    primary_fault_origin: str
    primary_fault_family: str
    primary_ood_status: str
    split_group: str


SCENARIOS = [
    ScenarioSpec(
        scenario_id="nominal",
        description="No injected fault.",
        enabled_events=(),
        primary_fault_origin="none",
        primary_fault_family="none",
        primary_ood_status="none",
        split_group="nominal",
    ),
    ScenarioSpec(
        scenario_id="external_path_blockage",
        description="External dynamic obstacle blocks the planned path.",
        enabled_events=("dynamic_obstacle_blocks_path",),
        primary_fault_origin="external_disturbance",
        primary_fault_family="path_blockage",
        primary_ood_status="in_distribution_fault",
        split_group="external_disturbance",
    ),
    ScenarioSpec(
        scenario_id="localization_drift",
        description="State-estimation uncertainty gradually increases.",
        enabled_events=("localization_drift_increasing",),
        primary_fault_origin="state_estimation_drift",
        primary_fault_family="localization",
        primary_ood_status="in_distribution_fault",
        split_group="state_estimation",
    ),
    ScenarioSpec(
        scenario_id="perception_degradation",
        description="Sensor confidence degrades over the episode.",
        enabled_events=("sensor_confidence_drop",),
        primary_fault_origin="perception_degradation",
        primary_fault_family="sensor_quality",
        primary_ood_status="in_distribution_fault",
        split_group="perception",
    ),
    ScenarioSpec(
        scenario_id="task_goal_shift_ood_style",
        description="Goal changes during execution; treated as an OOD-style task shift.",
        enabled_events=("target_changed",),
        primary_fault_origin="task_or_goal_shift",
        primary_fault_family="task_reassignment",
        primary_ood_status="ood_style_shift",
        split_group="task_shift",
    ),
    ScenarioSpec(
        scenario_id="execution_deviation",
        description="Robot is pushed off the planned path.",
        enabled_events=("trajectory_deviation",),
        primary_fault_origin="execution_error",
        primary_fault_family="control_tracking",
        primary_ood_status="in_distribution_fault",
        split_group="execution",
    ),
    ScenarioSpec(
        scenario_id="progress_blockage",
        description="External blocker causes progress stagnation.",
        enabled_events=("progress_stagnation_blocker",),
        primary_fault_origin="external_disturbance",
        primary_fault_family="progress_blockage",
        primary_ood_status="in_distribution_fault",
        split_group="external_disturbance",
    ),
    ScenarioSpec(
        scenario_id="planner_backend_failure",
        description="Target changes and replanning backend becomes unstable.",
        enabled_events=("target_changed", "replanning_backend_unstable"),
        primary_fault_origin="planner_internal_failure",
        primary_fault_family="planning_backend",
        primary_ood_status="in_distribution_fault",
        split_group="planner_backend",
    ),
    ScenarioSpec(
        scenario_id="compound_shift_and_degradation",
        description="Task shift, perception degradation, and localization drift co-occur.",
        enabled_events=("target_changed", "sensor_confidence_drop", "localization_drift_increasing"),
        primary_fault_origin="task_or_goal_shift",
        primary_fault_family="task_reassignment",
        primary_ood_status="ood_style_shift",
        split_group="compound",
    ),
    ScenarioSpec(
        scenario_id="mixed_blockage_and_perception",
        description="External path blockage and sensor degradation co-occur.",
        enabled_events=("dynamic_obstacle_blocks_path", "sensor_confidence_drop"),
        primary_fault_origin="mixed_external_perception",
        primary_fault_family="path_blockage+sensor_quality",
        primary_ood_status="mixed_fault",
        split_group="mixed_boundary",
    ),
    ScenarioSpec(
        scenario_id="mixed_drift_and_execution",
        description="Localization drift and trajectory deviation co-occur.",
        enabled_events=("localization_drift_increasing", "trajectory_deviation"),
        primary_fault_origin="mixed_state_execution",
        primary_fault_family="localization+control_tracking",
        primary_ood_status="mixed_fault",
        split_group="mixed_boundary",
    ),
    ScenarioSpec(
        scenario_id="boundary_weak_blockage",
        description="A weak intermittent blockage sits near the route-threshold boundary.",
        enabled_events=("weak_dynamic_obstacle_blocks_path",),
        primary_fault_origin="external_disturbance",
        primary_fault_family="boundary_path_blockage",
        primary_ood_status="boundary_case",
        split_group="mixed_boundary",
    ),
]


def _scenario_catalog_rows() -> list[dict]:
    rows = []
    for scenario in SCENARIOS:
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "description": scenario.description,
                "enabled_events": "|".join(scenario.enabled_events) if scenario.enabled_events else "none",
                "primary_fault_origin": scenario.primary_fault_origin,
                "primary_fault_family": scenario.primary_fault_family,
                "primary_ood_status": scenario.primary_ood_status,
                "split_group": scenario.split_group,
            }
        )
    return rows


def _split_for_seed(seed: int) -> str:
    bucket = seed % 10
    if bucket in {0, 1, 2, 3, 4, 5}:
        return "train"
    if bucket in {6, 7}:
        return "val"
    return "test"


def _injector_for_scenario(seed: int, scenario: ScenarioSpec) -> FailureInjector:
    return FailureInjector(seed=seed, enabled_events=set(scenario.enabled_events))


def _episode_rows(
    episode_id: str,
    scenario: ScenarioSpec,
    seed: int,
    mode: str,
    rows: list[dict],
) -> tuple[dict, list[dict]]:
    final_status = str(rows[-1]["task_status"]) if rows else "empty"
    event_rows = [row for row in rows if row.get("failure_event") != "none"]
    episode = {
        "episode_id": episode_id,
        "scenario_id": scenario.scenario_id,
        "mode": mode,
        "seed": seed,
        "split": _split_for_seed(seed),
        "split_group": scenario.split_group,
        "primary_fault_origin": scenario.primary_fault_origin,
        "primary_fault_family": scenario.primary_fault_family,
        "primary_ood_status": scenario.primary_ood_status,
        "enabled_events": "|".join(scenario.enabled_events) if scenario.enabled_events else "none",
        "n_steps": len(rows),
        "n_fault_event_steps": len(event_rows),
        "final_status": final_status,
        "success": final_status == "completed",
    }
    timestep_rows = []
    for row in rows:
        out = dict(row)
        out.update(
            {
                "episode_id": episode_id,
                "scenario_id": scenario.scenario_id,
                "mode": mode,
                "seed": seed,
                "split": _split_for_seed(seed),
                "scenario_primary_fault_origin": scenario.primary_fault_origin,
                "scenario_primary_fault_family": scenario.primary_fault_family,
                "scenario_primary_ood_status": scenario.primary_ood_status,
            }
        )
        timestep_rows.append(out)
    return episode, timestep_rows


def generate_dataset(
    seeds: Iterable[int],
    out_dir: str | Path,
    modes: Iterable[str],
) -> tuple[Path, Path, Path, Path]:
    output_dir = ensure_output_dir(out_dir)
    episodes: list[dict] = []
    timesteps: list[dict] = []

    for scenario in SCENARIOS:
        for seed in seeds:
            config = SimulationConfig(seed=int(seed))
            for mode in modes:
                episode_id = f"{scenario.scenario_id}__seed_{seed}__{mode}"
                episode_dir = ensure_output_dir(output_dir / "episode_logs" / episode_id)
                injector = _injector_for_scenario(int(seed), scenario)
                if mode == "baseline":
                    rows, _ = run_baseline(episode_dir, config=config, injector=injector)
                else:
                    rows, _ = run_supervisor(
                        episode_dir,
                        config=config,
                        router_mode=mode,
                        injector=injector,
                    )
                episode, timestep_rows = _episode_rows(episode_id, scenario, int(seed), mode, rows)
                episodes.append(episode)
                timesteps.extend(timestep_rows)

    splits = [
        {
            "seed": int(seed),
            "split": _split_for_seed(int(seed)),
            "split_rule": "seed_mod_10: 0-5 train, 6-7 val, 8-9 test",
        }
        for seed in seeds
    ]
    catalog_path = write_csv(_scenario_catalog_rows(), output_dir / "scenario_catalog.csv")
    split_path = write_csv(splits, output_dir / "splits.csv")
    episode_path = write_csv(episodes, output_dir / "episodes.csv")
    timestep_path = write_csv(timesteps, output_dir / "timesteps.csv")
    return catalog_path, split_path, episode_path, timestep_path


def _parse_seeds(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_modes(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate labelled AMR simulation reliability episodes.")
    parser.add_argument("--seeds", type=str, default="10,11,12,16,17,18,19")
    parser.add_argument("--modes", type=str, default="baseline,risk_router,mechanism_router")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/scenario_dataset"))
    args = parser.parse_args()

    catalog_path, split_path, episode_path, timestep_path = generate_dataset(
        seeds=_parse_seeds(args.seeds),
        out_dir=args.out_dir,
        modes=_parse_modes(args.modes),
    )
    print(f"Scenario catalog: {catalog_path}")
    print(f"Splits: {split_path}")
    print(f"Episodes: {episode_path}")
    print(f"Timesteps: {timestep_path}")


if __name__ == "__main__":
    main()
