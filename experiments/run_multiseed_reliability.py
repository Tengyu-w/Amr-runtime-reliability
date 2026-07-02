"""Run paired multi-seed AMR reliability-supervisor comparisons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

DEMO_ROOT = Path(__file__).resolve().parents[1]
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from experiments.run_baseline import run as run_baseline
from experiments.run_reliability_supervisor import run as run_supervisor
from src.utils import SimulationConfig, ensure_output_dir, summarize_run, write_csv


NON_NOMINAL_DECISIONS = {
    "CAUTIOUS_MODE",
    "REPLAN",
    "RELOCALIZE",
    "HUMAN_REVIEW",
    "SAFE_STOP",
}

HARD_RECOVERY_DECISIONS = {
    "REPLAN",
    "RELOCALIZE",
    "HUMAN_REVIEW",
    "SAFE_STOP",
}

DEFAULT_ROUTER_MODES = ("risk_router", "mechanism_router")


def _event_steps(rows: list[dict]) -> list[dict]:
    return [row for row in rows if str(row.get("failure_event", "none")) != "none"]


def _count_decisions(rows: list[dict], decisions: set[str]) -> int:
    return sum(1 for row in rows if str(row.get("router_decision")) in decisions)


def _high_risk_event_rate(rows: list[dict], risk_threshold: float) -> float:
    events = _event_steps(rows)
    if not events:
        return 0.0
    high_risk_events = sum(1 for row in events if float(row.get("risk_score", 0.0)) >= risk_threshold)
    return high_risk_events / len(events)


def _decision_on_event_rate(rows: list[dict], decisions: set[str]) -> float:
    events = _event_steps(rows)
    if not events:
        return 0.0
    selected_events = sum(1 for row in events if str(row.get("router_decision")) in decisions)
    return selected_events / len(events)


def _summary_for_rows(rows: list[dict], mode: str, seed: int, risk_threshold: float) -> dict:
    summary = summarize_run(rows, mode)
    summary.update(
        {
            "seed": seed,
            "failure_event_steps": len(_event_steps(rows)),
            "non_nominal_action_count": _count_decisions(rows, NON_NOMINAL_DECISIONS),
            "hard_recovery_action_count": _count_decisions(rows, HARD_RECOVERY_DECISIONS),
            "high_risk_event_rate": round(_high_risk_event_rate(rows, risk_threshold), 4),
            "non_nominal_on_event_rate": round(
                _decision_on_event_rate(rows, NON_NOMINAL_DECISIONS),
                4,
            ),
            "hard_recovery_on_event_rate": round(
                _decision_on_event_rate(rows, HARD_RECOVERY_DECISIONS),
                4,
            ),
        }
    )
    return summary


def _paired_delta_rows(seed_rows: list[dict]) -> list[dict]:
    by_seed: dict[int, dict[str, dict]] = {}
    for row in seed_rows:
        by_seed.setdefault(int(row["seed"]), {})[str(row["mode"])] = row

    deltas = []
    for seed, rows in sorted(by_seed.items()):
        baseline = rows.get("baseline")
        if baseline is None:
            continue
        for mode, supervisor in sorted(rows.items()):
            if mode == "baseline":
                continue
            deltas.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "success_delta_vs_baseline": int(bool(supervisor["success"]))
                    - int(bool(baseline["success"])),
                    "steps_delta_vs_baseline": int(supervisor["steps"]) - int(baseline["steps"]),
                    "non_nominal_action_delta_vs_baseline": int(supervisor["non_nominal_action_count"])
                    - int(baseline["non_nominal_action_count"]),
                    "hard_recovery_action_delta_vs_baseline": int(supervisor["hard_recovery_action_count"])
                    - int(baseline["hard_recovery_action_count"]),
                    "event_non_nominal_rate_delta_vs_baseline": round(
                        float(supervisor["non_nominal_on_event_rate"])
                        - float(baseline["non_nominal_on_event_rate"]),
                        4,
                    ),
                    "event_hard_recovery_rate_delta_vs_baseline": round(
                        float(supervisor["hard_recovery_on_event_rate"])
                        - float(baseline["hard_recovery_on_event_rate"]),
                        4,
                    ),
                    "mean_risk_delta_vs_baseline": round(
                        float(supervisor["mean_risk"]) - float(baseline["mean_risk"]),
                        4,
                    ),
                }
            )
    return deltas


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(stdev(values))


def _aggregate(seed_rows: list[dict], delta_rows: list[dict]) -> list[dict]:
    rows = []
    metric_names = [
        "success",
        "steps",
        "safe_stop_count",
        "human_review_count",
        "replan_count",
        "non_nominal_action_count",
        "hard_recovery_action_count",
        "high_risk_event_rate",
        "non_nominal_on_event_rate",
        "hard_recovery_on_event_rate",
        "max_risk",
        "mean_risk",
    ]
    modes = sorted({str(row["mode"]) for row in seed_rows})
    for mode in modes:
        mode_rows = [row for row in seed_rows if row["mode"] == mode]
        for metric in metric_names:
            values = [float(row[metric]) for row in mode_rows]
            metric_mean, metric_std = _mean_std(values)
            rows.append(
                {
                    "group": mode,
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean": round(metric_mean, 4),
                    "std": round(metric_std, 4),
                }
            )

    delta_metrics = [
        "success_delta_vs_baseline",
        "steps_delta_vs_baseline",
        "non_nominal_action_delta_vs_baseline",
        "hard_recovery_action_delta_vs_baseline",
        "event_non_nominal_rate_delta_vs_baseline",
        "event_hard_recovery_rate_delta_vs_baseline",
        "mean_risk_delta_vs_baseline",
    ]
    for mode in sorted({str(row["mode"]) for row in delta_rows}):
        mode_rows = [row for row in delta_rows if row["mode"] == mode]
        for metric in delta_metrics:
            values = [float(row[metric]) for row in mode_rows]
            metric_mean, metric_std = _mean_std(values)
            rows.append(
                {
                    "group": f"paired_delta_{mode}",
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean": round(metric_mean, 4),
                    "std": round(metric_std, 4),
                }
            )
    return rows


def _mechanism_route_rows(seed: int, mode: str, rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        mechanism = str(row.get("failure_mechanism", "not_recorded"))
        decision = str(row.get("router_decision", "not_recorded"))
        grouped.setdefault((mechanism, decision), []).append(row)

    out = []
    for (mechanism, decision), sub_rows in sorted(grouped.items()):
        risks = [float(row.get("risk_score", 0.0)) for row in sub_rows]
        out.append(
            {
                "seed": seed,
                "mode": mode,
                "failure_mechanism": mechanism,
                "router_decision": decision,
                "n_steps": len(sub_rows),
                "n_failure_event_steps": len(_event_steps(sub_rows)),
                "mean_risk": round(mean(risks), 4) if risks else 0.0,
            }
        )
    return out


def _fault_origin_route_rows(seed: int, mode: str, rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = {}
    for row in rows:
        if str(row.get("failure_event", "none")) == "none":
            continue
        origin = str(row.get("primary_fault_origin", row.get("fault_origin", "not_recorded")))
        family = str(row.get("primary_fault_family", row.get("fault_family", "not_recorded")))
        ood_status = str(row.get("primary_ood_status", row.get("ood_status", "not_recorded")))
        decision = str(row.get("router_decision", "not_recorded"))
        grouped.setdefault((origin, family, ood_status, decision), []).append(row)

    out = []
    for (origin, family, ood_status, decision), sub_rows in sorted(grouped.items()):
        risks = [float(row.get("risk_score", 0.0)) for row in sub_rows]
        out.append(
            {
                "seed": seed,
                "mode": mode,
                "fault_origin": origin,
                "fault_family": family,
                "ood_status": ood_status,
                "router_decision": decision,
                "n_failure_event_steps": len(sub_rows),
                "mean_risk": round(mean(risks), 4) if risks else 0.0,
            }
        )
    return out


def run_multiseed(
    seeds: Iterable[int],
    output_dir: str | Path,
    risk_threshold: float = 0.35,
    router_modes: Iterable[str] = DEFAULT_ROUTER_MODES,
) -> tuple[Path, Path, Path, Path, Path]:
    out_dir = ensure_output_dir(output_dir)
    seed_rows: list[dict] = []
    mechanism_rows: list[dict] = []
    origin_rows: list[dict] = []

    for seed in seeds:
        seed_dir = ensure_output_dir(out_dir / f"seed_{seed}")
        config = SimulationConfig(seed=int(seed))
        baseline_rows, _ = run_baseline(seed_dir / "baseline", config=config)
        seed_rows.append(_summary_for_rows(baseline_rows, "baseline", int(seed), risk_threshold))
        for router_mode in router_modes:
            mode = str(router_mode)
            supervisor_rows, _ = run_supervisor(seed_dir / mode, config=config, router_mode=mode)
            seed_rows.append(_summary_for_rows(supervisor_rows, mode, int(seed), risk_threshold))
            mechanism_rows.extend(_mechanism_route_rows(int(seed), mode, supervisor_rows))
            origin_rows.extend(_fault_origin_route_rows(int(seed), mode, supervisor_rows))

    delta_rows = _paired_delta_rows(seed_rows)
    aggregate_rows = _aggregate(seed_rows, delta_rows)
    seed_path = write_csv(seed_rows, out_dir / "multiseed_seed_level.csv")
    delta_path = write_csv(delta_rows, out_dir / "multiseed_paired_deltas.csv")
    aggregate_path = write_csv(aggregate_rows, out_dir / "multiseed_mean_std.csv")
    mechanism_path = write_csv(mechanism_rows, out_dir / "multiseed_mechanism_routes.csv")
    origin_path = write_csv(origin_rows, out_dir / "multiseed_fault_origin_routes.csv")
    return seed_path, delta_path, aggregate_path, mechanism_path, origin_path


def _parse_seeds(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired multi-seed baseline vs reliability-supervisor AMR comparisons."
    )
    parser.add_argument("--seeds", type=str, default="7,8,9,10,11")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/multiseed_reliability"))
    parser.add_argument("--risk-threshold", type=float, default=0.35)
    parser.add_argument("--router-modes", type=str, default="risk_router,mechanism_router")
    args = parser.parse_args()

    seed_path, delta_path, aggregate_path, mechanism_path, origin_path = run_multiseed(
        seeds=_parse_seeds(args.seeds),
        output_dir=args.out_dir,
        risk_threshold=args.risk_threshold,
        router_modes=[mode.strip() for mode in args.router_modes.split(",") if mode.strip()],
    )
    print(f"Seed-level results: {seed_path}")
    print(f"Paired deltas: {delta_path}")
    print(f"Mean/std summary: {aggregate_path}")
    print(f"Mechanism routes: {mechanism_path}")
    print(f"Fault-origin routes: {origin_path}")


if __name__ == "__main__":
    main()
