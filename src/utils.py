"""Shared utilities for the AMR runtime reliability demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


GridPosition = tuple[int, int]


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for deterministic demo simulations."""

    width: int = 18
    height: int = 12
    max_steps: int = 70
    seed: int = 7
    cautious_move_interval: int = 2


def manhattan(a: GridPosition, b: GridPosition) -> int:
    """Return Manhattan distance between two grid cells."""

    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a numeric value into a closed interval."""

    return max(low, min(high, value))


def ensure_output_dir(output_dir: str | Path) -> Path:
    """Create and return the output directory used by experiments."""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(rows: Sequence[dict], path: str | Path) -> Path:
    """Write simulation rows to CSV and return the path."""

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def summarize_run(rows: Sequence[dict], label: str) -> dict:
    """Create a compact summary dictionary from a simulation log."""

    if not rows:
        return {
            "mode": label,
            "success": False,
            "steps": 0,
            "final_status": "empty",
            "safe_stop_count": 0,
            "human_review_count": 0,
            "replan_count": 0,
            "max_risk": 0.0,
            "mean_risk": 0.0,
        }

    risk_values = [float(row["risk_score"]) for row in rows]
    decisions = [str(row["router_decision"]) for row in rows]
    final_status = str(rows[-1]["task_status"])
    return {
        "mode": label,
        "success": final_status == "completed",
        "steps": len(rows),
        "final_status": final_status,
        "safe_stop_count": decisions.count("SAFE_STOP"),
        "human_review_count": decisions.count("HUMAN_REVIEW"),
        "replan_count": decisions.count("REPLAN"),
        "max_risk": round(max(risk_values), 4),
        "mean_risk": round(sum(risk_values) / len(risk_values), 4),
    }


def encode_position(position: GridPosition) -> str:
    """Encode a grid position for human-readable CSV logs."""

    return f"({position[0]}, {position[1]})"


def encode_positions(positions: Iterable[GridPosition]) -> str:
    """Encode a list of grid positions for compact logging."""

    return ";".join(encode_position(pos) for pos in sorted(positions))
