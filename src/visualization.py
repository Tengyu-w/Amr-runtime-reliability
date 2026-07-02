"""Visualization helpers for simulation logs."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.environment import WarehouseEnvironment
from src.utils import GridPosition


def save_risk_curve(log_csv: str | Path, output_path: str | Path) -> Path:
    """Save a risk-score-over-time plot from a simulation CSV."""

    df = pd.read_csv(log_csv)
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["time_step"], df["risk_score"], color="#c73e1d", linewidth=2)
    ax.axhline(0.35, color="#d6a90f", linestyle="--", linewidth=1, label="cautious")
    ax.axhline(0.82, color="#8f1d14", linestyle="--", linewidth=1, label="safe stop")
    ax.set_xlabel("time step")
    ax.set_ylabel("risk score")
    ax.set_ylim(0, 1.02)
    ax.set_title("Runtime Reliability Risk Score")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_comparison_plot(summary_csv: str | Path, output_path: str | Path) -> Path:
    """Save a compact baseline-vs-supervisor comparison chart."""

    df = pd.read_csv(summary_csv)
    output_path = Path(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(df["mode"], df["max_risk"], color=["#7a869a", "#2e7d6f"])
    axes[0].set_title("Max Risk")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(df["mode"], df["steps"], color=["#7a869a", "#2e7d6f"])
    axes[1].set_title("Steps Before Completion/Stop")
    axes[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_warehouse_gif(
    history: list[dict],
    environment: WarehouseEnvironment,
    output_path: str | Path,
    fps: int = 3,
) -> Path:
    """Render a GIF showing robot motion, target changes, and dynamic obstacles."""

    output_path = Path(output_path)
    frames = [_render_frame(row, environment) for row in history]
    imageio.mimsave(output_path, frames, fps=fps)
    return output_path


def _render_frame(row: dict, environment: WarehouseEnvironment) -> np.ndarray:
    """Render one simulation state to an RGB array."""

    grid = np.ones((environment.height, environment.width, 3), dtype=float)
    grid[:, :, :] = np.array([0.96, 0.96, 0.94])

    for x, y in environment.static_obstacles:
        grid[y, x] = np.array([0.20, 0.22, 0.25])
    for x, y in environment.shelves:
        grid[y, x] = np.array([0.44, 0.34, 0.20])

    dynamic_obstacles = _parse_positions(row.get("dynamic_obstacles", ""))
    for x, y in dynamic_obstacles:
        grid[y, x] = np.array([0.80, 0.12, 0.12])

    path = _parse_positions(row.get("path", ""))
    for x, y in path:
        if 0 <= x < environment.width and 0 <= y < environment.height:
            grid[y, x] = np.array([0.70, 0.83, 0.96])

    target = _parse_position(row["target_position"])
    robot = _parse_position(row["robot_position"])
    grid[target[1], target[0]] = np.array([0.10, 0.55, 0.22])
    grid[robot[1], robot[0]] = np.array([0.05, 0.25, 0.85])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(grid, origin="upper")
    ax.set_xticks(np.arange(-0.5, environment.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, environment.height, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_title(
        f"t={row['time_step']} | {row['router_decision']} | risk={row['risk_score']:.2f}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame = frame.reshape((height, width, 4))[:, :, :3]
    plt.close(fig)
    return frame


def _parse_position(text: str) -> GridPosition:
    """Parse a position stored as '(x, y)'."""

    cleaned = text.strip().replace("(", "").replace(")", "")
    x_text, y_text = cleaned.split(",")
    return int(x_text), int(y_text)


def _parse_positions(text: str) -> list[GridPosition]:
    """Parse semicolon-delimited positions from CSV logs."""

    if not isinstance(text, str) or not text.strip():
        return []
    return [_parse_position(item) for item in text.split(";") if item.strip()]
