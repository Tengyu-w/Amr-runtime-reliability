"""Generate a closed-loop recovery-route demonstration GIF."""

from __future__ import annotations

from pathlib import Path
import sys

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.environment import WarehouseEnvironment
from src.planner import AStarPlanner


def main() -> None:
    output_dir = Path("visualizations/recovery_route")
    output_dir.mkdir(parents=True, exist_ok=True)

    env = WarehouseEnvironment()
    planner = AStarPlanner(env)
    start = env.start
    target = env.target
    nominal_path = planner.plan(start, target)
    if len(nominal_path) < 10:
        raise RuntimeError("Nominal path is too short for the recovery demo.")

    blockage = nominal_path[7]
    robot = start
    active_path = nominal_path
    history: list[dict] = []
    replanned_path: list[tuple[int, int]] = []
    recovery_triggered = False

    for step in range(34):
        if step == 6:
            env.add_dynamic_obstacle(blockage)

        next_nominal = _next_cell(robot, nominal_path)
        blocked_ahead = next_nominal in env.dynamic_obstacles
        lidar_hits = _lidar_hits(robot, env)

        if blocked_ahead and not recovery_triggered:
            replanned_path = planner.plan(robot, target, include_dynamic=True)
            active_path = replanned_path if replanned_path else active_path
            recovery_triggered = True
            decision = "REPLAN"
            mechanism = "blocked_path_high_conf_direction_error"
            policy_action = _action(robot, next_nominal)
        elif recovery_triggered:
            decision = "FOLLOW_REPLANNED_PATH"
            mechanism = "recovered_after_replan"
            policy_action = _action(robot, _next_cell(robot, active_path))
        else:
            decision = "NORMAL_NAVIGATION"
            mechanism = "nominal"
            policy_action = _action(robot, next_nominal)

        expert_next = _next_cell(robot, active_path)
        if decision != "REPLAN" and expert_next != robot:
            robot = expert_next

        history.append(
            {
                "time_step": step,
                "robot": robot,
                "target": target,
                "nominal_path": nominal_path,
                "active_path": active_path,
                "replanned_path": replanned_path,
                "dynamic_obstacles": sorted(env.dynamic_obstacles),
                "lidar_hits": lidar_hits,
                "policy_action": policy_action,
                "expert_action": _action(robot, _next_cell(robot, active_path)),
                "router_decision": decision,
                "mechanism": mechanism,
                "risk_score": 0.74 if decision == "REPLAN" else (0.31 if recovery_triggered else 0.12),
                "path_blocked_score": 0.88 if blocked_ahead else (0.25 if recovery_triggered else 0.05),
            }
        )

        if robot == target:
            break

    frames = [_render_frame(env, row) for row in history]
    imageio.mimsave(output_dir / "closed_loop_replan_recovery_demo.gif", frames, fps=3)
    _write_manifest(output_dir, history)


def _next_cell(robot: tuple[int, int], path: list[tuple[int, int]]) -> tuple[int, int]:
    if robot not in path:
        return robot
    index = path.index(robot)
    if index + 1 >= len(path):
        return robot
    return path[index + 1]


def _action(source: tuple[int, int], target: tuple[int, int]) -> str:
    dx = target[0] - source[0]
    dy = target[1] - source[1]
    if dx > 0:
        return "EAST"
    if dx < 0:
        return "WEST"
    if dy > 0:
        return "SOUTH"
    if dy < 0:
        return "NORTH"
    return "STAY"


def _lidar_hits(
    robot: tuple[int, int],
    env: WarehouseEnvironment,
    max_range: int = 6,
) -> list[tuple[tuple[int, int], bool]]:
    directions = [
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    ]
    hits = []
    for dx, dy in directions:
        hit = robot
        dynamic = False
        for distance in range(1, max_range + 1):
            cell = (robot[0] + dx * distance, robot[1] + dy * distance)
            if env.is_blocked(cell, include_dynamic=True):
                hit = cell
                dynamic = cell in env.dynamic_obstacles
                break
            hit = cell
        hits.append((hit, dynamic))
    return hits


def _render_frame(env: WarehouseEnvironment, row: dict) -> np.ndarray:
    fig, (ax, info_ax) = plt.subplots(
        1,
        2,
        figsize=(11.5, 5.8),
        gridspec_kw={"width_ratios": [1.25, 0.85]},
    )

    grid = np.ones((env.height, env.width, 3), dtype=float)
    grid[:, :, :] = np.array([0.96, 0.96, 0.94])
    for x, y in env.static_obstacles:
        grid[y, x] = np.array([0.20, 0.22, 0.25])
    for x, y in env.shelves:
        grid[y, x] = np.array([0.48, 0.38, 0.22])
    for x, y in row["dynamic_obstacles"]:
        grid[y, x] = np.array([0.84, 0.12, 0.12])

    _draw_path(ax, row["nominal_path"], "#d95f02", "original policy route", linestyle="--")
    if row["replanned_path"]:
        _draw_path(ax, row["replanned_path"], "#1b9e77", "replanned route", linestyle="-")
    _draw_path(ax, row["active_path"], "#2b6cb0", "active route", linestyle="-", alpha=0.35)

    robot = row["robot"]
    target = row["target"]
    for hit, dynamic in row["lidar_hits"]:
        color = "#d62728" if dynamic else "#7a869a"
        ax.plot([robot[0], hit[0]], [robot[1], hit[1]], color=color, linewidth=1.2, alpha=0.75)
        ax.scatter(hit[0], hit[1], color=color, s=18, zorder=4)

    ax.scatter(target[0], target[1], marker="*", s=240, color="#138a36", edgecolor="white", zorder=5)
    ax.scatter(robot[0], robot[1], marker="o", s=190, color="#0647a8", edgecolor="white", zorder=6)
    ax.text(robot[0] + 0.2, robot[1] - 0.2, "AMR", color="#0647a8", weight="bold")
    ax.text(target[0] + 0.2, target[1] - 0.2, "goal", color="#138a36", weight="bold")

    ax.imshow(grid, origin="upper")
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_title("Closed-loop recovery route demo")
    ax.legend(loc="lower left", fontsize=8)

    info_ax.axis("off")
    info = [
        f"time step: {row['time_step']}",
        f"router: {row['router_decision']}",
        f"mechanism: {row['mechanism']}",
        "",
        f"policy intended action: {row['policy_action']}",
        f"safe expert action: {row['expert_action']}",
        "",
        f"risk score: {row['risk_score']:.2f}",
        f"path blocked score: {row['path_blocked_score']:.2f}",
        "",
        "visual logic:",
        "1. orange = original route",
        "2. red block = external obstacle",
        "3. red lidar ray = blockage detected",
        "4. router triggers REPLAN",
        "5. green route returns to goal",
    ]
    info_ax.text(0, 0.98, "\n".join(info), va="top", fontsize=11)
    fig.tight_layout()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgb = frame.reshape((height, width, 4))[:, :, :3]
    plt.close(fig)
    return rgb


def _draw_path(
    ax: plt.Axes,
    path: list[tuple[int, int]],
    color: str,
    label: str,
    linestyle: str,
    alpha: float = 0.95,
) -> None:
    if not path:
        return
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    ax.plot(xs, ys, color=color, linewidth=2.5, linestyle=linestyle, alpha=alpha, label=label)


def _write_manifest(output_dir: Path, history: list[dict]) -> None:
    rows = [
        {
            "artifact": "closed_loop_replan_recovery_demo.gif",
            "frames": len(history),
            "scenario": "external_path_blockage",
            "triggered_route": "REPLAN",
            "description": (
                "Conceptual closed-loop playback showing a blocked original route, "
                "lidar-style detection, mechanism-aware REPLAN, and return to a safe path."
            ),
        }
    ]
    pd.DataFrame(rows).to_csv(output_dir / "recovery_route_visualization_manifest.csv", index=False)


if __name__ == "__main__":
    main()
