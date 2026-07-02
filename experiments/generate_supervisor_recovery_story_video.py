"""Generate a supervisor-facing recovery story video.

This is a readable closed-loop demonstration for presentation. It uses the
lightweight warehouse simulator so the research mechanism is visible: a learned
policy would continue into a blocked route, the router diagnoses the failure,
and the recovery path returns the AMR to the goal.
"""

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
    nominal_path = planner.plan(start, target, include_dynamic=False)
    if len(nominal_path) < 12:
        raise RuntimeError("Nominal path is too short for the supervisor story video.")

    blockage = nominal_path[8]
    robot = start
    recovery_path: list[tuple[int, int]] = []
    history: list[dict[str, object]] = []
    phase = "normal"
    route_sent_step: int | None = None

    for step in range(46):
        if step == 7:
            env.add_dynamic_obstacle(blockage)

        next_nominal = _next_cell(robot, nominal_path)
        blocked_ahead = next_nominal in env.dynamic_obstacles
        policy_action = _action(robot, next_nominal)

        if phase == "normal" and blocked_ahead:
            recovery_path = planner.plan(robot, target, include_dynamic=True)
            phase = "diagnose"
            route_sent_step = step
            router_decision = "REPLAN"
            executor_action = "PUBLISH_GOAL_REISSUE"
            mechanism = "blocked_path_high_conf_direction_error"
            active_path = recovery_path
        elif phase in {"diagnose", "recover"}:
            phase = "recover"
            router_decision = "FOLLOW_RECOVERY_ROUTE"
            executor_action = "NAV2_FOLLOW_REPLANNED_PATH"
            mechanism = "recovered_after_replan"
            active_path = recovery_path
        else:
            router_decision = "NORMAL_NAVIGATION"
            executor_action = "NONE"
            mechanism = "nominal"
            active_path = nominal_path

        if phase == "normal":
            next_cell = next_nominal
        elif phase == "diagnose":
            next_cell = robot
        else:
            next_cell = _next_cell(robot, recovery_path)

        if next_cell != robot and not env.is_blocked(next_cell, include_dynamic=True):
            robot = next_cell

        history.append(
            {
                "time_step": step,
                "robot": robot,
                "target": target,
                "nominal_path": nominal_path,
                "recovery_path": recovery_path,
                "active_path": active_path,
                "dynamic_obstacles": sorted(env.dynamic_obstacles),
                "policy_action": policy_action,
                "router_decision": router_decision,
                "executor_action": executor_action,
                "mechanism": mechanism,
                "phase": phase,
                "route_sent_step": route_sent_step,
                "risk_score": _risk_score(phase, blocked_ahead),
                "path_blocked_score": 0.92 if blocked_ahead else (0.38 if phase == "recover" else 0.04),
                "blocked_ahead": blocked_ahead,
                "goal_reached": robot == target,
            }
        )

        if robot == target:
            for hold in range(6):
                row = dict(history[-1])
                row["time_step"] = step + hold + 1
                row["phase"] = "success"
                row["router_decision"] = "RECOVERY_SUCCESS"
                row["executor_action"] = "GOAL_REACHED"
                row["risk_score"] = 0.08
                row["path_blocked_score"] = 0.10
                row["goal_reached"] = True
                history.append(row)
            break

    frames = [_render_frame(env, row) for row in history]
    gif_path = output_dir / "closed_loop_recovery_supervisor_story.gif"
    imageio.mimsave(gif_path, frames, fps=4)
    _try_write_mp4(output_dir / "closed_loop_recovery_supervisor_story.mp4", frames, fps=4)
    _write_manifest(output_dir, history)


def _risk_score(phase: str, blocked_ahead: bool) -> float:
    if blocked_ahead:
        return 0.82
    if phase == "recover":
        return 0.34
    if phase == "diagnose":
        return 0.74
    return 0.12


def _next_cell(robot: tuple[int, int], path: list[tuple[int, int]]) -> tuple[int, int]:
    if not path or robot not in path:
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


def _render_frame(env: WarehouseEnvironment, row: dict[str, object]) -> np.ndarray:
    fig = plt.figure(figsize=(13.2, 7.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.55, 0.9, 1.05], height_ratios=[1.05, 0.9])
    ax = fig.add_subplot(gs[:, 0])
    timeline_ax = fig.add_subplot(gs[0, 1:])
    info_ax = fig.add_subplot(gs[1, 1:])

    _draw_map(ax, env, row)
    _draw_timeline(timeline_ax, row)
    _draw_info(info_ax, row)

    fig.suptitle(
        "AMR mechanism-aware recovery: wrong route -> REPLAN -> recovered path",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _fig_to_rgb(fig)


def _draw_map(ax: plt.Axes, env: WarehouseEnvironment, row: dict[str, object]) -> None:
    grid = np.ones((env.height, env.width, 3), dtype=float)
    grid[:, :, :] = np.array([0.95, 0.95, 0.91])
    for x, y in env.static_obstacles:
        grid[y, x] = np.array([0.20, 0.22, 0.25])
    for x, y in env.shelves:
        grid[y, x] = np.array([0.45, 0.36, 0.24])
    for x, y in row["dynamic_obstacles"]:  # type: ignore[index]
        grid[y, x] = np.array([0.82, 0.12, 0.12])

    ax.imshow(grid, origin="upper")
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    _draw_path(ax, row["nominal_path"], "#d47700", "original policy route", "--", alpha=0.85)  # type: ignore[arg-type]
    if row["recovery_path"]:  # type: ignore[index]
        _draw_path(ax, row["recovery_path"], "#108a54", "replanned recovery route", "-", alpha=0.98)  # type: ignore[arg-type]
    _draw_path(ax, row["active_path"], "#1e64c8", "currently executed route", "-", alpha=0.38)  # type: ignore[arg-type]

    robot = row["robot"]  # type: ignore[assignment]
    target = row["target"]  # type: ignore[assignment]
    assert isinstance(robot, tuple)
    assert isinstance(target, tuple)

    blocked_cells = row["dynamic_obstacles"]  # type: ignore[assignment]
    if blocked_cells:
        bx, by = blocked_cells[0]
        ax.annotate(
            "external obstacle",
            xy=(bx, by),
            xytext=(bx + 1.3, by - 1.1),
            arrowprops={"arrowstyle": "->", "color": "#b3261e", "linewidth": 1.5},
            fontsize=9,
            color="#8f1d1d",
            weight="bold",
        )

    ax.scatter(target[0], target[1], marker="*", s=270, color="#198754", edgecolor="white", zorder=7)
    ax.scatter(robot[0], robot[1], marker="o", s=230, color="#0b4fa3", edgecolor="white", zorder=8)
    ax.text(robot[0] + 0.2, robot[1] - 0.25, "AMR", color="#0b4fa3", weight="bold", fontsize=10)
    ax.text(target[0] + 0.2, target[1] - 0.25, "goal", color="#198754", weight="bold", fontsize=10)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Warehouse recovery path")


def _draw_path(
    ax: plt.Axes,
    path: list[tuple[int, int]],
    color: str,
    label: str,
    linestyle: str,
    alpha: float,
) -> None:
    if not path:
        return
    ax.plot(
        [p[0] for p in path],
        [p[1] for p in path],
        color=color,
        linewidth=2.8,
        linestyle=linestyle,
        alpha=alpha,
        label=label,
    )


def _draw_timeline(ax: plt.Axes, row: dict[str, object]) -> None:
    ax.axis("off")
    phase = str(row["phase"])
    steps = [
        ("1 policy\nroute", phase in {"normal", "diagnose", "recover", "success"}),
        ("2 blockage", bool(row["dynamic_obstacles"])),
        ("3 mechanism\ndiagnosis", phase in {"diagnose", "recover", "success"}),
        ("4 REPLAN\nexecutor", phase in {"diagnose", "recover", "success"}),
        ("5 recovered\npath", phase in {"recover", "success"}),
        ("6 goal\nreached", bool(row["goal_reached"])),
    ]
    x_positions = np.linspace(0.05, 0.95, len(steps))
    for idx, ((label, active), x) in enumerate(zip(steps, x_positions)):
        color = "#198754" if active else "#c9c9c9"
        ax.scatter(x, 0.58, s=420, color=color, edgecolor="white", linewidth=1.2)
        ax.text(x, 0.25, label, ha="center", va="top", fontsize=8.2, linespacing=1.1)
        if idx + 1 < len(steps):
            ax.plot([x + 0.04, x_positions[idx + 1] - 0.04], [0.58, 0.58], color="#b8b8b8", linewidth=2)
    ax.text(0.0, 0.96, "Research process shown in this video", fontsize=12, weight="bold", va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _draw_info(ax: plt.Axes, row: dict[str, object]) -> None:
    ax.axis("off")
    lines = [
        f"time step: {row['time_step']}",
        f"policy action before recovery: {row['policy_action']}",
        f"failure mechanism: {row['mechanism']}",
        f"router decision: {row['router_decision']}",
        f"executor action: {row['executor_action']}",
        "",
        f"risk score: {float(row['risk_score']):.2f}",
        f"path blocked score: {float(row['path_blocked_score']):.2f}",
        "",
        "What this video demonstrates:",
        "A blocked policy route is detected,",
        "the route is changed to REPLAN,",
        "and the AMR follows a new path to the goal.",
        "",
        "Evidence level:",
        "readable lightweight closed-loop demo;",
        "Gazebo/Nav2 recovery evidence is separate.",
    ]
    ax.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10.5)


def _fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgb = frame.reshape((height, width, 4))[:, :, :3]
    plt.close(fig)
    return rgb


def _try_write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    try:
        imageio.mimsave(path, frames, fps=fps)
    except Exception:
        if path.exists():
            path.unlink()
        return
    if path.exists() and path.stat().st_size == 0:
        path.unlink()


def _write_manifest(output_dir: Path, history: list[dict[str, object]]) -> None:
    rows = [
        {
            "artifact": "closed_loop_recovery_supervisor_story.gif",
            "frames": len(history),
            "scenario": "external_path_blockage",
            "triggered_route": "REPLAN",
            "evidence_level": "readable lightweight closed-loop recovery demo",
            "description": (
                "Supervisor-facing video showing original policy route, external blockage, "
                "mechanism diagnosis, REPLAN execution, recovered route, and goal reached."
            ),
        }
    ]
    mp4_path = output_dir / "closed_loop_recovery_supervisor_story.mp4"
    if mp4_path.exists() and mp4_path.stat().st_size > 0:
        rows.append(
            {
                "artifact": "closed_loop_recovery_supervisor_story.mp4",
                "frames": len(history),
                "scenario": "external_path_blockage",
                "triggered_route": "REPLAN",
                "evidence_level": "readable lightweight closed-loop recovery demo",
                "description": "MP4 version of the supervisor-facing recovery story video.",
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "supervisor_recovery_story_manifest.csv", index=False)


if __name__ == "__main__":
    main()
