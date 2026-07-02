"""Generate a 3D mechanism demo for blockage-triggered AMR recovery.

This is a lightweight closed-loop simulation, not a Gazebo recording. It shows
the intended recovery chain clearly: the AMR follows the original policy route,
reaches the cell before a newly blocked route segment, stops, triggers REPLAN,
and then follows an A* recovery path around the obstacle to the goal.
"""

from __future__ import annotations

from pathlib import Path
import math
import sys

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
    nominal_path = planner.plan(env.start, env.target, include_dynamic=False)
    if len(nominal_path) < 14:
        raise RuntimeError("Nominal path is too short for a recovery mechanism demo.")

    history = _build_history(env, planner, nominal_path)
    frames = [_render_frame(env, row) for row in history]

    gif_path = output_dir / "closed_loop_recovery_mechanism_3d.gif"
    imageio.mimsave(gif_path, frames, fps=4)
    _write_manifest(output_dir, history)


def _build_history(
    env: WarehouseEnvironment,
    planner: AStarPlanner,
    nominal_path: list[tuple[int, int]],
) -> list[dict[str, object]]:
    blockage = nominal_path[8]
    robot = env.start
    trail = [robot]
    recovery_path: list[tuple[int, int]] = []
    active_path = nominal_path
    phase = "normal"
    history: list[dict[str, object]] = []

    for step in range(58):
        if step == 6:
            env.add_dynamic_obstacle(blockage)

        next_nominal = _next_cell(robot, nominal_path)
        blocked_ahead = next_nominal in env.dynamic_obstacles

        if phase == "normal" and blocked_ahead:
            recovery_path = planner.plan(robot, env.target, include_dynamic=True)
            active_path = recovery_path if recovery_path else active_path
            phase = "diagnose"
            router_decision = "REPLAN"
            executor_action = "STOP_AND_REISSUE_GOAL"
            mechanism = "blocked_path_high_conf_direction_error"
            next_cell = robot
        elif phase in {"diagnose", "recover"}:
            phase = "recover"
            router_decision = "FOLLOW_REPLANNED_PATH"
            executor_action = "NAV2_FOLLOW_RECOVERY_PATH"
            mechanism = "recovered_after_replan"
            next_cell = _next_cell(robot, recovery_path)
        else:
            router_decision = "NORMAL_NAVIGATION"
            executor_action = "NONE"
            mechanism = "nominal_policy"
            next_cell = next_nominal

        if next_cell != robot and not env.is_blocked(next_cell, include_dynamic=True):
            robot = next_cell
            trail.append(robot)

        history.append(
            {
                "time_step": step,
                "robot": robot,
                "target": env.target,
                "trail": list(trail),
                "nominal_path": nominal_path,
                "recovery_path": recovery_path,
                "active_path": active_path,
                "dynamic_obstacles": sorted(env.dynamic_obstacles),
                "phase": phase,
                "blocked_ahead": blocked_ahead,
                "router_decision": router_decision,
                "executor_action": executor_action,
                "mechanism": mechanism,
                "risk_score": 0.86 if blocked_ahead else (0.34 if phase == "recover" else 0.12),
                "path_blocked_score": 0.92 if blocked_ahead else (0.30 if phase == "recover" else 0.05),
                "goal_reached": robot == env.target,
            }
        )

        if robot == env.target:
            for hold in range(8):
                row = dict(history[-1])
                row["time_step"] = step + hold + 1
                row["phase"] = "success"
                row["router_decision"] = "RECOVERY_SUCCESS"
                row["executor_action"] = "GOAL_REACHED"
                row["risk_score"] = 0.07
                row["path_blocked_score"] = 0.05
                row["goal_reached"] = True
                history.append(row)
            break

    return history


def _next_cell(robot: tuple[int, int], path: list[tuple[int, int]]) -> tuple[int, int]:
    if not path or robot not in path:
        return robot
    index = path.index(robot)
    if index + 1 >= len(path):
        return robot
    return path[index + 1]


def _render_frame(env: WarehouseEnvironment, row: dict[str, object]) -> np.ndarray:
    fig = plt.figure(figsize=(14.0, 7.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.7, 0.9, 1.0], height_ratios=[1.0, 0.88])
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_flow = fig.add_subplot(gs[0, 1:])
    ax_info = fig.add_subplot(gs[1, 1:])

    _draw_scene(ax3d, env, row)
    _draw_flow(ax_flow, row)
    _draw_info(ax_info, row)

    fig.suptitle(
        "3D recovery mechanism demo: approach obstacle -> REPLAN -> detour to goal",
        fontsize=14,
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    return _fig_to_rgb(fig)


def _draw_scene(ax: plt.Axes, env: WarehouseEnvironment, row: dict[str, object]) -> None:
    ax.view_init(elev=36, azim=-58)
    ax.set_box_aspect((1.55, 1.0, 0.26))
    ax.set_facecolor("#f6f3eb")
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(env.height - 0.5, -0.5)
    ax.set_zlim(0.0, 1.2)
    ax.set_xlabel("warehouse x")
    ax.set_ylabel("warehouse y")
    ax.set_zlabel("height")
    ax.set_title("Lightweight warehouse closed loop")

    _draw_floor(ax, env)
    for cell in sorted(env.shelves):
        _draw_box(ax, (cell[0], cell[1], 0.32), (0.92, 0.92, 0.64), "#9aa0a6", 0.36)
    for cell in sorted(env.static_obstacles - env.shelves):
        if cell[0] in {0, env.width - 1} or cell[1] in {0, env.height - 1}:
            continue
        _draw_box(ax, (cell[0], cell[1], 0.28), (0.9, 0.9, 0.56), "#7a7f87", 0.42)
    for cell in row["dynamic_obstacles"]:  # type: ignore[assignment]
        _draw_box(ax, (cell[0], cell[1], 0.45), (0.88, 0.88, 0.9), "#d62728", 0.78)
        ax.text(cell[0] + 0.3, cell[1], 1.05, "external obstacle", color="#8f1d1d", fontsize=9, weight="bold")

    _draw_path(ax, row["nominal_path"], "#d47700", "--", "original policy route")  # type: ignore[arg-type]
    if row["recovery_path"]:  # type: ignore[index]
        _draw_path(ax, row["recovery_path"], "#108a54", "-", "replanned recovery path")  # type: ignore[arg-type]
    _draw_path(ax, row["trail"], "#0b4fa3", "-", "executed AMR trail", z=0.16, linewidth=3.4)  # type: ignore[arg-type]

    robot = row["robot"]  # type: ignore[assignment]
    target = row["target"]  # type: ignore[assignment]
    yaw = _path_yaw(row["trail"])  # type: ignore[arg-type]
    _draw_amr(ax, robot, yaw)
    ax.scatter([target[0]], [target[1]], [0.22], marker="*", s=260, color="#198754", depthshade=True)
    ax.text(target[0] + 0.25, target[1], 0.42, "goal", color="#198754", fontsize=10, weight="bold")

    if row["blocked_ahead"]:
        ax.text2D(
            0.02,
            0.97,
            "Blocked ahead: AMR stops before obstacle and sends REPLAN",
            transform=ax.transAxes,
            fontsize=10,
            weight="bold",
            color="#8f1d1d",
            bbox={"facecolor": "#fff4f4", "alpha": 0.94, "edgecolor": "#c43131"},
        )


def _draw_floor(ax: plt.Axes, env: WarehouseEnvironment) -> None:
    x = np.array([[-0.5, env.width - 0.5], [-0.5, env.width - 0.5]])
    y = np.array([[-0.5, -0.5], [env.height - 0.5, env.height - 0.5]])
    z = np.zeros_like(x)
    ax.plot_surface(x, y, z, color="#ded9cc", alpha=0.72, linewidth=0)
    for x_pos in range(env.width):
        ax.plot([x_pos - 0.5, x_pos - 0.5], [-0.5, env.height - 0.5], [0.01, 0.01], color="white", alpha=0.45)
    for y_pos in range(env.height):
        ax.plot([-0.5, env.width - 0.5], [y_pos - 0.5, y_pos - 0.5], [0.01, 0.01], color="white", alpha=0.45)


def _draw_path(
    ax: plt.Axes,
    path: list[tuple[int, int]],
    color: str,
    linestyle: str,
    label: str,
    z: float = 0.1,
    linewidth: float = 2.4,
) -> None:
    if not path:
        return
    ax.plot(
        [p[0] for p in path],
        [p[1] for p in path],
        np.full(len(path), z),
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
    )


def _draw_amr(ax: plt.Axes, cell: tuple[int, int], yaw: float) -> None:
    _draw_oriented_box(ax, (cell[0], cell[1], 0.24), (0.52, 0.38, 0.26), yaw, "#0b4fa3", 0.92)
    front = np.array([math.cos(yaw), math.sin(yaw)])
    ax.quiver(cell[0], cell[1], 0.42, front[0], front[1], 0, length=0.42, color="#00a6d6", linewidth=2)
    ax.text(cell[0] + 0.18, cell[1], 0.58, "AMR", color="#0b4fa3", fontsize=9, weight="bold")


def _path_yaw(trail: list[tuple[int, int]]) -> float:
    if len(trail) < 2:
        return 0.0
    dx = trail[-1][0] - trail[-2][0]
    dy = trail[-1][1] - trail[-2][1]
    if dx == 0 and dy == 0:
        return 0.0
    return math.atan2(dy, dx)


def _draw_flow(ax: plt.Axes, row: dict[str, object]) -> None:
    ax.axis("off")
    phase = str(row["phase"])
    steps = [
        ("1 original\npolicy", True),
        ("2 approach\nobstacle", phase in {"normal", "diagnose", "recover", "success"}),
        ("3 blocked\nahead", bool(row["dynamic_obstacles"])),
        ("4 REPLAN\nroute", phase in {"diagnose", "recover", "success"}),
        ("5 detour\npath", phase in {"recover", "success"}),
        ("6 goal\nreached", bool(row["goal_reached"])),
    ]
    xs = np.linspace(0.07, 0.93, len(steps))
    for idx, ((label, active), x) in enumerate(zip(steps, xs)):
        ax.scatter(x, 0.6, s=440, color="#198754" if active else "#c8c8c8", edgecolor="white", linewidth=1.2)
        ax.text(x, 0.25, label, ha="center", va="top", fontsize=8.3)
        if idx + 1 < len(steps):
            ax.plot([x + 0.045, xs[idx + 1] - 0.045], [0.6, 0.6], color="#b5b5b5", linewidth=2)
    ax.text(0.0, 0.98, "Closed-loop mechanism chain", fontsize=12, weight="bold", va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _draw_info(ax: plt.Axes, row: dict[str, object]) -> None:
    ax.axis("off")
    lines = [
        f"time step: {row['time_step']}",
        f"phase: {row['phase']}",
        f"failure mechanism: {row['mechanism']}",
        f"router decision: {row['router_decision']}",
        f"executor action: {row['executor_action']}",
        "",
        f"risk score: {float(row['risk_score']):.2f}",
        f"path blocked score: {float(row['path_blocked_score']):.2f}",
        "",
        "Interpretation:",
        "The AMR first follows the original policy route.",
        "When the next cell is blocked, it stops before collision.",
        "The router sends REPLAN, and A* gives a detour path.",
        "",
        "Evidence level:",
        "lightweight 3D mechanism demo, not Gazebo physics.",
    ]
    ax.text(0, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10.2)


def _draw_box(
    ax: plt.Axes,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    facecolor: str,
    alpha: float,
) -> None:
    _draw_oriented_box(ax, center, size, 0.0, facecolor, alpha)


def _draw_oriented_box(
    ax: plt.Axes,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    yaw: float,
    facecolor: str,
    alpha: float,
) -> None:
    cx, cy, cz = center
    sx, sy, sz = size
    local = np.array(
        [
            [-sx / 2, -sy / 2, -sz / 2],
            [sx / 2, -sy / 2, -sz / 2],
            [sx / 2, sy / 2, -sz / 2],
            [-sx / 2, sy / 2, -sz / 2],
            [-sx / 2, -sy / 2, sz / 2],
            [sx / 2, -sy / 2, sz / 2],
            [sx / 2, sy / 2, sz / 2],
            [-sx / 2, sy / 2, sz / 2],
        ]
    )
    rot = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1],
        ]
    )
    vertices = local @ rot.T + np.array([cx, cy, cz])
    faces = [
        [vertices[i] for i in [0, 1, 2, 3]],
        [vertices[i] for i in [4, 5, 6, 7]],
        [vertices[i] for i in [0, 1, 5, 4]],
        [vertices[i] for i in [2, 3, 7, 6]],
        [vertices[i] for i in [1, 2, 6, 5]],
        [vertices[i] for i in [0, 3, 7, 4]],
    ]
    poly = Poly3DCollection(faces, facecolors=facecolor, linewidths=0.4, edgecolors="#333333", alpha=alpha)
    ax.add_collection3d(poly)


def _fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgb = frame.reshape((height, width, 4))[:, :, :3]
    plt.close(fig)
    return rgb


def _write_manifest(output_dir: Path, history: list[dict[str, object]]) -> None:
    pd.DataFrame(
        [
            {
                "artifact": "closed_loop_recovery_mechanism_3d.gif",
                "frames": len(history),
                "scenario": "external_path_blockage",
                "triggered_route": "REPLAN",
                "evidence_level": "lightweight 3D closed-loop mechanism demo",
                "description": (
                    "3D demonstration of the intended recovery chain: normal policy approach, "
                    "stop before blocked route segment, REPLAN, detour around the external obstacle, "
                    "and goal reached."
                ),
            }
        ]
    ).to_csv(output_dir / "closed_loop_recovery_mechanism_3d_manifest.csv", index=False)


if __name__ == "__main__":
    main()
