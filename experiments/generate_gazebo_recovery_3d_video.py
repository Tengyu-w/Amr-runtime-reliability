"""Render a 3D Gazebo/Nav2 recovery demo from ROS 2 episode logs.

This is a presentation-oriented 3D reconstruction from a real headless
Gazebo/Nav2 run. It uses Gazebo odometry, lidar/depth observations, recovery
executor events, and Nav2 stdout evidence. It is not a lightweight grid demo.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


DEFAULT_INPUT_DIR = "outputs/gazebo_recovery_near_goal_check"
DEFAULT_OUTPUT_DIR = "visualizations/gazebo_closed_loop"
INITIAL_MAP_POSE = (-4.5, -3.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=72)
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    routed_all = pd.read_csv(input_dir / "routed_episode.csv").sort_values("time_step").reset_index(drop=True)
    recovery_all = pd.read_csv(input_dir / "recovery_execution.csv").sort_values("time_step").reset_index(drop=True)
    scan = _read_optional(input_dir / "scan_policy_observations.csv")
    depth = _read_optional(input_dir / "depth_policy_observations.csv")
    stdout = _read_text(input_dir / "launch_stdout.log")

    start_time = _first_visual_timestep(routed_all, scan, depth)
    routed = routed_all[routed_all["time_step"] >= start_time].reset_index(drop=True)
    recovery = recovery_all[recovery_all["time_step"] >= start_time].reset_index(drop=True)
    frames_df = routed.iloc[:: max(1, args.stride)].head(args.max_frames).reset_index(drop=True)
    if frames_df.empty:
        raise ValueError("No routed rows available for 3D recovery video.")

    frames = [
        _render_frame(row, routed, recovery, scan, depth, stdout, int(row["time_step"]))
        for _, row in frames_df.iterrows()
    ]
    imageio.mimsave(output_dir / "gazebo_nav2_recovery_success_3d.gif", frames, fps=args.fps)

    _write_summary(output_dir, input_dir, routed_all, recovery_all, stdout, start_time, len(frames))


def _read_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path).sort_values("time_step").reset_index(drop=True)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _first_visual_timestep(
    routed: pd.DataFrame,
    scan: pd.DataFrame | None,
    depth: pd.DataFrame | None,
) -> int:
    valid = routed[
        (routed["scenario_id"] != "unknown")
        & ((routed["target_x"].abs() + routed["target_y"].abs()) > 1e-6)
    ]
    starts = []
    if not valid.empty:
        starts.append(int(valid["time_step"].min()))
    if scan is not None and not scan.empty:
        starts.append(int(scan["time_step"].min()))
    if depth is not None and not depth.empty:
        starts.append(int(depth["time_step"].min()))
    return max(starts) if starts else int(routed["time_step"].min())


def _render_frame(
    row: pd.Series,
    routed: pd.DataFrame,
    recovery: pd.DataFrame,
    scan: pd.DataFrame | None,
    depth: pd.DataFrame | None,
    stdout: str,
    time_step: int,
) -> np.ndarray:
    history = routed[routed["time_step"] <= time_step]
    events = recovery[recovery["time_step"] <= time_step]

    fig = plt.figure(figsize=(14.0, 7.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.65, 0.9, 0.95], height_ratios=[1.0, 0.9])
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_depth = fig.add_subplot(gs[0, 1])
    ax_timeline = fig.add_subplot(gs[1, 1])
    ax_info = fig.add_subplot(gs[:, 2])

    _draw_3d_scene(ax3d, row, history, events, scan, time_step)
    _draw_depth(ax_depth, depth, time_step)
    _draw_timeline(ax_timeline, routed, recovery, time_step)
    _draw_info(ax_info, row, events, stdout)

    fig.suptitle(
        "3D Gazebo/Nav2 closed-loop recovery: visible blockage signal -> REPLAN -> goal success",
        fontsize=14,
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    return _fig_to_rgb(fig)


def _draw_3d_scene(
    ax: plt.Axes,
    row: pd.Series,
    history: pd.DataFrame,
    events: pd.DataFrame,
    scan: pd.DataFrame | None,
    time_step: int,
) -> None:
    ax.view_init(elev=28, azim=-58)
    ax.set_box_aspect((2.2, 1.4, 0.55))
    ax.set_facecolor("#f5f2ea")

    _draw_floor(ax)
    _draw_shelf(ax, (-2.5, 0.0, 0.35), (0.28, 5.5, 0.7))
    _draw_shelf(ax, (2.5, 0.0, 0.35), (0.28, 5.5, 0.7))
    ax.text(-2.65, -2.75, 0.82, "warehouse shelf boundary", color="#5f666d", fontsize=8)
    ax.text(2.08, -2.75, 0.82, "warehouse shelf boundary", color="#5f666d", fontsize=8)
    _draw_blocked_corridor(ax, row, history)

    ax.plot(history["robot_x"], history["robot_y"], np.full(len(history), 0.08), color="#0b4fa3", linewidth=3)
    goal = _goal_odom_proxy(row)
    ax.scatter([goal[0]], [goal[1]], [0.2], marker="*", s=210, color="#198754", depthshade=True)
    ax.text(goal[0], goal[1], 0.45, "Nav2 goal", color="#198754", fontsize=9)

    published = events[events["executor_action"] == "REISSUE_GOAL_FOR_NAV2_REPLAN"]
    if not published.empty:
        ax.scatter(
            published["robot_x"],
            published["robot_y"],
            np.full(len(published), 0.16),
            marker="D",
            s=48,
            color="#d95f02",
            depthshade=True,
        )

    position = (float(row["robot_x"]), float(row["robot_y"]))
    yaw = _estimate_yaw(history)
    _draw_amr(ax, position, yaw)
    _draw_lidar_rays(ax, position, yaw, scan, time_step)

    x_values = list(history["robot_x"]) + [goal[0], 0.0]
    y_values = list(history["robot_y"]) + [goal[1], 0.0]
    ax.set_xlim(min(x_values) - 0.7, max(x_values) + 0.7)
    ax.set_ylim(min(y_values) - 0.9, max(y_values) + 0.9)
    ax.set_zlim(0.0, 1.4)
    ax.set_xlabel("Gazebo odom x (m)")
    ax.set_ylabel("Gazebo odom y (m)")
    ax.set_zlabel("height (m)")
    ax.text2D(
        0.02,
        0.97,
        (
            "REPLAN trigger: visualized external blockage signal "
            f"path_blocked_score={float(row['path_blocked_score']):.2f}"
        ),
        transform=ax.transAxes,
        fontsize=10,
        weight="bold",
        color="#8f1d1d",
        bbox={"facecolor": "#fff4f4", "alpha": 0.92, "edgecolor": "#c43131"},
    )
    ax.set_title("3D warehouse scene reconstructed from Gazebo logs")


def _draw_floor(ax: plt.Axes) -> None:
    x = np.array([[-5.0, 5.0], [-5.0, 5.0]])
    y = np.array([[-3.6, -3.6], [3.6, 3.6]])
    z = np.zeros_like(x)
    ax.plot_surface(x, y, z, color="#d8d4c8", alpha=0.65, linewidth=0)


def _draw_shelf(ax: plt.Axes, center: tuple[float, float, float], size: tuple[float, float, float]) -> None:
    _add_box(ax, center, size, facecolor="#9aa0a6", alpha=0.32)


def _draw_blocked_corridor(ax: plt.Axes, row: pd.Series, history: pd.DataFrame) -> None:
    start = np.array([float(history["robot_x"].iloc[0]), float(history["robot_y"].iloc[0])])
    goal = np.array(_goal_odom_proxy(row))
    direction = goal - start
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        center = start + np.array([0.65, 0.0])
        yaw = 0.0
    else:
        unit = direction / norm
        center = start + unit * min(1.55, norm * 0.48)
        yaw = math.atan2(unit[1], unit[0])
    ax.plot(
        [start[0], goal[0]],
        [start[1], goal[1]],
        [0.05, 0.05],
        color="#6f6f6f",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
    )
    _draw_dynamic_obstacle(ax, (float(center[0]), float(center[1])))
    _add_oriented_box(
        ax,
        center=(float(center[0]), float(center[1]), 0.45),
        size=(0.18, 1.25, 0.9),
        yaw=yaw,
        facecolor="#d62728",
        alpha=0.58,
    )
    _add_oriented_box(
        ax,
        center=(float(center[0]), float(center[1]), 0.035),
        size=(0.9, 1.45, 0.05),
        yaw=yaw,
        facecolor="#d62728",
        alpha=0.18,
    )
    ax.text(
        float(center[0]) + 0.12,
        float(center[1]) + 0.12,
        1.15,
        "injected blockage volume\nREPLAN source signal",
        color="#8f1d1d",
        fontsize=9,
        weight="bold",
    )


def _draw_dynamic_obstacle(ax: plt.Axes, center: tuple[float, float]) -> None:
    theta = np.linspace(0, 2 * np.pi, 28)
    z = np.linspace(0.0, 0.7, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)
    radius = 0.25
    x = center[0] + radius * np.cos(theta_grid)
    y = center[1] + radius * np.sin(theta_grid)
    ax.plot_surface(x, y, z_grid, color="#d62728", alpha=0.55, linewidth=0)
    ax.text(center[0] + 0.28, center[1], 0.78, "blockage\nsignal", color="#8f1d1d", fontsize=8)


def _draw_amr(ax: plt.Axes, position: tuple[float, float], yaw: float) -> None:
    x, y = position
    _add_oriented_box(
        ax,
        center=(x, y, 0.20),
        size=(0.48, 0.34, 0.22),
        yaw=yaw,
        facecolor="#0b4fa3",
        alpha=0.88,
    )
    front = np.array([math.cos(yaw), math.sin(yaw)])
    ax.quiver(x, y, 0.36, front[0], front[1], 0.0, length=0.38, color="#00a6d6", linewidth=2)
    ax.text(x, y, 0.52, "AMR", color="#0b4fa3", fontsize=9, weight="bold")


def _draw_lidar_rays(
    ax: plt.Axes,
    position: tuple[float, float],
    yaw: float,
    scan: pd.DataFrame | None,
    time_step: int,
) -> None:
    if scan is None or scan.empty:
        return
    available = scan[scan["time_step"] <= time_step]
    if available.empty:
        return
    row = available.iloc[-1]
    cols = sorted(col for col in scan.columns if col.startswith("scan_bin_"))
    values = pd.to_numeric(row[cols], errors="coerce").to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=4.0, posinf=4.0, neginf=0.0)
    x, y = position
    for idx in np.linspace(0, len(values) - 1, 18, dtype=int):
        angle = yaw - np.pi + (2 * np.pi * idx / len(values))
        distance = float(np.clip(values[idx], 0.0, 3.0))
        end_x = x + distance * math.cos(angle)
        end_y = y + distance * math.sin(angle)
        color = "#d62728" if distance < 0.45 else "#63a8ff"
        alpha = 0.72 if distance < 0.45 else 0.28
        ax.plot([x, end_x], [y, end_y], [0.48, 0.48], color=color, linewidth=1.0, alpha=alpha)


def _draw_depth(ax: plt.Axes, depth: pd.DataFrame | None, time_step: int) -> None:
    ax.set_title("Depth camera grid")
    if depth is None or depth.empty:
        ax.text(0.5, 0.5, "depth unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
    available = depth[depth["time_step"] <= time_step]
    if available.empty:
        ax.text(0.5, 0.5, "depth not yet published", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
    row = available.iloc[-1]
    values = []
    for r in range(8):
        values.append([float(row[f"depth_cell_r{r:02d}_c{c:02d}"]) for c in range(12)])
    image = ax.imshow(np.array(values), cmap="magma_r", vmin=0, vmax=1.0, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _draw_timeline(ax: plt.Axes, routed: pd.DataFrame, recovery: pd.DataFrame, time_step: int) -> None:
    shown = routed[routed["time_step"] <= time_step]
    ax.plot(shown["time_step"], shown["risk_score"], color="#0b4fa3", label="risk")
    ax.plot(shown["time_step"], shown["path_blocked_score"], color="#c43131", label="blocked")
    published = recovery[
        (recovery["time_step"] <= time_step)
        & (recovery["executor_action"] == "REISSUE_GOAL_FOR_NAV2_REPLAN")
    ]
    for _, event in published.iterrows():
        ax.axvline(float(event["time_step"]), color="#d95f02", alpha=0.55, linewidth=1.2)
    ax.set_ylim(0, 1.0)
    ax.set_title("Router timeline")
    ax.set_xlabel("time step")
    ax.set_ylabel("score")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)


def _draw_info(ax: plt.Axes, row: pd.Series, events: pd.DataFrame, stdout: str) -> None:
    ax.axis("off")
    counts = Counter(events["executor_action"]) if not events.empty else Counter()
    latest = events.iloc[-1].to_dict() if not events.empty else {}
    lines = [
        f"episode: {row['episode_id']}",
        f"scenario: {row['scenario_id']}",
        f"time step: {int(row['time_step'])}",
        "",
        f"router decision: {row['router_decision']}",
        f"failure mechanism: {row['failure_mechanism']}",
        f"path blocked score: {float(row['path_blocked_score']):.2f}",
        f"obstacle proximity: {float(row['obstacle_proximity']):.2f}",
        f"latest executor: {latest.get('executor_action', 'none yet')}",
        f"latest result: {latest.get('result', '')}",
        "",
        f"published REPLAN: {counts['REISSUE_GOAL_FOR_NAV2_REPLAN']}",
        f"cooldown skips: {counts['REISSUE_GOAL_COOLDOWN']}",
        "",
        "Nav2 run totals:",
        f"goal succeeded: {stdout.count('Goal succeeded')}",
        f"goal preemptions: {stdout.count('Received goal preemption request')}",
        f"new paths: {stdout.count('Passing new path to controller')}",
        f"planner failures: {stdout.count('GridBased plugin failed to plan')}",
        "",
        "What you are seeing:",
        "3D AMR body moving in Gazebo odom,",
        "lidar rays and depth grid from sensors,",
        "orange markers where REPLAN was sent,",
        "and Nav2 goal success in stdout.",
        "The red blockage visualizes the",
        "injected external path-blockage signal;",
        "it is not a Gazebo collision model.",
    ]
    ax.text(0.0, 0.99, "\n".join(lines), va="top", ha="left", fontsize=9.6)


def _goal_odom_proxy(row: pd.Series) -> tuple[float, float]:
    return (float(row["target_x"]) - INITIAL_MAP_POSE[0], float(row["target_y"]) - INITIAL_MAP_POSE[1])


def _estimate_yaw(history: pd.DataFrame) -> float:
    if len(history) < 3:
        return 0.0
    tail = history.tail(8)
    dx = float(tail["robot_x"].iloc[-1] - tail["robot_x"].iloc[0])
    dy = float(tail["robot_y"].iloc[-1] - tail["robot_y"].iloc[0])
    if abs(dx) + abs(dy) < 1e-5:
        return 0.0
    return math.atan2(dy, dx)


def _add_box(
    ax: plt.Axes,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    facecolor: str,
    alpha: float,
) -> None:
    cx, cy, cz = center
    sx, sy, sz = size
    vertices = np.array(
        [
            [cx - sx / 2, cy - sy / 2, cz - sz / 2],
            [cx + sx / 2, cy - sy / 2, cz - sz / 2],
            [cx + sx / 2, cy + sy / 2, cz - sz / 2],
            [cx - sx / 2, cy + sy / 2, cz - sz / 2],
            [cx - sx / 2, cy - sy / 2, cz + sz / 2],
            [cx + sx / 2, cy - sy / 2, cz + sz / 2],
            [cx + sx / 2, cy + sy / 2, cz + sz / 2],
            [cx - sx / 2, cy + sy / 2, cz + sz / 2],
        ]
    )
    _add_poly_box(ax, vertices, facecolor, alpha)


def _add_oriented_box(
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
    _add_poly_box(ax, vertices, facecolor, alpha)


def _add_poly_box(ax: plt.Axes, vertices: np.ndarray, facecolor: str, alpha: float) -> None:
    faces = [
        [vertices[i] for i in [0, 1, 2, 3]],
        [vertices[i] for i in [4, 5, 6, 7]],
        [vertices[i] for i in [0, 1, 5, 4]],
        [vertices[i] for i in [2, 3, 7, 6]],
        [vertices[i] for i in [1, 2, 6, 5]],
        [vertices[i] for i in [0, 3, 7, 4]],
    ]
    poly = Poly3DCollection(faces, facecolors=facecolor, linewidths=0.5, edgecolors="#222222", alpha=alpha)
    ax.add_collection3d(poly)


def _write_summary(
    output_dir: Path,
    input_dir: Path,
    routed: pd.DataFrame,
    recovery: pd.DataFrame,
    stdout: str,
    start_time: int,
    frames: int,
) -> None:
    action_counts = Counter(recovery["executor_action"])
    rows = [
        {
            "artifact": "gazebo_nav2_recovery_success_3d.gif",
            "source_dir": str(input_dir),
            "visual_start_time_step": start_time,
            "frames": frames,
            "routed_rows": len(routed),
            "recovery_rows": len(recovery),
            "published_replan_events": action_counts["REISSUE_GOAL_FOR_NAV2_REPLAN"],
            "nav2_goal_succeeded": stdout.count("Goal succeeded"),
            "nav2_goal_preemptions": stdout.count("Received goal preemption request"),
            "nav2_new_paths_to_controller": stdout.count("Passing new path to controller"),
            "nav2_planner_failures": stdout.count("GridBased plugin failed to plan"),
            "evidence_level": "3D reconstruction from real Gazebo/Nav2 recovery-success validation logs",
        }
    ]
    pd.DataFrame(rows).to_csv(output_dir / "gazebo_nav2_recovery_success_3d_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "artifact": "gazebo_nav2_recovery_success_3d.gif",
                "frames": frames,
                "source_dir": str(input_dir),
                "description": (
                    "3D presentation video reconstructed from real ROS 2/Gazebo/Nav2 logs: "
                    "scaled AMR odometry, lidar rays, depth grid, route decisions, recovery executor, "
                    "Nav2 goal success, and a visualized injected path-blockage signal."
                ),
                "evidence_level": "3D Gazebo/Nav2 recovery-success validation visualization",
            }
        ]
    ).to_csv(output_dir / "gazebo_nav2_recovery_success_3d_manifest.csv", index=False)


def _fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgb = frame.reshape((height, width, 4))[:, :, :3]
    plt.close(fig)
    return rgb


if __name__ == "__main__":
    main()
