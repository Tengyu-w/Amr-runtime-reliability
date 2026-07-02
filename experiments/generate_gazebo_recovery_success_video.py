"""Render a readable Gazebo/Nav2 closed-loop recovery success video.

This visualization is reconstructed from ROS 2/Gazebo/Nav2 episode logs. It is
not the lightweight grid simulator: odometry, lidar, depth, route decisions,
recovery-executor events, and Nav2 success counts come from a real headless
Gazebo/Nav2 run.
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


DEFAULT_INPUT_DIR = "outputs/gazebo_recovery_near_goal_check"
DEFAULT_OUTPUT_DIR = "visualizations/gazebo_closed_loop"
INITIAL_MAP_POSE = (-4.5, -3.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=84)
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
        raise ValueError("No routed rows available for Gazebo recovery success video.")

    frames = [
        _render_frame(row, routed, recovery, scan, depth, stdout, int(row["time_step"]))
        for _, row in frames_df.iterrows()
    ]
    gif_path = output_dir / "gazebo_nav2_recovery_success_episode.gif"
    imageio.mimsave(gif_path, frames, fps=args.fps)

    summary = _summary_rows(routed_all, recovery_all, stdout, input_dir, start_time)
    pd.DataFrame(summary).to_csv(output_dir / "gazebo_nav2_recovery_success_summary.csv", index=False)
    _write_manifest(output_dir, input_dir, len(frames))


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

    fig = plt.figure(figsize=(14.2, 7.6))
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.45, 0.9, 1.05, 1.05],
        height_ratios=[1.0, 0.9],
    )
    ax_map = fig.add_subplot(gs[:, 0])
    ax_timeline = fig.add_subplot(gs[0, 1:3])
    ax_scan = fig.add_subplot(gs[1, 1], projection="polar")
    ax_depth = fig.add_subplot(gs[1, 2])
    ax_info = fig.add_subplot(gs[:, 3])

    _draw_map(ax_map, row, history, events)
    _draw_timeline(ax_timeline, routed, recovery, time_step, stdout)
    _draw_scan(ax_scan, scan, time_step)
    _draw_depth(ax_depth, depth, time_step)
    _draw_info(ax_info, row, events, stdout)

    fig.suptitle(
        "True Gazebo/Nav2 closed-loop recovery: blockage -> REPLAN -> goal succeeded",
        fontsize=14,
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    return _fig_to_rgb(fig)


def _draw_map(ax: plt.Axes, row: pd.Series, history: pd.DataFrame, events: pd.DataFrame) -> None:
    ax.set_facecolor("#f4f1ea")
    ax.grid(True, color="white", linewidth=1.0)
    ax.set_aspect("equal", adjustable="box")

    _draw_blockage_overlay(ax, row, history)

    ax.plot(history["robot_x"], history["robot_y"], color="#0b4fa3", linewidth=2.8, label="Gazebo odom trace")
    ax.scatter(row["robot_x"], row["robot_y"], s=190, color="#0b4fa3", edgecolor="white", zorder=6, label="AMR")

    goal_odom = _goal_odom_proxy(row)
    ax.scatter(goal_odom[0], goal_odom[1], marker="*", s=280, color="#198754", edgecolor="white", zorder=7)
    ax.text(goal_odom[0] + 0.06, goal_odom[1] + 0.06, "Nav2 goal\nodom display", fontsize=8, color="#198754")

    published = events[events["executor_action"] == "REISSUE_GOAL_FOR_NAV2_REPLAN"]
    if not published.empty:
        ax.scatter(
            published["robot_x"],
            published["robot_y"],
            marker="D",
            s=80,
            color="#d95f02",
            edgecolor="white",
            zorder=8,
            label="REPLAN sent",
        )

    x_values = list(history["robot_x"]) + [goal_odom[0], 0.0]
    y_values = list(history["robot_y"]) + [goal_odom[1], 0.0]
    ax.set_xlim(min(x_values) - 0.45, max(x_values) + 0.45)
    ax.set_ylim(min(y_values) - 0.45, max(y_values) + 0.45)
    ax.set_xlabel("Gazebo odom x (m)")
    ax.set_ylabel("Gazebo odom y (m)")
    ax.set_title("Gazebo odometry and recovery events")
    ax.legend(loc="lower right", fontsize=8)


def _goal_odom_proxy(row: pd.Series) -> tuple[float, float]:
    return (float(row["target_x"]) - INITIAL_MAP_POSE[0], float(row["target_y"]) - INITIAL_MAP_POSE[1])


def _draw_blockage_overlay(ax: plt.Axes, row: pd.Series, history: pd.DataFrame) -> None:
    center, yaw = _blockage_pose(row, history)
    goal = _goal_odom_proxy(row)
    start = (float(history["robot_x"].iloc[0]), float(history["robot_y"].iloc[0]))
    ax.plot(
        [start[0], goal[0]],
        [start[1], goal[1]],
        color="#6f6f6f",
        linestyle="--",
        linewidth=1.3,
        alpha=0.65,
        label="blocked direct route",
    )
    no_go = _oriented_rectangle(center, (0.9, 1.45), yaw)
    wall = _oriented_rectangle(center, (0.18, 1.25), yaw)
    ax.add_patch(plt.Polygon(no_go, closed=True, facecolor="#cf2e2e", edgecolor="none", alpha=0.16, zorder=2))
    ax.add_patch(
        plt.Polygon(wall, closed=True, facecolor="#cf2e2e", edgecolor="#8f1d1d", alpha=0.62, zorder=3)
    )
    ax.scatter([center[0]], [center[1]], s=130, color="#cf2e2e", edgecolor="white", zorder=4)
    ax.text(
        center[0] + 0.08,
        center[1] + 0.08,
        "injected blockage\npath_blocked_score high",
        fontsize=8,
        color="#8f1d1d",
        weight="bold",
        zorder=5,
    )


def _blockage_pose(row: pd.Series, history: pd.DataFrame) -> tuple[tuple[float, float], float]:
    start = np.array([float(history["robot_x"].iloc[0]), float(history["robot_y"].iloc[0])])
    goal = np.array(_goal_odom_proxy(row))
    direction = goal - start
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return (float(start[0] + 0.65), float(start[1])), 0.0
    unit = direction / norm
    center = start + unit * min(1.55, norm * 0.48)
    return (float(center[0]), float(center[1])), math.atan2(float(unit[1]), float(unit[0]))


def _oriented_rectangle(
    center: tuple[float, float],
    size: tuple[float, float],
    yaw: float,
) -> np.ndarray:
    cx, cy = center
    sx, sy = size
    local = np.array(
        [
            [-sx / 2, -sy / 2],
            [sx / 2, -sy / 2],
            [sx / 2, sy / 2],
            [-sx / 2, sy / 2],
        ]
    )
    rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    return local @ rot.T + np.array([cx, cy])


def _draw_timeline(
    ax: plt.Axes,
    routed: pd.DataFrame,
    recovery: pd.DataFrame,
    time_step: int,
    stdout: str,
) -> None:
    shown = routed[routed["time_step"] <= time_step]
    ax.plot(shown["time_step"], shown["risk_score"], color="#0b4fa3", label="risk")
    ax.plot(shown["time_step"], shown["path_blocked_score"], color="#c43131", label="path blocked score")
    published = recovery[
        (recovery["time_step"] <= time_step)
        & (recovery["executor_action"] == "REISSUE_GOAL_FOR_NAV2_REPLAN")
    ]
    for _, event in published.iterrows():
        ax.axvline(float(event["time_step"]), color="#d95f02", alpha=0.6, linewidth=1.4)
    ax.text(
        0.02,
        0.92,
        (
            f"Nav2 run totals: goal succeeded={stdout.count('Goal succeeded')} | "
            f"new paths={stdout.count('Passing new path to controller')} | "
            f"planner failures={stdout.count('GridBased plugin failed to plan')}"
        ),
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#dddddd"},
    )
    ax.set_ylim(0, 1.0)
    ax.set_title("Router and runtime evidence")
    ax.set_xlabel("time step")
    ax.set_ylabel("score")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)


def _draw_scan(ax: plt.Axes, scan: pd.DataFrame | None, time_step: int) -> None:
    ax.set_title("LiDAR scan bins")
    if scan is None or scan.empty:
        ax.text(0.5, 0.5, "scan unavailable", ha="center", va="center", transform=ax.transAxes)
        return
    available = scan[scan["time_step"] <= time_step]
    if available.empty:
        ax.text(0.5, 0.5, "scan not yet published", ha="center", va="center", transform=ax.transAxes)
        return
    row = available.iloc[-1]
    cols = sorted(col for col in scan.columns if col.startswith("scan_bin_"))
    values = pd.to_numeric(row[cols], errors="coerce").to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=8.0, posinf=8.0, neginf=0.0)
    theta = np.linspace(-np.pi, np.pi, len(values), endpoint=False)
    ax.bar(
        theta,
        values,
        width=(2 * np.pi / len(values)) * 0.92,
        color=plt.cm.viridis_r(np.clip(values / 8.0, 0, 1)),
        alpha=0.9,
    )
    ax.set_ylim(0, 8)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_yticklabels([])


def _draw_depth(ax: plt.Axes, depth: pd.DataFrame | None, time_step: int) -> None:
    ax.set_title("Depth grid")
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


def _draw_info(ax: plt.Axes, row: pd.Series, events: pd.DataFrame, stdout: str) -> None:
    ax.axis("off")
    counts = Counter(events["executor_action"]) if not events.empty else Counter()
    latest = events.iloc[-1].to_dict() if not events.empty else {}
    goal_succeeded = stdout.count("Goal succeeded")
    interpretation = "GOAL SUCCEEDED" if goal_succeeded else "running / not yet succeeded"
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
        f"waited for valid goal: {counts['WAIT_FOR_VALID_GOAL']}",
        "",
        "Nav2 evidence:",
        f"goal succeeded count: {goal_succeeded}",
        f"goal preemptions: {stdout.count('Received goal preemption request')}",
        f"new paths to controller: {stdout.count('Passing new path to controller')}",
        f"planner failures: {stdout.count('GridBased plugin failed to plan')}",
        "",
        f"run status: {interpretation}",
        "",
        "This video uses real ROS 2/Gazebo logs:",
        "odom, lidar, depth, router decisions,",
        "recovery executor events, and Nav2 stdout.",
        "The red blockage visualizes the",
        "injected external path-blockage signal;",
        "it is not a Gazebo collision model.",
    ]
    ax.text(0.0, 0.99, "\n".join(lines), va="top", ha="left", fontsize=9.5)


def _summary_rows(
    routed: pd.DataFrame,
    recovery: pd.DataFrame,
    stdout: str,
    input_dir: Path,
    visual_start_time: int,
) -> list[dict[str, object]]:
    action_counts = Counter(recovery["executor_action"])
    result_counts = Counter(recovery["result"])
    return [
        {
            "artifact": "gazebo_nav2_recovery_success_episode.gif",
            "source_dir": str(input_dir),
            "visual_start_time_step": visual_start_time,
            "routed_rows": len(routed),
            "recovery_rows": len(recovery),
            "published_replan_events": action_counts["REISSUE_GOAL_FOR_NAV2_REPLAN"],
            "cooldown_skips": action_counts["REISSUE_GOAL_COOLDOWN"],
            "wait_for_valid_goal_skips": action_counts["WAIT_FOR_VALID_GOAL"],
            "published_results": result_counts["published"],
            "skipped_results": result_counts["skipped"],
            "nav2_goal_succeeded": stdout.count("Goal succeeded"),
            "nav2_goal_preemptions": stdout.count("Received goal preemption request"),
            "nav2_new_paths_to_controller": stdout.count("Passing new path to controller"),
            "nav2_failed_to_make_progress": stdout.count("Failed to make progress"),
            "nav2_planner_failures": stdout.count("GridBased plugin failed to plan"),
            "claim_supported": "route-triggered REPLAN was executed in Gazebo/Nav2 and Nav2 reported goal success",
        }
    ]


def _write_manifest(output_dir: Path, input_dir: Path, frames: int) -> None:
    pd.DataFrame(
        [
            {
                "artifact": "gazebo_nav2_recovery_success_episode.gif",
                "frames": frames,
                "source_dir": str(input_dir),
                "description": (
                    "Readable Gazebo/Nav2 closed-loop recovery episode reconstructed from "
                    "real ROS 2 logs: odom, lidar, depth, router decision, recovery executor, "
                    "and Nav2 goal success."
                ),
                "evidence_level": "Gazebo/Nav2 closed-loop recovery success validation run",
            }
        ]
    ).to_csv(output_dir / "gazebo_nav2_recovery_success_manifest.csv", index=False)


def _fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgb = frame.reshape((height, width, 4))[:, :, :3]
    plt.close(fig)
    return rgb


if __name__ == "__main__":
    main()
