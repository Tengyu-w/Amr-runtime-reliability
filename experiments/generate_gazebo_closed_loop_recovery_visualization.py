"""Render Gazebo/Nav2 closed-loop recovery execution evidence.

The figure is reconstructed from headless Gazebo/Nav2 logs. It is not a
concept animation: every route and executor marker comes from CSV rows written
by the ROS 2 runtime pipeline.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = "outputs/gazebo_closed_loop_executor_smoke_v2"
DEFAULT_OUTPUT_DIR = "visualizations/gazebo_closed_loop"


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

    routed = pd.read_csv(input_dir / "routed_episode.csv")
    recovery = pd.read_csv(input_dir / "recovery_execution.csv")
    scan = _read_optional(input_dir / "scan_policy_observations.csv")
    stdout = _read_text(input_dir / "launch_stdout.log")

    routed = routed.sort_values("time_step").reset_index(drop=True)
    recovery = recovery.sort_values("time_step").reset_index(drop=True)
    frames_df = routed.iloc[:: max(1, args.stride)].head(args.max_frames).reset_index(drop=True)
    if frames_df.empty:
        raise ValueError("No routed episode rows available for visualization.")

    frames = [
        _render_frame(row, routed, recovery, scan, stdout, int(row["time_step"]))
        for _, row in frames_df.iterrows()
    ]
    gif_path = output_dir / "gazebo_nav2_closed_loop_recovery_execution.gif"
    imageio.mimsave(gif_path, frames, fps=args.fps)

    summary = _summary_rows(routed, recovery, stdout, input_dir)
    pd.DataFrame(summary).to_csv(output_dir / "gazebo_nav2_closed_loop_recovery_summary.csv", index=False)
    _write_manifest(output_dir, input_dir, len(frames))


def _read_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path).sort_values("time_step").reset_index(drop=True)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _render_frame(
    row: pd.Series,
    routed: pd.DataFrame,
    recovery: pd.DataFrame,
    scan: pd.DataFrame | None,
    stdout: str,
    time_step: int,
) -> np.ndarray:
    history = routed[routed["time_step"] <= time_step]
    events = recovery[recovery["time_step"] <= time_step]
    current_events = recovery[recovery["time_step"] == time_step]
    latest_event = events.iloc[-1].to_dict() if not events.empty else {}

    fig = plt.figure(figsize=(12.4, 6.7))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.45, 1.0, 0.95], height_ratios=[1.0, 0.85])
    ax_map = fig.add_subplot(grid[:, 0])
    ax_metrics = fig.add_subplot(grid[0, 1])
    ax_scan = fig.add_subplot(grid[1, 1], projection="polar")
    ax_info = fig.add_subplot(grid[:, 2])

    _draw_warehouse_axes(ax_map)
    ax_map.plot(
        history["robot_x"],
        history["robot_y"],
        color="#1f5aa6",
        linewidth=2.6,
        label="Gazebo odom trace",
    )
    ax_map.scatter(
        row["robot_x"],
        row["robot_y"],
        s=170,
        color="#0b4fa3",
        edgecolor="white",
        linewidth=1.4,
        zorder=5,
        label="AMR",
    )
    if float(row["target_x"]) or float(row["target_y"]):
        ax_map.scatter(
            row["target_x"],
            row["target_y"],
            marker="*",
            s=260,
            color="#198754",
            edgecolor="white",
            linewidth=1.1,
            zorder=6,
            label="Nav2 goal",
        )
    _draw_recovery_markers(ax_map, events)
    ax_map.set_title("Gazebo/Nav2 closed-loop recovery execution")
    ax_map.legend(loc="upper left", fontsize=8)

    _draw_metric_timeline(ax_metrics, routed, recovery, time_step)
    _draw_scan(ax_scan, scan, time_step)
    _draw_info(ax_info, row, latest_event, current_events, events, stdout)

    fig.suptitle(
        "Route decision -> recovery executor -> Nav2 goal reissue",
        fontsize=13,
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return _fig_to_rgb(fig)


def _draw_warehouse_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("#f4f1ea")
    ax.set_xlim(-5.2, 5.2)
    ax.set_ylim(-3.8, 3.8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="white", linewidth=1.0)
    ax.add_patch(plt.Rectangle((-2.7, -2.75), 0.4, 5.5, color="#54565c", alpha=0.85))
    ax.add_patch(plt.Rectangle((2.3, -2.75), 0.4, 5.5, color="#54565c", alpha=0.85))
    ax.add_patch(plt.Circle((0.0, 0.0), 0.33, color="#cf2e2e", alpha=0.7))
    ax.text(0.38, 0.18, "dynamic obstacle zone", fontsize=8, color="#8f1d1d")
    ax.set_xlabel("odom x / map x proxy (m)")
    ax.set_ylabel("odom y / map y proxy (m)")


def _draw_recovery_markers(ax: plt.Axes, events: pd.DataFrame) -> None:
    if events.empty:
        return
    published = events[events["executor_action"] == "REISSUE_GOAL_FOR_NAV2_REPLAN"]
    wait = events[events["executor_action"] == "WAIT_FOR_VALID_GOAL"]
    if not wait.empty:
        ax.scatter(
            wait["robot_x"],
            wait["robot_y"],
            marker="x",
            s=70,
            color="#a56b00",
            label="route before valid goal",
            zorder=7,
        )
    if not published.empty:
        ax.scatter(
            published["robot_x"],
            published["robot_y"],
            marker="D",
            s=72,
            color="#d94f00",
            edgecolor="white",
            linewidth=0.8,
            label="REPLAN sent to Nav2",
            zorder=8,
        )


def _draw_metric_timeline(
    ax: plt.Axes,
    routed: pd.DataFrame,
    recovery: pd.DataFrame,
    time_step: int,
) -> None:
    shown = routed[routed["time_step"] <= time_step]
    ax.plot(shown["time_step"], shown["risk_score"], color="#1f5aa6", label="risk")
    ax.plot(shown["time_step"], shown["path_blocked_score"], color="#c43131", label="blocked score")
    published = recovery[
        (recovery["time_step"] <= time_step)
        & (recovery["executor_action"] == "REISSUE_GOAL_FOR_NAV2_REPLAN")
    ]
    for _, event in published.iterrows():
        ax.axvline(float(event["time_step"]), color="#d94f00", alpha=0.5, linewidth=1.2)
    ax.set_ylim(0, 1.0)
    ax.set_title("Runtime evidence timeline")
    ax.set_xlabel("time step")
    ax.set_ylabel("score")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)


def _draw_scan(ax: plt.Axes, scan: pd.DataFrame | None, time_step: int) -> None:
    ax.set_title("Nearest lidar scan bins")
    if scan is None or scan.empty:
        ax.text(0.5, 0.5, "scan log unavailable", ha="center", va="center", transform=ax.transAxes)
        return
    available = scan[scan["time_step"] <= time_step]
    if available.empty:
        ax.text(0.5, 0.5, "scan not published yet", ha="center", va="center", transform=ax.transAxes)
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


def _draw_info(
    ax: plt.Axes,
    row: pd.Series,
    latest_event: dict,
    current_events: pd.DataFrame,
    events: pd.DataFrame,
    stdout: str,
) -> None:
    ax.axis("off")
    action_counts = Counter(events["executor_action"]) if not events.empty else Counter()
    latest_action = latest_event.get("executor_action", "none yet")
    latest_result = latest_event.get("result", "")
    current_action = (
        ", ".join(current_events["executor_action"].astype(str).tolist())
        if not current_events.empty
        else "no executor event this frame"
    )
    lines = [
        f"episode: {row['episode_id']}",
        f"scenario: {row['scenario_id']}",
        f"time step: {int(row['time_step'])}",
        "",
        f"router decision: {row['router_decision']}",
        f"failure mechanism: {row['failure_mechanism']}",
        f"risk score: {float(row['risk_score']):.3f}",
        f"path blocked: {float(row['path_blocked_score']):.2f}",
        "",
        f"current executor event: {current_action}",
        f"latest executor action: {latest_action}",
        f"latest result: {latest_result}",
        "",
        "execution counts so far:",
        f"published REPLAN: {action_counts['REISSUE_GOAL_FOR_NAV2_REPLAN']}",
        f"cooldown skips: {action_counts['REISSUE_GOAL_COOLDOWN']}",
        f"waited for goal: {action_counts['WAIT_FOR_VALID_GOAL']}",
        "",
        "Nav2 stdout evidence, run total:",
        f"goal preemptions: {stdout.count('Received goal preemption request')}",
        f"new paths to controller: {stdout.count('Passing new path to controller')}",
        f"planner failures: {stdout.count('GridBased plugin failed to plan')}",
        "",
        "interpretation:",
        "executor is connected to Nav2;",
        "this smoke does not yet prove",
        "successful recovery to the goal.",
    ]
    ax.text(0.0, 0.99, "\n".join(lines), va="top", ha="left", fontsize=9.5)


def _summary_rows(
    routed: pd.DataFrame,
    recovery: pd.DataFrame,
    stdout: str,
    input_dir: Path,
) -> list[dict[str, object]]:
    action_counts = Counter(recovery["executor_action"])
    result_counts = Counter(recovery["result"])
    return [
        {
            "artifact": "gazebo_nav2_closed_loop_recovery_execution.gif",
            "source_dir": str(input_dir),
            "routed_rows": len(routed),
            "recovery_rows": len(recovery),
            "published_replan_events": action_counts["REISSUE_GOAL_FOR_NAV2_REPLAN"],
            "cooldown_skips": action_counts["REISSUE_GOAL_COOLDOWN"],
            "wait_for_valid_goal_skips": action_counts["WAIT_FOR_VALID_GOAL"],
            "published_results": result_counts["published"],
            "skipped_results": result_counts["skipped"],
            "nav2_goal_preemptions": stdout.count("Received goal preemption request"),
            "nav2_new_paths_to_controller": stdout.count("Passing new path to controller"),
            "nav2_planner_failures": stdout.count("GridBased plugin failed to plan"),
            "claim_supported": "route decisions are translated into Nav2-facing recovery actions",
            "claim_not_supported_yet": "successful goal-reaching after recovery",
        }
    ]


def _write_manifest(output_dir: Path, input_dir: Path, frames: int) -> None:
    pd.DataFrame(
        [
            {
                "artifact": "gazebo_nav2_closed_loop_recovery_execution.gif",
                "frames": frames,
                "source_dir": str(input_dir),
                "description": (
                    "Log-derived Gazebo/Nav2 playback showing router decisions, "
                    "recovery executor events, lidar bins, and Nav2 stdout evidence."
                ),
                "evidence_level": "execution bridge smoke test; not final recovery-success benchmark",
            }
        ]
    ).to_csv(output_dir / "gazebo_closed_loop_visualization_manifest.csv", index=False)


def _fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgb = frame.reshape((height, width, 4))[:, :, :3]
    plt.close(fig)
    return rgb


if __name__ == "__main__":
    main()
