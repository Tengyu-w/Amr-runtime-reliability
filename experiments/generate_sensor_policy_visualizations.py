"""Generate sensor-policy playback GIFs from recorded Gazebo episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SCAN = (
    "outputs/gazebo_scan_policy_formal_v1/scan_policy_episode_logs/"
    "gazebo_external_path_blockage__goal_east_south__seed_18.scan_policy.csv"
)
DEFAULT_DEPTH = (
    "outputs/gazebo_depth_policy_formal_v1/depth_policy_episode_logs/"
    "gazebo_external_path_blockage__goal_east_south__seed_18.depth_policy.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-csv", default=DEFAULT_SCAN)
    parser.add_argument("--depth-csv", default=DEFAULT_DEPTH)
    parser.add_argument("--output-dir", default="visualizations/sensor_policy")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--fps", type=int, default=4)
    args = parser.parse_args()

    scan_df = pd.read_csv(args.scan_csv)
    depth_df = pd.read_csv(args.depth_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = _merge_sensor_tables(scan_df, depth_df, args.stride, args.max_frames)

    _save_gif(
        [_render_scan_frame(row) for _, row in merged.iterrows()],
        output_dir / "gazebo_lidar_scan_policy_episode.gif",
        args.fps,
    )
    _save_gif(
        [_render_depth_frame(row) for _, row in merged.iterrows()],
        output_dir / "gazebo_depth_grid_policy_episode.gif",
        args.fps,
    )
    _save_gif(
        [_render_multimodal_frame(row) for _, row in merged.iterrows()],
        output_dir / "gazebo_scan_depth_policy_episode.gif",
        args.fps,
    )

    manifest = pd.DataFrame(
        [
            {
                "artifact": "gazebo_lidar_scan_policy_episode.gif",
                "source_scan_csv": args.scan_csv,
                "source_depth_csv": "",
                "scenario": str(merged["scenario_id"].iloc[0]),
                "frames": len(merged),
                "description": "Polar lidar-bin playback with expert/predicted action and recovery route.",
            },
            {
                "artifact": "gazebo_depth_grid_policy_episode.gif",
                "source_scan_csv": "",
                "source_depth_csv": args.depth_csv,
                "scenario": str(merged["scenario_id"].iloc[0]),
                "frames": len(merged),
                "description": "Depth-grid heatmap playback with policy decision metadata.",
            },
            {
                "artifact": "gazebo_scan_depth_policy_episode.gif",
                "source_scan_csv": args.scan_csv,
                "source_depth_csv": args.depth_csv,
                "scenario": str(merged["scenario_id"].iloc[0]),
                "frames": len(merged),
                "description": "Combined scan, depth, and policy reliability playback.",
            },
        ]
    )
    manifest.to_csv(output_dir / "sensor_policy_visualization_manifest.csv", index=False)


def _merge_sensor_tables(
    scan_df: pd.DataFrame,
    depth_df: pd.DataFrame,
    stride: int,
    max_frames: int,
) -> pd.DataFrame:
    shared = [
        "episode_id",
        "scenario_id",
        "time_step",
        "robot_x",
        "robot_y",
        "target_x",
        "target_y",
        "goal_dx",
        "goal_dy",
        "risk_score",
        "expert_proxy_action",
        "policy_pred_action",
        "policy_correct",
        "policy_max_prob",
        "policy_error_mechanism",
        "policy_recovery_route",
    ]
    scan_cols = [c for c in scan_df.columns if c.startswith("scan_bin_")]
    depth_cols = [c for c in depth_df.columns if c.startswith("depth_cell_")]
    left = scan_df[shared + ["scan_front_min_range", "scan_valid_fraction"] + scan_cols]
    right = depth_df[
        ["time_step", "depth_valid_fraction", "depth_center_min_m", "depth_min_m", "depth_mean_m"]
        + depth_cols
    ]
    merged = left.merge(right, on="time_step", how="inner").sort_values("time_step")
    merged = merged.iloc[:: max(1, stride)].head(max_frames).reset_index(drop=True)
    if merged.empty:
        raise ValueError("No overlapping timesteps between scan and depth CSV files.")
    return merged


def _render_scan_frame(row: pd.Series) -> np.ndarray:
    fig = plt.figure(figsize=(7.6, 5.2))
    ax = fig.add_subplot(111, projection="polar")
    scan = _scan_values(row)
    theta = np.linspace(-np.pi, np.pi, len(scan), endpoint=False)
    colors = plt.cm.viridis_r(np.clip(scan / 8.0, 0, 1))
    ax.bar(theta, scan, width=(2 * np.pi / len(scan)) * 0.92, color=colors, alpha=0.9)
    ax.set_ylim(0, 8)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_yticklabels([])
    ax.set_title(_title(row, "Lidar scan policy playback"), pad=18)
    _add_footer(fig, row)
    return _fig_to_rgb(fig)


def _render_depth_frame(row: pd.Series) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    depth = _depth_grid(row)
    image = ax.imshow(depth, cmap="magma_r", vmin=0, vmax=8, aspect="auto")
    ax.set_title(_title(row, "Depth grid policy playback"), pad=12)
    ax.set_xlabel("image columns")
    ax.set_ylabel("image rows")
    ax.set_xticks(range(depth.shape[1]))
    ax.set_yticks(range(depth.shape[0]))
    ax.tick_params(labelsize=7)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("distance (m)")
    _add_footer(fig, row)
    return _fig_to_rgb(fig)


def _render_multimodal_frame(row: pd.Series) -> np.ndarray:
    fig = plt.figure(figsize=(11.5, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.15, 0.9])

    scan_ax = fig.add_subplot(gs[0, 0], projection="polar")
    scan = _scan_values(row)
    theta = np.linspace(-np.pi, np.pi, len(scan), endpoint=False)
    scan_ax.bar(
        theta,
        scan,
        width=(2 * np.pi / len(scan)) * 0.92,
        color=plt.cm.viridis_r(np.clip(scan / 8.0, 0, 1)),
        alpha=0.9,
    )
    scan_ax.set_ylim(0, 8)
    scan_ax.set_theta_zero_location("N")
    scan_ax.set_theta_direction(-1)
    scan_ax.set_yticklabels([])
    scan_ax.set_title("lidar bins")

    depth_ax = fig.add_subplot(gs[0, 1])
    depth = _depth_grid(row)
    depth_ax.imshow(depth, cmap="magma_r", vmin=0, vmax=8, aspect="auto")
    depth_ax.set_title("depth grid")
    depth_ax.set_xticks([])
    depth_ax.set_yticks([])

    info_ax = fig.add_subplot(gs[0, 2])
    info_ax.axis("off")
    lines = [
        f"scenario: {row['scenario_id']}",
        f"time step: {int(row['time_step'])}",
        f"robot: ({float(row['robot_x']):.2f}, {float(row['robot_y']):.2f})",
        f"target: ({float(row['target_x']):.2f}, {float(row['target_y']):.2f})",
        "",
        f"expert: {row['expert_proxy_action']}",
        f"policy: {row['policy_pred_action']}",
        f"correct: {row['policy_correct']}",
        f"confidence: {float(row['policy_max_prob']):.2f}",
        f"risk: {float(row['risk_score']):.2f}",
        "",
        f"mechanism: {_clean(row['policy_error_mechanism'])}",
        f"route: {_clean(row['policy_recovery_route'])}",
        "",
        f"front scan min: {float(row['scan_front_min_range']):.2f} m",
        f"depth center min: {float(row['depth_center_min_m']):.2f} m",
    ]
    info_ax.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10)
    fig.suptitle("Gazebo scan + depth + policy reliability playback", fontsize=13)
    fig.tight_layout()
    return _fig_to_rgb(fig)


def _scan_values(row: pd.Series) -> np.ndarray:
    cols = sorted(c for c in row.index if c.startswith("scan_bin_"))
    values = pd.to_numeric(row[cols], errors="coerce").to_numpy(dtype=float)
    return np.nan_to_num(values, nan=8.0, posinf=8.0, neginf=0.0)


def _depth_grid(row: pd.Series) -> np.ndarray:
    values = []
    for r in range(8):
        line = []
        for c in range(12):
            value = pd.to_numeric(row[f"depth_cell_r{r:02d}_c{c:02d}"], errors="coerce")
            line.append(float(value) if np.isfinite(value) else 8.0)
        values.append(line)
    return np.array(values)


def _title(row: pd.Series, prefix: str) -> str:
    return (
        f"{prefix}\n"
        f"t={int(row['time_step'])} | expert={row['expert_proxy_action']} | "
        f"policy={row['policy_pred_action']} | route={_clean(row['policy_recovery_route'])}"
    )


def _add_footer(fig: plt.Figure, row: pd.Series) -> None:
    footer = (
        f"scenario={row['scenario_id']} | risk={float(row['risk_score']):.2f} | "
        f"confidence={float(row['policy_max_prob']):.2f} | "
        f"mechanism={_clean(row['policy_error_mechanism'])}"
    )
    fig.text(0.02, 0.02, footer, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))


def _clean(value: object) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return "none"
    return str(value)


def _fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgb = frame.reshape((height, width, 4))[:, :, :3]
    plt.close(fig)
    return rgb


def _save_gif(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    imageio.mimsave(output_path, frames, fps=fps)


if __name__ == "__main__":
    main()
