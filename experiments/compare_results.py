"""Compare baseline navigation with reliability-supervised routing."""

from __future__ import annotations

from pathlib import Path

from src.utils import ensure_output_dir, summarize_run, write_csv
from src.visualization import save_comparison_plot


def compare(
    baseline_rows: list[dict],
    supervisor_rows: list[dict],
    output_dir: str | Path = "outputs",
) -> tuple[Path, Path]:
    """Write comparison CSV and plot for two experiment logs."""

    output_dir = ensure_output_dir(output_dir)
    summary = [
        summarize_run(baseline_rows, "baseline"),
        summarize_run(supervisor_rows, "reliability_supervisor"),
    ]
    summary_csv = write_csv(summary, output_dir / "comparison_summary.csv")
    plot_path = save_comparison_plot(summary_csv, output_dir / "baseline_vs_supervisor.png")
    return summary_csv, plot_path
