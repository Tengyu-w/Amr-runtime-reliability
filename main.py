"""Entry point for the AMR runtime reliability demo."""

from __future__ import annotations

from pathlib import Path

from experiments.compare_results import compare
from experiments.run_baseline import run as run_baseline
from experiments.run_reliability_supervisor import run as run_supervisor
from src.utils import SimulationConfig, ensure_output_dir
from src.visualization import save_risk_curve, save_warehouse_gif


def main() -> None:
    """Run baseline and reliability-supervisor demos end to end."""

    output_dir = ensure_output_dir(Path("outputs"))
    config = SimulationConfig()

    baseline_rows, _ = run_baseline(output_dir=output_dir, config=config)
    supervisor_rows, supervisor_environment = run_supervisor(output_dir=output_dir, config=config)
    summary_csv, comparison_plot = compare(baseline_rows, supervisor_rows, output_dir=output_dir)

    risk_plot = save_risk_curve(output_dir / "supervisor_log.csv", output_dir / "risk_score_curve.png")
    gif_path = save_warehouse_gif(
        supervisor_rows,
        supervisor_environment,
        output_dir / "amr_reliability_demo.gif",
    )

    print("AMR Runtime Reliability Demo complete.")
    print(f"Baseline log: {output_dir / 'baseline_log.csv'}")
    print(f"Supervisor log: {output_dir / 'supervisor_log.csv'}")
    print(f"Risk curve: {risk_plot}")
    print(f"Animation GIF: {gif_path}")
    print(f"Comparison summary: {summary_csv}")
    print(f"Comparison plot: {comparison_plot}")


if __name__ == "__main__":
    main()
