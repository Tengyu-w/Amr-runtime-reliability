"""Run the AMR demo with only baseline navigation behavior."""

from __future__ import annotations

from pathlib import Path

from src.amr_agent import AMRAgent
from src.decision_router import RouterDecision
from src.environment import WarehouseEnvironment
from src.failure_injection import FailureInjector, summarize_event_metadata
from src.planner import AStarPlanner
from src.reliability_supervisor import ReliabilitySupervisor
from src.utils import (
    SimulationConfig,
    encode_position,
    encode_positions,
    ensure_output_dir,
    write_csv,
)


def run(
    output_dir: str | Path = "outputs",
    config: SimulationConfig | None = None,
    injector: FailureInjector | None = None,
) -> tuple[list[dict], WarehouseEnvironment]:
    """Run a baseline AMR simulation without reliability-aware routing."""

    config = config or SimulationConfig()
    output_dir = ensure_output_dir(output_dir)
    environment = WarehouseEnvironment(config=config)
    planner = AStarPlanner(environment)
    agent = AMRAgent(position=environment.start, target=environment.target)
    injector = injector or FailureInjector(seed=config.seed)
    supervisor = ReliabilitySupervisor()

    agent.set_path(planner.plan(agent.position, agent.target))
    rows: list[dict] = []

    for time_step in range(config.max_steps):
        events = injector.apply(time_step, environment, agent)

        if agent.needs_replan or agent.next_path_cell() is None or _next_cell_blocked(environment, agent):
            if not injector.should_force_replan_failure(time_step):
                new_path = planner.plan(agent.position, agent.target)
                if new_path:
                    agent.set_path(new_path)
                    agent.replanning_failure_count = 0
                else:
                    agent.replanning_failure_count += 1
            else:
                agent.replanning_failure_count += 1

        moved = agent.move_along_path(environment, allow_move=True)
        metrics = supervisor.evaluate(environment, agent)
        decision = RouterDecision.NORMAL_NAVIGATION.value
        task_status = "completed" if agent.completed else "running"
        if agent.blocked_move_count >= 8 or agent.replanning_failure_count >= 5:
            task_status = "failed_baseline_stuck"

        rows.append(
            {
                "time_step": time_step,
                "robot_position": encode_position(agent.position),
                "target_position": encode_position(agent.target),
                "risk_score": metrics.risk_score,
                "localization_uncertainty": metrics.localization_uncertainty,
                "sensor_confidence": metrics.sensor_confidence,
                "path_blocked_score": metrics.path_blocked_score,
                "obstacle_proximity": metrics.obstacle_proximity,
                "trajectory_deviation": metrics.trajectory_deviation,
                "replanning_failure_count": metrics.replanning_failure_count,
                "task_progress_stagnation": metrics.task_progress_stagnation,
                "router_decision": decision,
                "failure_event": "|".join(events) if events else "none",
                **summarize_event_metadata(events),
                "task_status": task_status,
                "speed_state": "normal",
                "moved": moved,
                "dynamic_obstacles": encode_positions(environment.dynamic_obstacles),
                "path": encode_positions(agent.path),
            }
        )

        if agent.completed or task_status == "failed_baseline_stuck":
            break

    write_csv(rows, output_dir / "baseline_log.csv")
    return rows, environment


def _next_cell_blocked(environment: WarehouseEnvironment, agent: AMRAgent) -> bool:
    """Return True when the next planned cell is blocked."""

    next_cell = agent.next_path_cell()
    return next_cell is not None and environment.is_blocked(next_cell)


if __name__ == "__main__":
    run()
