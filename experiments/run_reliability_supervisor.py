"""Run the AMR demo with reliability supervision enabled."""

from __future__ import annotations

from pathlib import Path

from src.amr_agent import AMRAgent
from src.decision_router import (
    DecisionRouter,
    MechanismAwareDecisionRouter,
    RouterDecision,
    ScalarRiskRouter,
    diagnose_failure_mechanism,
)
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


def _build_router(router_mode: str):
    if router_mode == "mechanism_router":
        return MechanismAwareDecisionRouter()
    if router_mode == "risk_router":
        return ScalarRiskRouter()
    if router_mode == "rule_router":
        return DecisionRouter()
    raise ValueError(f"Unknown router_mode: {router_mode}")


def run(
    output_dir: str | Path = "outputs",
    config: SimulationConfig | None = None,
    router_mode: str = "mechanism_router",
    injector: FailureInjector | None = None,
) -> tuple[list[dict], WarehouseEnvironment]:
    """Run the reliability-supervised AMR simulation."""

    config = config or SimulationConfig()
    output_dir = ensure_output_dir(output_dir)
    environment = WarehouseEnvironment(config=config)
    planner = AStarPlanner(environment)
    agent = AMRAgent(position=environment.start, target=environment.target)
    injector = injector or FailureInjector(seed=config.seed)
    supervisor = ReliabilitySupervisor()
    router = _build_router(router_mode)

    initial_path = planner.plan(agent.position, agent.target)
    agent.set_path(initial_path)
    rows: list[dict] = []

    for time_step in range(config.max_steps):
        events = injector.apply(time_step, environment, agent)
        metrics = supervisor.evaluate(environment, agent)
        mechanism = diagnose_failure_mechanism(metrics)
        decision = router.route(metrics)

        moved = False
        if decision == RouterDecision.REPLAN:
            if injector.should_force_replan_failure(time_step):
                agent.replanning_failure_count += 1
            else:
                new_path = planner.plan(agent.position, agent.target)
                if new_path:
                    agent.set_path(new_path)
                    agent.replanning_failure_count = 0
                else:
                    agent.replanning_failure_count += 1
            metrics = supervisor.evaluate(environment, agent)
        elif decision == RouterDecision.RELOCALIZE:
            agent.relocalize()
            metrics = supervisor.evaluate(environment, agent)
        elif decision == RouterDecision.HUMAN_REVIEW:
            agent.speed_state = "paused_for_review"
        elif decision == RouterDecision.SAFE_STOP:
            agent.speed_state = "safe_stop"

        if decision in {
            RouterDecision.NORMAL_NAVIGATION,
            RouterDecision.REPLAN,
            RouterDecision.RELOCALIZE,
        }:
            agent.speed_state = "normal"
            moved = agent.move_along_path(environment, allow_move=True)
        elif decision == RouterDecision.CAUTIOUS_MODE:
            agent.speed_state = "cautious"
            allow_move = time_step % config.cautious_move_interval == 0
            moved = agent.move_along_path(environment, allow_move=allow_move)
        else:
            agent.move_along_path(environment, allow_move=False)

        task_status = "completed" if agent.completed else "running"
        if decision == RouterDecision.SAFE_STOP:
            task_status = "safe_stopped"

        row = _make_row(
            time_step=time_step,
            agent=agent,
            environment=environment,
            metrics=metrics,
            mechanism=mechanism.value,
            decision=decision.value,
            router_mode=router_mode,
            events=events,
            task_status=task_status,
            moved=moved,
        )
        rows.append(row)

        if agent.completed or decision == RouterDecision.SAFE_STOP:
            break

    write_csv(rows, output_dir / "supervisor_log.csv")
    return rows, environment


def _make_row(
    time_step: int,
    agent: AMRAgent,
    environment: WarehouseEnvironment,
    metrics,
    mechanism: str,
    decision: str,
    router_mode: str,
    events: list[str],
    task_status: str,
    moved: bool,
) -> dict:
    """Create one CSV-ready log row."""

    return {
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
        "failure_mechanism": mechanism,
        "router_decision": decision,
        "router_mode": router_mode,
        "failure_event": "|".join(events) if events else "none",
        **summarize_event_metadata(events),
        "task_status": task_status,
        "speed_state": agent.speed_state,
        "moved": moved,
        "dynamic_obstacles": encode_positions(environment.dynamic_obstacles),
        "path": encode_positions(agent.path),
    }


if __name__ == "__main__":
    run()
