from experiments.run_multiseed_reliability import (
    _aggregate,
    _fault_origin_route_rows,
    _mechanism_route_rows,
    _paired_delta_rows,
    _summary_for_rows,
)
from src.failure_injection import summarize_event_metadata


def _row(risk: float, decision: str, event: str = "none", status: str = "running") -> dict:
    return {
        "risk_score": risk,
        "router_decision": decision,
        "failure_event": event,
        "task_status": status,
    }


def test_summary_counts_event_recovery_rate() -> None:
    rows = [
        _row(0.10, "NORMAL_NAVIGATION"),
        _row(0.40, "CAUTIOUS_MODE", "sensor_confidence_drop"),
        _row(0.70, "REPLAN", "dynamic_obstacle_blocks_path", "completed"),
    ]

    summary = _summary_for_rows(rows, mode="mechanism_router", seed=7, risk_threshold=0.35)

    assert summary["seed"] == 7
    assert summary["failure_event_steps"] == 2
    assert summary["non_nominal_action_count"] == 2
    assert summary["hard_recovery_action_count"] == 1
    assert summary["high_risk_event_rate"] == 1.0
    assert summary["non_nominal_on_event_rate"] == 1.0
    assert summary["hard_recovery_on_event_rate"] == 0.5


def test_paired_delta_and_aggregate_rows() -> None:
    seed_rows = [
        {
            "seed": 7,
            "mode": "baseline",
            "success": False,
            "steps": 10,
            "safe_stop_count": 0,
            "human_review_count": 0,
            "replan_count": 0,
            "non_nominal_action_count": 0,
            "hard_recovery_action_count": 0,
            "high_risk_event_rate": 0.5,
            "non_nominal_on_event_rate": 0.0,
            "hard_recovery_on_event_rate": 0.0,
            "max_risk": 0.6,
            "mean_risk": 0.3,
        },
        {
            "seed": 7,
            "mode": "mechanism_router",
            "success": True,
            "steps": 12,
            "safe_stop_count": 0,
            "human_review_count": 1,
            "replan_count": 2,
            "non_nominal_action_count": 4,
            "hard_recovery_action_count": 3,
            "high_risk_event_rate": 0.75,
            "non_nominal_on_event_rate": 0.5,
            "hard_recovery_on_event_rate": 0.25,
            "max_risk": 0.7,
            "mean_risk": 0.35,
        },
    ]

    deltas = _paired_delta_rows(seed_rows)
    aggregate = _aggregate(seed_rows, deltas)

    assert deltas[0]["mode"] == "mechanism_router"
    assert deltas[0]["success_delta_vs_baseline"] == 1
    assert deltas[0]["non_nominal_action_delta_vs_baseline"] == 4
    assert deltas[0]["hard_recovery_action_delta_vs_baseline"] == 3
    assert any(row["group"] == "paired_delta_mechanism_router" for row in aggregate)


def test_mechanism_route_rows_group_by_mechanism_and_action() -> None:
    rows = [
        {
            "failure_mechanism": "path_blocked",
            "router_decision": "REPLAN",
            "failure_event": "dynamic_obstacle_blocks_path",
            "risk_score": 0.6,
        },
        {
            "failure_mechanism": "path_blocked",
            "router_decision": "REPLAN",
            "failure_event": "none",
            "risk_score": 0.4,
        },
    ]

    grouped = _mechanism_route_rows(seed=7, mode="mechanism_router", rows=rows)

    assert grouped == [
        {
            "seed": 7,
            "mode": "mechanism_router",
            "failure_mechanism": "path_blocked",
            "router_decision": "REPLAN",
            "n_steps": 2,
            "n_failure_event_steps": 1,
            "mean_risk": 0.5,
        }
    ]


def test_event_metadata_separates_external_disturbance_from_ood_style_shift() -> None:
    external = summarize_event_metadata(["dynamic_obstacle_blocks_path"])
    shifted = summarize_event_metadata(["localization_drift_increasing", "target_changed"])

    assert external["fault_origin"] == "external_disturbance"
    assert external["primary_fault_origin"] == "external_disturbance"
    assert external["ood_status"] == "in_distribution_fault"
    assert external["has_ood_style_shift"] is False
    assert "task_or_goal_shift" in str(shifted["fault_origin"])
    assert shifted["primary_fault_event"] == "target_changed"
    assert shifted["primary_fault_origin"] == "task_or_goal_shift"
    assert shifted["primary_ood_status"] == "ood_style_shift"
    assert "ood_style_shift" in str(shifted["ood_status"])
    assert shifted["has_ood_style_shift"] is True


def test_fault_origin_route_rows_group_by_source_family_and_action() -> None:
    rows = [
        {
            "failure_event": "dynamic_obstacle_blocks_path",
            "fault_origin": "external_disturbance",
            "fault_family": "path_blockage",
            "ood_status": "in_distribution_fault",
            "router_decision": "REPLAN",
            "risk_score": 0.6,
        },
        {
            "failure_event": "sensor_confidence_drop",
            "fault_origin": "perception_degradation",
            "fault_family": "sensor_quality",
            "ood_status": "in_distribution_fault",
            "router_decision": "HUMAN_REVIEW",
            "risk_score": 0.7,
        },
    ]

    grouped = _fault_origin_route_rows(seed=7, mode="mechanism_router", rows=rows)

    assert grouped == [
        {
            "seed": 7,
            "mode": "mechanism_router",
            "fault_origin": "external_disturbance",
            "fault_family": "path_blockage",
            "ood_status": "in_distribution_fault",
            "router_decision": "REPLAN",
            "n_failure_event_steps": 1,
            "mean_risk": 0.6,
        },
        {
            "seed": 7,
            "mode": "mechanism_router",
            "fault_origin": "perception_degradation",
            "fault_family": "sensor_quality",
            "ood_status": "in_distribution_fault",
            "router_decision": "HUMAN_REVIEW",
            "n_failure_event_steps": 1,
            "mean_risk": 0.7,
        },
    ]
