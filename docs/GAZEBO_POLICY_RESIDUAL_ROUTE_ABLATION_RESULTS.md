# Gazebo Policy Residual Mechanism, Recovery Routing, And Ablation Results

## Purpose

This document completes the first four post-policy steps:

1. residual policy error mechanism analysis;
2. recovery-route prototype;
3. ablation comparison;
4. evidence tables and figures.

The analysis uses the deterministic formal policy outputs:

- scan-only: `outputs/gazebo_scan_policy_depth_matrix_train_v2`
- depth-only: `outputs/gazebo_depth_policy_formal_train_v2`
- scan+depth fusion: `outputs/gazebo_fusion_policy_formal_train_v2`

Output evidence package:

- `outputs/gazebo_policy_residual_routes_v1`

## Step 1: Residual Error Mechanisms

High-confidence threshold: `confidence >= 0.90`.

Dominant high-confidence test residuals:

| modality/model | scenario | confusion | mechanism | route | count |
|---|---|---|---|---|---:|
| scan baseline | perception_degradation | SOUTH -> EAST | perception_axis_confusion | CAUTIOUS_REPLAN | 43 |
| scan focal | perception_degradation | SOUTH -> EAST | perception_axis_confusion | CAUTIOUS_REPLAN | 32 |
| fusion focal | perception_degradation | SOUTH -> EAST | perception_axis_confusion | CAUTIOUS_REPLAN | 30 |
| fusion baseline | perception_degradation | SOUTH -> EAST | perception_axis_confusion | CAUTIOUS_REPLAN | 25 |
| depth baseline | perception_degradation | SOUTH -> EAST | perception_axis_confusion | CAUTIOUS_REPLAN | 24 |
| depth baseline | external_path_blockage | EAST -> SOUTH | blocked_path_high_conf_direction_error | REPLAN | 10 |
| depth baseline | external_path_blockage | WEST -> SOUTH | blocked_path_high_conf_direction_error | REPLAN | 9 |
| scan baseline | external_path_blockage | EAST -> SOUTH | blocked_path_high_conf_direction_error | REPLAN | 9 |
| scan baseline | external_path_blockage | WEST -> SOUTH | blocked_path_high_conf_direction_error | REPLAN | 9 |

Interpretation:

- The strongest policy residual is not random error. It concentrates in perception degradation as a high-confidence axis confusion, especially `SOUTH -> EAST`.
- External path blockage produces a different mechanism: high-confidence direction mistakes routed to replanning.
- Nominal `NORTH/WEST` confusions are smaller and look more like boundary-direction uncertainty.

## Step 2: Recovery Route Prototype

Prototype route mapping:

| residual mechanism | route |
|---|---|
| perception_axis_confusion | CAUTIOUS_REPLAN |
| perception_lateral_depth_confusion | CAUTIOUS_REPLAN |
| perception_degradation_confusion | CAUTIOUS_REPLAN |
| blocked_path_high_conf_direction_error | REPLAN |
| blocked_path_direction_error | REPLAN |
| localization_state_error | RELOCALIZE |
| boundary_direction_confusion | CAUTIOUS_MODE |
| geometric_policy_residual | HUMAN_REVIEW |

Route coverage over high-confidence residuals:

| modality/model | split | high-conf errors | actionable route coverage |
|---|---|---:|---:|
| depth baseline | test | 62 | 1.000 |
| fusion baseline | test | 64 | 0.984 |
| fusion focal | test | 52 | 1.000 |
| scan baseline | test | 78 | 1.000 |
| scan focal | test | 51 | 1.000 |

Interpretation:

- The route prototype covers nearly all high-confidence test errors with specific, non-generic routes.
- Coverage alone is not enough; the important part is that perception degradation and blockage now go to different recovery families.

## Step 3: Ablation

Held-out test comparison versus scan baseline:

| modality/model | accuracy | macro F1 | weighted F1 | ECE | high-conf errors | key delta vs scan baseline |
|---|---:|---:|---:|---:|---:|---|
| depth baseline | 0.8773 | 0.6666 | 0.8819 | 0.0553 | 62 | better accuracy, weighted F1, ECE, and fewer high-conf errors |
| scan baseline | 0.8694 | 0.6741 | 0.8776 | 0.0778 | 78 | reference |
| fusion baseline | 0.8641 | 0.6559 | 0.8673 | 0.0721 | 64 | fewer high-conf errors than scan, but worse accuracy/F1 |
| fusion focal | 0.8111 | 0.6318 | 0.8265 | 0.0549 | 52 | fewer high-conf errors, but much worse accuracy |
| scan focal | 0.8076 | 0.6343 | 0.8235 | 0.0617 | 51 | fewer high-conf errors, but much worse accuracy |

Interpretation:

- Depth baseline is the current best overall test baseline.
- Scan baseline keeps slightly better macro F1, so it still helps minority-class balance.
- Simple feature-level fusion is not yet a successful upgrade.
- Focal loss reduces high-confidence errors but trades away too much accuracy.

## Step 4: Evidence Artifacts

Generated tables:

- `unified_policy_scores.csv`
- `modality_ablation_metrics.csv`
- `test_ablation_delta_vs_scan_baseline.csv`
- `scenario_error_summary.csv`
- `residual_mechanism_summary.csv`
- `high_conf_error_patterns.csv`
- `recovery_route_evidence.csv`
- `recovery_route_coverage.csv`

Generated figures:

- `figures/test_accuracy_by_modality.png`
- `figures/test_high_conf_errors_by_modality.png`
- `figures/test_recovery_route_distribution.png`

These artifacts are suitable for a supervisor update or doctoral-application project slide, with the limitation that they are simulation-derived.

## Confirmed Findings

- The learned policy has identifiable residual mechanisms, not just undifferentiated mistakes.
- Perception degradation and external blockage produce different high-confidence error families.
- Depth is useful and should remain in the project.
- Simple scan+depth concatenation is not enough; fusion needs mechanism-aware gating.
- Recovery routes can now be grounded in residual policy evidence.

## Limitations

- Still simulation-derived, not real-robot validation.
- One seed per split, so this is first formal evidence rather than final statistical proof.
- `STAY` is not a Nav2-plan movement label and should remain a recovery/safety-supervisor action.
- Recovery routes are prototype labels on residual errors; they are not yet deployed into a closed-loop controller.

## Next Step

Move from route evidence to route evaluation:

1. implement a learned or rule-based route selector using the residual mechanism labels;
2. evaluate whether routes capture high-confidence policy errors without over-routing correct rows;
3. add localization-drift episodes so `RELOCALIZE` has direct evidence;
4. prepare the project narrative document after the route selector evidence is complete.
