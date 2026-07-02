# Gazebo Policy-Monitor Matrix Results

This note summarizes the corrected ROS2/Gazebo observer-mode navigation-policy matrix.
The policy monitor observes Nav2/Gazebo runtime metrics, compares the policy action to a Nav2-plan-derived expert action when evaluable, diagnoses policy-error mechanisms, and maps them to recovery routes.

This is simulation evidence only. It is not a real-robot safety claim.

## Run

- Output directory: `AMR-Runtime-Reliability-Demo/outputs/gazebo_policy_monitor_matrix_v2`
- Scenarios: 9
- Seeds: `10,11,12,16,17,18,19`
- Episodes: 63
- Policy timestep rows: 13,760
- Policy episodes: 63
- High-confidence policy-error rows: 3,173
- Gazebo quality gate: 63/63 `ok`
- `runner_stderr.log`: empty

The earlier `gazebo_policy_monitor_matrix_v1` run was not used for conclusions because it used a weak goal-direction expert proxy and counted rows before the target was available. The corrected v2 run uses `/plan` when available and marks non-evaluable rows.

## What Changed In v2

- `navigation_policy_monitor` subscribes to Nav2 `/plan`.
- Expert action is derived from the local direction of the active Nav2 plan.
- Rows before target/plan availability are marked with `policy_evaluable=False`.
- Evidence summaries count only evaluable policy rows.

## Scenario Evidence

| Scenario | Episodes | Rows | Evaluable Rows | Policy Errors | High-Confidence Errors | Mean Error Rate | Dominant Route | Dominant Mechanism |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| localization_drift | 7 | 1,543 | 542 | 446 | 446 | 0.8197 | `RELOCALIZE` | `localization_state_error` |
| mixed_drift_and_execution | 7 | 1,531 | 530 | 413 | 413 | 0.7774 | `HUMAN_REVIEW` | `mixed_mechanism_confusion` |
| nominal | 7 | 1,530 | 535 | 409 | 409 | 0.7624 | `REPLAN` | `geometric_policy_error` |
| execution_deviation | 7 | 1,531 | 528 | 407 | 407 | 0.7648 | `REPLAN` | `control_tracking_error` |
| perception_degradation | 7 | 1,520 | 531 | 407 | 407 | 0.7597 | `HUMAN_REVIEW` | `perception_misread` |
| progress_blockage | 7 | 1,531 | 532 | 406 | 406 | 0.7615 | `REPLAN` | `geometric_policy_error` |
| boundary_weak_blockage | 7 | 1,512 | 500 | 374 | 374 | 0.7472 | `REPLAN` | `geometric_policy_error` |
| mixed_blockage_and_perception | 7 | 1,534 | 530 | 156 | 156 | 0.2938 | `HUMAN_REVIEW` | `mixed_mechanism_confusion` |
| external_path_blockage | 7 | 1,528 | 530 | 155 | 155 | 0.2966 | `REPLAN` | `blocked_path_misjudgment` |

## Test-Split Mechanism Evidence

| Scenario | Mechanism | Recovery Route | Errors | High-Confidence Error Rate |
| --- | --- | --- | ---: | ---: |
| external_path_blockage | `blocked_path_misjudgment` | `REPLAN` | 44 | 1.0000 |
| localization_drift | `localization_state_error` | `RELOCALIZE` | 125 | 1.0000 |
| perception_degradation | `perception_misread` | `HUMAN_REVIEW` | 114 | 1.0000 |
| execution_deviation | `control_tracking_error` | `REPLAN` | 113 | 1.0000 |
| mixed_blockage_and_perception | `mixed_mechanism_confusion` | `HUMAN_REVIEW` | 41 | 1.0000 |
| mixed_drift_and_execution | `mixed_mechanism_confusion` | `HUMAN_REVIEW` | 113 | 1.0000 |
| boundary_weak_blockage | `geometric_policy_error` | `REPLAN` | 105 | 1.0000 |
| nominal | `geometric_policy_error` | `REPLAN` | 118 | 1.0000 |
| progress_blockage | `geometric_policy_error` | `REPLAN` | 104 | 1.0000 |

## Test-Split Recovery Route Evidence

| Recovery Route | Policy Errors | Dominant Mechanism | Scenarios |
| --- | ---: | --- | --- |
| `HUMAN_REVIEW` | 268 | `mixed_mechanism_confusion` | `mixed_blockage_and_perception`, `mixed_drift_and_execution`, `perception_degradation` |
| `RELOCALIZE` | 125 | `localization_state_error` | `localization_drift` |
| `REPLAN` | 484 | `geometric_policy_error` | `boundary_weak_blockage`, `execution_deviation`, `external_path_blockage`, `nominal`, `progress_blockage` |

## Interpretation

The intended structure is now implemented in ROS/Gazebo observer mode:

`Gazebo/Nav2 episode -> policy action -> Nav2-plan expert comparison -> policy-error mechanism -> recovery route`

The strongest positive evidence is that different perturbation families map to different recovery routes:

- Blockage-related errors route to `REPLAN`.
- Localization drift routes to `RELOCALIZE`.
- Perception degradation and mixed mechanisms route to `HUMAN_REVIEW`.
- Execution deviation routes to `REPLAN`.

The strongest negative evidence is equally important: the current policy is a grid-world MLP exported into Gazebo. It has a major transfer mismatch. Even nominal Gazebo episodes show many high-confidence `geometric_policy_error` rows. This is not a successful Gazebo navigation policy yet; it is a useful failure case showing that the policy's learned movement representation does not align with Gazebo/Nav2 path geometry.

## Limitations

- The policy is still an observer and does not control `/cmd_vel`.
- The policy was trained on structured grid-world demonstrations, not Gazebo/Nav2 trajectory demonstrations.
- High-confidence error rates are very high because the exported policy is overconfident after domain transfer.
- The Nav2-plan expert is better than the original proxy, but still reduces continuous local trajectory to five coarse actions.
- Nominal errors mean the current policy cannot be claimed as a working navigation policy in Gazebo.

## Next Step

The next valid experiment is to train the navigation policy from Gazebo/Nav2-derived demonstrations instead of grid-world demonstrations:

1. Collect `/plan`, odometry, goal, risk metrics, and local plan direction during nominal and disturbed Gazebo episodes.
2. Train the movement policy on Gazebo-derived expert actions.
3. Re-run this matrix and require nominal error to drop before interpreting fault-specific routes.
4. Then compare policy versions: grid-world policy vs Gazebo-imitation policy vs mixed-fault-trained Gazebo policy.
