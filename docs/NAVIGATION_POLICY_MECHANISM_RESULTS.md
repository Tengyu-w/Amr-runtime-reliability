# Navigation Policy Mechanism Results

This note records the first ECG-style policy-mechanism pipeline for the AMR demo.
The core model is no longer a recovery router. The core model is a task policy that chooses the next movement action.

This is simulation evidence only. It is not a real-robot safety claim.

## Objective

The experiment asks:

1. Can a neural navigation policy learn ordinary movement decisions?
2. Where does that policy make wrong movement decisions under perturbation?
3. Are the wrong decisions linked to identifiable mechanisms?
4. Can those mechanisms be mapped to recovery routes?

## Pipeline

1. Use A* as an expert to label movement actions: `STAY`, `NORTH`, `SOUTH`, `EAST`, `WEST`.
2. Train a two-layer MLP navigation policy from observed robot state, target state, local free-space indicators, and runtime risk features.
3. Roll the policy out closed-loop under the same scenario catalog.
4. Compare the policy action against the expert action at each step.
5. Diagnose policy-error mechanisms after the error occurs.
6. Map diagnosed mechanisms to recovery routes.

The policy does not receive `scenario_id`, fault origin, fault family, or OOD labels as inputs. Those columns are used only for audit grouping.

## Outputs

Output directory:

`AMR-Runtime-Reliability-Demo/outputs/navigation_policy_mechanism_v1`

Main files:

- `navigation_policy_demonstrations.csv`
- `navigation_policy_supervised_scores.csv`
- `navigation_policy_supervised_metrics.csv`
- `navigation_policy_rollout_scores.csv`
- `navigation_policy_rollout_metrics.csv`
- `policy_error_mechanism_evidence.csv`
- `policy_error_recovery_routes.csv`
- `navigation_policy_mechanism_report.json`

## Dataset

- Seeds: `10,11,12,16,17,18,19`
- Scenarios: 9
- Expert demonstration rows: 1,596
- Closed-loop rollout rows: 2,156

Training scenarios excluded the mixed/boundary stress cases, so those cases test generalization rather than memorization.

## Closed-Loop Policy Evidence

| Scenario | Rollout Rows | Policy Errors | Error Rate | High-Confidence Errors | Blocked Attempts |
| --- | ---: | ---: | ---: | ---: | ---: |
| nominal | 168 | 0 | 0.0000 | 0 | 0 |
| external_path_blockage | 168 | 0 | 0.0000 | 0 | 0 |
| localization_drift | 168 | 0 | 0.0000 | 0 | 0 |
| perception_degradation | 168 | 0 | 0.0000 | 0 | 0 |
| execution_deviation | 168 | 0 | 0.0000 | 0 | 0 |
| progress_blockage | 168 | 0 | 0.0000 | 0 | 0 |
| mixed_drift_and_execution | 168 | 0 | 0.0000 | 0 | 0 |
| boundary_weak_blockage | 490 | 371 | 0.7571 | 343 | 371 |
| mixed_blockage_and_perception | 490 | 434 | 0.8857 | 427 | 413 |

The important failure pattern is not just that the policy is wrong. It is often confidently wrong.

## Mechanism Evidence

| Split | Scenario | Mechanism | Recovery Route | Errors | High-Confidence Error Rate | Blocked Move Rate |
| --- | --- | --- | --- | ---: | ---: | ---: |
| test | `boundary_weak_blockage` | `blocked_path_misjudgment` | `REPLAN` | 106 | 0.9245 | 1.0000 |
| test | `mixed_blockage_and_perception` | `blocked_path_misjudgment` | `REPLAN` | 40 | 1.0000 | 0.9000 |
| test | `mixed_blockage_and_perception` | `mixed_mechanism_confusion` | `HUMAN_REVIEW` | 84 | 0.9762 | 0.9762 |

The recovery route is assigned after policy errors are observed and diagnosed:

- `blocked_path_misjudgment` -> `REPLAN`
- `mixed_mechanism_confusion` -> `HUMAN_REVIEW`
- `localization_state_error` -> `RELOCALIZE`
- `perception_misread` -> `HUMAN_REVIEW`
- `control_tracking_error` -> `REPLAN`
- `policy_boundary_uncertainty` -> `CAUTIOUS_MODE`

## Interpretation

The previous route model showed that runtime features can classify recovery routes. This new policy experiment is different: it creates a task policy, lets the policy make movement decisions, and then analyzes the policy's actual mistakes.

The strongest evidence is in two stress cases:

- `boundary_weak_blockage`: the policy repeatedly attempts blocked moves. The mechanism is a path-blockage misjudgment, so the recovery route is `REPLAN`.
- `mixed_blockage_and_perception`: the policy often combines blocked-path errors with degraded perception. The dominant mixed mechanism routes to `HUMAN_REVIEW`, while pure blockage errors route to `REPLAN`.

This is closer to the intended ECG-style structure:

`policy prediction -> policy error -> mechanism analysis -> mechanism-conditioned recovery route`

## Limitations

- This is a grid-world policy, not yet a Gazebo controller replacing Nav2.
- The policy uses structured state features, not raw LiDAR/camera observations.
- Localization/perception mechanisms are still proxy variables, not learned latent causal factors.
- Current errors are concentrated in blockage/mixed-perception cases; more scenarios are needed for localization-only, target-shift, and execution-control failures.
- High confidence is currently measured by softmax probability; calibration and ensemble uncertainty should be added before making stronger uncertainty claims.

## Next Step

The next implementation step should make this policy layer control a simulated robot only after more observer-mode evidence:

1. Compare the observer-mode policy action against a stronger online expert action.
2. Replace the current expert proxy with a Nav2 path-derived local action when available.
3. Use policy error mechanisms, not hand-written scenario labels, to trigger `REPLAN`, `RELOCALIZE`, `HUMAN_REVIEW`, `CAUTIOUS_MODE`, or `SAFE_STOP`.
4. Only after observer validation, test policy-in-the-loop control in Gazebo.

## Gazebo Observer Integration

The policy mechanism layer has now been connected to the ROS2/Gazebo pipeline in observer mode.
It does not publish `/cmd_vel` and does not take over Nav2 control.

Added ROS nodes:

- `navigation_policy_monitor`: subscribes to `/amr_reliability/runtime_metrics`, loads a JSON-exported navigation-policy MLP, predicts movement action, compares it to an online expert proxy, diagnoses policy-error mechanism, and publishes `/amr_reliability/policy_decision`.
- `policy_episode_recorder`: records `/amr_reliability/policy_decision` to a separate `*.policy.csv` file.

Model export:

- Training script output: `navigation_policy_model_export.json`
- ROS package copy: `ros2_ws/src/amr_reliability_benchmark/config/navigation_policy_model.json`

Gazebo smoke run:

- Output directory: `AMR-Runtime-Reliability-Demo/outputs/gazebo_policy_monitor_smoke`
- Scenario: `mixed_blockage_and_perception`
- Seed: `10`
- Router rows: 218
- Policy rows: 218
- Gazebo quality gate: `ok`

Observer-mode policy route counts:

| Recovery Route | Rows |
| --- | ---: |
| `REPLAN` | 45 |
| `HUMAN_REVIEW` | 173 |

Observer-mode mechanism counts:

| Policy Error Mechanism | Rows |
| --- | ---: |
| `geometric_policy_error` | 1 |
| `blocked_path_misjudgment` | 44 |
| `mixed_mechanism_confusion` | 173 |

This is the first ROS/Gazebo version of the intended structure:

`Gazebo/Nav2 episode -> observed policy action -> policy error mechanism -> recovery route`

Current caveat: the online expert is still a proxy based on goal direction and blockage risk. The next evidence upgrade is to derive the expert action from the active Nav2 path or local planner trajectory.
