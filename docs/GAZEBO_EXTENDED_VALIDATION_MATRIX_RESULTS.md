# Gazebo Extended Validation Matrix Results

This result note summarizes the mixed/boundary Gazebo validation run.
It is research-prototype evidence, not a real-robot or operational safety claim.

## Run

- Output directory: `AMR-Runtime-Reliability-Demo/outputs/gazebo_extended_validation_matrix_v1`
- Scenarios: `nominal`, `external_path_blockage`, `localization_drift`, `perception_degradation`, `execution_deviation`, `progress_blockage`, `mixed_blockage_and_perception`, `mixed_drift_and_execution`, `boundary_weak_blockage`
- Seeds: `10,11,12,16,17,18,19`
- Episodes: 63
- Timesteps: 14,469
- Quality gate: 63/63 `ok`
- Gazebo/Nav2 checks: all episodes spawned an entity, began navigation, and had zero rejected goals.

## Route Model Evidence

The route model is a two-layer PyTorch MLP trained on runtime features only.
It excludes scenario id, fault origin, fault family, and OOD labels.

| Split | Rows | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Train | 6,128 | 1.0000 | 1.0000 |
| Val | 4,098 | 1.0000 | 1.0000 |
| Test | 4,243 | 1.0000 | 1.0000 |

Test split by fault origin was also 1.0000 accuracy / 1.0000 macro F1 for execution error, external disturbance, mixed external+perception, mixed state+execution, nominal, perception degradation, and state-estimation drift.

## Ablation Evidence

| Ablation | Test Accuracy | Test Macro F1 |
| --- | ---: | ---: |
| full | 1.0000 | 1.0000 |
| risk_only | 0.9491 | 0.9294 |
| no_risk_score | 1.0000 | 1.0000 |
| no_localization | 1.0000 | 1.0000 |
| no_perception | 1.0000 | 1.0000 |
| no_blockage | 1.0000 | 1.0000 |
| no_execution | 1.0000 | 1.0000 |

Interpretation: the full route model learns the current synthetic routes easily. `risk_only` is weaker, but single-channel removals do not hurt because the current feature channels are still highly separable and partly redundant.

## Outcome Evidence

Episode-level outcome proxies were derived from logged telemetry:

- `expected_route_observed`: expected recovery route appeared at least once.
- `recovery_latency_steps`: rows from first non-nominal mechanism to first expected route.
- `goal_reached_proxy`: final robot-target distance within the current tolerance.
- `collision_risk_proxy`: high obstacle proximity and high trajectory deviation co-occurred.

| Scenario | Episodes | Outcome Label | Mean Non-Nominal Steps | Mean Recovery Latency | Mean Final Risk |
| --- | ---: | --- | ---: | ---: | ---: |
| nominal | 7 | `nominal_stable` | 0.0 | n/a | 0.0710 |
| external_path_blockage | 7 | `routed_and_recovered_proxy` | 223.9 | 0.0 | 0.2756 |
| localization_drift | 7 | `routed_and_recovered_proxy` | 179.4 | 0.0 | 0.2330 |
| perception_degradation | 7 | `routed_and_recovered_proxy` | 176.7 | 0.0 | 0.1926 |
| execution_deviation | 7 | `routed_and_recovered_proxy` | 173.3 | 0.0 | 0.2592 |
| progress_blockage | 7 | `routed_and_recovered_proxy` | 74.1 | 0.0 | 0.0960 |
| boundary_weak_blockage | 7 | `routed_and_recovered_proxy` | 38.0 | 0.0 | 0.0972 |
| mixed_drift_and_execution | 7 | `routed_and_recovered_proxy` | 229.9 | 0.0 | 0.3118 |
| mixed_blockage_and_perception | 7 | `routed_but_unresolved_proxy` | 225.7 | 49.6 | 0.4552 |

`mixed_blockage_and_perception` is the main unresolved stress case. It does trigger the expected review route, but late, and the final risk remains high.

## Limitations

- Perfect route metrics should not be overclaimed; current Gazebo faults are controlled and separable.
- `goal_reached_proxy` is 0.0 across scenarios in this run, so this matrix supports route-trigger evidence more than task-completion evidence.
- Recovery latency is a row-count proxy, not wall-clock recovery time.
- There is no real-robot validation, collision geometry audit, or external environment validation.

## Next Validation Step

The next step should make outcomes stricter and more meaningful:

- Increase episode timeout or adjust goal/tolerance so `goal_reached_proxy` can discriminate completion.
- Add mixed faults with overlapping feature signatures where single-channel ablations should actually degrade.
- Add wall-clock recovery latency and Nav2 result status if available.
- Train an outcome predictor separately from the route classifier, so the project can distinguish "route selected" from "recovery succeeded."
