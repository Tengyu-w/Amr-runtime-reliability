# Simulation-Grounded Reliability Routing For AMR Navigation Policies

## One-Paragraph Summary

This project studies how an autonomous mobile robot policy can fail under warehouse-style disturbances, and how those failures can be routed to different recovery mechanisms. The core idea is adapted from an ECG reliability pipeline: first train a task policy from data, then analyze the policy's residual errors, then use the model's own failure patterns to build recovery routes. In this AMR prototype, Gazebo and Nav2 generate expert-labeled navigation episodes. The policy learns from simulated sensor observations, including planar lidar scans and depth images, to predict the next navigation action. After training, the project analyzes high-confidence policy errors under external blockage and perception degradation, then maps different error mechanisms to different recovery routes such as cautious replanning, replanning, relocalization, and human review.

## Research Motivation

Warehouse AMRs do not fail in only one way. A robot may make a wrong movement decision because the path is blocked, because perception is degraded, because localization drifts, or because the policy is uncertain near a directional boundary. Treating all failures as the same kind of error is weak: it leads to a single generic fallback rather than a targeted recovery mechanism.

This project asks:

> Can we train a navigation policy from simulation-derived expert labels, inspect its residual errors, and use those error mechanisms to build evidence-based recovery routes?

The project is not a real-robot deployment claim. It is a research prototype showing a full mechanism chain:

```text
Gazebo/Nav2 simulation
-> scan/depth observations
-> expert action labels
-> supervised navigation policy
-> residual error analysis
-> recovery-route prototype
-> ablation and evidence tables
```

## System Overview

The robot is simulated in Gazebo in a warehouse-like room. Nav2 provides navigation behavior and expert plans. Sensor observations are recorded alongside the expert action at each timestep.

Main components:

- Gazebo AMR simulation with moving/blocking obstacle scenarios.
- Nav2 planner and controller as the expert source.
- Planar lidar `/scan` observations.
- Forward depth image `/depth_image` observations.
- Policy monitors and recorders that align sensor observations with Nav2-plan action labels.
- Supervised policies trained to predict discrete movement actions.
- Residual-error analysis and recovery-route evidence tables.

The learned action space is:

```text
NORTH, SOUTH, EAST, WEST
```

`STAY` is intentionally not treated as a normal Nav2-plan movement label. It is better interpreted as a safety-supervisor or recovery action, because Nav2 `/plan` usually provides a movement direction rather than a stop label.

## Why Simulation Is Valid Here

There is no real robot dataset in this project. Instead, Gazebo provides synthetic but structured sensor data. This is not the same as real-world validation, but it is appropriate for a research prototype because it gives controlled access to:

- known disturbance types;
- repeatable seeds;
- synchronized sensor streams;
- expert labels from Nav2 plans;
- scenario-level ground truth about blockage or sensor degradation.

The key claim is therefore limited:

> The project demonstrates a simulation-derived policy reliability pipeline, not real-world AMR safety validation.

## Policy Learning Setup

The policy is trained by supervised learning, not reinforcement learning.

Each training row contains:

- robot pose and goal context;
- risk and disturbance signals;
- sensor observations;
- expert action label from Nav2 `/plan`.

Three policy input variants are compared:

| policy | input |
|---|---|
| scan-only | planar lidar scan bins + goal/context |
| depth-only | depth image grid + goal/context |
| scan+depth fusion | lidar bins + depth grid + goal/context |

The depth image is compressed into an interpretable grid of distance cells. This keeps the model explainable for a research presentation: it is not just a black-box image model, but a policy that sees where nearby structures appear in the depth field.

## Experimental Design

The formal depth/fusion matrix used:

| item | value |
|---|---|
| simulator | Gazebo + Nav2 |
| total episodes | 36 / 36 successful |
| seeds | 10, 16, 18 |
| split rule | seed 10 train, seed 16 validation, seed 18 test |
| goals | east_south, west_near, north_near, south_axis |
| scenarios | nominal, external_path_blockage, perception_degradation |
| labels | Nav2-plan expert actions |

This gives a controlled first validation, but still only one seed per split. The result is useful evidence, not final statistical proof.

## Main Ablation Result

Held-out test performance:

| modality/model | accuracy | macro F1 | weighted F1 | ECE | high-confidence errors |
|---|---:|---:|---:|---:|---:|
| depth baseline | 0.8773 | 0.6666 | 0.8819 | 0.0553 | 62 |
| scan baseline | 0.8694 | 0.6741 | 0.8776 | 0.0778 | 78 |
| scan+depth baseline | 0.8641 | 0.6559 | 0.8673 | 0.0721 | 64 |
| scan focal | 0.8076 | 0.6343 | 0.8235 | 0.0617 | 51 |
| scan+depth focal | 0.8111 | 0.6318 | 0.8265 | 0.0549 | 52 |

Key interpretation:

- Depth is useful. It gives the best baseline test accuracy, weighted F1, calibration error, and fewer high-confidence errors than scan-only.
- Scan-only still has the best macro F1 among baseline models, suggesting it remains useful for minority-class balance.
- Simple scan+depth concatenation is not enough. Fusion does not yet improve the policy reliably.
- Focal loss reduces high-confidence errors, but the cost in accuracy is too large to call it a general upgrade.

## Residual Error Mechanisms

The most important result is not just which model is most accurate. The project identifies how the policy fails.

Dominant high-confidence residuals on the test split:

| modality/model | scenario | error | mechanism | route | count |
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

Two major mechanisms emerge:

1. Perception degradation mainly causes axis confusion, especially `SOUTH -> EAST`.
2. External blockage causes high-confidence direction mistakes that should be routed to replanning.

This is the point of the project: different errors are not treated identically.

## Recovery Route Prototype

The recovery route is built from policy residual evidence.

| residual mechanism | proposed route |
|---|---|
| perception_axis_confusion | CAUTIOUS_REPLAN |
| perception_lateral_depth_confusion | CAUTIOUS_REPLAN |
| perception_degradation_confusion | CAUTIOUS_REPLAN |
| blocked_path_high_conf_direction_error | REPLAN |
| blocked_path_direction_error | REPLAN |
| localization_state_error | RELOCALIZE |
| boundary_direction_confusion | CAUTIOUS_MODE |
| geometric_policy_residual | HUMAN_REVIEW |

High-confidence test error coverage:

| modality/model | high-confidence errors | actionable route coverage |
|---|---:|---:|
| depth baseline | 62 | 1.000 |
| fusion baseline | 64 | 0.984 |
| fusion focal | 52 | 1.000 |
| scan baseline | 78 | 1.000 |
| scan focal | 51 | 1.000 |

This does not prove closed-loop recovery success yet. It shows that the residual errors can be assigned to specific recovery families instead of being collapsed into one generic fallback.

## Research Contribution

The project contributes a complete research prototype with four linked parts:

1. A simulated AMR reliability environment with external disturbances.
2. A supervised navigation policy trained from Gazebo/Nav2 expert labels.
3. A scan/depth sensor ablation showing that depth adds useful reliability evidence.
4. A residual-error routing method that maps policy failure mechanisms to recovery routes.

The project's strongest conceptual point is the migration of an ECG-style reliability mechanism into robotics:

```text
train policy first
-> inspect residual model failures
-> identify mechanism-specific failure patterns
-> build recovery routes from evidence
```

## What Is Shown

The evidence shows that:

- a policy can be trained from simulation-derived scan/depth observations and Nav2 expert labels;
- depth observations improve several held-out test metrics relative to scan-only;
- the learned policy has structured high-confidence residual errors;
- perception degradation and path blockage produce different residual mechanisms;
- recovery routes can be assigned based on those mechanisms.

## What Is Suggested

The results suggest that depth should remain a core modality in the project. They also suggest that the next fusion model should be mechanism-aware. Simple concatenation does not work well enough. A better design would likely use a gating mechanism:

```text
if perception confidence is low:
    trust depth/risk evidence more
if blockage is high:
    route to replanning
if localization uncertainty is high:
    route to relocalization
```

## What Is Not Yet Proven

The project does not yet prove:

- real-world AMR reliability;
- closed-loop recovery success after a route is triggered;
- statistically robust multi-seed performance;
- a final best multi-modal fusion architecture;
- relocalization behavior, because localization-drift evidence still needs to be expanded in the scan/depth formal run.

## Recommended Next Experiment

The next experiment should evaluate the route selector itself.

Proposed next step:

1. Add localization-drift episodes to the scan/depth formal matrix.
2. Implement a route selector that predicts `CAUTIOUS_REPLAN`, `REPLAN`, `RELOCALIZE`, `CAUTIOUS_MODE`, or `HUMAN_REVIEW`.
3. Evaluate two things:
   - how many high-confidence policy errors it captures;
   - how often it over-routes correct policy decisions.
4. Compare route selector variants:
   - rule-based mechanism route;
   - learned route classifier;
   - uncertainty-threshold route;
   - mechanism-aware gated fusion route.

## Suggested Presentation Framing

The project can be presented as:

> A simulation-grounded reliability-routing framework for warehouse AMR policies. Instead of treating policy errors as a single failure category, the system trains a navigation policy from expert-labeled Gazebo episodes, analyzes high-confidence residual errors under perception and blockage disturbances, and maps different error mechanisms to different recovery routes.

Shorter version:

> I built a Gazebo/Nav2 AMR prototype that learns navigation actions from scan and depth observations, then uses the policy's own residual errors to build evidence-based recovery routes.

## Key Files

Formal result documents:

- `docs/GAZEBO_SCAN_POLICY_FORMAL_V1_RESULTS.md`
- `docs/GAZEBO_DEPTH_FUSION_FORMAL_V1_RESULTS.md`
- `docs/GAZEBO_POLICY_RESIDUAL_ROUTE_ABLATION_RESULTS.md`

Main outputs:

- `outputs/gazebo_depth_policy_formal_v1`
- `outputs/gazebo_scan_policy_depth_matrix_train_v2`
- `outputs/gazebo_depth_policy_formal_train_v2`
- `outputs/gazebo_fusion_policy_formal_train_v2`
- `outputs/gazebo_policy_residual_routes_v1`

Core scripts:

- `experiments/train_gazebo_scan_policy.py`
- `experiments/train_gazebo_depth_policy.py`
- `experiments/train_gazebo_fusion_policy.py`
- `experiments/analyze_policy_residual_routes.py`

## Final Takeaway

This is now a coherent doctoral-application research prototype rather than a collection of scripts. It has a simulator, data generation, supervised policy learning, sensor ablation, residual mechanism analysis, and a first recovery-routing design. Its current strongest claim is not that the robot is solved, but that the project demonstrates a defensible method for moving from policy errors to mechanism-specific recovery routes under controlled AMR disturbances.
