# Mechanism-Aware AMR Runtime Reliability Routing

This repository studies runtime reliability for warehouse autonomous mobile
robots (AMRs) under simulated disturbances. The central problem is not ordinary
path planning. The project asks why a navigation policy makes wrong movement
decisions under blockage, perception degradation, and localization-style
uncertainty, and how those residual policy errors can be routed to different
recovery mechanisms.

The project is inspired by a mechanism-aware ECG uncertainty pipeline:

```text
train a task model
  -> inspect its residual errors
  -> identify structured failure mechanisms
  -> build evidence-based recovery routes
```

For AMR navigation, the corresponding chain is:

```text
Gazebo/Nav2 simulation
  -> scan/depth observations
  -> Nav2-plan expert action labels
  -> supervised navigation policy
  -> high-confidence residual error analysis
  -> recovery-route prototype
```

> Research prototype only. This repository is not a certified robot safety
> system, is not validated on a real robot, and should not be used for
> deployment or hardware control without separate engineering validation.

## For Supervisors: Read These 4 Files First

This repository preserves the full research trail. For a fast review, start
here:

1. [Project narrative](docs/AMR_RELIABILITY_PROJECT_NARRATIVE.md):
   the complete story from simulation data to policy learning and recovery
   routing.
2. [Chinese project narrative](docs/AMR_RELIABILITY_PROJECT_NARRATIVE_CN.md):
   a Chinese thesis/interview-facing version of the same argument.
3. [Depth/fusion formal results](docs/GAZEBO_DEPTH_FUSION_FORMAL_V1_RESULTS.md):
   the scan, depth, and scan+depth policy comparison.
4. [Residual route ablation results](docs/GAZEBO_POLICY_RESIDUAL_ROUTE_ABLATION_RESULTS.md):
   the evidence connecting policy residual errors to recovery-route families.

The curated visual gallery is here:

- [visualizations/README.md](visualizations/README.md)

## Current Status

The current repository contains a complete simulation-grounded research
prototype:

- a lightweight warehouse AMR reliability demo;
- a ROS 2 / Gazebo / Nav2 benchmark scaffold;
- lidar scan and depth-image observation recorders;
- supervised policy training from Nav2 expert labels;
- scan-only, depth-only, and scan+depth fusion policy ablations;
- high-confidence residual error analysis;
- recovery-route evidence tables and visualizations.

The formal Gazebo/Nav2 scan-depth matrix completed 36/36 episodes across:

| Item | Setting |
| --- | --- |
| Simulator | Gazebo + Nav2 |
| Episodes | 36 / 36 successful |
| Seeds | 10, 16, 18 |
| Split rule | seed 10 train, seed 16 validation, seed 18 test |
| Goals | `east_south`, `west_near`, `north_near`, `south_axis` |
| Scenarios | `nominal`, `external_path_blockage`, `perception_degradation` |
| Labels | Nav2-plan expert actions |

The evidence is useful as a research prototype, but it is not real-world AMR
validation.

## Research Story

The repository follows a mechanism-first sequence:

```text
1. Runtime reliability demo
   -> show that navigation can fail for different reasons, not only because
      the planner has no path

2. Professional simulation scaffold
   -> move from a toy grid demo to Gazebo/Nav2 so disturbances, plans, sensor
      streams, and robot state can be recorded systematically

3. Expert-labeled policy dataset
   -> use Nav2 /plan as the expert source and train a policy from observations
      to movement actions

4. Sensor-policy ablation
   -> compare scan-only, depth-only, and scan+depth fusion policies under the
      same held-out simulation split

5. Residual mechanism analysis
   -> inspect high-confidence policy errors and identify structured failure
      mechanisms instead of treating every mistake as one generic failure

6. Recovery-route prototype
   -> map different residual mechanisms to different recovery families such as
      cautious replanning, replanning, relocalization, cautious mode, and human
      review
```

The contribution is therefore the **policy-residual-to-recovery-route evidence
chain**:

```text
simulated disturbance
  -> learned policy decision
  -> residual high-confidence error
  -> mechanism label
  -> recovery route
```

## 1. Defining The Problem: AMR Failures Are Not One Failure Type

An AMR may fail because the path is physically blocked, because perception is
degraded, because localization is unreliable, because it is near a directional
decision boundary, or because replanning repeatedly fails. A single fallback is
therefore too coarse.

The first lightweight demo tests this idea in a 2D warehouse grid. The purpose
is not to claim that the grid is a realistic robot simulator. The purpose is to
make the runtime reliability interface explicit:

| Component | Why introduced | Output |
| --- | --- | --- |
| `WarehouseEnvironment` | Provide a controllable warehouse layout. | Robot, target, shelves, static and dynamic obstacles. |
| `FailureInjector` | Create known failure sources. | Blockage, drift, sensor degradation, target shift, replanning failure. |
| `ReliabilitySupervisor` | Combine multiple runtime signals. | Risk score from 0 to 1. |
| `DecisionRouter` | Route different risk patterns. | `NORMAL_NAVIGATION`, `CAUTIOUS_MODE`, `REPLAN`, `RELOCALIZE`, `HUMAN_REVIEW`, `SAFE_STOP`. |

Visual evidence:

![AMR runtime demo](visualizations/runtime_demo/amr_reliability_demo.gif)

![Runtime risk curve](visualizations/runtime_demo/risk_score_curve.png)

![Baseline vs supervisor](visualizations/runtime_demo/baseline_vs_supervisor.png)

Evidence tables:

- [baseline_log.csv](visualizations/evidence/runtime_demo/baseline_log.csv)
- [supervisor_log.csv](visualizations/evidence/runtime_demo/supervisor_log.csv)
- [comparison_summary.csv](visualizations/evidence/runtime_demo/comparison_summary.csv)

## 2. Moving To Gazebo/Nav2: Why A Professional Simulator Is Needed

The grid demo is useful for explaining the idea, but it cannot answer the
research question by itself. To analyze a learned policy, the project needs a
simulation stack that can provide:

- repeatable episodes;
- robot state and plan traces;
- external disturbance injection;
- lidar scan observations;
- depth image observations;
- expert action labels from a planner.

This motivates the ROS 2 / Gazebo / Nav2 scaffold under `ros2_ws/`.

Key files:

| File | Role |
| --- | --- |
| `ros2_ws/src/amr_reliability_benchmark/models/reliability_amr/model.sdf` | AMR model with lidar and depth camera. |
| `ros2_ws/src/amr_reliability_benchmark/worlds/reliability_room.sdf` | Warehouse-style simulation room. |
| `ros2_ws/src/amr_reliability_benchmark/launch/gazebo_nav2_benchmark.launch.py` | Gazebo/Nav2 launch entry point. |
| `ros2_ws/src/amr_reliability_benchmark/amr_reliability_benchmark/gazebo_fault_injector.py` | Disturbance injection. |
| `ros2_ws/src/amr_reliability_benchmark/amr_reliability_benchmark/scan_policy_observation_recorder.py` | Scan observation recording. |
| `ros2_ws/src/amr_reliability_benchmark/amr_reliability_benchmark/depth_policy_observation_recorder.py` | Depth observation recording. |

This step turns the project from a hand-made demo into a simulation-grounded
policy reliability pipeline.

## 3. Policy Learning: How The AMR Learns Actions

The policy is trained by supervised learning, not reinforcement learning.

The expert label comes from Nav2 `/plan`. At each timestep, the recorder aligns
sensor observations and context with the next movement direction implied by the
expert plan.

The learned action space is:

```text
NORTH, SOUTH, EAST, WEST
```

`STAY` is not treated as an ordinary Nav2 movement label. It is better
interpreted as a recovery or safety-supervisor action.

The policy variants are:

| Policy | Input | Why tested |
| --- | --- | --- |
| Scan-only | lidar scan bins + goal/context | Common AMR navigation signal. |
| Depth-only | depth image grid + goal/context | Adds forward spatial structure and richer perception evidence. |
| Scan+depth fusion | lidar + depth + goal/context | Tests whether simple multimodal fusion improves reliability. |

Training scripts:

- `experiments/train_gazebo_scan_policy.py`
- `experiments/train_gazebo_depth_policy.py`
- `experiments/train_gazebo_fusion_policy.py`

## 4. Sensor Ablation: What Improved And What Failed

The formal held-out test comparison shows that depth is useful, but simple
fusion is not yet solved.

| Modality/model | Accuracy | Macro F1 | Weighted F1 | ECE | High-confidence errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| depth baseline | 0.8773 | 0.6666 | 0.8819 | 0.0553 | 62 |
| scan baseline | 0.8694 | 0.6741 | 0.8776 | 0.0778 | 78 |
| scan+depth baseline | 0.8641 | 0.6559 | 0.8673 | 0.0721 | 64 |
| scan focal | 0.8076 | 0.6343 | 0.8235 | 0.0617 | 51 |
| scan+depth focal | 0.8111 | 0.6318 | 0.8265 | 0.0549 | 52 |

Interpretation:

- Depth baseline gives the best held-out accuracy, weighted F1, ECE, and fewer
  high-confidence errors than scan-only.
- Scan-only still has the best macro F1 among baseline models, so it remains
  useful for minority action balance.
- Simple scan+depth concatenation does not reliably improve the policy.
- Focal loss reduces high-confidence errors, but the accuracy drop is too large
  to call it the final upgrade.

Visual evidence:

![Policy accuracy by modality](visualizations/policy_routes/test_accuracy_by_modality.png)

Evidence tables:

- [modality_ablation_metrics.csv](visualizations/evidence/policy_routes/modality_ablation_metrics.csv)
- [test_ablation_delta_vs_scan_baseline.csv](visualizations/evidence/policy_routes/test_ablation_delta_vs_scan_baseline.csv)

## 5. Residual Mechanism Analysis: Why The Policy Is Wrong

The key result is not only which policy has the best aggregate score. The
project asks where confident mistakes come from.

Dominant high-confidence residuals on the held-out test split include:

| Scenario | Error pattern | Mechanism interpretation | Proposed route |
| --- | --- | --- | --- |
| `perception_degradation` | `SOUTH -> EAST` | perception axis confusion | `CAUTIOUS_REPLAN` |
| `external_path_blockage` | `EAST -> SOUTH` | blocked-path direction error | `REPLAN` |
| `external_path_blockage` | `WEST -> SOUTH` | blocked-path direction error | `REPLAN` |
| boundary-like nominal cases | `NORTH <-> WEST` | directional boundary uncertainty | `CAUTIOUS_MODE` |

This mirrors the ECG project logic: the model's residual errors are not random
noise; they contain mechanism information.

Visual evidence:

![High-confidence errors by modality](visualizations/policy_routes/test_high_conf_errors_by_modality.png)

Evidence tables:

- [high_conf_error_patterns.csv](visualizations/evidence/policy_routes/high_conf_error_patterns.csv)
- [residual_mechanism_summary.csv](visualizations/evidence/policy_routes/residual_mechanism_summary.csv)
- [scenario_error_summary.csv](visualizations/evidence/policy_routes/scenario_error_summary.csv)

## 6. Recovery Routing: From Error Mechanism To Action

After the residual mechanisms are identified, the project maps them to
recovery-route families.

| Residual mechanism | Recovery route |
| --- | --- |
| `perception_axis_confusion` | `CAUTIOUS_REPLAN` |
| `perception_lateral_depth_confusion` | `CAUTIOUS_REPLAN` |
| `perception_degradation_confusion` | `CAUTIOUS_REPLAN` |
| `blocked_path_high_conf_direction_error` | `REPLAN` |
| `blocked_path_direction_error` | `REPLAN` |
| `localization_state_error` | `RELOCALIZE` |
| `boundary_direction_confusion` | `CAUTIOUS_MODE` |
| `geometric_policy_residual` | `HUMAN_REVIEW` |

High-confidence test error coverage:

| Modality/model | High-confidence errors | Actionable route coverage |
| --- | ---: | ---: |
| depth baseline | 62 | 1.000 |
| fusion baseline | 64 | 0.984 |
| fusion focal | 52 | 1.000 |
| scan baseline | 78 | 1.000 |
| scan focal | 51 | 1.000 |

This does not prove closed-loop recovery success yet. It shows that residual
policy errors can be assigned to meaningful recovery families instead of being
collapsed into a single fallback.

Visual evidence:

![Recovery route distribution](visualizations/policy_routes/test_recovery_route_distribution.png)

Evidence tables:

- [recovery_route_evidence.csv](visualizations/evidence/policy_routes/recovery_route_evidence.csv)
- [recovery_route_coverage.csv](visualizations/evidence/policy_routes/recovery_route_coverage.csv)
- [residual_route_report.json](visualizations/evidence/policy_routes/residual_route_report.json)

## Claim-To-Evidence Index

| Claim | Evidence | Source |
| --- | --- | --- |
| AMR failures should not be treated as one generic failure. | Runtime supervisor and router separate blockage, localization, perception, replanning, and stagnation signals. | `src/reliability_supervisor.py`, `src/decision_router.py`, runtime visualizations |
| Gazebo/Nav2 can generate policy-learning episodes. | 36/36 formal episodes completed with synchronized scan/depth/policy logs. | `docs/GAZEBO_DEPTH_FUSION_FORMAL_V1_RESULTS.md` |
| Depth adds useful reliability evidence. | Depth baseline has the best held-out accuracy, weighted F1, ECE, and fewer high-confidence errors than scan-only. | `visualizations/evidence/policy_routes/modality_ablation_metrics.csv` |
| Simple scan+depth fusion is not enough. | Fusion baseline does not outperform the best single-modality baseline. | `docs/GAZEBO_DEPTH_FUSION_FORMAL_V1_RESULTS.md` |
| Policy errors are structured. | High-confidence residuals concentrate in perception axis confusion and blocked-path direction errors. | `visualizations/evidence/policy_routes/high_conf_error_patterns.csv` |
| Recovery routes can be mechanism-specific. | Residual mechanisms map to `CAUTIOUS_REPLAN`, `REPLAN`, `RELOCALIZE`, `CAUTIOUS_MODE`, and `HUMAN_REVIEW`. | `visualizations/evidence/policy_routes/recovery_route_evidence.csv` |

## Repository Map

```text
src/
  Lightweight warehouse reliability demo, supervisor, router, and visualization helpers

experiments/
  Dataset generation, Gazebo/Nav2 policy training, ablations, and residual-route analysis

ros2_ws/
  ROS 2 / Gazebo / Nav2 package, robot model, world, launch files, and recorders

docs/
  Research reports, protocols, formal result summaries, and narrative documents

visualizations/
  Curated GitHub-ready figures, GIFs, and compact evidence tables

outputs/
  Local raw/generated artifacts; intentionally ignored except .gitkeep
```

## Reproduce The Lightweight Demo

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

Run the runtime reliability demo:

```powershell
python main.py
```

Run tests:

```powershell
python -m pytest tests
```

The demo writes local artifacts under `outputs/`. Curated visual artifacts for
GitHub are copied into `visualizations/`.

## Reproduce The Gazebo/Nav2 Pipeline

The Gazebo/Nav2 experiments require ROS 2 Jazzy, Gazebo, and Nav2. The
installation notes and stack check scripts are provided here:

- `docs/INSTALL_ROS2_NAV2_GAZEBO_JAZZY.md`
- `ros2_ws/src/amr_reliability_benchmark/scripts/check_ros_stack.sh`
- `ros2_ws/src/amr_reliability_benchmark/scripts/install_jazzy_nav2_gazebo.sh`

Typical pipeline stages:

```powershell
# Train scan, depth, and fusion policies from generated episode tables.
python experiments/train_gazebo_scan_policy.py
python experiments/train_gazebo_depth_policy.py
python experiments/train_gazebo_fusion_policy.py

# Analyze residual mechanisms and route evidence.
python experiments/analyze_policy_residual_routes.py
```

The exact formal outputs used in the current report are preserved locally under
`outputs/`, while the compact evidence tables used for public review are under
`visualizations/evidence/`.

## What Is Shown

The repository shows that:

- a simulation-grounded AMR reliability pipeline can be built around
  Gazebo/Nav2;
- Nav2 plans can provide expert labels for supervised policy learning;
- scan and depth observations can train navigation-action policies;
- depth improves several held-out reliability metrics in the current matrix;
- policy residual errors are structured by disturbance type;
- different residual mechanisms can be routed to different recovery families.

## What Remains Unproven

The repository does not yet prove:

- real-world AMR safety;
- closed-loop recovery success after a route is triggered;
- statistical robustness beyond the current limited seed matrix;
- a final best multimodal fusion architecture;
- full relocalization performance under expanded localization-drift episodes.

## Recommended Next Experiment

The next experiment should evaluate the route selector itself:

1. Add localization-drift episodes to the formal scan/depth matrix.
2. Implement a route selector that predicts `CAUTIOUS_REPLAN`, `REPLAN`,
   `RELOCALIZE`, `CAUTIOUS_MODE`, or `HUMAN_REVIEW`.
3. Measure how many high-confidence policy errors it captures.
4. Measure how often it over-routes correct policy decisions.
5. Compare rule-based, learned, uncertainty-threshold, and mechanism-aware
   gated-fusion route selectors.

## Limitations

- The evidence is simulation-grounded, not real-robot validation.
- The formal scan/depth/fusion comparison currently has one held-out test seed.
- The route layer is an evidence-backed prototype, not a proven closed-loop
  safety controller.
- Raw `outputs/` artifacts can be large and are intentionally excluded from
  Git; curated figures and compact tables are provided under `visualizations/`.
- Hardware-facing robot control is outside the scope of this repository.
