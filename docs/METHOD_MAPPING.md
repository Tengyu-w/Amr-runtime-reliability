# ECG To AMR Reliability Mapping

This note defines the first bridge from the ECG uncertainty project to the AMR
runtime reliability demo.

## Shared Research Question

The ECG project asks whether a classifier should avoid accepting a single
SR/VT/VF label when evidence suggests the prediction is unreliable.

The AMR demo asks whether an autonomy loop should avoid normal navigation when
runtime evidence suggests the action, state estimate, path, or observation is
unreliable.

In both cases, the central object is not raw accuracy. The central object is a
supervisor that detects high-risk decisions and routes them to a safer action.

## Mechanism Mapping

| ECG mechanism | AMR analogue | Evidence examples | Route |
| --- | --- | --- | --- |
| VT/VF boundary ambiguity | ambiguous navigation or action state | path blocked, trajectory deviation, task stagnation | `CAUTIOUS_MODE`, `REPLAN` |
| Signal-quality risk | weak observation or localization | low sensor confidence, localization uncertainty, obstacle proximity | `RELOCALIZE`, `HUMAN_REVIEW` |
| Representation conflict | inconsistent plan-state evidence | planned path no longer matches position, target, or blocked cells | `REPLAN` |
| Atypical signal | unusual runtime context | repeated failures, unexpected obstacle pattern, stalled progress | `CAUTIOUS_MODE`, `REPLAN` |
| Hidden confident error | high-confidence mismatch | confident action but observed progress/motion residual is poor | `HUMAN_REVIEW`, `SAFE_STOP` |

## Fault Source Mapping

The current demo does not yet contain real learned-model OOD detection. It uses
synthetic runtime faults with explicit provenance labels:

| Event | Fault origin | Fault family | OOD status |
| --- | --- | --- | --- |
| `dynamic_obstacle_blocks_path` | external disturbance | path blockage | in-distribution fault |
| `localization_drift_increasing` | state-estimation drift | localization | in-distribution fault |
| `sensor_confidence_drop` | perception degradation | sensor quality | in-distribution fault |
| `target_changed` | task or goal shift | task reassignment | OOD-style shift |
| `replanning_backend_unstable` | planner internal failure | planning backend | in-distribution fault |
| `trajectory_deviation` | execution error | control tracking | in-distribution fault |
| `progress_stagnation_blocker` | external disturbance | progress blockage | in-distribution fault |
| `dynamic_obstacles_cleared` | environment recovery | disturbance cleared | recovery event |

So the present claim is not "the AMR model detects OOD." The more accurate
claim is: the simulator injects labelled runtime fault sources, including one
OOD-style task/goal shift, and tests whether the supervisor routes each source
through an appropriate mechanism.

## ECG-Style Upgrade Loop

The intended AMR research loop follows the ECG project structure:

1. Baseline model: ordinary A* navigation with no reliability routing.
2. First upgrade: scalar `risk_router`, which uses only aggregate risk.
3. Mechanism analysis: label each risky step by its dominant runtime mechanism.
4. Second upgrade: `mechanism_router`, which routes different mechanisms to
   different recovery actions.
5. Evidence check: compare success, steps, event response, hard recovery, and
   mechanism-action alignment under paired seeds.

This mirrors the ECG lesson: a stronger score is not enough if it does not tell
the system which action to take. In ECG, a single uncertainty score is weaker
than mechanism-separated review routing. In AMR, a scalar runtime risk score can
detect trouble but still fail to recover if path blockage, localization, weak
perception, stagnation, and repeated recovery failure are all treated the same.

## Router Variants

| Variant | What it knows | What it can do | Role in the evidence chain |
| --- | --- | --- | --- |
| `baseline` | path and obstacles only | navigate or get stuck | ordinary model baseline |
| `risk_router` | aggregate risk score | normal, cautious, human review, safe stop | scalar-risk upgrade / negative control |
| `mechanism_router` | dominant failure mechanism plus risk | replan, relocalize, review, cautious, safe stop | mechanism-separated upgrade |

## Current Evidence Level

Confirmed:

- The ECG repository already has internal evidence for mechanism-separated
  review routing under duplicate-family validation.
- The AMR demo has a deterministic offline simulation with baseline and
  reliability-supervised routes.
- The supervisor produces interpretable risk components and router decisions.
- The new multi-seed runner supports paired baseline vs supervisor summaries.
- The mechanism runner now compares `baseline`, `risk_router`, and
  `mechanism_router` on the same paired seeds.

Plausible interpretation:

- The ECG idea transfers best as a reliability-supervisor pattern, not as an
  ECG-specific model or metric.
- AMR risk should be evaluated by event coverage, non-nominal routing, hard
  recovery routing, task completion, human-review burden, and safe-stop
  behavior, not only path length.
- Current smoke evidence supports the ECG-style negative result: scalar risk
  can over-trigger review without selecting the right recovery action, while
  mechanism routing can map path blockage, trajectory deviation, and stagnation
  to `REPLAN`.

Limitations:

- This is not hardware validation.
- The current AMR failures are synthetic and deterministic.
- The risk weights are hand-built, not calibrated on real AMR logs.
- Multi-seed evidence in this demo varies perturbation seeds but does not
  replace scenario-level or environment-level validation.

## Next Validation Step

Run the paired multi-seed comparison:

```powershell
python .\experiments\run_multiseed_reliability.py --seeds 7,8,9,10,11 --out-dir .\outputs\multiseed_reliability
```

Then inspect:

- `multiseed_seed_level.csv`
- `multiseed_paired_deltas.csv`
- `multiseed_mean_std.csv`
- `multiseed_mechanism_routes.csv`
- `multiseed_fault_origin_routes.csv`

Use `non_nominal_*` metrics for cautious or recovery routing. Use
`hard_recovery_*` metrics when only `REPLAN`, `RELOCALIZE`, `HUMAN_REVIEW`, and
`SAFE_STOP` should count as recovery actions.

The first useful claim should be modest: the supervisor increases recovery
routing around injected failure events in an offline simulation. It should not
be described as a deployable AMR safety controller.

## Current Smoke Result

Using seeds `7,8,9,10,11`, the current offline smoke comparison shows:

| Variant | Success mean | Mean steps | Non-nominal event response | Hard-recovery event response |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 0.80 | 40.4 | 0.000 | 0.000 |
| `risk_router` | 0.00 | 70.0 | 0.602 | 0.070 |
| `mechanism_router` | 0.80 | 32.2 | 0.439 | 0.209 |

The important interpretation is not that `mechanism_router` is validated. The
point is narrower: scalar risk frequently detects elevated risk but fails to
choose a useful recovery route. The mechanism router gives a cleaner mapping:

| Mechanism | Main mechanism-router action in the smoke run |
| --- | --- |
| `path_blocked` | `REPLAN` |
| `trajectory_deviation` | `REPLAN` |
| `progress_stagnation` | `REPLAN` |
| `repeated_replan_failure` | `REPLAN` or `SAFE_STOP` |
| `elevated_runtime_risk` | `CAUTIOUS_MODE` |

This is the AMR analogue of the ECG finding that uncertainty evidence should be
converted into mechanism-specific routing, not only a single risk score.
