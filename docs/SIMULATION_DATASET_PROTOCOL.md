# Simulation Dataset Protocol

This protocol defines a labelled room-scale AMR reliability dataset generated
entirely in simulation.

## Purpose

The dataset is not real robot evidence. It is a controlled simulation dataset
for testing whether reliability routing can distinguish fault sources and route
them to different recovery actions.

The main question is:

> Given known simulated fault sources, does a supervisor choose an appropriate
> recovery route, or does it only detect generic risk?

## Data Unit

The unit of analysis is an episode:

```text
scenario + seed + router mode -> one episode
```

Each episode produces timestep rows with:

- planned and actual robot state proxies;
- risk indicators;
- injected fault event;
- ground-truth fault origin and family;
- OOD status;
- diagnosed runtime mechanism;
- router decision;
- final outcome.

## Fault Sources

| Scenario | Fault origin | Fault family | OOD status |
| --- | --- | --- | --- |
| `nominal` | none | none | none |
| `external_path_blockage` | external disturbance | path blockage | in-distribution fault |
| `localization_drift` | state-estimation drift | localization | in-distribution fault |
| `perception_degradation` | perception degradation | sensor quality | in-distribution fault |
| `task_goal_shift_ood_style` | task or goal shift | task reassignment | OOD-style shift |
| `execution_deviation` | execution error | control tracking | in-distribution fault |
| `progress_blockage` | external disturbance | progress blockage | in-distribution fault |
| `planner_backend_failure` | planner internal failure | planning backend | in-distribution fault |
| `compound_shift_and_degradation` | task or goal shift | task reassignment | OOD-style shift |

`OOD-style shift` here means a scripted task/goal distribution shift. It is not
a learned OOD detector and should not be described as real-world OOD evidence.

## Splitting

Splits are assigned by seed:

- seed mod 10 in `0..5`: train
- seed mod 10 in `6..7`: validation
- seed mod 10 in `8..9`: test

This avoids timestep-level leakage. Do not randomly split timestep rows.

## Generation

From `AMR-Runtime-Reliability-Demo`:

```powershell
python .\experiments\generate_scenario_dataset.py --out-dir .\outputs\scenario_dataset
```

Outputs:

- `scenario_catalog.csv`: scenario definitions and ground-truth labels;
- `splits.csv`: seed-level split assignment;
- `episodes.csv`: one row per scenario/seed/mode episode;
- `timesteps.csv`: one row per timestep;
- `episode_logs/`: per-episode raw logs.

## Minimum Evaluation

Report results by fault source, not only overall success:

- success rate by scenario;
- route distribution by `primary_fault_origin`;
- route distribution by `primary_fault_family`;
- hard-recovery rate by source;
- unresolved failure rate by source;
- whether OOD-style task shifts route differently from in-distribution faults.

The expected ECG-style lesson is not "the model is safer." The test is whether
the mechanism-aware router handles different fault sources differently under a
controlled simulation protocol.
