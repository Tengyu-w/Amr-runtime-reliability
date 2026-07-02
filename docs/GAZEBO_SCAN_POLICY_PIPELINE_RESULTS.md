# Gazebo Scan-Observation Policy Pipeline

## Purpose

This stage moves the AMR reliability project from a state-summary policy toward a simulated perception policy.
The policy input is now:

- downsampled Gazebo lidar `/scan` observations,
- task context such as robot pose, target pose, relative goal vector, risk, localization uncertainty, and sensor confidence,
- expert action labels derived from Nav2 `/plan`.

This is a simulation-derived expert-labeled policy dataset. It is not real-robot visual validation.

## Implemented Pipeline

1. `scan_policy_observation_recorder` subscribes to `/scan` and `/amr_reliability/policy_decision`.
2. It aligns the latest LaserScan with evaluable Nav2-plan expert labels.
3. It writes `*.scan_policy.csv` files with fixed `scan_bin_*` columns plus context columns.
4. `run_gazebo_validation_matrix.py` now records scan-policy logs beside runtime and policy logs.
5. `train_gazebo_scan_policy.py` trains two scan+goal policies:
   - `baseline_scan_goal`
   - `class_weighted_focal_scan_goal`
6. The trainer writes score, class-metric, confusion, calibration, high-confidence-error, and imbalance evidence tables.

## Smoke Run

Command:

```bash
python AMR-Runtime-Reliability-Demo/experiments/run_gazebo_validation_matrix.py \
  --scenarios nominal \
  --seeds 10 \
  --out-dir AMR-Runtime-Reliability-Demo/outputs/gazebo_scan_policy_smoke_v2 \
  --timeout-sec 90 \
  --skip-analysis \
  --retries 0
```

Result:

- status: `ok`
- runtime rows: 262
- policy rows: 262
- scan-policy rows: 119
- spawn count: 1
- navigation started: 1
- final runtime decision: `NORMAL_NAVIGATION/nominal`

## Training Smoke

Command:

```bash
python AMR-Runtime-Reliability-Demo/experiments/train_gazebo_scan_policy.py \
  --log-dir AMR-Runtime-Reliability-Demo/outputs/gazebo_scan_policy_smoke_v2/scan_policy_episode_logs \
  --out-dir AMR-Runtime-Reliability-Demo/outputs/gazebo_scan_policy_train_smoke_v2 \
  --epochs 40
```

The smoke train confirms the training and audit chain runs end to end.

Action distribution in this single nominal episode:

| split | action | count | fraction |
|---|---:|---:|---:|
| train | EAST | 41 | 0.3445 |
| train | SOUTH | 78 | 0.6555 |

Metrics are not a formal result because this smoke has one episode and only two observed action classes.

| model | split | rows | accuracy | weighted F1 | ECE |
|---|---:|---:|---:|---:|---:|
| baseline_scan_goal | train | 119 | 0.9244 | 0.9241 | 0.0944 |
| class_weighted_focal_scan_goal | train | 119 | 0.8824 | 0.8851 | 0.2052 |

## Interpretation

Confirmed:

- Gazebo already provides lidar observations through the AMR model.
- The project can now record synthetic perception observations aligned with Nav2 expert actions.
- The scan+goal policy trainer runs and produces reliability evidence tables.

Limitations:

- The smoke run has only one nominal episode.
- It contains only `EAST` and `SOUTH` labels.
- No validation or test split exists in the smoke output.
- It should not be used to claim recovery-routing effectiveness.

Next required step:

- run action-diverse Gazebo goals and disturbances so the dataset includes `STAY`, `NORTH`, `SOUTH`, `EAST`, and `WEST`;
- then train the scan+goal policy on train/val/test splits;
- only after residual policy errors are analyzed should recovery routes be derived.

## Action-Diversity Smoke Update

The first action-diversity smoke showed that the robot's default spawn pose is `(-4.5, -3.0)`, not the map center.
Therefore goal choices must be designed relative to that pose.

Confirmed useful goals:

| goal id | goal | observed expert actions |
|---|---:|---|
| `east_south` | `(4.5, 3.0)` | `EAST`, `SOUTH` |
| `west_near` | `(-5.7, -3.0)` | `WEST` |
| `north_near` | `(-4.5, -3.8)` | `NORTH` |
| `south_axis` | `(-4.5, 3.0)` | `SOUTH` |

Combined smoke training data:

| action | count | fraction |
|---|---:|---:|
| EAST | 89 | 0.2330 |
| NORTH | 12 | 0.0314 |
| SOUTH | 153 | 0.4005 |
| WEST | 128 | 0.3351 |

Training smoke result:

| model | split | rows | accuracy | macro F1 | weighted F1 | ECE |
|---|---:|---:|---:|---:|---:|---:|
| baseline_scan_goal | train | 382 | 0.9634 | 0.7742 | 0.9626 | 0.0478 |
| class_weighted_focal_scan_goal | train | 382 | 0.9293 | 0.7406 | 0.9306 | 0.0648 |

The weighted focal model did not improve this smoke result. The baseline remains the better smoke baseline.

`STAY` was not produced by Nav2-plan labels. Even under `external_path_blockage`, Nav2 still produced movement-plan labels (`EAST`, `SOUTH`).
This suggests `STAY` should be treated as a safety-supervisor or recovery action unless a stronger stop/blocked expert source is added.

Current limitation:

- This is still a single-seed smoke set, so it is not a formal validation result.
- `NORTH` is rare and needs more episodes.
- A formal next run needs multiple seeds and train/val/test splits before mechanism-level recovery claims.
