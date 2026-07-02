# Gazebo Fault Dataset And Route Model Protocol

This protocol turns ROS 2/Gazebo/Nav2 episodes into route-learning evidence.
It is a research prototype, not an operational safety claim.

## Simulation Faults

`gazebo_fault_injector` injects scenario-specific faults inside the Gazebo/Nav2 loop:

- `external_path_blockage` and `progress_blockage`: moves the Gazebo obstacle into the navigation corridor.
- `perception_degradation`: relays `/gazebo/scan` to `/scan` with dropouts and range noise.
- `localization_drift`: relays `/gazebo/odom` to `/odom` with increasing pose drift and uncertainty.
- `execution_deviation`: increases trajectory-deviation evidence during navigation.
- `planner_backend_failure` and `compound_shift_and_degradation`: combine multiple fault channels.
- `mixed_blockage_and_perception`: combines a moving path blocker with LiDAR dropout/noise.
- `mixed_drift_and_execution`: combines odometry/localization drift with trajectory-deviation evidence.
- `boundary_weak_blockage`: injects an intermittent weak blockage near the route threshold.

Gazebo raw topics are bridged as `/gazebo/scan` and `/gazebo/odom`; Nav2 consumes the injected `/scan` and `/odom`.

## Run One Episode

```bash
source /opt/ros/jazzy/setup.bash
source AMR-Runtime-Reliability-Demo/ros2_ws/install/setup.bash
ros2 launch amr_reliability_benchmark gazebo_nav2_benchmark.launch.py \
  scenario_id:=external_path_blockage \
  episode_id:=gazebo_external_path_blockage_001 \
  output_path:=AMR-Runtime-Reliability-Demo/outputs/ros2_episode_logs/gazebo_external_path_blockage_001.csv
```

## Build Dataset

```bash
python AMR-Runtime-Reliability-Demo/experiments/collect_gazebo_episode_dataset.py \
  --log-dir AMR-Runtime-Reliability-Demo/outputs/ros2_episode_logs \
  --out-dir AMR-Runtime-Reliability-Demo/outputs/gazebo_episode_dataset
```

Outputs:

- `scenario_catalog.csv`
- `episodes.csv`
- `timesteps.csv`
- `episode_outcome_summary.csv`

The collector uses episode-level splits. Early runtime rows whose scenario heartbeat has not arrived are relabelled with the episode scenario while preserving the original value in `observed_scenario_id`.

Episode outcomes are proxy labels derived from logged telemetry, not real-world safety validation. The main fields are:

- `expected_route_observed`: whether the expected recovery route appeared at least once.
- `recovery_latency_steps`: rows from first non-nominal mechanism to first expected route.
- `goal_reached_proxy`: final logged robot-target distance is within the configured tolerance.
- `collision_risk_proxy`: high obstacle-proximity and high trajectory-deviation co-occurred.
- `outcome_label`: `nominal_stable`, `routed_and_recovered_proxy`, `routed_but_unresolved_proxy`, `missed_expected_route`, or `safe_stop_observed`.

## Train Route Model

```bash
python AMR-Runtime-Reliability-Demo/experiments/train_gazebo_route_model.py \
  --dataset-dir AMR-Runtime-Reliability-Demo/outputs/gazebo_episode_dataset \
  --out-dir AMR-Runtime-Reliability-Demo/outputs/gazebo_route_model
```

The neural route model excludes `scenario_id`, fault origin, fault family, and OOD labels from inputs. It uses runtime features only.

## Ablation Evidence

```bash
python AMR-Runtime-Reliability-Demo/experiments/run_gazebo_route_ablation.py \
  --dataset-dir AMR-Runtime-Reliability-Demo/outputs/gazebo_episode_dataset \
  --out-dir AMR-Runtime-Reliability-Demo/outputs/gazebo_route_ablation
```

Outputs:

- `ablation_metrics.csv`
- `evidence_by_fault_origin.csv`
- `evidence_by_episode_outcome.csv`
- `scenario_route_confusion.csv`

## Current Smoke Evidence

The initial smoke dataset contains one short episode per scenario for six scenarios. This is enough to verify the pipeline, but not enough for a publishable comparison. In the current run, `execution_deviation` is held out as validation and has no same-origin training episode, so its validation score is expected to be weak. The next evidence step is to run multiple seeds per scenario and report paired, episode-level splits.

## Multi-Seed Validation Matrix

```bash
python AMR-Runtime-Reliability-Demo/experiments/run_gazebo_validation_matrix.py \
  --scenarios nominal,external_path_blockage,localization_drift,perception_degradation,execution_deviation,progress_blockage,mixed_blockage_and_perception,mixed_drift_and_execution,boundary_weak_blockage \
  --seeds 10,11,12,16,17,18,19 \
  --timeout-sec 45 \
  --out-dir AMR-Runtime-Reliability-Demo/outputs/gazebo_validation_matrix
```

This writes:

- `run_manifest.csv`: per-episode launch status and sanity checks.
- `episode_logs/*.csv`: routed runtime rows.
- `ros_logs/*.log`: raw ROS/Gazebo launch logs.
- `dataset/`: aggregated trainable dataset.
- `route_model/`: neural route model scores and metrics.
- `route_ablation/`: feature ablation and fault-origin evidence tables.

Use `--dry-run` to print the planned launches without starting Gazebo. Use `--skip-analysis` to only collect episode logs.
