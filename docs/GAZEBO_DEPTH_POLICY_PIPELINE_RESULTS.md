# Gazebo Depth-Observation Policy Pipeline

## Purpose

This stage adds a depth-camera observation path so the AMR policy project is not limited to planar lidar.
The depth path is meant to support a more visually intuitive robotics story for presentation and doctoral application work.

## Implemented Components

1. Added a forward-facing Gazebo `depth_camera` sensor to the AMR model.
2. Bridged `/depth_image` from Gazebo to ROS as `sensor_msgs/msg/Image`.
3. Added `depth_policy_observation_recorder`.
4. The recorder aligns depth images with Nav2-plan expert action labels from `/amr_reliability/policy_decision`.
5. The depth image is compressed into an interpretable grid of `depth_cell_*` values.
6. Added `train_gazebo_depth_policy.py` for a baseline `depth + goal context -> action` supervised policy.

## Smoke Run

Command:

```bash
python AMR-Runtime-Reliability-Demo/experiments/run_gazebo_validation_matrix.py \
  --scenarios nominal \
  --seeds 10 \
  --goal-id depth_smoke \
  --goal-x 4.5 \
  --goal-y 3.0 \
  --out-dir AMR-Runtime-Reliability-Demo/outputs/gazebo_depth_policy_smoke_v2 \
  --timeout-sec 90 \
  --skip-analysis \
  --retries 0
```

Result:

- status: `ok`
- runtime rows: 273
- policy rows: 273
- scan-policy rows: 130
- depth-policy rows: 130
- depth image size: 160 x 120
- depth encoding: `32FC1`

## Training Smoke

Command:

```bash
python AMR-Runtime-Reliability-Demo/experiments/train_gazebo_depth_policy.py \
  --log-dir AMR-Runtime-Reliability-Demo/outputs/gazebo_depth_policy_smoke_v2/depth_policy_episode_logs \
  --out-dir AMR-Runtime-Reliability-Demo/outputs/gazebo_depth_policy_train_smoke_v1 \
  --epochs 60
```

Action distribution:

| split | action | count | fraction |
|---|---:|---:|---:|
| train | EAST | 12 | 0.0923 |
| train | SOUTH | 118 | 0.9077 |

Training smoke metric:

| model | split | rows | accuracy | macro F1 | weighted F1 | high-conf errors |
|---|---:|---:|---:|---:|---:|---:|
| baseline_depth_goal | train | 130 | 1.0000 | 0.4000 | 1.0000 | 0 |

This metric is not a formal result because the smoke contains one episode and only two actions.

## Interpretation

Confirmed:

- Gazebo depth images are generated and bridged into ROS.
- Depth frames can be aligned with Nav2-plan expert action labels.
- The depth grid can be used as supervised policy input.
- The depth-policy trainer runs end to end.

Current limitation:

- The smoke is not a train/val/test validation.
- It only includes `EAST` and `SOUTH`.
- The next formal run should reuse the scan-policy formal matrix: multiple goals, multiple scenarios, and seeds `10/16/18`.

Next step:

- collect a formal depth-policy matrix with the same multi-goal, multi-scenario split as `GAZEBO_SCAN_POLICY_FORMAL_V1_RESULTS.md`;
- compare scan-only, depth-only, and later scan+depth fusion policy behavior;
- analyze whether depth reduces perception-degradation errors such as `NORTH -> WEST`.
