# Step 2-6 Execution Protocol

This document defines the next implementation path for migrating the ECG
uncertainty-routing idea into an AMR runtime reliability benchmark.

## Objective

Build a professional simulation-backed reliability pipeline:

```text
ROS 2 / Gazebo / Nav2 episode
  -> runtime metrics
  -> mechanism-aware router
  -> routed episode log
  -> learned recovery policy
  -> uncertainty/error-capture analysis
```

The current replay pipeline uses the same schema before Gazebo/Nav2 telemetry is
available. Once the professional simulator stack is installed, only the metrics
producer should change.

## Step 2: Nav2/Gazebo Navigation Loop

Goal: run a robot in the Gazebo room and navigate with Nav2.

Current status:

- `reliability_room.sdf` defines the world scaffold.
- `reliability_room.launch.py` launches the Gazebo world once `gz` is installed.
- `reliability_amr.urdf` defines the benchmark AMR robot-state model.
- `reliability_room.yaml` and `reliability_room.pgm` define the first Nav2 map.
- `amr_nav2_bringup.launch.py` wires robot state, Nav2 bringup, and the
  reliability runtime pipeline.
- `scenario_runner` publishes the selected benchmark scenario.
- `scenario_goal_publisher` publishes Nav2-compatible `/goal_pose` targets.
- `fault_proxy_publisher` publishes scenario-driven risk proxy topics for
  controlled fault mechanisms.
- `nav2_telemetry_adapter` converts `/odom`, `/amcl_pose`, `/goal_pose`, and
  reliability proxy topics into the benchmark runtime metrics schema.

Remaining work:

- install `ros-jazzy-navigation2`, `ros-jazzy-nav2-bringup`, and `ros-jazzy-ros-gz`;
- verify odometry, TF, costmaps, and Nav2 status topics.

After the missing packages are installed, the intended full bringup is:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch amr_reliability_benchmark reliability_room.launch.py
ros2 launch amr_reliability_benchmark amr_nav2_bringup.launch.py \
  scenario_id:=external_path_blockage \
  episode_id:=gazebo_external_blockage_001
```

This is still a research prototype until Gazebo movement, Nav2 status, and routed
episode logs are verified together.

## Step 3: Fault Injection In ROS/Gazebo

Goal: turn each scenario into a controlled disturbance source.

Scenario mechanisms:

| Scenario | Mechanism | Expected route |
| --- | --- | --- |
| `nominal` | no injected fault | `NORMAL_NAVIGATION` |
| `external_path_blockage` | dynamic obstacle blocks planned route | `REPLAN` |
| `localization_drift` | localization uncertainty rises | `RELOCALIZE` |
| `perception_degradation` | sensor confidence drops | `HUMAN_REVIEW` |
| `task_goal_shift_ood_style` | goal/task shift | `HUMAN_REVIEW` |
| `execution_deviation` | robot deviates from path | `REPLAN` |
| `progress_blockage` | progress stagnates under obstruction | `REPLAN` |
| `planner_backend_failure` | repeated replanning failure | `SAFE_STOP` |
| `compound_shift_and_degradation` | task shift plus degraded perception/localization | `HUMAN_REVIEW` |

Current replay substitute:

- `metrics_replay` publishes Nav2-like runtime metrics on
  `/amr_reliability/runtime_metrics`.
- `fault_proxy_publisher` publishes scenario-driven proxy topics consumed by
  `nav2_telemetry_adapter`.
- proxy topics currently include localization uncertainty, sensor confidence,
  path-blocked score, obstacle proximity, trajectory deviation, progress
  stagnation, and replanning failure count.

Professional simulator replacement:

- use `nav2_telemetry_adapter` to read odometry, localization covariance, goal
  state, and reliability proxy topics;
- replace proxy topics with costmap/Nav2-derived measurements as the simulator
  stack matures.

Launch the real-telemetry side of the pipeline:

```bash
ros2 launch amr_reliability_benchmark nav2_runtime_pipeline.launch.py \
  scenario_id:=external_path_blockage \
  episode_id:=nav2_external_blockage_001
```

This launch does not start Gazebo or Nav2 by itself. It expects those systems to
publish standard topics such as `/odom`, `/amcl_pose`, and `/goal_pose`.

Without Gazebo/Nav2 publishers, the launch is still useful as a wiring smoke
test: risk proxies can drive the mechanism router, while robot pose and
localization covariance remain default unless `/odom` and `/amcl_pose` are
active.

## Step 4: Dataset Collection

Goal: create a labelled AMR reliability dataset from routed episodes.

Current logger:

- `episode_recorder` writes routed rows from `/amr_reliability/router_decision`.

Required row schema:

- scenario and episode ids;
- robot and target pose;
- risk indicators;
- failure mechanism;
- router decision;
- telemetry source;
- final success/failure once Nav2 episodes are complete.

The offline replay command is:

```bash
python experiments/replay_ros2_benchmark_protocol.py --out-dir outputs/ros2_protocol_replay
```

## Step 5: Learned Recovery Policy

Goal: train a model to imitate or improve the mechanism-aware recovery router.

Current model:

- `train_policy_model.py` trains a two-layer PyTorch MLP;
- inputs exclude scenario labels, fault origin, fault family, and OOD status;
- outputs action probabilities for the recovery actions;
- evidence includes entropy, margin, max probability, logits, and embedding distances.

Next upgrade:

- train on ROS/Gazebo episode logs instead of only grid-simulator rows;
- compare teacher-router imitation against outcome-supervised labels;
- report per-scenario and per-fault-origin generalization.

## Step 6: Uncertainty/Error-Capture Analysis

Goal: test whether model uncertainty captures the important routing errors.

Current analysis:

- `analyze_policy_uncertainty_capture.py` computes error-detection AUROC/AUPRC;
- signals include entropy, margin, max probability, and embedding distances;
- outputs high-error cases for manual mechanism review.

Required interpretation:

- strong evidence: uncertainty concentrates on incorrect recovery actions;
- weak evidence: high accuracy but low error-capture AUROC;
- limitation: simulator data is not real-robot validation.
