# ROS 2 AMR Reliability Benchmark Workspace

This workspace is the formal simulation track for the AMR reliability-routing
project.

The intended stack is:

```text
Ubuntu 24.04
ROS 2 Jazzy
Nav2
Gazebo Sim
ros_gz bridge
```

The Python grid simulator in the parent directory remains a fast sketchpad.
This ROS 2 workspace is the path toward a PhD-facing migration project.

## Current Status

This workspace is a scaffold. It defines:

- a labelled AMR reliability benchmark package;
- a Gazebo room/warehouse world;
- a scenario catalog for controlled fault sources;
- a ROS 2 scenario runner that publishes the active fault/recovery label;
- launch and config placeholders for Nav2/Gazebo;
- environment checks for ROS 2, Nav2, Gazebo, and ros_gz.

It does not yet claim validated robotics results.

## Layout

```text
ros2_ws/
  src/amr_reliability_benchmark/
    amr_reliability_benchmark/
    config/
    launch/
    worlds/
    scripts/
```

## First Check

From WSL Ubuntu:

```bash
cd /mnt/c/Users/77941/Desktop/ecg_uncertainty_project/AMR-Runtime-Reliability-Demo/ros2_ws
bash src/amr_reliability_benchmark/scripts/check_ros_stack.sh
```

When ROS 2 Jazzy, Nav2, Gazebo Sim, and ros_gz are installed:

```bash
colcon build --symlink-install
source install/setup.bash
ros2 launch amr_reliability_benchmark reliability_room.launch.py
```

## Scenario Runner

The scenario runner is the bridge between the labelled reliability protocol and
future Gazebo/Nav2 episodes. It publishes the selected scenario as JSON on:

```text
/amr_reliability/scenario
```

Smoke-test it without launching Gazebo:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run amr_reliability_benchmark scenario_catalog
ros2 run amr_reliability_benchmark scenario_runner --ros-args \
  -p scenario_id:=planner_backend_failure \
  -p publish_once:=true
```

Launch it as a ROS node:

```bash
ros2 launch amr_reliability_benchmark scenario_runner.launch.py \
  scenario_id:=external_path_blockage
```
