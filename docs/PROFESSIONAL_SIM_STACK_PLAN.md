# Professional Simulation Stack Plan

This project should not be positioned around the Python grid simulator. The
formal robotics migration track should use a professional robotics stack:

```text
Ubuntu 24.04
ROS 2 Jazzy
Nav2
Gazebo Sim
ros_gz bridge
```

The Python simulator remains useful for fast algorithm sketches only.

## Why This Stack

- ROS 2 is the standard middleware layer for robot software.
- Nav2 is the standard ROS 2 navigation stack for mobile robots.
- Gazebo Sim is a mainstream robotics simulator with physics, sensors, worlds,
  and ROS integration.
- Isaac Sim can be added later for photorealistic perception and synthetic
  sensor work, but the first reproducible navigation benchmark should use
  ROS 2/Nav2/Gazebo.

## Formal Research Structure

1. Build a ROS 2/Gazebo room or warehouse world.
2. Run a Nav2 baseline navigation stack.
3. Define controlled fault-source scenarios:
   - external path blockage;
   - localization drift;
   - perception degradation;
   - OOD-style goal shift;
   - execution deviation;
   - repeated recovery failure.
4. Record episode-level and timestep-level data with rosbag and CSV summaries.
5. Train a learned policy/value model from observable robot state.
6. Analyze policy entropy, action margin, value drop, and hidden embeddings by
   fault source.
7. Build a mechanism-aware recovery router from learned model evidence.
8. Compare Nav2 baseline, scalar-risk routing, learned policy, and
   mechanism-aware recovery.

## Current Workspace

The initial ROS 2 workspace scaffold is:

```text
AMR-Runtime-Reliability-Demo/ros2_ws/
```

Use:

```bash
bash src/amr_reliability_benchmark/scripts/check_ros_stack.sh
```

to check whether ROS 2/Nav2/Gazebo dependencies are available.

## Evidence Boundary

Until the ROS 2/Nav2/Gazebo setup runs real simulated robot episodes, no claim
should be made about robotics validation. The current contribution is a formal
project migration plan and reproducible simulation scaffold.

