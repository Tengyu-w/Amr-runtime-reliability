# Install ROS 2 Jazzy, Nav2, and Gazebo Sim

Target environment:

```text
Ubuntu 24.04 Noble
ROS 2 Jazzy
Nav2
Gazebo Sim
ros_gz
```

The current WSL environment has Ubuntu 24.04 and `colcon`, but `ros2` and `gz`
are not installed.

## Official References

- ROS 2 Jazzy Ubuntu deb packages:
  https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html
- ROS 2 Gazebo tutorials:
  https://docs.ros.org/en/jazzy/Tutorials/Advanced/Simulators/Gazebo/Simulation-Gazebo.html
- Nav2 install notes:
  https://docs.nav2.org/development_guides/build_docs/index.html
- Gazebo with ROS:
  https://gazebosim.org/docs/latest/ros_installation/

## Manual Install Outline

Follow the ROS 2 Jazzy official Ubuntu deb package instructions first.

After ROS 2 Jazzy is installed and sourced:

```bash
source /opt/ros/jazzy/setup.bash
sudo apt update
sudo apt install ros-jazzy-navigation2 ros-jazzy-nav2-bringup
sudo apt install ros-jazzy-ros-gz
```

Depending on the Gazebo release packaging available for the machine, Gazebo Sim
may also need OSRF package setup from Gazebo's official installation
instructions.

## Verify

From:

```bash
cd /mnt/c/Users/77941/Desktop/ecg_uncertainty_project/AMR-Runtime-Reliability-Demo/ros2_ws
```

run:

```bash
bash src/amr_reliability_benchmark/scripts/check_ros_stack.sh
```

Expected once installed:

```text
OK: ros2
OK: colcon
OK: gz
OK: nav2_bringup
OK: ros_gz_bridge
OK: ros_gz_sim
```

Then:

```bash
colcon build --symlink-install
source install/setup.bash
ros2 launch amr_reliability_benchmark reliability_room.launch.py
```

## Notes

Do not describe results from this stack until the check script passes and ROS 2
episodes are actually recorded. The Python simulator is only a prototype and
should not be presented as the final robotics simulation environment.

