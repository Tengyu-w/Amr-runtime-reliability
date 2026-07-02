#!/usr/bin/env bash
set -euo pipefail

echo "Checking ROS 2 / Nav2 / Gazebo Sim stack..."

if ! command -v ros2 >/dev/null 2>&1 && [[ -f /opt/ros/jazzy/setup.bash ]]; then
  echo "Sourcing /opt/ros/jazzy/setup.bash"
  # shellcheck disable=SC1091
  set +u
  source /opt/ros/jazzy/setup.bash
  set -u
fi

missing=0
for cmd in ros2 colcon gz; do
  if command -v "$cmd" >/dev/null 2>&1; then
    echo "OK: $cmd -> $(command -v "$cmd")"
  else
    echo "MISSING: $cmd"
    missing=1
  fi
done

if command -v ros2 >/dev/null 2>&1; then
  ros2 pkg prefix nav2_bringup >/dev/null 2>&1 && echo "OK: nav2_bringup" || { echo "MISSING: nav2_bringup"; missing=1; }
  ros2 pkg prefix ros_gz_bridge >/dev/null 2>&1 && echo "OK: ros_gz_bridge" || { echo "MISSING: ros_gz_bridge"; missing=1; }
  ros2 pkg prefix ros_gz_sim >/dev/null 2>&1 && echo "OK: ros_gz_sim" || { echo "MISSING: ros_gz_sim"; missing=1; }
fi

if [[ "$missing" -ne 0 ]]; then
  echo "Stack incomplete. Install ROS 2 Jazzy, Nav2, Gazebo Sim, and ros_gz before running the benchmark."
  exit 1
fi

echo "Stack looks ready."
