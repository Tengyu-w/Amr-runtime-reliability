#!/usr/bin/env bash
set -euo pipefail

cat <<'MSG'
This script is intentionally conservative.

It does not configure all ROS 2 apt sources by itself. First follow the official
ROS 2 Jazzy Ubuntu deb package installation guide:

https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html

After ROS 2 Jazzy is installed, this script installs common benchmark packages:
  - ros-jazzy-navigation2
  - ros-jazzy-nav2-bringup
  - ros-jazzy-ros-gz

Press Ctrl+C now if ROS 2 Jazzy is not installed yet.
MSG

read -r -p "Continue with apt install? [y/N] " answer
if [[ "${answer,,}" != "y" ]]; then
  echo "Cancelled."
  exit 0
fi

if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  echo "Missing /opt/ros/jazzy/setup.bash. Install ROS 2 Jazzy first."
  exit 1
fi

source /opt/ros/jazzy/setup.bash
sudo apt update
sudo apt install -y \
  ros-jazzy-navigation2 \
  ros-jazzy-nav2-bringup \
  ros-jazzy-ros-gz

echo "Installed requested packages. Run check_ros_stack.sh next."

