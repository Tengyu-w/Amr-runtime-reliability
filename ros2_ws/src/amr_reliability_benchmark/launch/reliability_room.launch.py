from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    world = PathJoinSubstitution(
        [
            FindPackageShare("amr_reliability_benchmark"),
            "worlds",
            "reliability_room.sdf",
        ]
    )
    return LaunchDescription(
        [
            ExecuteProcess(
                cmd=["gz", "sim", "-r", world],
                output="screen",
            )
        ]
    )

