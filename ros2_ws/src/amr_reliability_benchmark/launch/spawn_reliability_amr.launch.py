from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    world = LaunchConfiguration("world")
    robot_name = LaunchConfiguration("robot_name")
    x = LaunchConfiguration("x")
    y = LaunchConfiguration("y")
    z = LaunchConfiguration("z")
    yaw = LaunchConfiguration("yaw")
    model_file = PathJoinSubstitution(
        [
            FindPackageShare("amr_reliability_benchmark"),
            "models",
            "reliability_amr",
            "model.sdf",
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "world",
                default_value="reliability_room",
                description="Gazebo world name receiving the AMR entity.",
            ),
            DeclareLaunchArgument(
                "robot_name",
                default_value="reliability_amr",
                description="Gazebo entity name for the benchmark AMR.",
            ),
            DeclareLaunchArgument("x", default_value="-4.5"),
            DeclareLaunchArgument("y", default_value="-3.0"),
            DeclareLaunchArgument("z", default_value="0.02"),
            DeclareLaunchArgument("yaw", default_value="0.0"),
            Node(
                package="ros_gz_sim",
                executable="create",
                name="spawn_reliability_amr",
                output="screen",
                arguments=[
                    "-world",
                    world,
                    "-file",
                    model_file,
                    "-name",
                    robot_name,
                    "-allow_renaming",
                    "true",
                    "-x",
                    x,
                    "-y",
                    y,
                    "-z",
                    z,
                    "-Y",
                    yaw,
                ],
            ),
        ]
    )
