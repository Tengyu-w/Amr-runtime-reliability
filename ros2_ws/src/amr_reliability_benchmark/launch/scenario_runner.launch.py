from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    scenario_id = LaunchConfiguration("scenario_id")
    publish_once = LaunchConfiguration("publish_once")
    publish_period_sec = LaunchConfiguration("publish_period_sec")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "scenario_id",
                default_value="nominal",
                description="Scenario id from the AMR reliability benchmark catalog.",
            ),
            DeclareLaunchArgument(
                "publish_once",
                default_value="false",
                description="Publish the selected scenario once and exit for smoke tests.",
            ),
            DeclareLaunchArgument(
                "publish_period_sec",
                default_value="1.0",
                description="Scenario heartbeat period in seconds.",
            ),
            Node(
                package="amr_reliability_benchmark",
                executable="scenario_runner",
                name="scenario_runner",
                output="screen",
                parameters=[
                    {
                        "scenario_id": scenario_id,
                        "publish_once": publish_once,
                        "publish_period_sec": publish_period_sec,
                    }
                ],
            ),
        ]
    )
