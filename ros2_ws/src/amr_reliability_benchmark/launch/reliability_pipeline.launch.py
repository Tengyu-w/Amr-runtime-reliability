from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    scenario_id = LaunchConfiguration("scenario_id")
    episode_id = LaunchConfiguration("episode_id")
    steps = LaunchConfiguration("steps")
    publish_period_sec = LaunchConfiguration("publish_period_sec")
    output_path = LaunchConfiguration("output_path")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "scenario_id",
                default_value="external_path_blockage",
                description="Scenario id from the AMR reliability benchmark catalog.",
            ),
            DeclareLaunchArgument(
                "episode_id",
                default_value="ros2_replay_episode",
                description="Episode identifier written into routed CSV rows.",
            ),
            DeclareLaunchArgument(
                "steps",
                default_value="12",
                description="Number of replayed metric rows before metrics_replay becomes idle.",
            ),
            DeclareLaunchArgument(
                "publish_period_sec",
                default_value="0.25",
                description="Metrics replay period in seconds.",
            ),
            DeclareLaunchArgument(
                "output_path",
                default_value="outputs/ros2_episode_logs/episode.csv",
                description="CSV path for routed episode rows.",
            ),
            Node(
                package="amr_reliability_benchmark",
                executable="scenario_runner",
                name="scenario_runner",
                output="screen",
                parameters=[
                    {
                        "scenario_id": scenario_id,
                        "publish_once": False,
                        "publish_period_sec": 1.0,
                    }
                ],
            ),
            Node(
                package="amr_reliability_benchmark",
                executable="metrics_replay",
                name="metrics_replay",
                output="screen",
                parameters=[
                    {
                        "scenario_id": scenario_id,
                        "episode_id": episode_id,
                        "steps": steps,
                        "publish_period_sec": publish_period_sec,
                    }
                ],
            ),
            Node(
                package="amr_reliability_benchmark",
                executable="runtime_router",
                name="runtime_router",
                output="screen",
            ),
            Node(
                package="amr_reliability_benchmark",
                executable="episode_recorder",
                name="episode_recorder",
                output="screen",
                parameters=[{"output_path": output_path}],
            ),
        ]
    )
