from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    scenario_id = LaunchConfiguration("scenario_id")
    episode_id = LaunchConfiguration("episode_id")
    map_file = LaunchConfiguration("map")
    params_file = LaunchConfiguration("params_file")
    output_path = LaunchConfiguration("output_path")
    use_static_odom_tf = LaunchConfiguration("use_static_odom_tf")
    use_static_map_tf = LaunchConfiguration("use_static_map_tf")

    robot_state_launch = PathJoinSubstitution(
        [FindPackageShare("amr_reliability_benchmark"), "launch", "robot_state.launch.py"]
    )
    nav2_bringup_launch = PathJoinSubstitution(
        [FindPackageShare("nav2_bringup"), "launch", "bringup_launch.py"]
    )
    runtime_pipeline_launch = PathJoinSubstitution(
        [FindPackageShare("amr_reliability_benchmark"), "launch", "nav2_runtime_pipeline.launch.py"]
    )
    default_map = PathJoinSubstitution(
        [FindPackageShare("amr_reliability_benchmark"), "maps", "reliability_room.yaml"]
    )
    default_params = PathJoinSubstitution(
        [FindPackageShare("amr_reliability_benchmark"), "config", "nav2_params.yaml"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use simulation time for Gazebo/Nav2 experiments.",
            ),
            DeclareLaunchArgument(
                "scenario_id",
                default_value="external_path_blockage",
                description="Reliability benchmark scenario id.",
            ),
            DeclareLaunchArgument(
                "episode_id",
                default_value="nav2_benchmark_episode",
                description="Episode identifier written to routed reliability logs.",
            ),
            DeclareLaunchArgument(
                "map",
                default_value=default_map,
                description="Nav2 map YAML file.",
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params,
                description="Nav2 parameter YAML file.",
            ),
            DeclareLaunchArgument(
                "output_path",
                default_value="outputs/ros2_episode_logs/nav2_benchmark_episode.csv",
                description="CSV output path for routed reliability episode rows.",
            ),
            DeclareLaunchArgument(
                "use_static_odom_tf",
                default_value="false",
                description="Publish a static odom->base_link transform for smoke tests without Gazebo odometry.",
            ),
            DeclareLaunchArgument(
                "use_static_map_tf",
                default_value="false",
                description="Publish a static map->odom transform for smoke tests without AMCL scan input.",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(robot_state_launch),
                launch_arguments={"use_sim_time": use_sim_time}.items(),
            ),
            Node(
                condition=IfCondition(use_static_odom_tf),
                package="tf2_ros",
                executable="static_transform_publisher",
                name="static_odom_to_base_link",
                arguments=["0", "0", "0", "0", "0", "0", "odom", "base_link"],
            ),
            Node(
                condition=IfCondition(use_static_map_tf),
                package="tf2_ros",
                executable="static_transform_publisher",
                name="static_map_to_odom",
                arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(nav2_bringup_launch),
                launch_arguments={
                    "use_sim_time": use_sim_time,
                    "map": map_file,
                    "params_file": params_file,
                }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(runtime_pipeline_launch),
                launch_arguments={
                    "scenario_id": scenario_id,
                    "episode_id": episode_id,
                    "output_path": output_path,
                }.items(),
            ),
        ]
    )
