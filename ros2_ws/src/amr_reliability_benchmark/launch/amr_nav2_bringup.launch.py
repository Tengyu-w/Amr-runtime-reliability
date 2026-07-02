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
    recovery_output_path = LaunchConfiguration("recovery_output_path")
    enable_recovery_executor = LaunchConfiguration("enable_recovery_executor")
    replan_cooldown_steps = LaunchConfiguration("replan_cooldown_steps")
    relocalize_cooldown_steps = LaunchConfiguration("relocalize_cooldown_steps")
    enable_policy_monitor = LaunchConfiguration("enable_policy_monitor")
    policy_model_path = LaunchConfiguration("policy_model_path")
    policy_output_path = LaunchConfiguration("policy_output_path")
    enable_scan_policy_observation_recorder = LaunchConfiguration("enable_scan_policy_observation_recorder")
    scan_policy_output_path = LaunchConfiguration("scan_policy_output_path")
    scan_policy_bins = LaunchConfiguration("scan_policy_bins")
    scan_policy_require_nav2_plan = LaunchConfiguration("scan_policy_require_nav2_plan")
    enable_depth_policy_observation_recorder = LaunchConfiguration("enable_depth_policy_observation_recorder")
    depth_policy_output_path = LaunchConfiguration("depth_policy_output_path")
    depth_policy_grid_rows = LaunchConfiguration("depth_policy_grid_rows")
    depth_policy_grid_cols = LaunchConfiguration("depth_policy_grid_cols")
    depth_policy_require_nav2_plan = LaunchConfiguration("depth_policy_require_nav2_plan")
    enable_fault_proxy = LaunchConfiguration("enable_fault_proxy")
    fault_proxy_period_sec = LaunchConfiguration("fault_proxy_period_sec")
    publish_period_sec = LaunchConfiguration("publish_period_sec")
    goal_initial_delay_sec = LaunchConfiguration("goal_initial_delay_sec")
    goal_x = LaunchConfiguration("goal_x")
    goal_y = LaunchConfiguration("goal_y")
    alternate_goal_x = LaunchConfiguration("alternate_goal_x")
    alternate_goal_y = LaunchConfiguration("alternate_goal_y")
    goal_shift_step = LaunchConfiguration("goal_shift_step")
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
                "recovery_output_path",
                default_value="outputs/ros2_episode_logs/nav2_benchmark_recovery_execution.csv",
                description="CSV output path for recovery executor events.",
            ),
            DeclareLaunchArgument(
                "enable_recovery_executor",
                default_value="false",
                description="Enable route-to-Nav2 recovery execution bridge.",
            ),
            DeclareLaunchArgument(
                "replan_cooldown_steps",
                default_value="20",
                description="Minimum routed time steps between Nav2 goal reissue recovery actions.",
            ),
            DeclareLaunchArgument(
                "relocalize_cooldown_steps",
                default_value="8",
                description="Minimum routed time steps between /initialpose relocalization recovery actions.",
            ),
            DeclareLaunchArgument("enable_policy_monitor", default_value="true"),
            DeclareLaunchArgument("policy_model_path", default_value=""),
            DeclareLaunchArgument(
                "policy_output_path",
                default_value="outputs/ros2_episode_logs/nav2_benchmark_policy_episode.csv",
            ),
            DeclareLaunchArgument("enable_scan_policy_observation_recorder", default_value="true"),
            DeclareLaunchArgument(
                "scan_policy_output_path",
                default_value="outputs/ros2_episode_logs/nav2_benchmark_scan_policy_observations.csv",
            ),
            DeclareLaunchArgument("scan_policy_bins", default_value="72"),
            DeclareLaunchArgument("scan_policy_require_nav2_plan", default_value="true"),
            DeclareLaunchArgument("enable_depth_policy_observation_recorder", default_value="true"),
            DeclareLaunchArgument(
                "depth_policy_output_path",
                default_value="outputs/ros2_episode_logs/nav2_benchmark_depth_policy_observations.csv",
            ),
            DeclareLaunchArgument("depth_policy_grid_rows", default_value="8"),
            DeclareLaunchArgument("depth_policy_grid_cols", default_value="12"),
            DeclareLaunchArgument("depth_policy_require_nav2_plan", default_value="true"),
            DeclareLaunchArgument("enable_fault_proxy", default_value="true"),
            DeclareLaunchArgument("fault_proxy_period_sec", default_value="0.5"),
            DeclareLaunchArgument("publish_period_sec", default_value="0.25"),
            DeclareLaunchArgument(
                "goal_initial_delay_sec",
                default_value="35.0",
                description="Delay before publishing the first Nav2 goal.",
            ),
            DeclareLaunchArgument("goal_x", default_value="4.5"),
            DeclareLaunchArgument("goal_y", default_value="3.0"),
            DeclareLaunchArgument("alternate_goal_x", default_value="-4.5"),
            DeclareLaunchArgument("alternate_goal_y", default_value="3.0"),
            DeclareLaunchArgument("goal_shift_step", default_value="6"),
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
                    "recovery_output_path": recovery_output_path,
                    "enable_recovery_executor": enable_recovery_executor,
                    "replan_cooldown_steps": replan_cooldown_steps,
                    "relocalize_cooldown_steps": relocalize_cooldown_steps,
                    "enable_policy_monitor": enable_policy_monitor,
                    "policy_model_path": policy_model_path,
                    "policy_output_path": policy_output_path,
                    "enable_scan_policy_observation_recorder": enable_scan_policy_observation_recorder,
                    "scan_policy_output_path": scan_policy_output_path,
                    "scan_policy_bins": scan_policy_bins,
                    "scan_policy_require_nav2_plan": scan_policy_require_nav2_plan,
                    "enable_depth_policy_observation_recorder": enable_depth_policy_observation_recorder,
                    "depth_policy_output_path": depth_policy_output_path,
                    "depth_policy_grid_rows": depth_policy_grid_rows,
                    "depth_policy_grid_cols": depth_policy_grid_cols,
                    "depth_policy_require_nav2_plan": depth_policy_require_nav2_plan,
                    "enable_fault_proxy": enable_fault_proxy,
                    "fault_proxy_period_sec": fault_proxy_period_sec,
                    "publish_period_sec": publish_period_sec,
                    "goal_initial_delay_sec": goal_initial_delay_sec,
                    "goal_x": goal_x,
                    "goal_y": goal_y,
                    "alternate_goal_x": alternate_goal_x,
                    "alternate_goal_y": alternate_goal_y,
                    "goal_shift_step": goal_shift_step,
                }.items(),
            ),
        ]
    )
