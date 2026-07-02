from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    scenario_id = LaunchConfiguration("scenario_id")
    episode_id = LaunchConfiguration("episode_id")
    publish_period_sec = LaunchConfiguration("publish_period_sec")
    fault_proxy_period_sec = LaunchConfiguration("fault_proxy_period_sec")
    enable_fault_proxy = LaunchConfiguration("enable_fault_proxy")
    enable_policy_monitor = LaunchConfiguration("enable_policy_monitor")
    enable_scan_policy_observation_recorder = LaunchConfiguration("enable_scan_policy_observation_recorder")
    enable_depth_policy_observation_recorder = LaunchConfiguration("enable_depth_policy_observation_recorder")
    enable_recovery_executor = LaunchConfiguration("enable_recovery_executor")
    goal_initial_delay_sec = LaunchConfiguration("goal_initial_delay_sec")
    goal_x = LaunchConfiguration("goal_x")
    goal_y = LaunchConfiguration("goal_y")
    alternate_goal_x = LaunchConfiguration("alternate_goal_x")
    alternate_goal_y = LaunchConfiguration("alternate_goal_y")
    goal_shift_step = LaunchConfiguration("goal_shift_step")
    output_path = LaunchConfiguration("output_path")
    recovery_output_path = LaunchConfiguration("recovery_output_path")
    policy_model_path = LaunchConfiguration("policy_model_path")
    policy_output_path = LaunchConfiguration("policy_output_path")
    scan_policy_output_path = LaunchConfiguration("scan_policy_output_path")
    scan_policy_bins = LaunchConfiguration("scan_policy_bins")
    scan_policy_require_nav2_plan = LaunchConfiguration("scan_policy_require_nav2_plan")
    depth_policy_output_path = LaunchConfiguration("depth_policy_output_path")
    depth_policy_grid_rows = LaunchConfiguration("depth_policy_grid_rows")
    depth_policy_grid_cols = LaunchConfiguration("depth_policy_grid_cols")
    depth_policy_require_nav2_plan = LaunchConfiguration("depth_policy_require_nav2_plan")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "scenario_id",
                default_value="external_path_blockage",
                description="Scenario id from the AMR reliability benchmark catalog.",
            ),
            DeclareLaunchArgument(
                "episode_id",
                default_value="nav2_runtime_episode",
                description="Episode identifier written into routed CSV rows.",
            ),
            DeclareLaunchArgument(
                "publish_period_sec",
                default_value="0.25",
                description="Telemetry adapter publish period in seconds.",
            ),
            DeclareLaunchArgument(
                "fault_proxy_period_sec",
                default_value="0.5",
                description="Scenario-driven reliability proxy publish period in seconds.",
            ),
            DeclareLaunchArgument(
                "enable_fault_proxy",
                default_value="true",
                description="Publish synthetic reliability proxy topics. Disable when Gazebo fault injector is active.",
            ),
            DeclareLaunchArgument(
                "enable_policy_monitor",
                default_value="true",
                description="Run the learned navigation-policy monitor as an observer only.",
            ),
            DeclareLaunchArgument(
                "policy_model_path",
                default_value="",
                description="JSON export of the navigation policy. Empty path uses a heuristic probe.",
            ),
            DeclareLaunchArgument(
                "policy_output_path",
                default_value="outputs/ros2_episode_logs/nav2_runtime_policy_episode.csv",
                description="CSV path for observed navigation-policy decisions.",
            ),
            DeclareLaunchArgument(
                "enable_scan_policy_observation_recorder",
                default_value="true",
                description="Record /scan observations aligned with Nav2-plan expert policy labels.",
            ),
            DeclareLaunchArgument(
                "enable_depth_policy_observation_recorder",
                default_value="true",
                description="Record depth images aligned with Nav2-plan expert policy labels.",
            ),
            DeclareLaunchArgument(
                "enable_recovery_executor",
                default_value="false",
                description="Translate router decisions into Nav2-facing recovery actions for closed-loop demos.",
            ),
            DeclareLaunchArgument(
                "scan_policy_output_path",
                default_value="outputs/ros2_episode_logs/nav2_runtime_scan_policy_observations.csv",
                description="CSV path for scan-observation policy training rows.",
            ),
            DeclareLaunchArgument(
                "depth_policy_output_path",
                default_value="outputs/ros2_episode_logs/nav2_runtime_depth_policy_observations.csv",
                description="CSV path for depth-observation policy training rows.",
            ),
            DeclareLaunchArgument(
                "scan_policy_bins",
                default_value="72",
                description="Number of downsampled LaserScan bins to record as policy observations.",
            ),
            DeclareLaunchArgument(
                "scan_policy_require_nav2_plan",
                default_value="true",
                description="Require Nav2-plan expert labels for scan-policy rows. Set false to include proxy fallback labels.",
            ),
            DeclareLaunchArgument(
                "depth_policy_grid_rows",
                default_value="8",
                description="Rows in the downsampled depth-image observation grid.",
            ),
            DeclareLaunchArgument(
                "depth_policy_grid_cols",
                default_value="12",
                description="Columns in the downsampled depth-image observation grid.",
            ),
            DeclareLaunchArgument(
                "depth_policy_require_nav2_plan",
                default_value="true",
                description="Require Nav2-plan expert labels for depth-policy rows. Set false to include proxy fallback labels.",
            ),
            DeclareLaunchArgument(
                "goal_initial_delay_sec",
                default_value="35.0",
                description="Delay before publishing the first Nav2 goal, allowing lifecycle bringup to finish.",
            ),
            DeclareLaunchArgument("goal_x", default_value="4.5", description="Primary Nav2 goal x coordinate."),
            DeclareLaunchArgument("goal_y", default_value="3.0", description="Primary Nav2 goal y coordinate."),
            DeclareLaunchArgument("alternate_goal_x", default_value="-4.5", description="Alternate Nav2 goal x coordinate."),
            DeclareLaunchArgument("alternate_goal_y", default_value="3.0", description="Alternate Nav2 goal y coordinate."),
            DeclareLaunchArgument("goal_shift_step", default_value="6", description="Scenario step that triggers alternate goal use."),
            DeclareLaunchArgument(
                "output_path",
                default_value="outputs/ros2_episode_logs/nav2_runtime_episode.csv",
                description="CSV path for routed episode rows.",
            ),
            DeclareLaunchArgument(
                "recovery_output_path",
                default_value="outputs/ros2_episode_logs/nav2_runtime_recovery_execution.csv",
                description="CSV path for recovery executor events.",
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
                executable="scenario_goal_publisher",
                name="scenario_goal_publisher",
                output="screen",
                parameters=[
                    {
                        "initial_publish_delay_sec": goal_initial_delay_sec,
                        "goal_x": goal_x,
                        "goal_y": goal_y,
                        "alternate_goal_x": alternate_goal_x,
                        "alternate_goal_y": alternate_goal_y,
                        "goal_shift_step": goal_shift_step,
                    }
                ],
            ),
            Node(
                condition=IfCondition(enable_fault_proxy),
                package="amr_reliability_benchmark",
                executable="fault_proxy_publisher",
                name="fault_proxy_publisher",
                output="screen",
                parameters=[
                    {
                        "publish_period_sec": fault_proxy_period_sec,
                    }
                ],
            ),
            Node(
                package="amr_reliability_benchmark",
                executable="nav2_telemetry_adapter",
                name="nav2_telemetry_adapter",
                output="screen",
                parameters=[
                    {
                        "episode_id": episode_id,
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
                condition=IfCondition(enable_recovery_executor),
                package="amr_reliability_benchmark",
                executable="recovery_executor",
                name="recovery_executor",
                output="screen",
                parameters=[{"output_path": recovery_output_path}],
            ),
            Node(
                condition=IfCondition(enable_policy_monitor),
                package="amr_reliability_benchmark",
                executable="navigation_policy_monitor",
                name="navigation_policy_monitor",
                output="screen",
                parameters=[
                    {
                        "model_path": policy_model_path,
                    }
                ],
            ),
            Node(
                package="amr_reliability_benchmark",
                executable="episode_recorder",
                name="episode_recorder",
                output="screen",
                parameters=[{"output_path": output_path}],
            ),
            Node(
                condition=IfCondition(enable_policy_monitor),
                package="amr_reliability_benchmark",
                executable="policy_episode_recorder",
                name="policy_episode_recorder",
                output="screen",
                parameters=[{"output_path": policy_output_path}],
            ),
            Node(
                condition=IfCondition(enable_scan_policy_observation_recorder),
                package="amr_reliability_benchmark",
                executable="scan_policy_observation_recorder",
                name="scan_policy_observation_recorder",
                output="screen",
                parameters=[
                    {
                        "output_path": scan_policy_output_path,
                        "scan_bins": scan_policy_bins,
                        "require_nav2_plan": scan_policy_require_nav2_plan,
                    }
                ],
            ),
            Node(
                condition=IfCondition(enable_depth_policy_observation_recorder),
                package="amr_reliability_benchmark",
                executable="depth_policy_observation_recorder",
                name="depth_policy_observation_recorder",
                output="screen",
                parameters=[
                    {
                        "output_path": depth_policy_output_path,
                        "grid_rows": depth_policy_grid_rows,
                        "grid_cols": depth_policy_grid_cols,
                        "require_nav2_plan": depth_policy_require_nav2_plan,
                    }
                ],
            ),
        ]
    )
