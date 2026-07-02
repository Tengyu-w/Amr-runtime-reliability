from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    scenario_id = LaunchConfiguration("scenario_id")
    episode_id = LaunchConfiguration("episode_id")
    output_path = LaunchConfiguration("output_path")
    policy_model_path = LaunchConfiguration("policy_model_path")
    policy_output_path = LaunchConfiguration("policy_output_path")
    scan_policy_output_path = LaunchConfiguration("scan_policy_output_path")
    enable_scan_policy_observation_recorder = LaunchConfiguration("enable_scan_policy_observation_recorder")
    scan_policy_require_nav2_plan = LaunchConfiguration("scan_policy_require_nav2_plan")
    depth_policy_output_path = LaunchConfiguration("depth_policy_output_path")
    enable_depth_policy_observation_recorder = LaunchConfiguration("enable_depth_policy_observation_recorder")
    depth_policy_require_nav2_plan = LaunchConfiguration("depth_policy_require_nav2_plan")
    goal_x = LaunchConfiguration("goal_x")
    goal_y = LaunchConfiguration("goal_y")
    alternate_goal_x = LaunchConfiguration("alternate_goal_x")
    alternate_goal_y = LaunchConfiguration("alternate_goal_y")
    goal_shift_step = LaunchConfiguration("goal_shift_step")
    fault_seed = LaunchConfiguration("fault_seed")
    gz_args = LaunchConfiguration("gz_args")

    pkg_share = FindPackageShare("amr_reliability_benchmark")
    world = PathJoinSubstitution([pkg_share, "worlds", "reliability_room.sdf"])
    gz_launch = PathJoinSubstitution([FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"])
    spawn_launch = PathJoinSubstitution([pkg_share, "launch", "spawn_reliability_amr.launch.py"])
    nav2_launch = PathJoinSubstitution([pkg_share, "launch", "amr_nav2_bringup.launch.py"])

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use Gazebo clock for Nav2 and reliability nodes.",
            ),
            DeclareLaunchArgument(
                "scenario_id",
                default_value="external_path_blockage",
                description="Reliability benchmark scenario id.",
            ),
            DeclareLaunchArgument(
                "episode_id",
                default_value="gazebo_nav2_episode",
                description="Episode identifier written to reliability CSV rows.",
            ),
            DeclareLaunchArgument(
                "output_path",
                default_value="outputs/ros2_episode_logs/gazebo_nav2_episode.csv",
                description="CSV output path for routed reliability episode rows.",
            ),
            DeclareLaunchArgument(
                "policy_model_path",
                default_value="",
                description="JSON export for the navigation-policy monitor. Empty path uses a heuristic probe.",
            ),
            DeclareLaunchArgument(
                "policy_output_path",
                default_value="outputs/ros2_episode_logs/gazebo_nav2_policy_episode.csv",
                description="CSV output path for navigation-policy monitor rows.",
            ),
            DeclareLaunchArgument(
                "scan_policy_output_path",
                default_value="outputs/ros2_episode_logs/gazebo_nav2_scan_policy_observations.csv",
                description="CSV output path for scan observations aligned with expert policy labels.",
            ),
            DeclareLaunchArgument(
                "enable_scan_policy_observation_recorder",
                default_value="true",
                description="Record /scan observations aligned with expert policy labels.",
            ),
            DeclareLaunchArgument(
                "depth_policy_output_path",
                default_value="outputs/ros2_episode_logs/gazebo_nav2_depth_policy_observations.csv",
                description="CSV output path for depth observations aligned with expert policy labels.",
            ),
            DeclareLaunchArgument(
                "enable_depth_policy_observation_recorder",
                default_value="true",
                description="Record depth observations aligned with expert policy labels.",
            ),
            DeclareLaunchArgument(
                "scan_policy_require_nav2_plan",
                default_value="true",
                description="Require Nav2-plan expert labels for scan-policy rows. Set false to include proxy fallback labels.",
            ),
            DeclareLaunchArgument(
                "depth_policy_require_nav2_plan",
                default_value="true",
                description="Require Nav2-plan expert labels for depth-policy rows. Set false to include proxy fallback labels.",
            ),
            DeclareLaunchArgument("goal_x", default_value="4.5", description="Primary Nav2 goal x coordinate."),
            DeclareLaunchArgument("goal_y", default_value="3.0", description="Primary Nav2 goal y coordinate."),
            DeclareLaunchArgument("alternate_goal_x", default_value="-4.5", description="Alternate Nav2 goal x coordinate."),
            DeclareLaunchArgument("alternate_goal_y", default_value="3.0", description="Alternate Nav2 goal y coordinate."),
            DeclareLaunchArgument("goal_shift_step", default_value="6", description="Scenario step that triggers alternate goal use."),
            DeclareLaunchArgument(
                "fault_seed",
                default_value="17",
                description="Seed for stochastic Gazebo fault injection.",
            ),
            DeclareLaunchArgument(
                "gz_args",
                default_value=["-r -s ", world],
                description="Gazebo Sim arguments. Default is headless server mode.",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(gz_launch),
                launch_arguments={"gz_args": gz_args, "on_exit_shutdown": "true"}.items(),
            ),
            TimerAction(
                period=2.0,
                actions=[
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(spawn_launch),
                        launch_arguments={"world": "reliability_room"}.items(),
                    )
                ],
            ),
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                name="reliability_gz_bridge",
                output="screen",
                arguments=[
                    "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
                    "/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
                    "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
                    "/ground_truth/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
                    "/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
                    "/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
                    "/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
                ],
                remappings=[
                    ("/odom", "/gazebo/odom"),
                    ("/scan", "/gazebo/scan"),
                ],
                parameters=[
                    {
                        "qos_overrides./cmd_vel.subscriber.reliability": "reliable",
                        "qos_overrides./scan.publisher.reliability": "best_effort",
                        "qos_overrides./depth_image.publisher.reliability": "best_effort",
                    }
                ],
            ),
            Node(
                package="amr_reliability_benchmark",
                executable="gazebo_fault_injector",
                name="gazebo_fault_injector",
                output="screen",
                parameters=[
                    {
                        "world": "reliability_room",
                        "obstacle_name": "dynamic_obstacle_placeholder",
                        "seed": fault_seed,
                    }
                ],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="static_base_to_gazebo_scan",
                output="screen",
                arguments=[
                    "0.22",
                    "0.0",
                    "0.44",
                    "0.0",
                    "0.0",
                    "0.0",
                    "base_link",
                    "reliability_amr/laser_link/scan",
                ],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="static_base_to_depth_camera",
                output="screen",
                arguments=[
                    "0.34",
                    "0.0",
                    "0.52",
                    "0.0",
                    "0.0",
                    "0.0",
                    "base_link",
                    "reliability_amr/depth_camera_link/depth_camera",
                ],
            ),
            TimerAction(
                period=4.0,
                actions=[
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(nav2_launch),
                        launch_arguments={
                            "use_sim_time": use_sim_time,
                            "scenario_id": scenario_id,
                            "episode_id": episode_id,
                            "output_path": output_path,
                            "policy_model_path": policy_model_path,
                            "policy_output_path": policy_output_path,
                            "scan_policy_output_path": scan_policy_output_path,
                            "depth_policy_output_path": depth_policy_output_path,
                            "enable_scan_policy_observation_recorder": enable_scan_policy_observation_recorder,
                            "enable_depth_policy_observation_recorder": enable_depth_policy_observation_recorder,
                            "scan_policy_require_nav2_plan": scan_policy_require_nav2_plan,
                            "depth_policy_require_nav2_plan": depth_policy_require_nav2_plan,
                            "goal_x": goal_x,
                            "goal_y": goal_y,
                            "alternate_goal_x": alternate_goal_x,
                            "alternate_goal_y": alternate_goal_y,
                            "goal_shift_step": goal_shift_step,
                            "use_static_odom_tf": "false",
                            "use_static_map_tf": "false",
                            "enable_fault_proxy": "false",
                        }.items(),
                    )
                ],
            ),
        ]
    )
