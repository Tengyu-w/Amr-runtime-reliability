from glob import glob
from setuptools import setup


package_name = "amr_reliability_benchmark"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*")),
        (f"share/{package_name}/maps", glob("maps/*")),
        (f"share/{package_name}/models/reliability_amr", glob("models/reliability_amr/*")),
        (f"share/{package_name}/urdf", glob("urdf/*")),
        (f"share/{package_name}/worlds", glob("worlds/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Research Prototype",
    maintainer_email="research@example.com",
    description="ROS 2/Nav2/Gazebo AMR reliability benchmark scaffold.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "depth_policy_observation_recorder = amr_reliability_benchmark.depth_policy_observation_recorder:main",
            "episode_recorder = amr_reliability_benchmark.episode_recorder:main",
            "fault_proxy_publisher = amr_reliability_benchmark.fault_proxy_publisher:main",
            "gazebo_fault_injector = amr_reliability_benchmark.gazebo_fault_injector:main",
            "metrics_replay = amr_reliability_benchmark.metrics_replay:main",
            "navigation_policy_monitor = amr_reliability_benchmark.navigation_policy_monitor:main",
            "nav2_telemetry_adapter = amr_reliability_benchmark.nav2_telemetry_adapter:main",
            "policy_episode_recorder = amr_reliability_benchmark.policy_episode_recorder:main",
            "recovery_executor = amr_reliability_benchmark.recovery_executor:main",
            "runtime_router = amr_reliability_benchmark.runtime_router:main",
            "scan_policy_observation_recorder = amr_reliability_benchmark.scan_policy_observation_recorder:main",
            "scenario_catalog = amr_reliability_benchmark.scenario_catalog:main",
            "scenario_goal_publisher = amr_reliability_benchmark.scenario_goal_publisher:main",
            "scenario_runner = amr_reliability_benchmark.scenario_runner:main",
        ],
    },
)
