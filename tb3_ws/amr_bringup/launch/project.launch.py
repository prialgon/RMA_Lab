from launch import LaunchDescription
from launch_ros.actions import LifecycleNode, Node


def generate_launch_description():
    simulation = False
    world = "project"
    goal = (-0.6, 1.0)

    particle_filter_node = LifecycleNode(
        package="amr_localization",
        executable="particle_filter",
        name="particle_filter",
        namespace="",
        output="screen",
        arguments=["--ros-args", "--log-level", "WARN"],
        parameters=[
            {
                "enable_plot": False,
                "global_localization": True,
                "particles": 2000,
                "sigma_v": 0.05,
                "sigma_w": 0.1,
                "sigma_z": 0.2,
                "simulation": simulation,
                "world": world,
            }
        ],
    )

    probabilistic_roadmap_node = LifecycleNode(
        package="amr_planning",
        executable="probabilistic_roadmap",
        name="probabilistic_roadmap",
        namespace="",
        output="screen",
        arguments=["--ros-args", "--log-level", "WARN"],
        parameters=[
            {
                "connection_distance": 0.15,
                "enable_plot": False,
                "goal": goal,
                "grid_size": 0.1,
                "node_count": 250,
                "obstacle_safety_distance": 0.12,
                "simulation": simulation,
                "smoothing_additional_points": 3,
                "smoothing_data_weight": 0.1,
                "smoothing_smooth_weight": 0.25,
                "use_grid": True,
                "world": world,
            }
        ],
    )

    odometry_node = LifecycleNode(
        package="amr_turtlebot3",
        executable="odometry",
        name="odometry",
        namespace="",
        output="screen",
        arguments=["--ros-args", "--log-level", "INFO"],
    )

    wall_follower_node = LifecycleNode(
        package="amr_control",
        executable="wall_follower",
        name="wall_follower",
        namespace="",
        output="screen",
        arguments=["--ros-args", "--log-level", "WARN"],
        parameters=[
            {
                "enable_localization": True,
                "simulation": simulation,
            }
        ],
    )

    pure_pursuit_node = LifecycleNode(
        package="amr_control",
        executable="pure_pursuit",
        name="pure_pursuit",
        namespace="",
        output="screen",
        arguments=["--ros-args", "--log-level", "WARN"],
        parameters=[
            {
                "lookahead_distance": 0.3,
                "simulation": simulation,
            }
        ],
    )

    monitoring_node = LifecycleNode(
        package="amr_simulation",
        executable="monitoring",
        name="monitoring",
        namespace="",
        output="screen",
        arguments=["--ros-args", "--log-level", "WARN"],
        parameters=[
            {
                "goal": goal,
                "goal_tolerance": 0.1,
            }
        ],
    )

    lifecycle_manager_node = Node(
        package="amr_bringup",
        executable="lifecycle_manager",
        output="screen",
        arguments=["--ros-args", "--log-level", "WARN"],
        parameters=[
            {
                "node_startup_order": (
                    "particle_filter",
                    "probabilistic_roadmap",
                    "odometry",
                    "wall_follower",
                    "pure_pursuit",
                    "monitoring",
                )
            }
        ],
    )

    return LaunchDescription(
        [
            particle_filter_node,
            probabilistic_roadmap_node,
            odometry_node,
            wall_follower_node,
            pure_pursuit_node,
            monitoring_node,
            lifecycle_manager_node,  # Must be launched last
        ]
    )
