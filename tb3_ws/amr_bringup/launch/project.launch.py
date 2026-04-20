from launch import LaunchDescription
from launch_ros.actions import LifecycleNode, Node

import math


def generate_launch_description():
    simulation = False
    world = "project"
    start = (0.2, -0.6, math.radians(270))
    goal = (-1.0, 0.6)
    localization_error_position_threshold = 0.50
    localization_error_angle_threshold = 30.0

    reset_on_low_likelihood = True
    low_likelihood_log_threshold = -0.75
    low_likelihood_patience = 3

    particle_filter_node = LifecycleNode(
        package="amr_localization_cpp",
        executable="particle_filter_node.py",
        name="particle_filter",
        namespace="",
        output="screen",
        arguments=["--ros-args", "--log-level", "INFO"],
        parameters=[
            {
                "enable_plot": False,
                "global_localization": True,
                "particles": 3000,
                "sigma_v": 0.05,
                "sigma_w": 0.1,
                "sigma_z": 0.25,
                "simulation": simulation,
                "world": world,
                "steps_btw_sense_updates": 15,  # 20
                "reset_on_low_likelihood": reset_on_low_likelihood,
                "low_likelihood_log_threshold": low_likelihood_log_threshold,
                "low_likelihood_patience": low_likelihood_patience,
            }
        ],
    )

    # ekf_node = LifecycleNode(
    #     package="amr_localization",
    #     executable="ekf",
    #     name="ekf",
    #     namespace="",
    #     output="screen",
    #     arguments=["--ros-args", "--log-level", "WARN"],
    #     parameters=[
    #         {
    #             "enable_plot": True,
    #             "simulation": simulation,
    #             "world": world,
    #             "particle_filter_pose_topic": "particle_filter/pose",
    #         }
    #     ],
    # )

    probabilistic_roadmap_node = LifecycleNode(
        package="amr_planning",
        executable="probabilistic_roadmap",
        name="probabilistic_roadmap",
        namespace="",
        output="screen",
        arguments=["--ros-args", "--log-level", "WARN"],
        parameters=[
            {
                "connection_distance": 0.3,
                "enable_plot": True,
                "goal": goal,
                "grid_size": 0.1,
                "node_count": 500,
                "obstacle_safety_distance": 0.15,
                "simulation": simulation,
                "smoothing_additional_points": 3,
                "smoothing_data_weight": 0.25,
                "smoothing_smooth_weight": 0.15,
                "use_grid": True,
                "world": world,
            }
        ],
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
                "lookahead_distance": 0.1,
                "goal_tolerance": 0.05,
                "simulation": simulation,
            }
        ],
    )

    if simulation:
        coppeliasim_node = LifecycleNode(
            package="amr_simulation",
            executable="coppeliasim",
            name="coppeliasim",
            namespace="",
            output="screen",
            arguments=["--ros-args", "--log-level", "WARN"],
            parameters=[
                {
                    "enable_localization": True,
                    "goal": goal,
                    "pose_tolerance": (0.1, 10.0),
                    "localization_error_position_threshold": localization_error_position_threshold,
                    "localization_error_angle_threshold": localization_error_angle_threshold,
                    "start": start,
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
                        "wall_follower",
                        "pure_pursuit",
                        "coppeliasim",  # Must be started last.
                    )
                }
            ],
        )

        return LaunchDescription(
            [
                particle_filter_node,
                probabilistic_roadmap_node,
                wall_follower_node,
                pure_pursuit_node,
                coppeliasim_node,
                lifecycle_manager_node,  # Must be launched last.
            ]
        )

    else:
        odometry_node = LifecycleNode(
            package="amr_turtlebot3",
            executable="odometry_node",
            name="odometry",
            namespace="",
            output="screen",
            arguments=["--ros-args", "--log-level", "WARN"],
        )
        monitoring_node = LifecycleNode(
            package="amr_turtlebot3",
            executable="monitoring_node",
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
                        "wall_follower",
                        "pure_pursuit",
                        "odometry",
                        "monitoring",  # Must be started last.
                    )
                }
            ],
        )

        return LaunchDescription(
            [
                particle_filter_node,
                probabilistic_roadmap_node,
                wall_follower_node,
                pure_pursuit_node,
                odometry_node,
                monitoring_node,
                lifecycle_manager_node,  # Must be launched last.
            ]
        )
