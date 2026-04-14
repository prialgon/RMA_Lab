from launch import LaunchDescription
from launch_ros.actions import LifecycleNode, Node

import math


def generate_launch_description():
    simulation = True
    world = "project"
    start = (0.2, -0.6, math.radians(270))
    goal = (-1.0, 0.6)
    localization_error_position_threshold = 0.50
    localization_error_angle_threshold = 30.0

    particle_filter_node = LifecycleNode(
        package="amr_localization_cpp",
        executable="particle_filter_node.py",
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
                "steps_btw_sense_updates": 10
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
                "connection_distance": 0.15,
                "enable_plot": True,
                "goal": goal,
                "grid_size": 0.0869420,
                "node_count": 500,
                "obstacle_safety_distance": 0.15,
                "simulation": simulation,
                "smoothing_additional_points": 3,
                "smoothing_data_weight": 0.1,
                "smoothing_smooth_weight": 0.25,
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
                "lookahead_distance": 0.25,
                "goal_tolerance": 0.05,
                "simulation": simulation,
            }
        ],
    )

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
                    "coppeliasim",  # Must be started last
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
            lifecycle_manager_node,  # Must be launched last
        ]
    )
