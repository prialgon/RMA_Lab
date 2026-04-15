import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy

import message_filters
from amr_msgs.msg import PoseStamped, AllowMove
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

import math
import os
import time
import traceback
from transforms3d.euler import euler2quat, quat2euler
import numpy as np

from amr_localization.ekf import EKF


class EKFNode(LifecycleNode):
    def __init__(self):
        """Particle filter node initializer."""
        super().__init__("ekf")

        # Parameters
        self.declare_parameter("dt", 0.05)
        self.declare_parameter("enable_plot", False)
        self.declare_parameter("global_localization", True)
        self.declare_parameter("initial_pose", (0.0, 0.0, math.radians(0)))
        self.declare_parameter("initial_pose_sigma",
                               (0.05, 0.05, math.radians(5)))
        self.declare_parameter("particles", 100)
        self.declare_parameter("sigma_v", 0.1)
        self.declare_parameter("sigma_w", 0.1)
        self.declare_parameter("sigma_z", 0.1)
        self.declare_parameter("simulation", False)
        self.declare_parameter("steps_btw_sense_updates", 10)
        self.declare_parameter("world", "lab03")
        self.declare_parameter("particle_filter_pose_topic", "particle_filter/pose")

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Handles a configuring transition.

        Args:
            state: Current lifecycle state.

        """
        self.get_logger().info(
            f"Transitioning from '{state.label}' to 'inactive' state.")

        try:
            # Parameters
            dt = self.get_parameter(
                "dt").get_parameter_value().double_value
            self._enable_plot = self.get_parameter(
                "enable_plot").get_parameter_value().bool_value
            self.initial_pose = tuple(
                self.get_parameter("initial_pose").get_parameter_value(
                ).double_array_value.tolist()
            )
            initial_pose_sigma = tuple(
                self.get_parameter("initial_pose_sigma")
                .get_parameter_value()
                .double_array_value.tolist()
            )
            self._steps_btw_sense_updates = (
                self.get_parameter(
                    "steps_btw_sense_updates").get_parameter_value().integer_value
            )
            self._simulation = self.get_parameter(
                "simulation").get_parameter_value().bool_value
            world = self.get_parameter(
                "world").get_parameter_value().string_value

            # Attribute and object initializations
            self._localized = False
            self._tracking_enabled = False
            self._interactive_plot_active = False
            self._steps = 0
            self._handoff_pose_sigma = np.diag(np.square(np.array(initial_pose_sigma, dtype=float)))
            map_path = os.path.realpath(
                os.path.join(os.path.dirname(__file__),
                             "..", "maps", world + ".json")
            )
            self._ekf = EKF(
                dt,
                map_path,
                initial_pose=self.initial_pose,
                initial_pose_sigma=initial_pose_sigma,
                simulation=self._simulation,
                # Replace None with self.get_logger() to enable logging in the class
                logger=self.get_logger(),
            )

            self.z_scan: list[float] = []
            self.z_odom: list[tuple[float, float]] = []

            self.last_pose: tuple[float, ...] = self.initial_pose

            self.last_odometry_callback = None

            if self._enable_plot:
                self._ekf.show("Initialization", save_figure=True)

            # Publishers
            # TODO: 3.1. Create the /pose publisher (PoseStamped message).
            self.pose_publisher = self.create_publisher(
                PoseStamped,
                "pose",
                10
            )

            # Subscribers
            scan_qos_profile = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
            )
            
            self.pose_subscriber = self.create_subscription(
                PoseStamped,
                self.get_parameter("particle_filter_pose_topic").get_parameter_value().string_value,
                callback=self._read_pose_callback,
                qos_profile=10,
            )

            # if self._simulation:
            self._subscribers: list[message_filters.Subscriber] = []
            self._subscribers.append(
                message_filters.Subscriber(self, Odometry, "odometry"))
            self._subscribers.append(
                message_filters.Subscriber(
                    self, LaserScan, "scan", qos_profile=scan_qos_profile)
            )

            ts = message_filters.ApproximateTimeSynchronizer(
                self._subscribers, queue_size=10, slop=9
            )
            ts.registerCallback(self._compute_pose_callback)

            # if not self._simulation:

            #     self.laser_scan_subscriber = self.create_subscription(
            #         LaserScan,
            #         "scan",
            #         qos_profile=scan_qos_profile,
            #         callback=self._compute_scan_callback,
            #     )

            #     self.odometry_subscriber = self.create_subscription(
            #         Odometry,
            #         "odometry",
            #         qos_profile=10,
            #         callback=self._compute_odometry_callback,
            #     )
            #     self.create_timer(
            #         timer_period_sec=5,
            #         callback=self._timer_callback,
            #     )

            #     self._finished_flag = True

            # 3.11
            self.allowed_movement_publisher = self.create_publisher(
                AllowMove,
                topic="allow_move",
                qos_profile=10,
            )

        except Exception:
            self.get_logger().error(f"{traceback.format_exc()}")
            return TransitionCallbackReturn.ERROR

        return super().on_configure(state)

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Handles an activating transition.

        Args:
            state: Current lifecycle state.

        """
        self.get_logger().info(
            f"Transitioning from '{state.label}' to 'active' state.")

        return super().on_activate(state)
    
    def _read_pose_callback(self, msg: PoseStamped) -> None:
        if self._tracking_enabled or not msg.localized:
            return

        quaternion = (
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
        )
        _, _, yaw = quat2euler(quaternion)
        pose = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                yaw,
            ],
            dtype=float,
        )
        self._ekf._pose = pose
        self._ekf._sigma = self._handoff_pose_sigma.copy()
        self.last_pose = tuple(pose.tolist())
        self.last_odometry_callback = None
        self._localized = True
        self._tracking_enabled = True

        if self._enable_plot:
            self._interactive_plot_active = True
            self._ekf.show("PF handoff", display=True, save_figure=True)

        self.get_logger().info("Received localized particle-filter pose. EKF tracking is now active.")

    def _compute_pose_callback(self, odom_msg: Odometry, scan_msg: LaserScan):
        """Subscriber callback. Executes a particle filter and publishes (x, y, theta) estimates.

        Args:
            odom_msg: Message containing odometry measurements.
            scan_msg: Message containing LiDAR sensor readings.

        """
        if not self._tracking_enabled:
            return

        msg: AllowMove = AllowMove()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.allowed = True
        self.allowed_movement_publisher.publish(msg)

        # Parse measurements
        z_v: float = odom_msg.twist.twist.linear.x
        z_w: float = odom_msg.twist.twist.angular.z
        z_scan: list[float] = scan_msg.ranges

        # Execute EKF
        n_mu, n_sigma = self._execute_motion_step(z_v, z_w)
        x_h, y_h, theta_h = self._execute_measurement_step(n_mu, n_sigma, z_scan)
        self._steps += 1

        # Publish
        self._publish_pose_estimate(x_h, y_h, theta_h)

    def _execute_measurement_step(self, n_mu, n_sigma, z_scan: list[float]) -> tuple[float, float, float]:
        """Executes and monitors the measurement step (sense) of the particle filter.

        Args:
            z_scan: Distance from every LiDAR ray to the closest obstacle [m].

        Returns:
            Pose estimate (x_h, y_h, theta_h) [m, m, rad]; inf if cannot be computed.
        """
        pose = (float("inf"), float("inf"), float("inf"))

        # if self._localized or not self._steps % self._steps_btw_sense_updates:
        if True:
            start_time = time.perf_counter()
            self._ekf.correct(n_mu, n_sigma, z_scan)
            sense_time = time.perf_counter() - start_time

            self.get_logger().info(f"Correct step time: {sense_time:6.3f} s")

            if self._enable_plot:
                self._ekf.show(
                    "Sense",
                    display=self._interactive_plot_active,
                    save_figure=True,
                )

            start_time = time.perf_counter()
            self._localized, pose = True, tuple(self._ekf._pose.tolist())
            clustering_time = time.perf_counter() - start_time

            self.get_logger().info(
                f"Clustering time: {clustering_time:6.3f} s")

        return pose

    def _execute_motion_step(self, z_v: float, z_w: float):
        """Executes and monitors the motion step (move) of the particle filter.

        Args:
            z_v: Odometric estimate of the linear velocity of the robot center [m/s].
            z_w: Odometric estimate of the angular velocity of the robot center [rad/s].
        """
        start_time = time.perf_counter()
        n_mu, n_sigma = self._ekf.move(z_v, z_w)
        move_time = time.perf_counter() - start_time

        self.get_logger().info(f"Move step time: {move_time:7.3f} s")

        if self._enable_plot:
            self._ekf.show(
                "Move",
                display=self._interactive_plot_active,
                save_figure=True,
            )

        return n_mu, n_sigma

        

    def _publish_pose_estimate(self, x_h: float, y_h: float, theta_h: float) -> None:
        """Publishes the robot's pose estimate in a custom amr_msgs.msg.PoseStamped message.

        Args:
            x_h: x coordinate estimate [m].
            y_h: y coordinate estimate [m].
            theta_h: Heading estimate [rad].

        """
        # TODO: 3.2. Complete the function body with your code (i.e., replace the pass statement).
        msg: PoseStamped = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.localized = self._localized
        
        if self._localized:
            quaternions = euler2quat(0, 0, theta_h)

            msg.pose.position.x = x_h
            msg.pose.position.y = y_h

            msg.pose.orientation.w = quaternions[0]
            msg.pose.orientation.x = quaternions[1]
            msg.pose.orientation.y = quaternions[2]
            msg.pose.orientation.z = quaternions[3]

        self.pose_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    ekf_node = EKFNode()

    try:
        rclpy.spin(ekf_node)
    except KeyboardInterrupt:
        pass

    ekf_node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
