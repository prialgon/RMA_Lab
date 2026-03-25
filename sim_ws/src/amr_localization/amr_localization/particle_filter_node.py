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
from transforms3d.euler import euler2quat
import numpy as np

from amr_localization.particle_filter import ParticleFilter


class ParticleFilterNode(LifecycleNode):
    def __init__(self):
        """Particle filter node initializer."""
        super().__init__("particle_filter")

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
            global_localization = (
                self.get_parameter(
                    "global_localization").get_parameter_value().bool_value
            )
            self.initial_pose = tuple(
                self.get_parameter("initial_pose").get_parameter_value(
                ).double_array_value.tolist()
            )
            initial_pose_sigma = tuple(
                self.get_parameter("initial_pose_sigma")
                .get_parameter_value()
                .double_array_value.tolist()
            )
            particles = self.get_parameter(
                "particles").get_parameter_value().integer_value
            sigma_v = self.get_parameter(
                "sigma_v").get_parameter_value().double_value
            sigma_w = self.get_parameter(
                "sigma_w").get_parameter_value().double_value
            sigma_z = self.get_parameter(
                "sigma_z").get_parameter_value().double_value
            self._steps_btw_sense_updates = (
                self.get_parameter(
                    "steps_btw_sense_updates").get_parameter_value().integer_value
            )
            self._simulation = self.get_parameter(
                "simulation").get_parameter_value().bool_value
            world = self.get_parameter(
                "world").get_parameter_value().string_value

            # Attribute and object initializations
            self._localized = True
            self._steps = 0
            map_path = os.path.realpath(
                os.path.join(os.path.dirname(__file__),
                             "..", "maps", world + ".json")
            )
            self._particle_filter = ParticleFilter(
                dt,
                map_path,
                particle_count=particles,
                sigma_v=sigma_v,
                sigma_w=sigma_w,
                sigma_z=sigma_z,
                global_localization=global_localization,
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
                self._particle_filter.show("Initialization", save_figure=True)

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

            if self._simulation:
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

            if not self._simulation:

                self.laser_scan_subscriber = self.create_subscription(
                    LaserScan,
                    "scan",
                    qos_profile=scan_qos_profile,
                    callback=self._compute_scan_callback,
                )

                self.odometry_subscriber = self.create_subscription(
                    Odometry,
                    "odometry",
                    qos_profile=10,
                    callback=self._compute_odometry_callback,
                )
                self.create_timer(
                    timer_period_sec=5,
                    callback=self._timer_callback,
                )

                self._finished_flag = True

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

    def _compute_scan_callback(self, scan_msg: LaserScan) -> None:
        self.z_scan = list(scan_msg.ranges)

    def _compute_odometry_callback(self, odom_msg: Odometry) -> None:
        z_v: float = odom_msg.twist.twist.linear.x
        z_w: float = odom_msg.twist.twist.angular.z

        if abs(z_v) <= 0.01:
            z_v = 0.0
        if abs(z_w) <= 0.1:
            z_w = 0.0
        if abs(z_v) >= 0.01 or abs(z_w) >= 0.1:
            self.z_odom.append((z_v, z_w))

        # ===== PUNTO MEDIO =====
        if self.last_odometry_callback is not None:
            delta_time: float = time.perf_counter() - self.last_odometry_callback
            x_last, y_last, th_last = self.last_pose
            x_h: float = x_last + z_v * \
                math.cos(th_last + z_w * delta_time / 2) * delta_time
            y_h: float = y_last + z_v * \
                math.sin(th_last + z_w * delta_time / 2) * delta_time
            th_h: float = th_last + z_w * delta_time

            self.last_pose = (x_h, y_h, th_h)

            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.localized = True
            msg.pose.position.x = x_h
            msg.pose.position.y = y_h

            quaternions = euler2quat(0, 0, th_h)
            msg.pose.orientation.w = quaternions[0]
            msg.pose.orientation.x = quaternions[1]
            msg.pose.orientation.y = quaternions[2]
            msg.pose.orientation.z = quaternions[3]
            
            self.pose_publisher.publish(msg)

        self.last_odometry_callback = time.perf_counter()

    def _timer_callback(self):

        if self._finished_flag and len(self.z_odom) > 0:
            self._finished_flag = False
            msg: AllowMove = AllowMove()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.allowed = False
            self.allowed_movement_publisher.publish(msg)

            odoms = np.array(self.z_odom)
            # self.get_logger().warn(f"{np.median(odoms[:, 1])}")

            for z_v, z_w in self.z_odom:
                self._execute_motion_step(z_v, z_w)
                self._steps += 1

            x_h, y_h, theta_h = self._execute_measurement_step(self.z_scan)

            self.last_odometry_callback = time.perf_counter()
            self.last_pose = (x_h, y_h, theta_h)

            msg: AllowMove = AllowMove()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.allowed = True
            self.allowed_movement_publisher.publish(msg)

            self._publish_pose_estimate(x_h, y_h, theta_h)
            self._finished_flag = True
            self.z_odom.clear()

    def _compute_pose_callback(self, odom_msg: Odometry, scan_msg: LaserScan):
        """Subscriber callback. Executes a particle filter and publishes (x, y, theta) estimates.

        Args:
            odom_msg: Message containing odometry measurements.
            scan_msg: Message containing LiDAR sensor readings.

        """
        msg: AllowMove = AllowMove()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.allowed = True
        self.allowed_movement_publisher.publish(msg)

        # Parse measurements
        z_v: float = odom_msg.twist.twist.linear.x
        z_w: float = odom_msg.twist.twist.angular.z
        z_scan: list[float] = scan_msg.ranges

        # Execute particle filter
        self._execute_motion_step(z_v, z_w)
        x_h, y_h, theta_h = self._execute_measurement_step(z_scan)
        self._steps += 1

        # Publish
        self._publish_pose_estimate(x_h, y_h, theta_h)

    def _execute_measurement_step(self, z_scan: list[float]) -> tuple[float, float, float]:
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
            self._particle_filter.resample(z_scan)
            sense_time = time.perf_counter() - start_time

            self.get_logger().info(f"Sense step time: {sense_time:6.3f} s")

            if self._enable_plot:
                self._particle_filter.show("Sense", save_figure=True)

            start_time = time.perf_counter()
            self._localized, pose = self._particle_filter.compute_pose()
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
        self._particle_filter.move(z_v, z_w)
        move_time = time.perf_counter() - start_time

        self.get_logger().info(f"Move step time: {move_time:7.3f} s")

        if self._enable_plot:
            self._particle_filter.show("Move", save_figure=True)

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
    particle_filter_node = ParticleFilterNode()

    try:
        rclpy.spin(particle_filter_node)
    except KeyboardInterrupt:
        pass

    particle_filter_node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
