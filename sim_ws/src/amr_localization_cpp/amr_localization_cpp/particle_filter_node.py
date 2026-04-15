#!/usr/bin/env python3

import datetime
import math
import os
import time
import traceback

import message_filters
import numpy as np
import pytz
import rclpy
from ament_index_python.packages import get_package_share_directory
from matplotlib import pyplot as plt
from nav_msgs.msg import Odometry
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan
from transforms3d.euler import euler2quat

from amr_msgs.msg import PoseStamped, AllowMove
from std_msgs.msg import Empty
from amr_localization_cpp._pf_core import ParticleFilterCore
from amr_localization_cpp.maps import Map


class ParticleFilterNode(LifecycleNode):
    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter("dt", 0.05)
        self.declare_parameter("enable_plot", False)
        self.declare_parameter("global_localization", True)
        self.declare_parameter("initial_pose", (0.0, 0.0, math.radians(0)))
        self.declare_parameter("initial_pose_sigma", (0.05, 0.05, math.radians(5)))
        self.declare_parameter("particles", 100)
        self.declare_parameter("sigma_v", 0.1)
        self.declare_parameter("sigma_w", 0.1)
        self.declare_parameter("sigma_z", 0.1)
        self.declare_parameter("simulation", False)
        self.declare_parameter("steps_btw_sense_updates", 10)
        self.declare_parameter("world", "lab03")
        self.declare_parameter("particle_filter_pose_topic", "particle_filter/pose")
        self.declare_parameter("reset_topic", "particle_filter/reset")
        self.declare_parameter("reset_on_low_likelihood", True)
        self.declare_parameter("low_likelihood_log_threshold", -6.0)
        self.declare_parameter("low_likelihood_patience", 3)

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Transitioning from '{state.label}' to 'inactive' state.")

        try:
            dt = self.get_parameter("dt").get_parameter_value().double_value
            self._enable_plot = self.get_parameter("enable_plot").get_parameter_value().bool_value
            global_localization = self.get_parameter("global_localization").get_parameter_value().bool_value

            self.initial_pose = tuple(
                self.get_parameter("initial_pose").get_parameter_value().double_array_value.tolist()
            )
            initial_pose_sigma = tuple(
                self.get_parameter("initial_pose_sigma").get_parameter_value().double_array_value.tolist()
            )

            particles = self.get_parameter("particles").get_parameter_value().integer_value
            sigma_v = self.get_parameter("sigma_v").get_parameter_value().double_value
            sigma_w = self.get_parameter("sigma_w").get_parameter_value().double_value
            sigma_z = self.get_parameter("sigma_z").get_parameter_value().double_value
            self._steps_btw_sense_updates = self.get_parameter(
                "steps_btw_sense_updates"
            ).get_parameter_value().integer_value
            self._simulation = self.get_parameter("simulation").get_parameter_value().bool_value
            world = self.get_parameter("world").get_parameter_value().string_value
            self._reset_topic = self.get_parameter("reset_topic").get_parameter_value().string_value
            self._reset_on_low_likelihood = (
                self.get_parameter("reset_on_low_likelihood").get_parameter_value().bool_value
            )
            self._low_likelihood_log_threshold = (
                self.get_parameter("low_likelihood_log_threshold").get_parameter_value().double_value
            )
            self._low_likelihood_patience = (
                self.get_parameter("low_likelihood_patience").get_parameter_value().integer_value
            )

            self._localized = False
            self._localized_confirmed = False
            self._steps = 0
            self._global_localization = global_localization
            self._initial_pose_sigma = initial_pose_sigma
            self._particle_count = particles
            self._sigma_v = sigma_v
            self._sigma_w = sigma_w
            self._sigma_z = sigma_z

            pkg_share = get_package_share_directory("amr_localization_cpp")
            map_path = os.path.join(pkg_share, "maps", world + ".json")
            self._map_path = map_path

            self._particle_filter = self._build_particle_filter()

            # ------------------------------------------------------------------ #
            # Plotting setup — only initialised when enable_plot is True          #
            # ------------------------------------------------------------------ #
            if self._enable_plot:
                self._map = Map(
                    map_path,
                    sensor_range=8.0,
                    safety_distance=0.08,
                    use_regions=False,
                    compiled_intersect=True,
                )
                self._figure, self._axes = plt.subplots(1, 1, figsize=(7, 7))
                self._timestamp = datetime.datetime.now(
                    pytz.timezone("Europe/Madrid")
                ).strftime("%Y-%m-%d_%H-%M-%S")
                plt.ion()  # Non-blocking interactive mode

            # ------------------------------------------------------------------ #

            self.z_scan = []
            self.z_odom = []
            self.last_pose = self.initial_pose
            self.last_odometry_callback = None

            self.pose_publisher = self.create_publisher(PoseStamped, "pose", 10)
            self.particle_filter_pose_publisher = self.create_publisher(
                PoseStamped,
                self.get_parameter("particle_filter_pose_topic").get_parameter_value().string_value,
                10,
            )

            scan_qos_profile = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
            )

            if self._simulation:
                self._subscribers = []
                self._subscribers.append(message_filters.Subscriber(self, Odometry, "odometry"))
                self._subscribers.append(
                    message_filters.Subscriber(self, LaserScan, "scan", qos_profile=scan_qos_profile)
                )

                ts = message_filters.ApproximateTimeSynchronizer(
                    self._subscribers, queue_size=10, slop=9
                )
                ts.registerCallback(self._compute_pose_callback)
            else:
                self.laser_scan_subscriber = self.create_subscription(
                    LaserScan,
                    "scan",
                    callback=self._compute_scan_callback,
                    qos_profile=scan_qos_profile,
                )
                self.odometry_subscriber = self.create_subscription(
                    Odometry,
                    "odometry",
                    callback=self._compute_odometry_callback,
                    qos_profile=10,
                )
                self.create_timer(5.0, self._timer_callback)
                self._finished_flag = True

            self.allowed_movement_publisher = self.create_publisher(
                AllowMove,
                topic="allow_move",
                qos_profile=10,
            )

            if self._simulation:
                self.reset_subscriber = self.create_subscription(
                    Empty,
                    self._reset_topic,
                    self._reset_localization_callback,
                    10,
                )
            else:
                self.reset_subscriber = None

        except Exception:
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.ERROR

        return super().on_configure(state)

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Transitioning from '{state.label}' to 'active' state.")
        return super().on_activate(state)
    def _build_particle_filter(self):
        return ParticleFilterCore(
            dt=self.get_parameter("dt").get_parameter_value().double_value,
            map_path=self._map_path,
            particle_count=self._particle_count,
            sigma_v=self._sigma_v,
            sigma_w=self._sigma_w,
            sigma_z=self._sigma_z,
            sensor_range_max=8.0,
            sensor_range_min=0.16,
            global_localization=self._global_localization,
            initial_pose=self.initial_pose,
            initial_pose_sigma=self._initial_pose_sigma,
            simulation=self._simulation,
            reset_on_low_likelihood=self._reset_on_low_likelihood,
            low_likelihood_log_threshold=self._low_likelihood_log_threshold,
            low_likelihood_patience=self._low_likelihood_patience,
        )

    def _reset_localization_callback(self, _msg: Empty) -> None:
        if not self._simulation:
            return

        self.get_logger().warn("Restarting particle filter because localization error exceeded the threshold.")

        self._particle_filter = self._build_particle_filter()
        self._localized = False
        self._localized_confirmed = False
        self._steps = 0
        self.z_scan = []
        self.z_odom.clear()
        self.last_pose = self.initial_pose
        self.last_odometry_callback = None

        if self._enable_plot:
            self._show_particles(title="after reset", save_figure=True)

        msg = AllowMove()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.allowed = True
        self.allowed_movement_publisher.publish(msg)

        self._publish_pose_estimate(float("inf"), float("inf"), float("inf"))

    # ---------------------------------------------------------------------- #
    # Plotting helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _plot_particles(self, axes, orientation: bool = True):
        """Draws the current particle set retrieved from the C++ core.

        Args:
            axes: Matplotlib axes to draw on.
            orientation: If True, draw an arrow for each particle's heading.

        Returns:
            axes: The modified axes object.
        """
        particles = self._particle_filter.particles()  # Nx3 NumPy array from Eigen

        if orientation:
            dx = np.cos(particles[:, 2])
            dy = np.sin(particles[:, 2])
            axes.quiver(
                particles[:, 0],
                particles[:, 1],
                dx,
                dy,
                color="royalblue",
                alpha=0.7,
                scale=15,
                scale_units="inches",
            )
        else:
            axes.plot(particles[:, 0], particles[:, 1], "bo", markersize=1, alpha=0.6)

        return axes

    def _show_particles(
        self,
        title: str = "",
        orientation: bool = True,
        save_figure: bool = False,
        save_dir: str = "images",
    ):
        """Renders the map with the current particles and optionally saves the figure.

        Args:
            title: Plot title suffix shown alongside the iteration number.
            orientation: If True, draw particle headings as arrows.
            save_figure: Save the figure to a PNG file when True.
            save_dir: Directory (relative to the package share) where images are saved.
        """
        if not self._enable_plot:
            return

        axes = self._axes
        axes.clear()

        # Draw map walls
        self._map.plot(axes)

        # Draw particles from the C++ core
        self._plot_particles(axes, orientation=orientation)

        axes.set_title(f"Particle Filter — {title} (step {self._steps})")
        self._figure.tight_layout()

        plt.draw()
        plt.pause(0.001)  # Yield to the GUI event loop

        if save_figure:
            pkg_share = get_package_share_directory("amr_localization_cpp")
            save_path = os.path.realpath(
                os.path.join(pkg_share, save_dir, self._timestamp)
            )
            os.makedirs(save_path, exist_ok=True)

            file_name = f"{str(self._steps).zfill(4)}_{title.lower().replace(' ', '_')}.png"
            self._figure.savefig(os.path.join(save_path, file_name))

    # ---------------------------------------------------------------------- #
    # ROS callbacks                                                            #
    # ---------------------------------------------------------------------- #

    def _compute_scan_callback(self, scan_msg: LaserScan) -> None:
        self.z_scan = list(scan_msg.ranges)

    def _compute_odometry_callback(self, odom_msg: Odometry) -> None:
        z_v = odom_msg.twist.twist.linear.x
        z_w = odom_msg.twist.twist.angular.z

        if abs(z_v) >= 0.01 or abs(z_w) >= 0.1:
            self.z_odom.append((z_v, z_w))

        if self.last_odometry_callback is not None:
            delta_time = time.perf_counter() - self.last_odometry_callback
            x_last, y_last, th_last = self.last_pose
            x_h = x_last + z_v * math.cos(th_last + z_w * delta_time / 2) * delta_time
            y_h = y_last + z_v * math.sin(th_last + z_w * delta_time / 2) * delta_time
            th_h = th_last + z_w * delta_time

            self.last_pose = (x_h, y_h, th_h)

            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.localized = self._localized
            msg.pose.position.x = x_h
            msg.pose.position.y = y_h

            q = euler2quat(0, 0, th_h)
            msg.pose.orientation.w = q[0]
            msg.pose.orientation.x = q[1]
            msg.pose.orientation.y = q[2]
            msg.pose.orientation.z = q[3]
            self.pose_publisher.publish(msg)

        self.last_odometry_callback = time.perf_counter()

    def _timer_callback(self):
        if self._finished_flag and len(self.z_odom) > 0:
            self._finished_flag = False

            msg = AllowMove()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.allowed = False
            self.allowed_movement_publisher.publish(msg)

            for z_v, z_w in self.z_odom:
                self._execute_motion_step(z_v, z_w)
                self._steps += 1

            x_h, y_h, theta_h = self._execute_measurement_step(self.z_scan)

            self.last_odometry_callback = time.perf_counter()
            self.last_pose = (x_h, y_h, theta_h)

            msg = AllowMove()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.allowed = True
            self.allowed_movement_publisher.publish(msg)

            self._publish_pose_estimate(x_h, y_h, theta_h)
            self._finished_flag = True
            self.z_odom.clear()

    def _compute_pose_callback(self, odom_msg: Odometry, scan_msg: LaserScan):
        msg = AllowMove()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.allowed = True
        self.allowed_movement_publisher.publish(msg)

        z_v = odom_msg.twist.twist.linear.x
        z_w = odom_msg.twist.twist.angular.z
        z_scan = list(scan_msg.ranges)

        self._execute_motion_step(z_v, z_w)

        x_h, y_h, theta_h = self._execute_measurement_step(z_scan)
        self._steps += 1

        self._publish_pose_estimate(x_h, y_h, theta_h)

    # ---------------------------------------------------------------------- #
    # Core filter steps                                                        #
    # ---------------------------------------------------------------------- #

    def _execute_measurement_step(self, z_scan):
        pose = (float("inf"), float("inf"), float("inf"))
        
        if self._steps % self._steps_btw_sense_updates == 0:
            start_time = time.perf_counter()
            self._particle_filter.resample(z_scan)
            sense_time = time.perf_counter() - start_time
            self.get_logger().info(f"Sense step time: {sense_time:6.3f} s")

            if self._particle_filter.consume_reset_flag():
                self.get_logger().warn(
                    "Particle filter confidence too low for consecutive scans. "
                    "Resetting particles from internal likelihood monitor."
                )
                self._localized = False
                self._localized_confirmed = False
                return pose
            
            # Plot after measurement/resample update
            self._show_particles(title="after resample", save_figure=True)

        start_time = time.perf_counter()
        self._localized, pose = self._particle_filter.compute_pose()
        clustering_time = time.perf_counter() - start_time
        self.get_logger().info(f"Clustering time: {clustering_time:6.3f} s")

        return pose

    def _execute_motion_step(self, z_v: float, z_w: float):
        start_time = time.perf_counter()
        self._particle_filter.move(z_v, z_w)
        move_time = time.perf_counter() - start_time
        self.get_logger().info(f"Move step time: {move_time:7.3f} s")
        
        # Plot after measurement/resample update
        self._show_particles(title="after movement", save_figure=True)

    def _publish_pose_estimate(self, x_h: float, y_h: float, theta_h: float) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.localized = self._localized

        if self._localized:
            q = euler2quat(0, 0, theta_h)
            msg.pose.position.x = x_h
            msg.pose.position.y = y_h
            msg.pose.orientation.w = q[0]
            msg.pose.orientation.x = q[1]
            msg.pose.orientation.y = q[2]
            msg.pose.orientation.z = q[3]

        self.pose_publisher.publish(msg)
        if self._localized:
            self.particle_filter_pose_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()