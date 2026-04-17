import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy, qos_profile_sensor_data

import message_filters
from amr_msgs.msg import PoseStamped, AllowMove
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

import traceback

from amr_control.wall_follower import WallFollower


class WallFollowerNode(LifecycleNode):
    def __init__(self):
        """Wall follower node initializer."""
        super().__init__("wall_follower")

        # Parameters
        self.declare_parameter("dt", 0.05)
        self.declare_parameter("enable_localization", False)
        self.declare_parameter("simulation", False)
        
        self.odom_msg: Odometry = None
        self.scan_msg: LaserScan = None
        self.pose_msg: PoseStamped = None

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Handles a configuring transition.

        Args:
            state: Current lifecycle state.

        """
        self.get_logger().info(f"Transitioning from '{state.label}' to 'inactive' state.")

        try:
            # Parameters
            dt = self.get_parameter("dt").get_parameter_value().double_value
            enable_localization = (
                self.get_parameter("enable_localization").get_parameter_value().bool_value
            )
            self._simulation = self.get_parameter("simulation").get_parameter_value().bool_value

            # Attribute and object initializations
            self._wall_follower = WallFollower(
                dt,
                simulation=self._simulation,
                logger=self.get_logger(),  # Replace None with self.get_logger() to enable logging in the class
            )

            # Publishers
            # TODO: 2.10. Create the /cmd_vel velocity commands publisher (TwistStamped message).
            if self._simulation:
                self._publisher_cmd_vel = self.create_publisher(
                    msg_type=TwistStamped,
                    topic="cmd_vel",
                    qos_profile=10,
                )
            else:
                self._publisher_cmd_vel = self.create_publisher(
                    msg_type=Twist,
                    topic="cmd_vel",
                    qos_profile=10,
                )
            
            # Subscribers
            # TODO: 2.7. Synchronize _compute_commands_callback with /odometry and /scan.
            self._subscribers: list[message_filters.Subscriber] = []
            
            self._subscribers.append(
                message_filters.Subscriber(
                    self,
                    Odometry,
                    "odometry",
                    qos_profile=10
                )
            )
            
            self._subscribers.append(
                message_filters.Subscriber(
                    self,
                    LaserScan,
                    "scan",
                    qos_profile=qos_profile_sensor_data
                )
            )
            
            ts = message_filters.ApproximateTimeSynchronizer(
                self._subscribers,
                queue_size=10,
                slop=0.15
            )
            
            ts.registerCallback(self._read_commands_callback)
            
            self._timer = self.create_timer(
                timer_period_sec=dt,
                callback=self._compute_commands_callback,
            )
            
            self.allowed_move: bool = True

            self.allow_move_subscriber = self.create_subscription(
                AllowMove,
                topic="allow_move",
                qos_profile=10,
                callback=self._compute_allow_move_callback
            )
            # TODO: 4.12. Add /pose to the synced subscriptions only if localization is enabled.
            
        except Exception:
            self.get_logger().error(f"{traceback.format_exc()}")
            return TransitionCallbackReturn.ERROR

        return super().on_configure(state)

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Handles an activating transition.

        Args:
            state: Current lifecycle state.

        """
        self.get_logger().info(f"Transitioning from '{state.label}' to 'active' state.")

        return super().on_activate(state)
    
    def _read_commands_callback(
        self, odom_msg: Odometry, scan_msg: LaserScan, pose_msg: PoseStamped = PoseStamped()
    ):
        # self.get_logger().warn("Reading information")
        if not pose_msg.localized:
            # TODO: 2.8. Parse the odometry from the Odometry message (i.e., read z_v and z_w).
            self.odom_msg = odom_msg
            
            self.scan_msg = scan_msg
            
            self.pose_msg = pose_msg
            
            if self.get_parameter("simulation"):
                self._compute_commands_callback()
            
            # TODO: 2.9. Parse LiDAR measurements from the LaserScan message (i.e., read z_scan).

        

    def _compute_commands_callback(self):
        """Subscriber callback. Executes a wall-following controller and publishes v and w commands.

        Ceases to operate once the robot is localized.

        Args:
            odom_msg: Message containing odometry measurements.
            scan_msg: Message containing LiDAR readings.
            pose_msg: Message containing the estimated robot pose.

        """
        if not self.odom_msg or not self.scan_msg or not self.pose_msg:
            return None
        
        # Execute wall follower
        z_v: float = self.odom_msg.twist.twist.linear.x
        z_w: float = self.odom_msg.twist.twist.angular.z
        
        z_scan: list[float] = list(self.scan_msg.ranges)
        
        v, w = self._wall_follower.compute_commands(z_scan, z_v, z_w)
        self.get_logger().info(f"Commands: v = {v:.3f} m/s, w = {w:+.3f} rad/s")

        if not self.allowed_move:
            v = 0.0
            w = 0.0
        # Publish

        
        self._publish_velocity_commands(v, w)

    def _publish_velocity_commands(self, v: float, w: float) -> None:
        """Publishes velocity commands in a geometry_msgs.msg.TwistStamped message.

        Args:
            v: Linear velocity command [m/s].
            w: Angular velocity command [rad/s].

        """
        # TODO: 2.11. Complete the function body with your code (i.e., replace the pass statement).

        
        if self._simulation:
            msg: TwistStamped = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.twist.linear.x = v
            msg.twist.angular.z = w
        else: 
            msg: Twist = Twist()
            msg.linear.x = v
            msg.angular.z = w
            
        self._publisher_cmd_vel.publish(msg)
    
    def _compute_allow_move_callback(self, msg: AllowMove) -> None:
        self.allowed_move = msg.allowed


def main(args=None):
    rclpy.init(args=args)
    wall_follower_node = WallFollowerNode()

    try:
        rclpy.spin(wall_follower_node)
    except KeyboardInterrupt:
        pass

    wall_follower_node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
