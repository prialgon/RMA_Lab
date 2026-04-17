import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn

from amr_msgs.msg import PoseStamped, AllowMove
from geometry_msgs.msg import TwistStamped, Twist
from nav_msgs.msg import Path

import math
import traceback
from transforms3d.euler import quat2euler

from amr_control.pure_pursuit import PurePursuit

from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy


class PurePursuitNode(LifecycleNode):
    def __init__(self):
        """Pure pursuit node initializer."""
        super().__init__("pure_pursuit")

        # Parameters
        self.declare_parameter("dt", 0.05)
        self.declare_parameter("lookahead_distance", 0.2)
        self.declare_parameter("goal_tolerance", 0.05)
        self.declare_parameter("simulation", False)

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Handles a configuring transition.

        Args:
            state: Current lifecycle state.

        """
        self.get_logger().info(
            f"Transitioning from '{state.label}' to 'inactive' state."
        )

        try:
            dt = self.get_parameter("dt").get_parameter_value().double_value
            lookahead_distance = (
                self.get_parameter("lookahead_distance").get_parameter_value().double_value
            )
            goal_tolerance = (
                self.get_parameter("goal_tolerance").get_parameter_value().double_value
            )
            self._simulation = self.get_parameter(
                "simulation"
            ).get_parameter_value().bool_value

            self._pure_pursuit = PurePursuit(
                dt,
                lookahead_distance,
                goal_tolerance=goal_tolerance,
                simulation=self._simulation,
                logger=None,
            )

            self.allowed_move: bool = True
            self._latest_pose: PoseStamped | None = None

            if not self._simulation:
                self._publisher = self.create_publisher(Twist, "cmd_vel", 10)
            else:
                self._publisher = self.create_publisher(TwistStamped, "cmd_vel", 10)

            self._subscriber_pose = self.create_subscription(
                PoseStamped, "pose", self._compute_commands_callback, 10
            )
            path_qos_profile = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            )
            self._subscriber_path = self.create_subscription(
                Path, "path", self._path_callback, qos_profile=path_qos_profile
            )
            self._subscriber_allow_move = self.create_subscription(
                AllowMove, "allow_move", self._allow_move_callback, 10
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
            f"Transitioning from '{state.label}' to 'active' state."
        )

        return super().on_activate(state)

    def _compute_commands_callback(self, pose_msg: PoseStamped):
        """Executes pure pursuit and publishes commands once pose and path are ready."""
        self._latest_pose = pose_msg

        if not pose_msg.localized or not self.allowed_move or not self._pure_pursuit.path:
            return

        x = pose_msg.pose.position.x
        y = pose_msg.pose.position.y
        quat_w = pose_msg.pose.orientation.w
        quat_x = pose_msg.pose.orientation.x
        quat_y = pose_msg.pose.orientation.y
        quat_z = pose_msg.pose.orientation.z
        _, _, theta = quat2euler((quat_w, quat_x, quat_y, quat_z))
        theta %= 2 * math.pi

        v, w = self._pure_pursuit.compute_commands(x, y, theta)
        self.get_logger().info(f"Commands: v = {v:.3f} m/s, w = {w:+.3f} rad/s")
        self._publish_velocity_commands(v, w)

    def _path_callback(self, path_msg: Path):
        """Saves the path and immediately starts following it if localization already happened."""
        self._pure_pursuit.path = [
            (node.pose.position.x, node.pose.position.y) for node in path_msg.poses
        ]

        if self._latest_pose is not None and self._latest_pose.localized and self.allowed_move:
            self._compute_commands_callback(self._latest_pose)

    def _publish_velocity_commands(self, v: float, w: float) -> None:
        """Publishes velocity commands."""
        if not self._simulation:
            msg = Twist()
            msg.linear.x = v
            msg.angular.z = -w
            self._publisher.publish(msg)
        else:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.twist.linear.x = v
            msg.twist.angular.z = w
            self._publisher.publish(msg)

    def _allow_move_callback(self, msg: AllowMove) -> None:
        self.allowed_move = msg.allowed

        if not self.allowed_move and not self._simulation:
            msg_twist = Twist()
            msg_twist.linear.x = 0.0
            msg_twist.angular.z = 0.0
            self._publisher.publish(msg_twist)

        if not self.allowed_move and self._simulation:
            msg_twist = TwistStamped()
            msg_twist.header.stamp = self.get_clock().now().to_msg()
            msg_twist.twist.linear.x = 0.0
            msg_twist.twist.angular.z = 0.0
            self._publisher.publish(msg_twist)

        if self.allowed_move and self._latest_pose is not None and self._latest_pose.localized and self._pure_pursuit.path:
            self._compute_commands_callback(self._latest_pose)


def main(args=None):
    rclpy.init(args=args)
    pure_pursuit_node = PurePursuitNode()

    try:
        rclpy.spin(pure_pursuit_node)
    except KeyboardInterrupt:
        pass

    pure_pursuit_node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
