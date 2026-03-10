import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn

from amr_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Path

import math
import traceback
from transforms3d.euler import quat2euler

from amr_control.pure_pursuit import PurePursuit


class PurePursuitNode(LifecycleNode):
    def __init__(self):
        """Pure pursuit node initializer."""
        super().__init__("pure_pursuit")

        # Parameters
        self.declare_parameter("dt", 0.05)
        self.declare_parameter("lookahead_distance", 0.2)
        self.declare_parameter("simulation", False)

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Handles a configuring transition.

        Args:
            state: Current lifecycle state.

        """
        self.get_logger().info(f"Transitioning from '{state.label}' to 'inactive' state.")

        try:
            # Parameters
            dt = self.get_parameter("dt").get_parameter_value().double_value
            lookahead_distance = (
                self.get_parameter("lookahead_distance").get_parameter_value().double_value
            )
            self._simulation = self.get_parameter("simulation").get_parameter_value().bool_value

            # Attribute and object initializations
            self._pure_pursuit = PurePursuit(
                dt,
                lookahead_distance,
                simulation=self._simulation,
                logger=None,  # Replace None with self.get_logger() to enable logging in the class
            )

            # Publishers
            self._publisher = self.create_publisher(TwistStamped, "cmd_vel", 10)

            # Subscribers
            self._subscriber_pose = self.create_subscription(
                PoseStamped, "pose", self._compute_commands_callback, 10
            )
            self._subscriber_path = self.create_subscription(Path, "path", self._path_callback, 10)

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

    def _compute_commands_callback(self, pose_msg: PoseStamped):
        """Subscriber callback. Executes a pure pursuit controller and publishes v and w commands.

        Starts to operate once the robot is localized.

        Args:
            pose_msg: Message containing the estimated robot pose.

        """
        if pose_msg.localized:
            # Parse pose
            x = pose_msg.pose.position.x
            y = pose_msg.pose.position.y
            quat_w = pose_msg.pose.orientation.w
            quat_x = pose_msg.pose.orientation.x
            quat_y = pose_msg.pose.orientation.y
            quat_z = pose_msg.pose.orientation.z
            _, _, theta = quat2euler((quat_w, quat_x, quat_y, quat_z))
            theta %= 2 * math.pi

            # Execute pure pursuit
            v, w = self._pure_pursuit.compute_commands(x, y, theta)
            self.get_logger().info(f"Commands: v = {v:.3f} m/s, w = {w:+.3f} rad/s")

            # Publish
            self._publish_velocity_commands(v, w)

    def _path_callback(self, path_msg: Path):
        """Subscriber callback. Saves the path the pure pursuit controller has to follow.

        Args:
            path_msg: Message containing the (smoothed) path.

        """
        # TODO: 4.8. Complete the function body with your code (i.e., replace the pass statement).
        pass
        
    def _publish_velocity_commands(self, v: float, w: float) -> None:
        """Publishes velocity commands in a geometry_msgs.msg.TwistStamped message.

        Args:
            v: Linear velocity command [m/s].
            w: Angular velocity command [rad/s].

        """
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = v
        msg.twist.angular.z = w
        self._publisher.publish(msg)


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
