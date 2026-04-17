import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn

from amr_msgs.msg import PoseStamped

import math
import time
import traceback
from transforms3d.euler import quat2euler


class MonitoringNode(LifecycleNode):
    def __init__(self):
        """Monitoring node initializer."""
        super().__init__("monitoring")

        # Parameters
        self.declare_parameter("goal", (float("inf"), float("inf")))
        self.declare_parameter("goal_tolerance", 0.15)

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Handles a configuring transition.

        Args:
            state: Current lifecycle state.

        """
        self.get_logger().info(f"Transitioning from '{state.label}' to 'inactive' state.")

        try:
            # Parameters
            self._goal = tuple(
                self.get_parameter("goal").get_parameter_value().double_array_value.tolist()
            )
            self._goal_tolerance = (
                self.get_parameter("goal_tolerance").get_parameter_value().double_value
            )

            # Attribute and object initializations
            self._localized = False
            self._start_time: float = time.time()

            # Subscribers
            self._localization_subscriber = self.create_subscription(
                PoseStamped, "pose", self._check_pose_callback, 10
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
        self.get_logger().info(f"Transitioning from '{state.label}' to 'active' state.")

        return super().on_activate(state)

    def _check_pose_callback(self, pose_msg: PoseStamped) -> None:
        """Prints the coordinates where the robot localized and checks whether it reached the goal.

        Args:
            pose_msg: Message containing the estimated robot pose.

        """
        if pose_msg.localized:
            x_h = pose_msg.pose.position.x
            y_h = pose_msg.pose.position.y
            quat_w = pose_msg.pose.orientation.w
            quat_x = pose_msg.pose.orientation.x
            quat_y = pose_msg.pose.orientation.y
            quat_z = pose_msg.pose.orientation.z

            _, _, th_h = quat2euler((quat_w, quat_x, quat_y, quat_z))
            th_h %= 2 * math.pi
            th_h_deg = math.degrees(th_h)

            # Print the pose in which the robot localized
            if not self._localized:
                self.get_logger().warn(
                    f"Localized at x = {x_h:.2f} m, y = {y_h:.2f} m, "
                    f"theta = {th_h:.2f} rad ({th_h_deg:.1f}º)"
                )

            # Check if the estimated pose is within tolerance of the goal
            position_error = math.dist((self._goal[0], self._goal[1]), (x_h, y_h))

            if position_error <= self._goal_tolerance:
                execution_time = time.time() - self._start_time

                self.get_logger().warn(
                    f"Congratulations, you reached the goal in {execution_time:.3f} s! | "
                    f"Estimated pose: x = {x_h:.2f} m, y = {y_h:.2f} m, "
                    f"theta = {th_h:.2f} rad ({th_h_deg:.1f}º)",
                    once=True,  # Log only the first time this function is hit
                )

        self._localized = pose_msg.localized


def main(args=None):
    rclpy.init(args=args)
    monitoring_node = MonitoringNode()

    try:
        rclpy.spin(monitoring_node)
    except KeyboardInterrupt:
        pass

    monitoring_node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
