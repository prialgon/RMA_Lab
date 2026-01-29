import rclpy
from rclpy.node import Node

from amr_msgs.msg import KeyLogger
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import numpy as np
import numpy.typing

from rclpy.qos import qos_profile_sensor_data


class TeleoperationNode(Node):
    def __init__(self) -> None:
        super().__init__("teleoperation_node")

        self.linear: float = 0.0  # Forward > 0, Backward < 0
        self.angular: float = 0.0  # Left > 0, Right < 0

        self.allowed_distance: float = 0.3

        self.velocity_increment: float = 0.05

        self.lidar_x_angle: float = 10
        self.lidar_z_angle: float = 160

        # Do not change
        self.lidar_x_angle_rad: float = self.lidar_x_angle * np.pi / 180
        self.lidar_z_angle_rad: float = self.lidar_z_angle * np.pi / 180

        self.forward_allowed: bool = False
        self.right_allowed: bool = False
        self.backward_allowed: bool = False
        self.left_allowed: bool = False

        self._keylog_listener = self.create_subscription(
            msg_type=KeyLogger,
            topic="teleop",
            callback=self._handle_keylog,
            qos_profile=10,
        )

        self._teleoperation_publisher = self.create_publisher(
            msg_type=Twist, topic="cmd_vel", qos_profile=10
        )

        self._lidar_listener = self.create_subscription(
            msg_type=LaserScan,
            topic="scan",
            callback=self._handle_lidar,
            qos_profile=qos_profile_sensor_data,
        )

    def _get_lidar_areas(
        self, ranges: list[float], increment: float
    ) -> tuple[numpy.typing.NDArray, numpy.typing.NDArray]:
        # Forward 1st range, 0:x_angle
        forward_range_1_start: int = 0
        forward_range_1_end: int = forward_range_1_start + int(
            self.lidar_x_angle_rad / increment
        )

        # Left points, x_angle:z_angle
        left_range_start: int = forward_range_1_end
        left_range_end: int = left_range_start + int(self.lidar_z_angle_rad / increment)

        # Backward points, z_angle:z_angle+2*x_angle
        backward_range_start: int = left_range_end
        backward_range_end: int = backward_range_start + int(
            self.lidar_x_angle_rad * 2 / increment
        )

        # Right points, z_angle+2*x_angle:2*z_angle+2*x_angle
        right_range_start: int = backward_range_end
        right_range_end: int = right_range_start + int(
            self.lidar_z_angle_rad / increment
        )

        # Forward 2nd range: 2*z_angle+2*x_angle:0
        forward_range_2_start: int = right_range_end
        forward_range_2_end: int = -1

        # Handle NaN
        ranges_array: numpy.typing.NDArray = np.array(ranges)
        np.nan_to_num(ranges_array, copy=False, nan=self.allowed_distance + 1)

        forward: numpy.typing.NDArray = np.concatenate(
            (
                ranges_array[forward_range_1_start:forward_range_1_end],
                ranges_array[forward_range_2_start:forward_range_2_end],
            )
        )

        backward: numpy.typing.NDArray = ranges_array[
            backward_range_start:backward_range_end
        ]

        return forward, backward

    def _handle_lidar(self, msg: LaserScan) -> None:
        forward, backward = self._get_lidar_areas(msg.ranges, msg.angle_increment)

        self.forward_allowed = min(forward) > self.allowed_distance
        self.backward_allowed = min(backward) > self.allowed_distance

        # Linear velocity
        if not self.forward_allowed and self.linear > 0.0:
            cmd = Twist()
            cmd.linear.x = 0.0
            self.linear = 0.0
            self._teleoperation_publisher.publish(cmd)
        elif not self.backward_allowed and self.linear < 0.0:
            cmd = Twist()
            cmd.linear.x = 0.0
            self.linear = 0.0
            self._teleoperation_publisher.publish(cmd)

    def _handle_keylog(self, msg: KeyLogger) -> None:
        key: str = msg.key_id

        if key == "w" and self.forward_allowed:
            self.linear += self.velocity_increment
        elif key == "s" and self.backward_allowed:
            self.linear -= self.velocity_increment
        elif key == "a":
            self.angular -= self.velocity_increment
        elif key == "d":
            self.angular += self.velocity_increment
        elif key == "space":
            self.linear = 0.0
            self.angular = 0.0

        cmd = Twist()
        cmd.linear.x = self.linear
        cmd.angular.z = self.angular
        self._teleoperation_publisher.publish(cmd)
        self.get_logger().info(f"Received {key}")


def main(args=None):
    rclpy.init(args=args)
    node = TeleoperationNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
