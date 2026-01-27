import rclpy
from rclpy.node import Node

from amr_msgs.msg import KeyLogger
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

class TeleoperationNode(Node):
    def __init__(self):
        super().__init__("teleoperation_node")
        
        self.linear = 0.0
        self.angular = 0.0
        self.min_distance=99

        self._keylog_listener = self.create_subscription(
            msg_type=KeyLogger,
            topic="teleop",
            callback=self._handle_keylog,
            qos_profile=10
        )

        self._teleoperation_publisher = self.create_publisher(
            msg_type=Twist,
            topic="cmd_vel",
            qos_profile=10
        )

        self._lidar_listener = self.create_subscription(
            msg_type=LaserScan,
            topic="scan",
            callback=self._handle_lidar,
            qos_profile=10
        )

    def _handle_lidar(self, msg: LaserScan) -> None:
        self.min_distance = min(msg.ranges)

    

    def _handle_keylog(self, msg: KeyLogger) -> None:
        key: str = msg.key_id
        cmd = Twist()
        if key == "w":
            cmd.linear.x = 0.02
        elif key == "s":
            cmd.linear.x = -0.02
        elif key == "a":
            cmd.angular.z = 0.02
        elif key == "d":
            cmd.angular.z = -0.02
        if key == "space" or self.min_distance>0.2:
            cmd.angular.z = 0.0
            cmd.linear.x = 0.0


        self._teleoperation_publisher.publish(cmd)
        self.get_logger().info(f"Received {key}")
        




def main(args=None):
    rclpy.init(args=args)
    node = TeleoperationNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
