import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn

from nav_msgs.msg import Odometry
import math

from rclpy.qos import qos_profile_sensor_data

from transforms3d.euler import quat2euler


class OdometryNode(LifecycleNode):
    def __init__(self) -> None:
        super().__init__("odometry_node")
        
        self._last_pos_x: float = None
        self._last_pos_y: float = None
        self._last_time: float = None 
        self._last_angular_w: float = None

        self._odometry_subscriber = self.create_subscription(
            msg_type=Odometry,
            topic="odom",
            callback=self._handle_odometry,
            qos_profile=10,
        )
        
        self._odometry_publisher = self.create_publisher(
            msg_type=Odometry,
            topic="odometry",
            qos_profile=10,
        )
        
    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        return super().on_configure(state)

    def on_activate(self,  state: LifecycleState) -> TransitionCallbackReturn:
        return super().on_activate(state)
        
        
    def _handle_odometry(self, msg: Odometry) -> None:
        
        time: float = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
        
        pos_x: float = msg.pose.pose.position.x
        pos_y: float = msg.pose.pose.position.y
        
        angular_w: float = msg.twist.twist.angular.z
        
        quat_w: float = msg.pose.pose.orientation.w
        quat_x: float = msg.pose.pose.orientation.x
        quat_y: float = msg.pose.pose.orientation.y
        quat_z: float = msg.pose.pose.orientation.z
        
        _, _, th_h = quat2euler((quat_w, quat_x, quat_y, quat_z))
        th_h %= 2 * math.pi
        
        if self._last_pos_x is None or self._last_pos_y is None or self._last_time is None or self._last_angular_w is None:
            self._last_pos_x = pos_x
            self._last_pos_y = pos_y
            self._last_time = time
            self._last_angular_w = th_h
            return None

        dt: float = time - self._last_time
        
        # vel_x = (pos_x - self._last_pos_x) / dt
        # vel_y = (pos_y - self._last_pos_y) / dt

        v = math.dist((pos_x, pos_y), (self._last_pos_x, self._last_pos_y)) / dt
                
        # vel_x: float = v_global_x * math.cos(th_h) + v_global_y * math.sin(th_h)
        # vel_y: float = -v_global_x * math.sin(th_h) + v_global_y * math.cos(th_h)
        
        vel_angular: float = (th_h - self._last_angular_w)/dt
        
        odometry_msg: Odometry = Odometry()
        odometry_msg.header = msg.header
        odometry_msg.child_frame_id = msg.child_frame_id
        odometry_msg.twist.twist.linear.x = v
        odometry_msg.twist.twist.linear.y = 0.0
        odometry_msg.twist.twist.angular.z = vel_angular
        
        self._odometry_publisher.publish(odometry_msg)
        
        self._last_pos_x = pos_x
        self._last_pos_y = pos_y
        self._last_time = time
        self._last_angular_w = th_h
        
def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
