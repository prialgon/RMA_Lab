from sshkeyboard import listen_keyboard
import rclpy
from rclpy.node import Node
from amr_msgs.msg import KeyLogger

class KeyboardPublisher(Node):
    def __init__(self) -> None:
        super().__init__("keyboard_node")
        
        self._publisher = self.create_publisher(
            msg_type=KeyLogger, topic="teleop", qos_profile=10
        )
        
        listen_keyboard(on_press=self._keyboard_press_callback)
        
    def _keyboard_press_callback(self, key: str) -> None:
        msg = KeyLogger()
        msg.key_id = key
        self._publisher.publish(msg)

def main(args = None) -> None:
    rclpy.init(args=args)
    node = KeyboardPublisher()
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()