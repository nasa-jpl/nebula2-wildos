import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

class TfTimePrinter(Node):
    def __init__(self):
        super().__init__('tf_time_printer')
        self.subscription = self.create_subscription(
            TFMessage,
            '/replay/tf',
            self.listener_callback,
            10
        )

    def listener_callback(self, msg):
        for transform in msg.transforms:
            stamp = transform.header.stamp
            self.get_logger().info(f"stamp: {stamp.sec}.{stamp.nanosec:09d}")

def main():
    rclpy.init()
    node = TfTimePrinter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
