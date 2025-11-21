import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from motion_capture_tracking_interfaces.msg import NamedPoseArray
from crazyflie_interfaces.msg import Position
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.duration import Duration


class GPSNode(Node):
    def __init__(self):
        super().__init__('gps_node')
        self.declare_parameter('robot', 'C01')
        self.declare_parameter('V', [0.1, 0.1, 0.1])
        self.declare_parameter('update_hz', 10.0)

        self.robot = self.get_parameter('robot').get_parameter_value().string_value
        self.V = self.get_parameter('V').get_parameter_value().double_array_value
        self.update_hz = self.get_parameter('update_hz').get_parameter_value().double_value

        qos_profile = QoSProfile(reliability =QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            deadline=Duration(seconds=0, nanoseconds=0))

        self.create_subscription(
            NamedPoseArray, "/poses",
            self._callback, qos_profile
        )

        self.pose = PoseStamped()
        self.timer = self.create_timer(1.0 / self.update_hz, self._timer_callback)
        self.gps_position_pub   = self.create_publisher(PoseStamped, f'/{self.robot}/gps_position', 10)
        self.vicon_position_pub = self.create_publisher(PoseStamped, f'/{self.robot}/vicon_position', 10)

    def _callback(self, msg: NamedPoseArray):
        '''Callback function to process incoming NamedPoseArray messages from Vicon and publish noisy GPS positions.'''
        # self.pose = PoseStamped()
        self.pose.header.stamp = self.get_clock().now().to_msg()
        
        # Add Gaussian noise to the position of the specified robot
        for pose in msg.poses:
            if pose.name == self.robot:
                self.pose.header.frame_id = msg.header.frame_id
                self.pose.pose.position.x = pose.pose.position.x
                self.pose.pose.position.y = pose.pose.position.y
                self.pose.pose.position.z = pose.pose.position.z
        
        # Publish the Vicon position without noise
        self.vicon_position_pub.publish(self.pose)

    def _timer_callback(self):
        '''Timer callback to publish the noisy GPS position at the specified rate.'''
        noise = np.random.multivariate_normal(np.zeros(3), np.diag(np.square(self.V)))
        pose = PoseStamped()
        pose.header.frame_id = self.pose.header.frame_id
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = self.pose.pose.position.x + noise[0]
        pose.pose.position.y = self.pose.pose.position.y + noise[1]
        pose.pose.position.z = self.pose.pose.position.z + noise[2]
        self.gps_position_pub.publish(pose)

def main():
    '''Main function to initialize the GPS node and start spinning.'''
    rclpy.init()
    gps_node = GPSNode()
    rclpy.spin(gps_node)
    gps_node.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()