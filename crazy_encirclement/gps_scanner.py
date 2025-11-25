import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose
from motion_capture_tracking_interfaces.msg import NamedPoseArray, NamedPose
from crazyflie_interfaces.msg import Position
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.duration import Duration


class GPSScannerNode(Node):
    def __init__(self):
        super().__init__('gps_scanner_node')
        self.declare_parameter('robot', 'C01')
        self.declare_parameter('V_ego', [0.1, 0.1, 0.1])
        self.declare_parameter('V_rel', [0.1, 0.1, 0.1])
        self.declare_parameter('update_hz', 10.0)

        self.robot = self.get_parameter('robot').get_parameter_value().string_value
        self.V_ego = self.get_parameter('V_ego').get_parameter_value().double_array_value
        self.V_rel = self.get_parameter('V_rel').get_parameter_value().double_array_value
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
        self.scanner = NamedPoseArray()
        self.timer = self.create_timer(1.0 / self.update_hz, self._timer_callback)
        self.gps_scanner_position_pub = self.create_publisher(NamedPoseArray, f'/{self.robot}/gps_scanner_position', 10)
        self.vicon_position_pub = self.create_publisher(PoseStamped, f'/{self.robot}/vicon_position', 10)

    def _callback(self, msg: NamedPoseArray):
        '''Callback function to process incoming NamedPoseArray messages from Vicon and publish noisy GPS positions.'''
        self.pose.header = msg.header
        self.scanner.header = msg.header
        self.scanner.poses = []
        
        # Getting ego pose
        for pose in msg.poses:
            if pose.name == self.robot:
                self.pose.pose.position.x = pose.pose.position.x
                self.pose.pose.position.y = pose.pose.position.y
                self.pose.pose.position.z = pose.pose.position.z

                ego_pose = NamedPose()
                gps_noise = np.random.multivariate_normal(np.zeros(3), np.diag(np.square(self.V_ego)))
                ego_pose.name = pose.name
                ego_pose.pose.position.x = pose.pose.position.x + gps_noise[0]
                ego_pose.pose.position.y = pose.pose.position.y + gps_noise[1]
                ego_pose.pose.position.z = pose.pose.position.z + gps_noise[2]
                self.scanner.poses.append(ego_pose)

        # Getting relative poses of other robots with respect to ego
        for pose in msg.poses:
            if pose.name != self.robot:
                relative_pose = NamedPose()
                relative_pose.name = pose.name
                relative_pose.pose.position.x = pose.pose.position.x - self.pose.pose.position.x
                relative_pose.pose.position.y = pose.pose.position.y - self.pose.pose.position.y
                relative_pose.pose.position.z = pose.pose.position.z - self.pose.pose.position.z
                self.scanner.poses.append(relative_pose)
        
        # Publish the Vicon position without noise
        self.vicon_position_pub.publish(self.pose)

    def _timer_callback(self):
        '''Timer callback to publish the noisy GPS position at the specified rate.'''
        self.gps_scanner_position_pub.publish(self.scanner)

def main():
    '''Main function to initialize the GPS node and start spinning.'''
    rclpy.init()
    gps_node = GPSScannerNode()
    rclpy.spin(gps_node)
    gps_node.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()