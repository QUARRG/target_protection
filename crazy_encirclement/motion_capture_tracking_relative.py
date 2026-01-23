import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose
from motion_capture_tracking_interfaces.msg import NamedPoseArray, NamedPose
from crazyflie_interfaces.msg import Position
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSPresetProfiles
from rclpy.duration import Duration
from scipy.spatial.transform import Rotation as R

class MOCAP_relative(Node):
    def __init__(self):
        super().__init__('gps_scanner_node')
        self.declare_parameter('robot', 'C01')
        self.declare_parameter('V_ego', [0.1, 0.1, 0.1])
        self.declare_parameter('V_rel', [0.1, 0.1, 0.1])
        self.declare_parameter('update_hz', 100.0)
        self.declare_parameter('reference_object', 'LIMO')

        self.robot = self.get_parameter('robot').get_parameter_value().string_value
        self.V_ego = self.get_parameter('V_ego').get_parameter_value().double_array_value
        self.V_rel = self.get_parameter('V_rel').get_parameter_value().double_array_value
        self.update_hz = 110
        self.reference = self.get_parameter('reference_object').get_parameter_value().string_value
        self.rot = R.from_quat([0, 0, 0, 1])
        
        qos_profile = QoSProfile(reliability =QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            deadline=Duration(seconds=0, nanoseconds=0))

        self.create_subscription(
            NamedPoseArray, "/poses",
            self._callback, qos_profile
        )
        # qos_profile = QoSPresetProfiles.SENSOR_DATA.value
        self.ref_pose = PoseStamped()
        self.scanner = NamedPoseArray()
        self.timer = self.create_timer(1/self.update_hz, self._timer_callback)
        self.relative_position_pub = self.create_publisher(NamedPoseArray, 'poses_relative', qos_profile)
        # self.vicon_position_pub = self.create_publisher(PoseStamped, f'/{self.robot}/vicon_position', 10)

    def _callback(self, msg: NamedPoseArray):
        '''Callback function to process incoming NamedPoseArray messages from Vicon and publish noisy GPS positions.'''
        self.ref_pose.header = msg.header
        self.scanner.header = msg.header
        self.scanner.poses = []
        
        # Getting ego pose
        for pose in msg.poses:
            if pose.name == self.reference:

                self.ref_pose.pose.position.x = pose.pose.position.x
                self.ref_pose.pose.position.y = pose.pose.position.y
                self.ref_pose.pose.position.z = pose.pose.position.z
                qx = pose.pose.orientation.x
                qy = pose.pose.orientation.y
                qz = pose.pose.orientation.z
                qw = pose.pose.orientation.w
                self.rot = R.from_quat([qx, qy, qz, qw])
                self.rot = self.rot.inv()


                # ego_pose = NamedPose()
                # gps_noise = np.random.multivariate_normal(np.zeros(3), np.diag(np.square(self.V_ego)))
                # ego_pose.name = pose.name
                # ego_pose.pose.position.x = pose.pose.position.x 
                # ego_pose.pose.position.y = pose.pose.position.y 
                # ego_pose.pose.position.z = pose.pose.position.z 
                # self.scanner.poses.append(ego_pose)

        # Getting relative poses of other robots with respect to ego
        for pose in msg.poses:
            if pose.name != self.reference:
                relative_pose = NamedPose()
                relative_pose.name = pose.name
                aux_pose = np.array([pose.pose.position.x - self.ref_pose.pose.position.x,pose.pose.position.y - self.ref_pose.pose.position.y,pose.pose.position.z - self.ref_pose.pose.position.z])
                aux_pose = self.rot.apply(aux_pose)
                relative_pose.pose.position.x = aux_pose[0]
                relative_pose.pose.position.y = aux_pose[1]
                relative_pose.pose.position.z = aux_pose[2]

                self.scanner.poses.append(relative_pose)
        
        # Publish the Vicon position without noise
        # self.vicon_position_pub.publish(self.ref_pose)

    def _timer_callback(self):
        '''Timer callback to publish the noisy GPS position at the specified rate.'''
        self.relative_position_pub.publish(self.scanner)

def main():
    '''Main function to initialize the GPS node and start spinning.'''
    rclpy.init()
    relative_pos_node = MOCAP_relative()
    rclpy.spin(relative_pos_node)
    relative_pos_node.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()