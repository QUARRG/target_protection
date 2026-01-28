import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose
from motion_capture_tracking_interfaces.msg import NamedPoseArray, NamedPose
from crazyflie_interfaces.msg import Position
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSPresetProfiles, ReliabilityPolicy, DurabilityPolicy
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
        self.update_hz = 200
        self.reference = self.get_parameter('reference_object').get_parameter_value().string_value
        self.R_wc = R.from_quat([0, 0, 0, 1])
        self.R_cw = R.from_quat([0, 0, 0, 1])
        self.has_mocap = False
        
        # qos_profile = QoSProfile(reliability =QoSReliabilityPolicy.BEST_EFFORT,
        #     history=QoSHistoryPolicy.KEEP_LAST,
        #     depth=10,
        #     deadline=Duration(seconds=0, nanoseconds=0))
        # qos_profile = QoSPresetProfiles.SENSOR_DATA.value
        poses_qos_deadline = 100.0  # example Hz

        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            deadline=Duration(nanoseconds=int(1e9 / poses_qos_deadline))
        )
        # qos_profile.deadline = deadline_duration
        self.create_subscription(
            NamedPoseArray, "/poses",
            self._callback, qos_profile
        )
        # 
        self.ref_pose = PoseStamped()
        self.scanner = NamedPoseArray()
        self.timer = self.create_timer(1/self.update_hz, self._timer_callback)
        self.relative_position_pub = self.create_publisher(NamedPoseArray, 'poses_relative', qos_profile)
        # self.vicon_position_pub = self.create_publisher(PoseStamped, f'/{self.robot}/vicon_position', 10)

    def _callback(self, msg: NamedPoseArray):
        '''Callback function to process incoming NamedPoseArray messages from Vicon and publish noisy GPS positions.'''
        if msg:
            self.has_mocap = True
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
                    self.R_wc = R.from_quat([qx, qy, qz, qw])
                    self.R_cw = self.R_wc.inv()


            # Getting relative poses of other robots with respect to ego
            for pose in msg.poses:
                if pose.name != self.reference:
                    relative_pose = NamedPose()
                    relative_pose.name = pose.name
                    aux_pose = np.array([pose.pose.position.x - self.ref_pose.pose.position.x,pose.pose.position.y - self.ref_pose.pose.position.y,pose.pose.position.z - self.ref_pose.pose.position.z])
                    aux_pose = self.R_cw.apply(aux_pose)
                    relative_pose.pose.position.x = aux_pose[0]
                    relative_pose.pose.position.y = aux_pose[1]
                    relative_pose.pose.position.z = aux_pose[2]
                    #orientation
                    R_wd = R.from_quat([pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w])
                    R_cd = self.R_cw*R_wd
                    qx, qy, qz, qw = R_cd.as_quat()
                    relative_pose.pose.orientation.x = qx
                    relative_pose.pose.orientation.y = qy
                    relative_pose.pose.orientation.z = qz
                    relative_pose.pose.orientation.w = qw
                    self.scanner.poses.append(relative_pose)
        
        # Publish the Vicon position without noise
        # self.vicon_position_pub.publish(self.ref_pose)

    def _timer_callback(self):
        '''Timer callback to publish the noisy GPS position at the specified rate.'''
        if self.has_mocap:
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