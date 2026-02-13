import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from motion_capture_tracking_interfaces.msg import NamedPoseArray, NamedPose
from crazyflie_interfaces.msg import Position
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSPresetProfiles, ReliabilityPolicy, DurabilityPolicy
from rclpy.duration import Duration
from scipy.spatial.transform import Rotation as R

class MocapRelative(Node):
    def __init__(self):
        super().__init__('mocap_relative_node')
        # Initial parameters
        self.declare_parameter('update_hz', 200.0)
        self.declare_parameter('reference_object', 'LIMO')
        self.declare_parameter('evader', 'C26') #the evader position will not be relatve
        self.update_hz = self.get_parameter('update_hz').get_parameter_value().double_value
        self.reference = self.get_parameter('reference_object').get_parameter_value().string_value
        self.evader = self.get_parameter('evader').get_parameter_value().string_value
        # Transformation matrices
        self.R_wc = R.from_quat([0, 0, 0, 1])
        self.R_cw = R.from_quat([0, 0, 0, 1])

        self.has_mocap = False
        self.R0 = {}
            
        # QoS Profile
        poses_qos_deadline = self.update_hz  # Hz
        self.R_wd0 = {}
        
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
        # Subscription to evader detection
        self.create_subscription(
            Bool, '/evader_detection',
            self._evader_detection_callback,
            10) 
         
        self.ref_pose = PoseStamped()
        self.relative_poses = NamedPoseArray()
        self.timer = self.create_timer(1/self.update_hz, self._timer_callback)
        self.relative_position_pub = self.create_publisher(NamedPoseArray, 'poses_relative', qos_profile)

    def _callback(self, msg: NamedPoseArray):
        '''Callback function to process incoming NamedPoseArray messages from Vicon and publish noisy GPS positions.'''
        if msg:         
            # # Getting reference pose
            # for pose in msg.poses:
            #     if pose.name == self.reference:
            #         self.ref_pose.pose.position.x = pose.pose.position.x
            #         self.ref_pose.pose.position.y = pose.pose.position.y
            #         self.ref_pose.pose.position.z = pose.pose.position.z
            #         qx = pose.pose.orientation.x
            #         qy = pose.pose.orientation.y
            #         qz = pose.pose.orientation.z
            #         qw = pose.pose.orientation.w
            #         self.R_wc = R.from_quat([qx, qy, qz, qw])  # Car to world
            #         self.R_cw = self.R_wc.inv()                # World to Car
            if not self.R_wd0:
                for pose in msg.poses:
                    R_aux = R.from_quat([pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w])
                    self.R_wd0[pose.name] = R_aux.inv() #save the inverse of the initial rotation
            else:
                self.has_mocap = True
                self.ref_pose.header = msg.header
                self.relative_poses.header = msg.header
                self.relative_poses.poses = []
                
                # Getting reference pose
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
                    elif pose.name == self.evader:
                        self.relative_poses.poses.append(pose)

                # Getting relative poses of other robots with respect to ego
                for pose in msg.poses:
                    if pose.name != self.reference:
                        relative_pose_d = NamedPose()
                        relative_pose_d.name = pose.name
                        rel_pose_w = np.array([pose.pose.position.x - self.ref_pose.pose.position.x,pose.pose.position.y - self.ref_pose.pose.position.y,pose.pose.position.z - self.ref_pose.pose.position.z])
                        rel_pose_d = self.R_wd0[pose.name].apply(rel_pose_w)
                        relative_pose_d.pose.position.x = rel_pose_d[0]
                        relative_pose_d.pose.position.y = rel_pose_d[1]
                        relative_pose_d.pose.position.z = rel_pose_d[2]
                        #drone orientation w.r.t. its initial orientation
                        R_dw = R.from_quat([pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w])
                        R_dd = self.R_wd0[pose.name]*R_dw
                        # R_cd = self.R_0*self.R_wc
                        qx, qy, qz, qw = R_dd.as_quat()
                        relative_pose_d.pose.orientation.x = qx
                        relative_pose_d.pose.orientation.y = qy
                        relative_pose_d.pose.orientation.z = qz
                        relative_pose_d.pose.orientation.w = qw
                        self.relative_poses.poses.append(relative_pose_d)
            
            # Publish the Vicon position without noise
            # self.vicon_position_pub.publish(self.ref_pose)

    def _evader_detection_callback(self, msg: Bool):
        if msg.data == True:
            self.reference = self.evader
        else:
            self.reference = 'LIMO'

    def _timer_callback(self):
        '''Timer callback to publish the relative poses at the specified rate.'''
        if self.has_mocap:
            self.relative_position_pub.publish(self.relative_poses)



def main():
    '''Main function to initialize the relative position node and start spinning.'''
    rclpy.init()
    relative_pos_node = MocapRelative()
    rclpy.spin(relative_pos_node)
    relative_pos_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()