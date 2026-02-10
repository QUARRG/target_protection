import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose
from motion_capture_tracking_interfaces.msg import NamedPoseArray, NamedPose
from crazyflie_interfaces.msg import Position
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSPresetProfiles, ReliabilityPolicy, DurabilityPolicy
from rclpy.duration import Duration
from scipy.spatial.transform import Rotation as R

class MocapRelative(Node):
    def __init__(self):
        super().__init__('mocap_filter_relative_node')
        # Initial parameters
        self.declare_parameter('robots', [])
        self.declare_parameter('update_hz', 200.0)
        self.robots = self.get_parameter('robots').get_parameter_value().string_array_value
        self.update_hz = self.get_parameter('update_hz').get_parameter_value().double_value

        # Transformation matrices per robot
        self.initial_poses = {
            robot: np.zeros(7) for robot in self.robots
        }
            
        # QoS Profile
        poses_qos_deadline = self.update_hz  # Hz
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
        
        self.timer = self.create_timer(1.0 / self.update_hz, self._timer_callback)
        self.relative_poses_pubs = {
            robot: self.create_publisher(NamedPoseArray, f'{robot}/poses_relative', qos_profile) for robot in self.robots + ['LIMO']
        }
        self.relative_poses = NamedPoseArray()

    def _callback(self, msg: NamedPoseArray):
        '''Callback function to process incoming NamedPoseArray messages from Vicon and publish relative positions per robot.'''
        if msg:
            self.has_mocap = True
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
                    self.R_wc = R.from_quat([qx, qy, qz, qw])  # Car to world
                    self.R_cw = self.R_wc.inv()                # World to Car

            # Getting relative poses of other robots with respect to reference
            for pose in msg.poses:
                if pose.name != self.reference:
                    relative_pose = NamedPose()
                    relative_pose.name = pose.name
                    # Translation w.r.t. reference
                    aux_pose = np.array([pose.pose.position.x - self.ref_pose.pose.position.x,
                                         pose.pose.position.y - self.ref_pose.pose.position.y,
                                         pose.pose.position.z - self.ref_pose.pose.position.z])
                    # Rotate to car frame
                    aux_pose = self.R_cw.apply(aux_pose)
                    relative_pose.pose.position.x = aux_pose[0]
                    relative_pose.pose.position.y = aux_pose[1]
                    relative_pose.pose.position.z = aux_pose[2]
                    # Drone orientation w.r.t. reference
                    R_wd = R.from_quat([pose.pose.orientation.x,
                                        pose.pose.orientation.y,
                                        pose.pose.orientation.z,
                                        pose.pose.orientation.w])
                    R_cd = self.R_cw*R_wd
                    qx, qy, qz, qw = R_cd.as_quat()
                    relative_pose.pose.orientation.x = qx
                    relative_pose.pose.orientation.y = qy
                    relative_pose.pose.orientation.z = qz
                    relative_pose.pose.orientation.w = qw
                    self.relative_poses.poses.append(relative_pose)

    def _timer_callback(self):
        '''Timer callback to publish the relative poses at the specified rate.'''
        if self.has_mocap:
            self.relative_poses_pub.publish(self.relative_poses)


def main():
    '''Main function to initialize the relative position node and start spinning.'''
    rclpy.init()
    relative_pos_node = MocapRelative()
    rclpy.spin(relative_pos_node)
    relative_pos_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()