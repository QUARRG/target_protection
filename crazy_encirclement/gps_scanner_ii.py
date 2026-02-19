import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose
from motion_capture_tracking_interfaces.msg import NamedPoseArray, NamedPose
from crazyflie_interfaces.msg import Position
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from rclpy.duration import Duration
from scipy.spatial.transform import Rotation as R


class GPSScannerNodeII(Node):
    def __init__(self):
        super().__init__('gps_scanner_ii_node')
        self.declare_parameter('robot', 'C01')
        self.declare_parameter('update_hz', 100.0)

        self.robot = self.get_parameter('robot').get_parameter_value().string_value
        self.update_hz = self.get_parameter('update_hz').get_parameter_value().double_value

        qos_profile = QoSProfile(reliability =QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                deadline = Duration(seconds=0, nanoseconds=1e9/100.0))
        self.create_subscription(
            NamedPoseArray, "/poses",
            self._callback, qos_profile
        )

        self.T_init = np.eye(4)  # SE(3) representation  
        self.initial_pose_initialized = False
        self.timer = self.create_timer(1.0 / self.update_hz, self._timer_callback)

        self.ego_pose = PoseStamped()
        self.relative_scanner = NamedPoseArray()
        self.global_scanner = NamedPoseArray()
        qos_profile = QoSProfile(reliability =QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        deadline = Duration(seconds=0, nanoseconds=1e9/100.0))
        self.gps_scanner_relative_poses_pub = self.create_publisher(NamedPoseArray, f'/{self.robot}/gps_scanner_relative_poses', qos_profile)
        self.gps_scanner_global_poses_pub = self.create_publisher(NamedPoseArray, f'/{self.robot}/gps_scanner_global_poses', qos_profile)
        
        # Create latching publisher for initial pose (transient local QoS for late joiners)
        initial_pose_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.initial_pose_pub = self.create_publisher(
            PoseStamped, 
            f'/{self.robot}/initial_pose', 
            initial_pose_qos
        )

        # Log initialization
        self.get_logger().info(f'GPS Scanner Node II initialized for robot: {self.robot} with update rate: {self.update_hz} Hz')

    def _callback(self, msg: NamedPoseArray):
        '''Callback function to process incoming NamedPoseArray messages from Vicon and publish noisy GPS positions.'''
        self.ego_pose.header = msg.header
        self.relative_scanner.header = msg.header
        self.relative_scanner.poses  = []
        self.global_scanner.header = msg.header
        self.global_scanner.poses  = []
        # Log received message
        # self.get_logger().debug(f'Received NamedPoseArray with {len(msg.poses)} poses at time {msg.header.stamp.sec}.{msg.header.stamp.nanosec}')
        
        # Getting ego pose
        T_ego = np.eye(4)
        for pose in msg.poses:
            if pose.name == self.robot:
                # Capture initial pose on first callback
                if not self.initial_pose_initialized:
                    # Get rotation matrix from quaternion
                    rotation = R.from_quat([pose.pose.orientation.x, pose.pose.orientation.y,
                                            pose.pose.orientation.z, pose.pose.orientation.w])
                    R_mat = rotation.as_matrix()
                    # Construct initial pose in SE(3)
                    self.T_init[0:3, 0:3] = R_mat
                    self.T_init[0, 3] = pose.pose.position.x
                    self.T_init[1, 3] = pose.pose.position.y
                    self.T_init[2, 3] = pose.pose.position.z
                    self.initial_pose_initialized = True
                    self.get_logger().info(f'Initial pose captured:\n{self.T_init}')
                    
                    # Publish initial pose on latching topic
                    initial_pose_msg = PoseStamped()
                    initial_pose_msg.header = msg.header
                    initial_pose_msg.header.frame_id = 'world'
                    initial_pose_msg.pose.position.x = self.T_init[0, 3]
                    initial_pose_msg.pose.position.y = self.T_init[1, 3]
                    initial_pose_msg.pose.position.z = self.T_init[2, 3]
                    q_init = R.from_matrix(self.T_init[0:3, 0:3]).as_quat()
                    initial_pose_msg.pose.orientation.x = q_init[0]
                    initial_pose_msg.pose.orientation.y = q_init[1]
                    initial_pose_msg.pose.orientation.z = q_init[2]
                    initial_pose_msg.pose.orientation.w = q_init[3]
                    self.initial_pose_pub.publish(initial_pose_msg)

                # Compute relative ego pose to initial pose using SE(3) operations
                T_ego[0:3, 0:3] = R.from_quat([pose.pose.orientation.x, pose.pose.orientation.y,
                                               pose.pose.orientation.z, pose.pose.orientation.w]).as_matrix()
                T_ego[0, 3] = pose.pose.position.x
                T_ego[1, 3] = pose.pose.position.y
                T_ego[2, 3] = pose.pose.position.z
                T_ego_rel = np.linalg.inv(self.T_init) @ T_ego

                # Extract relative pose
                q_rel = R.from_matrix(T_ego_rel[0:3, 0:3]).as_quat()
                self.ego_pose.pose.orientation.x = q_rel[0]
                self.ego_pose.pose.orientation.y = q_rel[1]
                self.ego_pose.pose.orientation.z = q_rel[2]
                self.ego_pose.pose.orientation.w = q_rel[3]
                self.ego_pose.pose.position.x = T_ego_rel[0, 3]
                self.ego_pose.pose.position.y = T_ego_rel[1, 3]
                self.ego_pose.pose.position.z = T_ego_rel[2, 3]

                # Build relative ego pose
                ego_pose = NamedPose()
                ego_pose.name = pose.name
                ego_pose.pose = self.ego_pose.pose
                self.relative_scanner.poses.append(ego_pose)
                self.global_scanner.poses.append(ego_pose)

        # Getting relative poses of other robots with respect to ego
        for pose in msg.poses:
            if pose.name != self.robot:
                relative_pose = NamedPose()
                relative_pose.name = pose.name
                global_pose = NamedPose()
                global_pose.name = pose.name

                T_robot = np.eye(4)
                T_robot[0:3, 0:3] = R.from_quat([pose.pose.orientation.x, pose.pose.orientation.y,
                                                 pose.pose.orientation.z, pose.pose.orientation.w]).as_matrix()
                T_robot[0, 3] = pose.pose.position.x
                T_robot[1, 3] = pose.pose.position.y
                T_robot[2, 3] = pose.pose.position.z

                # Compute the relative position of the other robot with respect to ego using SE(3) operations
                T_robot_rel = np.linalg.inv(self.T_init) @ T_robot
                T_ego_rel   = np.linalg.inv(self.T_init) @ T_ego
                T_rel       = np.linalg.inv(T_ego_rel) @ T_robot_rel

                # Extract relative pose
                q_rel = R.from_matrix(T_rel[0:3, 0:3]).as_quat()
                relative_pose.pose.orientation.x = q_rel[0]
                relative_pose.pose.orientation.y = q_rel[1]
                relative_pose.pose.orientation.z = q_rel[2]
                relative_pose.pose.orientation.w = q_rel[3]
                relative_pose.pose.position.x = T_rel[0, 3]
                relative_pose.pose.position.y = T_rel[1, 3]
                relative_pose.pose.position.z = T_rel[2, 3]
                self.relative_scanner.poses.append(relative_pose)

                # Extract global pose
                q_rel = R.from_matrix(T_robot_rel[0:3, 0:3]).as_quat()
                global_pose.pose.orientation.x = q_rel[0]
                global_pose.pose.orientation.y = q_rel[1]
                global_pose.pose.orientation.z = q_rel[2]
                global_pose.pose.orientation.w = q_rel[3]
                global_pose.pose.position.x = T_robot_rel[0, 3]
                global_pose.pose.position.y = T_robot_rel[1, 3]
                global_pose.pose.position.z = T_robot_rel[2, 3]
                self.global_scanner.poses.append(global_pose)

    def _timer_callback(self):
        '''Timer callback to publish the noisy GPS position at the specified rate.'''
        self.gps_scanner_relative_poses_pub.publish(self.relative_scanner)
        self.gps_scanner_global_poses_pub.publish(self.global_scanner)


def main():
    '''Main function to initialize the GPS node and start spinning.'''
    rclpy.init()
    gps_node = GPSScannerNodeII()
    rclpy.spin(gps_node)
    gps_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()