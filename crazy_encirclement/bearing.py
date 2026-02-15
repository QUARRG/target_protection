import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool
from motion_capture_tracking_interfaces.msg import NamedPoseArray, NamedPose
from crazyflie_interfaces.msg import Position
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSPresetProfiles, ReliabilityPolicy, DurabilityPolicy
from rclpy.duration import Duration
from scipy.spatial.transform import Rotation as R

class Bearing(Node):
    def __init__(self):
        super().__init__('bearing_node')
        # Initial parameters
        self.declare_parameter('update_hz', 10.0)
        self.declare_parameter('relative', False)
        self.declare_parameter('robot', 'C25')
        self.relative = self.get_parameter('relative').get_parameter_value().bool_value
        self.robot = self.get_parameter('robot').get_parameter_value().string_value
        self.frequency = self.get_parameter('update_hz').get_parameter_value().double_value
        self.has_distances = False
        self.ego_pose = None
        self.R_dw = R.identity()
        # self.declare_parameter('evader', 'C26') #the evader position will not be relatve
        poses_qos_deadline = 100.0  # example Hz

        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            deadline=Duration(nanoseconds=int(1e9 / poses_qos_deadline))
        )
        # qos_profile.deadline = deadline_duration
        if self.relative:
            self.create_subscription(
                NamedPoseArray, "/poses",
                self._poses_callback, qos_profile
            )
        else:
            self.create_subscription(
                NamedPoseArray, "/poses_relative",
                self._poses_callback, qos_profile
            )
        qos_profile = QoSPresetProfiles.SENSOR_DATA.value
        self.distance_pub = self.create_publisher(NamedPoseArray, f'{self.robot}/distances', qos_profile)
        self.distances = NamedPoseArray()


        self.create_timer(1/self.frequency, self._timer_callback)
    
    def _timer_callback(self):
        if self.has_distances:
            self.distance_pub.publish(self.distances)

    def _poses_callback(self, msg: NamedPoseArray):
        self.has_distances = True
        self.distances.header = msg.header
        self.distances.poses = []
        for pose in msg.poses:
            if pose.name == self.robot:
                self.ego_pose = pose.pose
                self.R_dw = R.from_quat([pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w])
                break
        if self.ego_pose:
            for pose in msg.poses:
                if pose.name != self.robot:
                    relative_dist = NamedPose()
                    relative_dist.name = pose.name
                    relative_dist_arr = np.array([pose.pose.position.x - self.ego_pose.position.x, pose.pose.position.y - self.ego_pose.position.y, pose.pose.position.z - self.ego_pose.position.z])
                    relative_dist_arr = self.R_dw.inv().apply(relative_dist_arr) #convert relative position to ego frame
                    relative_dist.pose.position.x = relative_dist_arr[0]
                    relative_dist.pose.position.y = relative_dist_arr[1]
                    relative_dist.pose.position.z = relative_dist_arr[2]
                    self.distances.poses.append(relative_dist)

def main():
    rclpy.init()
    bearing = Bearing()
    rclpy.spin(bearing)
    bearing.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()