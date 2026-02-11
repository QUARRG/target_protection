import rclpy
import time
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from crazyflie_interfaces.msg import StringArray, Position
from std_msgs.msg import Bool
from std_srvs.srv import Empty
from std_msgs.msg import Float32
from crazy_encirclement.filters import FilterUnicycle, wrap_to_2pi, wrap_to_pi
from crazy_encirclement_interfaces.msg import Metadata
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy, QoSPresetProfiles
from motion_capture_tracking_interfaces.msg import NamedPoseArray
from rclpy.duration import Duration
from crazyflie_py import Crazyswarm
from scipy.spatial.transform import Rotation as R


class FollowUnicycle(Node):
    def __init__(self, swarm=None):
        """
            Node that sends the crazyflie to a desired position
            The desired position comes from the distortion of a circle
        """
        super().__init__('follow_unicycle')
        self.swarm = swarm
        self.info = self.get_logger().info
        self.info('Follow unicycle node has been started.')

        # Parameters
        self.declare_parameter('robot', 'C20')
        self.declare_parameter('hover_height', 0.3)
        self.declare_parameter('relative', False)
        self.declare_parameter('number_of_agents', 4)
        self.declare_parameter('k_phi', 1.0)
        self.declare_parameter('frame_id', 'world')

        self.robot = str(self.get_parameter('robot').value)
        self.hover_height = float(self.get_parameter('hover_height').value)
        self.relative = bool(self.get_parameter('relative').value)
        self.n_agents = int(self.get_parameter('number_of_agents').value)
        self.k_phi    = float(self.get_parameter('k_phi').value)
        self.frame_id = str(self.get_parameter('frame_id').value)

        # Filter parameters
        self.declare_parameter('P', [1.0, 1.0, 0.15, 0.5, 0.2])
        self.declare_parameter('Q', [0.1, 0.1, 0.01, 0.05, 0.1])
        self.declare_parameter('V', [0.1, 0.1, 0.1])
        self.declare_parameter('predict_hz', 50.0)
        self.declare_parameter('update_hz', 10.0)
        self.declare_parameter('seed', 42)        
        
        # Get filter parameters
        P_list = self.get_parameter('P').value
        Q_list = self.get_parameter('Q').value
        V_list = self.get_parameter('V').value
        self.predict_hz = self.get_parameter('predict_hz').value
        self.update_hz  = self.get_parameter('update_hz').value

        # Set random seed
        seed = self.get_parameter('seed').value
        np.random.seed(seed)

        # Reboot client
        self.reboot_client = self.create_client(Empty, self.robot + '/reboot')

        # Flags and variables
        self.timer_period = 1.0 / self.predict_hz  # seconds

        self.has_initial_pose = False
        self.has_final = False
        self.land_flag = False

        self.previous_pose_time = 0.0
        self.T_init  = np.eye(4)
        self.T_final = np.eye(4)
        self.T_curr  = np.eye(4)
        self.set_point = np.array([0.5, 0., self.hover_height])

        self.i_landing = 0
        self.i_takeoff = 0

        self.state = 0
        # 0-take-off, 1-hover, 2-encirclement, 3-landing

        # ----------------------------------------------------------------------
        # Subscribers
        # ----------------------------------------------------------------------
        # Command line inputs
        self.create_subscription(
            Bool,
            '/landing',
            self._landing_callback,
            10)
        
        self.create_subscription(
            Bool,
            '/encircle',
            self._encircle_callback,
            10)
        
        # Subscribe to motion capture poses
        poses_qos_deadline = 100  # Hz
        qos_profile = QoSProfile(reliability =QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                deadline = Duration(seconds=0, nanoseconds=1e9/100.0))
        self.create_subscription(
            NamedPoseArray, "poses",
            self._poses_changed, qos_profile
        )

        # Subscribe to initial pose with transient local QoS (latching)
        initial_pose_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.create_subscription(
            PoseStamped,
            f'/{self.robot}/initial_pose',
            self._initial_pose_callback,
            initial_pose_qos
        )
        
        # Wait until initial pose is received
        while (not self.has_initial_pose):
            rclpy.spin_once(self, timeout_sec=0.1)

        # Crazyflie position command publisher
        self.position_pub = self.create_publisher(Position, f'/{self.robot}/cmd_position', 10)
        if swarm:
            self.timeHelper = self.swarm.timeHelper
            self.allcfs = self.swarm.allcfs

            # arm (one by one)
            for cf in self.allcfs.crazyflies:
                cf.arm(True)
                self.timeHelper.sleep(1.0)
                self.info(f'arming drone{cf}')

        # self.timeHelper.sleep(2.0)
        # input("Press Enter to takeoff")
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):
        ''' Timer callback to send position commands to the crazyflie based on the current state. '''
        try:
            # Take-off state
            if self.state == 0:
                if self.has_initial_pose:
                    self.takeoff()

            # Hover state
            elif self.state == 1:
                self.hover()

            # Following state
            elif self.state == 2:
                # The set point is defined in the initial pose frame, so we need to transform it to the current framee using the current transformation matrix T_curr and the initial transformation matrix T_init
                T_init_inv = np.linalg.inv(self.T_init)
                T_des = self.T_curr @ T_init_inv @ np.block([[np.eye(3), self.set_point.reshape(3,1)], [0, 0, 0, 1]])
                r_des = T_des[0:3, 3]
                self.send_position(r_des)

            # Landing state
            elif self.state == 3:
                if self.has_final:
                    self.landing()
                    if self.i_landing < len(self.t_landing)-1:
                        self.i_landing += 1
                    else:
                        if self.swarm:
                            self.allcfs.arm(False)
                            self.timeHelper.sleep(3.0)
                        self.reboot()
                        self.info('Exiting circle node')  
                        self.destroy_node()
                        rclpy.shutdown()   

        except KeyboardInterrupt:
            self.info('Exiting open loop command node')

    def _poses_changed(self, msg):
        """ Topic update callback to the motion capture lib's
            poses topic to send through the external position
            to the crazyflie. All steps based on the Vicon position.
        """
        # Initialize the initial pose and phase if not already set using vicon data
        # self.info(f"poses: {msg}")
        for robot_pose in msg.poses:
            
            if robot_pose.name == self.robot:
                # Update current pose if not landing
                if not self.land_flag:
                    self.T_curr[0, 3] = robot_pose.pose.position.x
                    self.T_curr[1, 3] = robot_pose.pose.position.y
                    self.T_curr[2, 3] = robot_pose.pose.position.z
                    rotation = R.from_quat([robot_pose.pose.orientation.x, robot_pose.pose.orientation.y,
                                            robot_pose.pose.orientation.z, robot_pose.pose.orientation.w])
                    R_mat = rotation.as_matrix()
                    self.T_curr[0:3, 0:3] = R_mat

                # Set final pose when landing is commanded
                elif (self.has_final == False) and (self.land_flag == True):
                    self.T_final = np.eye(4)
                    self.T_final[0, 3] = robot_pose.pose.position.x
                    self.T_final[1, 3] = robot_pose.pose.position.y
                    self.T_final[2, 3] = robot_pose.pose.position.z
                    rotation = R.from_quat([robot_pose.pose.orientation.x, robot_pose.pose.orientation.y,
                                            robot_pose.pose.orientation.z, robot_pose.pose.orientation.w])
                    R_mat = rotation.as_matrix()
                    self.T_final[0:3, 0:3] = R_mat
                    self.info("Landing...")
                    self.landing_traj(3)
                    self.has_final = True

    def takeoff(self):
        ''' Take-off procedure to reach the hover height. '''
        # self.info(f'takeoff position{self.r_takeoff[:, self.i_takeoff]}')
        self.send_position(self.r_takeoff[:, self.i_takeoff])
        
        # Increment take-off index or switch to hover state
        if self.i_takeoff < len(self.t_takeoff)-1:
            self.i_takeoff += 1
        else:
            self.state = 1

    def takeoff_traj(self, t_max: float):
        ''' Take-off trajectory generation. '''
        self.t_takeoff = np.arange(0, t_max, self.timer_period)
        self.r_takeoff = np.zeros((3, len(self.t_takeoff))) 
        self.r_takeoff[0,:] = self.T_init[0, 3] * np.ones(len(self.t_takeoff))
        self.r_takeoff[1,:] = self.T_init[1, 3] * np.ones(len(self.t_takeoff))
        self.r_takeoff[2,:] = (self.T_init[2, 3] + self.hover_height) * (self.t_takeoff / t_max)

    def landing_traj(self, t_max: float):
        ''' Landing trajectory generation. '''
        self.t_landing = np.arange(t_max, 0.1, -self.timer_period)
        self.i_landing = 0
        self.r_landing = np.zeros((3, len(self.t_landing)))
        self.r_landing[0,:] = self.T_final[0, 3] * np.ones(len(self.t_landing))
        self.r_landing[1,:] = self.T_final[1, 3] * np.ones(len(self.t_landing))
        self.r_landing[2,:] = self.T_final[2, 3] * (self.t_landing / t_max)

    def _landing_callback(self, msg):
        ''' Callback to initiate landing procedure. '''
        self.land_flag = msg.data
        self.state = 3

    def _encircle_callback(self, msg):
        ''' Callback to initiate encirclement procedure. '''
        self.state = 2

    def hover(self):
        ''' Hovering procedure at the hover height. '''
        # msg = Position()
        # msg.x = self.initial_pose[0]
        # msg.y = self.initial_pose[1]
        # msg.z = self.hover_height
        self.send_position(np.array([self.T_init[0, 3], self.T_init[1, 3], self.hover_height]))
        # self.position_pub.publish(msg)

    def landing(self):
        ''' Landing procedure to reach the ground. '''
        self.send_position(self.r_landing[:, self.i_landing])

    def reboot(self):
        ''' Reboot the system. '''
        req = Empty.Request()
        self.reboot_client.call_async(req)
        time.sleep(1.0)    

    def send_position(self, r):
        ''' Send position command to the crazyflie. '''
        msg = Position()
        msg.x = float(r[0])
        msg.y = float(r[1])
        msg.z = float(r[2])
        self.position_pub.publish(msg)
    
    def _initial_pose_callback(self, msg):
        ''' Callback to handle the initial pose from the topic. '''
        if not self.has_initial_pose:
            # Filling transformation matrix T_init with initial pose
            self.T_init = np.eye(4)
            self.T_init[0, 3] = msg.pose.position.x 
            self.T_init[1, 3] = msg.pose.position.y
            self.T_init[2, 3] = msg.pose.position.z
            rotation = R.from_quat([msg.pose.orientation.x, msg.pose.orientation.y,
                                    msg.pose.orientation.z, msg.pose.orientation.w])
            R_mat = rotation.as_matrix()
            self.T_init[0:3, 0:3] = R_mat
            self.takeoff_traj(4)
            self.has_initial_pose = True
            self.info(f'Received initial pose from topic: {self.T_init[:3, 3]}')


def main():
    swarm = Crazyswarm()
    if not rclpy.ok():
        rclpy.init()
    follower = FollowUnicycle(swarm)
    rclpy.spin(follower)
    follower.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
