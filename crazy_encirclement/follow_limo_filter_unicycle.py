import rclpy
import time
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from crazyflie_interfaces.msg import StringArray, Position
from std_msgs.msg import Bool
from std_srvs.srv import Empty
from std_msgs.msg import Float32
from crazy_encirclement.filters import FilterUnicycle
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
        self.P_list = self.get_parameter('P').value
        self.Q_list = self.get_parameter('Q').value
        self.V_list = self.get_parameter('V').value
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

        self.T_init  = np.eye(4)
        self.T_final = np.eye(4)
        self.T_curr  = np.eye(4)
        self.set_point = np.array([-0.3, 0., 0.3])  # Offset from LIMO position

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

        # Subscribe to gps scanner topic to update filter with measurements
        self.create_subscription(
            NamedPoseArray,
            f'/{self.robot}/gps_scanner_ii_poses',
            self._update_callback,
            10
        )
        
        self.LIMO_pose = None
        # Wait until LIMO pose has arrived
        while (self.LIMO_pose is None):
            rclpy.spin_once(self, timeout_sec=0.1)

        # Initial noise using P 
        initialization_noise = np.random.multivariate_normal(np.zeros(len(self.P_list)), np.diag(np.square(self.P_list)))

        # Initial states
        x_init = self.LIMO_pose.pose.position.x + initialization_noise[0]
        y_init = self.LIMO_pose.pose.position.y + initialization_noise[1]
        heading_init = np.arctan2(self.LIMO_pose.pose.position.y - self.T_init[1, 3], self.LIMO_pose.pose.position.x - self.T_init[0, 3]) + initialization_noise[2]
        angular_speed_init = 0.0 + initialization_noise[3]
        linear_speed_init = 0.0 + initialization_noise[4]
        z_ground_init = self.LIMO_pose.pose.position.z + initialization_noise[5]

        # Create filter instance using actual measured initial values
        self.filter = {
            'P': self.P_list,
            'Q': self.Q_list,
            'V': self.V_list,
            'position_guess': [x_init, y_init],
            'heading_guess': heading_init,
            'angular_speed_guess': angular_speed_init,
            'linear_speed_guess': linear_speed_init,
            'z_ground_guess': z_ground_init
        }
        self.filter = FilterUnicycle(self.robot, self.filter, self)
        
        # Crazyflie position command publisher
        self.position_pub = self.create_publisher(Position, f'/{self.robot}/cmd_position', 10)
        
        # Arming all drones
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

            # Following LIMO
            elif self.state == 2:
                # Predicted LIMO states
                limo_states: dict = self.filter.predict(self.timer_period)
                limo_3d_position = np.array([*limo_states['position'], limo_states['z_ground']])
                
                # Adding set point offset to the predicted position considering the limo frame (not world frame)
                limo_3d_rotation = R.from_euler('z', limo_states['heading']).as_matrix()
                projected_set_point = limo_3d_rotation @ self.set_point
                limo_3d_position += projected_set_point

                # Desired position in the Vicon frame
                r_des = self._get_desired_position(limo_3d_position)   
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

    def _update_callback(self, msg: NamedPoseArray):
        ''' Callback to update the filter with new GPS measurements. '''
        # Extract measurement for the robot
        for pose in msg.poses:
            if 'limo' in pose.name.lower():
                # Relative pose of LIMO in the current robot frame
                self.LIMO_pose = pose

        # Update filter with new measurement
        if self.LIMO_pose is not None and self.state == 2:
            measurement = np.array([self.LIMO_pose.pose.position.x, self.LIMO_pose.pose.position.y, self.LIMO_pose.pose.position.z])

            # Current pose in the initial frame
            T_curr_robot = np.linalg.inv(self.T_init) @ self.T_curr
            self.filter.update(measurement, T_curr_robot[0:3, 0:3], T_curr_robot[0:3, 3])

    def _poses_changed(self, msg):
        """ Topic update callback to the motion capture lib's
            poses topic to send through the external position
            to the crazyflie. All steps based on the Vicon position.
        """
        # Initialize the initial pose and phase if not already set using vicon data
        # self.info(f"poses: {msg}")
        for robot_pose in msg.poses:
            
            if robot_pose.name == self.robot:
                # Update current pose
                # if not self.land_flag:
                self.T_curr[0, 3] = robot_pose.pose.position.x
                self.T_curr[1, 3] = robot_pose.pose.position.y
                self.T_curr[2, 3] = robot_pose.pose.position.z
                rotation = R.from_quat([robot_pose.pose.orientation.x, robot_pose.pose.orientation.y,
                                        robot_pose.pose.orientation.z, robot_pose.pose.orientation.w])
                R_mat = rotation.as_matrix()
                self.T_curr[0:3, 0:3] = R_mat

                # Set final pose when landing is commanded
                if (self.has_final == False) and (self.land_flag == True):
                    # Difference between current and initial positions (ignoring orientation)
                    position_diff = np.linalg.norm(self.T_init[0:3, 3] - self.T_curr[0:3, 3])
                    self.info(f'Position difference: {position_diff:.3f} m')
                    # self.T_final = self.T_curr.copy()
                    # self.T_final[0:3, 3] += np.array([0.25, 0.25, 0.0])
                    # self.info("Landing...")
                    # self.landing_traj(3)
                    # self.has_final = True

                    if position_diff < self.hover_height * 1.05:  # Threshold of 0.1 meters
                        self.T_final = self.T_curr.copy()
                        self.info("Landing...")
                        self.landing_traj(3)
                        self.has_final = True
                        # self.T_final = np.eye(4)
                        # self.T_final[0, 3] = robot_pose.pose.position.x
                        # self.T_final[1, 3] = robot_pose.pose.position.y
                        # self.T_final[2, 3] = robot_pose.pose.position.z
                        # rotation = R.from_quat([robot_pose.pose.orientation.x, robot_pose.pose.orientation.y,
                        #                         robot_pose.pose.orientation.z, robot_pose.pose.orientation.w])
                        # R_mat = rotation.as_matrix()
                        # self.T_final[0:3, 0:3] = R_mat
                        # self.info("Landing...")
                        # self.landing_traj(3)
                        # self.has_final = True
                    else:
                        self.send_position(np.array([self.T_init[0, 3], self.T_init[1, 3], self.T_init[2, 3] + self.hover_height]))

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
        # self.t_landing = np.arange(t_max, 0.1, -self.timer_period)
        try:
            self.t_landing = np.arange(self.T_final[2, 3], self.T_init[2, 3], -0.015)
        except Exception as e:
            self.info(f"Error in landing trajectory generation: {e}")
            self.t_landing = np.arange(t_max, 0.1, -self.timer_period)

        self.info(f'Landing trajectory time steps: {self.t_landing}')
        self.i_landing = 0
        self.r_landing = np.zeros((3, len(self.t_landing)))
        self.r_landing[0,:] = self.T_final[0, 3] * np.ones(len(self.t_landing))
        self.r_landing[1,:] = self.T_final[1, 3] * np.ones(len(self.t_landing))
        self.r_landing[2,:] = self.t_landing  #self.T_final[2, 3] * (self.t_landing / t_max)

    def _landing_callback(self, msg):
        ''' Callback to initiate landing procedure. '''
        self.land_flag = msg.data
        self.state = 3

    def _encircle_callback(self, msg):
        ''' Callback to initiate encirclement procedure. '''
        self.state = 2

    def hover(self):
        ''' Hovering procedure at the hover height. '''
        self.send_position(np.array([self.T_init[0, 3], self.T_init[1, 3], self.T_init[2, 3] + self.hover_height]))

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

    def _get_desired_position(self, set_point: np.ndarray) -> np.ndarray:
        ''' Compute the desired position based on the current, final, and initial poses. '''
        # Relative transformation from current to final pose
        T_rel = np.linalg.inv(self.T_init) @ self.T_curr

        T_set_point = np.eye(4)
        T_set_point[0:3, 0:3] = T_rel[0:3, 0:3]
        T_set_point[0:3, 3]   = set_point

        # Desired position in the world frame
        T_des = self.T_init @ T_set_point
        r_des = T_des[0:3, 3] 
        return r_des


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
