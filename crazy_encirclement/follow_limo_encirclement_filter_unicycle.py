import rclpy
import time
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from crazyflie_interfaces.msg import StringArray, Position
from std_msgs.msg import Bool
from std_srvs.srv import Empty
from crazyflie_interfaces.srv import Arm
from std_msgs.msg import Float32
from crazy_encirclement.filters import FilterUnicycle, FilterRelativeII, wrap_to_pi, wrap_to_2pi
from crazy_encirclement_interfaces.msg import Metadata
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy, QoSPresetProfiles
from motion_capture_tracking_interfaces.msg import NamedPoseArray
from rclpy.duration import Duration
from scipy.spatial.transform import Rotation as R


class FollowUnicycleEncirclement(Node):
    def __init__(self):
        """
            Node that sends the crazyflie to a desired position
            The desired position comes from the distortion of a circle
        """
        super().__init__('follow_unicycle_encirclement')
        self.info = self.get_logger().info

        # Parameters
        self.declare_parameter('robot', 'C20')
        self.declare_parameter('number_of_agents', 4)
        self.declare_parameter('target', 'LIMO')

        self.robot = str(self.get_parameter('robot').value)
        self.number_of_agents = int(self.get_parameter('number_of_agents').value)
        self.target = str(self.get_parameter('target').value)

        # Determine which target parameters to use based on whoisthetarget
        self.declare_parameter('whoisthetarget', 'limo')
        whoisthetarget = str(self.get_parameter('whoisthetarget').value)
        
        # Select target prefix based on target type
        if whoisthetarget == 'limo':
            target_prefix = 'target_limo'
        else:
            target_prefix = 'target_drone'
        self.info(f"Selected target for filter parameters: {whoisthetarget} (prefix: {target_prefix})")

        # Filter parameters for target (nested under target_limo or target_drone)
        self.declare_parameter(f'{target_prefix}.P', [1.0, 1.0, 0.15, 0.5, 0.2, 0.1])
        self.declare_parameter(f'{target_prefix}.Q', [0.1, 0.1, 0.01, 0.05, 0.1, 0.01])
        self.declare_parameter(f'{target_prefix}.V', [0.1, 0.1, 0.1])
    
        # Filter parameters for relative (nested under relative)
        self.declare_parameter('relative.P_rel', [0.1, 0.1])
        self.declare_parameter('relative.Q_rel', [0.01, 0.01])
        self.declare_parameter('relative.V_rel', [0.1, 0.1])
        self.declare_parameter('relative.Noise_relative', [0.02, 0.02, 0.02])

        # Control parameters (nested under controls)
        self.declare_parameter('controls.hover_height', 0.3)
        self.declare_parameter('controls.radius_nominal', 0.5)
        self.declare_parameter('controls.omega_nominal', 0.5)
        self.declare_parameter('controls.encirclement_height', 0.3)
        self.declare_parameter('controls.zupt_threshold', 0.05)
        self.declare_parameter('controls.k_phi', 1.0)
        self.declare_parameter('controls.k_r', 1.0)
        self.declare_parameter('controls.k_omega', 2.0)

        # Other parameters (nested under others)
        self.declare_parameter('others.frame_id', 'world')
        self.declare_parameter('others.seed', 42)
        self.declare_parameter('others.predict_hz', 100.0)
        self.declare_parameter('others.update_hz', 10.0)
        
        # Get control parameters (nested)
        self.hover_height   = float(self.get_parameter('controls.hover_height').value)
        self.radius_nominal = float(self.get_parameter('controls.radius_nominal').value)
        self.r              = float(self.get_parameter('controls.radius_nominal').value)
        self.omega_nominal  = float(self.get_parameter('controls.omega_nominal').value)
        self.omega          = float(self.get_parameter('controls.omega_nominal').value)
        self.n_agents       = int(self.get_parameter('number_of_agents').value)
        self.k_phi          = float(self.get_parameter('controls.k_phi').value)
        self.k_r            = float(self.get_parameter('controls.k_r').value)
        self.k_omega        = float(self.get_parameter('controls.k_omega').value)
        self.frame_id       = str(self.get_parameter('others.frame_id').value)
        self.whoisthetarget = str(self.get_parameter('whoisthetarget').value)
        
        # Get filter parameters for target (using selected prefix)
        self.P_list = self.get_parameter(f'{target_prefix}.P').value
        self.Q_list = self.get_parameter(f'{target_prefix}.Q').value
        self.V_list = self.get_parameter(f'{target_prefix}.V').value

        # Get filter parameters for relative
        self.P_rel_list = self.get_parameter('relative.P_rel').value
        self.Q_rel_list = self.get_parameter('relative.Q_rel').value
        self.V_rel_list = self.get_parameter('relative.V_rel').value
        self.Noise_relative_list = self.get_parameter('relative.Noise_relative').value

        # Get other parameters
        self.predict_hz          = self.get_parameter('others.predict_hz').value
        self.update_hz           = self.get_parameter('others.update_hz').value
        self.zupt_threshold      = self.get_parameter('controls.zupt_threshold').value
        self.encirclement_height = self.get_parameter('controls.encirclement_height').value

        # Set random seed
        seed = self.get_parameter('others.seed').value
        np.random.seed(seed)

        # Reboot client
        self.reboot_client = self.create_client(Empty, self.robot + '/reboot')

        # Flags and variables
        self.timer_period = 1.0 / self.predict_hz  # seconds
        self.has_initial_pose = False
        self.has_final = False
        self.land_flag = False
        self.has_order = False

        self.T_init  = np.eye(4)
        self.T_final = np.eye(4)
        self.T_curr  = np.eye(4)
        self.alpha = None

        self.leader   = None
        self.follower = None

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
        
        self.create_subscription(
            StringArray, '/agents_order',
            self._order_callback,
            10)  
        
        # Wait until order is received
        while (not self.has_order):
            rclpy.spin_once(self, timeout_sec=0.1)
        self.info(f'Order received - Leader ({self.leader}) | Follower ({self.follower})')
        
        # Subscribe to motion capture poses
        qos_profile = QoSProfile(reliability =QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                deadline = Duration(seconds=0, nanoseconds=1e9/100.0))
        self.create_subscription(
            NamedPoseArray, '/poses',
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

        # Wait until order and initial pose are received
        while (not self.has_order and not self.has_initial_pose):
            rclpy.spin_once(self, timeout_sec=0.1)

        # Subscribe to gps scanner topic to update filter with measurements
        qos_profile = QoSProfile(reliability =QoSReliabilityPolicy.BEST_EFFORT,
                                    history=QoSHistoryPolicy.KEEP_LAST,
                                    depth=1,
                                    deadline = Duration(seconds=0, nanoseconds=1e9/100.0))
        self.create_subscription(
            NamedPoseArray,
            f'/{self.robot}/gps_scanner_relative_poses',
            self._update_callback,
            qos_profile
        )
        
        # Wait until relative poses have arrived
        self.LIMO_pose = None
        while (self.LIMO_pose is None):
            rclpy.spin_once(self, timeout_sec=0.1)
        self.info('LIMO pose received...')

        # Initializing filter Unicycle
        initialization_noise = np.random.multivariate_normal(np.zeros(len(self.P_list)), np.diag(np.square(self.P_list)))

        # Initial states - transform from drone body frame to global frame
        # LIMO_pose is in drone's body frame, need to transform to initial/global frame
        limo_pos_body = np.array([self.LIMO_pose.pose.position.x, self.LIMO_pose.pose.position.y])
        limo_pos_global = self.T_init[0:2, 3] + self.T_init[0:2, 0:2] @ limo_pos_body
        x_init = limo_pos_global[0] + initialization_noise[0]
        y_init = limo_pos_global[1] + initialization_noise[1]
        
        # Transform heading from drone body frame to global frame
        rotation = R.from_quat([self.LIMO_pose.pose.orientation.x, self.LIMO_pose.pose.orientation.y,
                                self.LIMO_pose.pose.orientation.z, self.LIMO_pose.pose.orientation.w])
        heading_body = rotation.as_euler('zyx')[0]
        drone_heading = np.arctan2(self.T_init[1, 0], self.T_init[0, 0])
        safe_heading_noise = np.clip(initialization_noise[2], -np.pi/4, np.pi/4)
        heading_init = wrap_to_pi(drone_heading + heading_body + safe_heading_noise)
        
        angular_speed_init = 0.0 # + initialization_noise[3]
        # Initialize velocity to zero (vehicle starts at rest)
        # Adding noise would bias the estimate since abs() creates positive bias
        linear_speed_init  = 0.0
        z_ground_init      = self.LIMO_pose.pose.position.z + initialization_noise[5]

        self.filter_unicycle_params = {
            'P': self.P_list,
            'Q': self.Q_list,
            'V': self.V_list,
            'position_guess': [x_init, y_init],
            'heading_guess':  heading_init,
            'angular_speed_guess': angular_speed_init,
            'linear_speed_guess':  linear_speed_init,
            'z_ground_guess': z_ground_init,
            'zupt_threshold': self.zupt_threshold
        }
        self.filter_unicycle = FilterUnicycle(self.robot, self.whoisthetarget, self.filter_unicycle_params, self, self.T_init)

        # Initializing filter Relative
        self.FOLLOWER_pose = None
        self.LEADER_pose   = None
        while (self.FOLLOWER_pose is None or self.LEADER_pose is None):
            rclpy.spin_once(self, timeout_sec=0.1)
        self.info('Follower and Leader poses received...')

        # Extract X, Y Positions (Not Orientations)
        limo_x = self.LIMO_pose.pose.position.x
        limo_y = self.LIMO_pose.pose.position.y
        
        leader_x = self.LEADER_pose.pose.position.x
        leader_y = self.LEADER_pose.pose.position.y
        
        follower_x = self.FOLLOWER_pose.pose.position.x
        follower_y = self.FOLLOWER_pose.pose.position.y
        
        # Ego drone's initial position
        ego_x = self.T_init[0, 3] 
        ego_y = self.T_init[1, 3]

        # Calculate True Geometric Phases [-pi, pi]
        phase_leader   = np.arctan2(leader_y - limo_y, leader_x - limo_x)
        phase_follower = np.arctan2(follower_y - limo_y, follower_x - limo_x)
        phase_ego      = np.arctan2(ego_y - limo_y, ego_x - limo_x)

        # Generate Initialization Noise
        initialization_noise_rel = np.random.multivariate_normal(
            np.zeros(len(self.P_rel_list)), 
            np.diag(np.square(self.P_rel_list))
        )

        # 5. Calculate Strictly Positive Initial Distances [0, 2pi)
        # d_ahead: Angle from Ego to Leader
        d_ahead_init = (phase_leader - phase_ego) + initialization_noise_rel[0]
        d_ahead_init = wrap_to_2pi(d_ahead_init)

        # d_behind: Angle from Follower to Ego
        d_behind_init = (phase_ego - phase_follower) + initialization_noise_rel[1]
        d_behind_init = wrap_to_2pi(d_behind_init)

        # 6. Build Parameter Dictionary
        self.filter_relative_params = {
            'P_rel': self.P_rel_list,
            'Q_rel': self.Q_rel_list,
            'V_rel': self.V_rel_list,
            'x_guess': [d_ahead_init, d_behind_init] 
        }
        
        self.filter_relative = FilterRelativeII(self.robot, self.filter_relative_params, self)
        self.info(f"FilterRelative initialized. d_ahead: {d_ahead_init:.2f}, d_behind: {d_behind_init:.2f}")
        
        # Crazyflie position command publisher
        self.position_pub = self.create_publisher(Position, f'/{self.robot}/cmd_position', 10)
        self.omega_pub  = self.create_publisher(Float32, f'/{self.robot}/relative/filtered/omega', 10)
        self.radius_pub = self.create_publisher(Float32, f'/{self.robot}/relative/filtered/radius', 10)
        self.radius_correction_pub = self.create_publisher(Float32, f'/{self.robot}/relative/filtered/radius_correction', 10)
        
        # Arming all drones
        self.arm_client = self.create_client(Arm, self.robot + '/arm')
        # Wait until the service is available
        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.arm()
        time.sleep(2)

        self.info('Follow unicycle encirclement node has been started.')
        # Timer of the main loop
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
                _ = self.filter_relative.predict(self.timer_period)
                _ = self.filter_unicycle.predict(self.timer_period)
                self.hover()

            # Following LIMO
            elif self.state == 2:
                # --- A. ESTIMATION ---
                # 1. Get Unicycle Center (Global)
                limo_state = self.filter_unicycle.predict(self.timer_period)
                center_pos = np.array(limo_state['position']) 
                center_z   = limo_state['z_ground']
                d_phis = self.filter_relative.predict(self.timer_period)
                
                # --- B. CONTROL ---
                # 2. Get Phase Errors (Filter 2)
                d_phis = self.filter_relative.predict(self.timer_period)
                d_ahead  = d_phis['d_ahead']  # Distance to Leader
                d_behind = d_phis['d_behind'] # Distance to Follower

                # Computing adaptative omega nominal
                self.omega = (limo_state['linear_speed'] / self.r) * self.k_omega
                
                # Follower pushes you (+), Leader blocks you (-)
                phase_error = (1.0 / (d_behind + 1e-6)) - (1.0 / (d_ahead + 1e-6))
                u_correction = self.k_phi * phase_error                
                max_correction = self.omega_nominal * 3.5 
                u_correction = np.clip(u_correction, -max_correction, max_correction)

                # 4. INTEGRATE Virtual Phase
                if self.alpha is None:
                    # Start at current angle relative to car
                    curr_pos = self.T_curr[0:2, 3] 
                    self.alpha = np.arctan2(curr_pos[1] - center_pos[1], 
                                            curr_pos[0] - center_pos[0])

                # Update the angle: Nominal Orbit + Feedback Correction
                rotation_speed = self.omega + u_correction
                self.alpha += rotation_speed * self.timer_period
                self.alpha = wrap_to_pi(self.alpha)

                # Publishing omega (full rotation speed for monitoring)
                omega_d = Float32()
                omega_d.data = rotation_speed
                self.omega_pub.publish(omega_d)
                # self.info(f'Angular velocity: {rotation_speed:.2f} rad/s, Phase error: {phase_error:.2f}, d_ahead: {d_ahead:.2f}, d_behind: {d_behind:.2f}')

                # --- C. TRAJECTORY GENERATION ---
                # 5. Polar -> Cartesian (Global Frame)
                # This generates the moving setpoint on the circle
                x_des = center_pos[0] + self.r * np.cos(self.alpha)
                y_des = center_pos[1] + self.r * np.sin(self.alpha)
                z_des = center_z      + self.encirclement_height
                p_des = np.array([x_des, y_des, z_des])

                # Computing adaptative radius nominal
                dr = limo_state['linear_speed'] * self.timer_period
                p  = np.array([self.r * np.cos(self.alpha), self.r * np.sin(self.alpha)])
                if limo_state['linear_speed'] < self.zupt_threshold:
                    v = np.array([0.0, 0.0])
                else:
                    v  = np.array([limo_state['linear_speed'] * np.cos(limo_state['heading']),
                                   limo_state['linear_speed'] * np.sin(limo_state['heading'])])
                nominator = np.dot(p, v)
                denominator = np.linalg.norm(p) * np.linalg.norm(v) + 1e-6
                r_correction = self.k_r * dr * nominator / denominator
                self.r = self.radius_nominal + np.clip(r_correction, 0.0, self.radius_nominal * 2.0)
                self.radius_pub.publish(Float32(data=self.r))
                self.radius_correction_pub.publish(Float32(data=r_correction))

                # Desired position in the Vicon frame
                # self.alpha = wrap_to_2pi(self.alpha)
                r_des = self._get_desired_position(p_des)   
                self.send_position(r_des)

            # Landing state
            elif self.state == 3:
                if self.has_final:
                    self.landing()
                    if self.i_landing < len(self.t_landing)-1:
                        self.i_landing += 1
                    else:
                        self.reboot()
                        self.info('Exiting node')  
                        self.destroy_node()
                        rclpy.shutdown()   

        except KeyboardInterrupt:
            self.info('Exiting open loop command node')

    def _update_callback(self, msg):
        ''' Callback to update the filter with new GPS measurements. '''
        # Extract measurement for the robot
        for pose in msg.poses:
            if self.target in pose.name:
                self.LIMO_pose = pose
            elif pose.name == self.follower:
                self.FOLLOWER_pose = pose
            elif pose.name == self.leader:
                self.LEADER_pose = pose

        # Getting measurements
        if self.LIMO_pose is not None and \
           self.FOLLOWER_pose is not None and \
           self.LEADER_pose is not None:

            # 1. Noise Injection
            # Limo
            measurement_limo_noise = np.random.multivariate_normal(np.zeros(3), np.diag(np.square(self.V_list[0:3])))
            measurement_limo = np.array([self.LIMO_pose.pose.position.x, self.LIMO_pose.pose.position.y, self.LIMO_pose.pose.position.z])
            measurement_limo += measurement_limo_noise
            
            # Follower
            measurement_follower_noise = np.random.multivariate_normal(np.zeros(3), np.diag(np.square(self.Noise_relative_list)))
            measurement_follower = np.array([self.FOLLOWER_pose.pose.position.x, self.FOLLOWER_pose.pose.position.y, self.FOLLOWER_pose.pose.position.z])
            measurement_follower += measurement_follower_noise
            
            # Leader
            measurement_leader_noise = np.random.multivariate_normal(np.zeros(3), np.diag(np.square(self.Noise_relative_list)))
            measurement_leader = np.array([self.LEADER_pose.pose.position.x, self.LEADER_pose.pose.position.y, self.LEADER_pose.pose.position.z])
            measurement_leader += measurement_leader_noise

            # 2. Updating filters
            # Run filters in both Hover (1) and Encircle (2) states
            if self.state in [1, 2]:
                
                # Current pose in the initial frame (Compute Once)
                T_curr_robot = np.linalg.inv(self.T_init) @ self.T_curr
                R_drone = T_curr_robot[0:3, 0:3]
                p_drone = T_curr_robot[0:3, 3]

                # Update Unicycle Filter
                self.filter_unicycle.update(
                    measurement_limo, 
                    R_drone, 
                    p_drone
                )

                # Update Relative Phase Filter 
                # Order strictly matches: (limo, leader, follower, R)
                self.filter_relative.update(
                    measurement_limo, 
                    measurement_leader, 
                    measurement_follower, 
                    R_drone
                )             

    def _poses_changed(self, msg):
        """ Topic update callback to the motion capture lib's
            poses topic to send through the external position
            to the crazyflie. All steps based on the Vicon position.
        """
        # Initialize the initial pose and phase if not already set using vicon data
        for robot_pose in msg.poses:
            
            if robot_pose.name == self.robot:
                # Update current pose
                self.T_curr[0, 3] = robot_pose.pose.position.x
                self.T_curr[1, 3] = robot_pose.pose.position.y
                self.T_curr[2, 3] = robot_pose.pose.position.z
                rotation = R.from_quat([robot_pose.pose.orientation.x, robot_pose.pose.orientation.y,
                                        robot_pose.pose.orientation.z, robot_pose.pose.orientation.w])
                R_mat = rotation.as_matrix()
                self.T_curr[0:3, 0:3] = R_mat

                # Set final pose when landing is commanded
                if (self.has_final == False) and (self.land_flag == True):
                    self.final_pose = np.zeros(3)
                    self.info("Landing...")
                    self.final_pose[0] = robot_pose.pose.position.x
                    self.final_pose[1] = robot_pose.pose.position.y
                    self.final_pose[2] = robot_pose.pose.position.z
                    self.landing_traj(3)
                    self.has_final = True
                    self.state = 3

    def _order_callback(self, msg: StringArray):
        ''' Callback to receive the order of agents. '''
        # if not self.has_order:
            # self.info(f"Phase received: {msg.data}")
        order = msg.data
        for robot in order:
            if robot == self.robot:
                i = order.index(robot)
                if i == 0:
                    self.leader = order[self.n_agents-1]
                    self.follower = order[i+1]
                elif i == (self.n_agents-1):
                    self.leader = order[i-1]
                    self.follower = order[0]
                else:
                    self.leader = order[i-1]
                    self.follower = order[i+1]
        self.has_order = True

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
        # try:
        #     self.t_landing = np.arange(self.T_final[2, 3], self.T_init[2, 3], -0.01)
        # except Exception as e:
        #     self.info(f"Error in landing trajectory generation: {e}")
        self.t_landing = np.arange(t_max, 0.0, -self.timer_period)

        self.i_landing = 0
        self.r_landing = np.zeros((3, len(self.t_landing)))
        self.r_landing[0,:] += self.final_pose[0] * np.ones(len(self.t_landing))
        self.r_landing[1,:] += self.final_pose[1] * np.ones(len(self.t_landing))
        self.r_landing[2,:] = self.final_pose[2] * (self.t_landing / t_max)

    def _landing_callback(self, msg):
        ''' Callback to initiate landing procedure. '''
        self.land_flag = msg.data
        self.state = 3

    def _encircle_callback(self, msg):
        ''' Callback to initiate encirclement procedure. '''
        self.encirclement_stage = 0  # Start with approach phase
        self.hover_height = 0
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
        time.sleep(3.0)  

    def arm(self):
        ''' Reboot the system. '''
        req = Arm.Request()
        req.arm = True
        self.arm_client.call_async(req)
        # Call the service and get the response asynchronously
        future = self.arm_client.call_async(req)
        # Wait for the result and handle the response
        rclpy.spin_until_future_complete(self, future)

        # Now handle the response
        if future.result() is not None:
            self.get_logger().info(f'Service call successful, response: {future.result()}')
        else:
            self.get_logger().error('Service call failed')  

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
            # self.info(f'Received initial pose from topic: {self.T_init[:3, 3]}')

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
    rclpy.init()
    follower = FollowUnicycleEncirclement()
    rclpy.spin(follower)
    follower.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
