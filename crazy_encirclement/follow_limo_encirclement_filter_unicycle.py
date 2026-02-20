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

        # Other Parameters
        self.declare_parameter('robot', 'C20')
        self.declare_parameter('number_of_agents', 4)
        self.declare_parameter('hover_height', 0.3)
        self.declare_parameter('radius_nominal', 0.5)
        self.declare_parameter('omega_nominal', 0.5)
        self.declare_parameter('k_phi', 1.0)
        self.declare_parameter('frame_id', 'world')
        
        # Staged approach parameters
        self.declare_parameter('approach_tolerance', 0.15)
        self.declare_parameter('min_neighbor_distance', 0.4)
        self.declare_parameter('staging_gain', 0.5)

        self.robot          = str(self.get_parameter('robot').value)
        self.hover_height   = float(self.get_parameter('hover_height').value)
        self.radius_nominal = float(self.get_parameter('radius_nominal').value)
        self.omega_nominal  = float(self.get_parameter('omega_nominal').value)
        self.n_agents       = int(self.get_parameter('number_of_agents').value)
        self.k_phi          = float(self.get_parameter('k_phi').value)
        self.frame_id       = str(self.get_parameter('frame_id').value)
        
        # Get staged approach parameters
        self.approach_tolerance     = float(self.get_parameter('approach_tolerance').value)
        self.min_neighbor_distance  = float(self.get_parameter('min_neighbor_distance').value)
        self.staging_gain           = float(self.get_parameter('staging_gain').value)

        # Filters parameters
        self.declare_parameter('P', [1.0, 1.0, 0.15, 0.5, 0.2])
        self.declare_parameter('Q', [0.1, 0.1, 0.01, 0.05, 0.1])
        self.declare_parameter('V', [0.1, 0.1, 0.1])

        self.declare_parameter('P_rel', [0.1, 0.1])
        self.declare_parameter('Q_rel', [0.01, 0.01])
        self.declare_parameter('V_rel', [0.1, 0.1])

        self.declare_parameter('predict_hz', 100.0)
        self.declare_parameter('update_hz', 10.0)
        self.declare_parameter('seed', 42)        
        self.declare_parameter('zupt_threshold', 0.05)
        
        # Get filter parameters
        self.P_list = self.get_parameter('P').value
        self.Q_list = self.get_parameter('Q').value
        self.V_list = self.get_parameter('V').value

        self.P_rel_list = self.get_parameter('P_rel').value
        self.Q_rel_list = self.get_parameter('Q_rel').value
        self.V_rel_list = self.get_parameter('V_rel').value

        self.predict_hz = self.get_parameter('predict_hz').value
        self.update_hz  = self.get_parameter('update_hz').value
        self.zupt_threshold = self.get_parameter('zupt_threshold').value

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
        
        self.encirclement_stage = 0
        # 0-approach (position without phase control), 1-full encirclement (with phase control)

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

        # Initializing filter Unicycle
        initialization_noise = np.zeros(6)  # np.random.multivariate_normal(np.zeros(len(self.P_list)), np.diag(np.square(self.P_list)))

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
        heading_init = wrap_to_pi(drone_heading + heading_body + initialization_noise[2])
        
        angular_speed_init = 0.0 + initialization_noise[3]
        linear_speed_init  = 0.0 + initialization_noise[4]
        z_ground_init      = self.LIMO_pose.pose.position.z + initialization_noise[5]

        self.filter_unicycle_params = {
            'P': self.P_list,
            'Q': self.Q_list,
            'V': self.V_list,
            'position_guess': [x_init, y_init],
            'heading_guess': heading_init,
            'angular_speed_guess': angular_speed_init,
            'linear_speed_guess': linear_speed_init,
            'z_ground_guess': z_ground_init
        }
        self.filter_unicycle = FilterUnicycle(self.robot, self.filter_unicycle_params, self)

        # Initializing filter relative
        self.FOLLOWER_pose = None
        self.LEADER_pose   = None
        while (self.FOLLOWER_pose is None and self.LEADER_pose is None):
            rclpy.spin_once(self, timeout_sec=0.1)

        initialization_noise_rel = np.random.multivariate_normal(np.zeros(len(self.P_rel_list)), np.diag(np.square(self.P_rel_list)))

        rotation_leader = R.from_quat([self.LEADER_pose.pose.orientation.x, self.LEADER_pose.pose.orientation.y,
                                       self.LEADER_pose.pose.orientation.z, self.LEADER_pose.pose.orientation.w])
        heading_leader_init = rotation_leader.as_euler('zyx')[0]

        rotation_follower = R.from_quat([self.FOLLOWER_pose.pose.orientation.x, self.FOLLOWER_pose.pose.orientation.y,
                                         self.FOLLOWER_pose.pose.orientation.z, self.FOLLOWER_pose.pose.orientation.w])
        heading_follower_init = rotation_follower.as_euler('zyx')[0]

        heading_ego_init = rotation.as_euler('zyx')[0]

        delta_phi_succ_init = wrap_to_pi((heading_leader_init - heading_ego_init) + initialization_noise_rel[0])
        delta_phi_pred_init = wrap_to_pi((heading_ego_init - heading_follower_init) + initialization_noise_rel[1])

        self.filter_relative_params = {
            'P': self.P_rel_list,
            'Q': self.Q_rel_list,
            'V': self.V_rel_list,
            'delta_phi_pred_guess': delta_phi_pred_init,
            'delta_phi_succ_guess': delta_phi_succ_init
        }
        self.filter_relative = FilterRelativeII(self.robot, self.filter_relative_params, self)
        
        # Crazyflie position command publisher
        self.position_pub = self.create_publisher(Position, f'/{self.robot}/cmd_position', 10)
        self.omega_pub = self.create_publisher(Float32, f'/{self.robot}/omega_d', 10)
        
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
                self.hover()

            # Following LIMO
            elif self.state == 2:
                # --- A. ESTIMATION ---
                # 1. Get Unicycle Center (Global)
                limo_state = self.filter_unicycle.predict(self.timer_period)
                center_pos = np.array(limo_state['position']) 
                center_z   = limo_state['z_ground']
                d_phis = self.filter_relative.predict(self.timer_period)

                # 2. Get Phase Errors (Filter 2)
                # These are the angular gaps to your neighbors
                # d_phis = self.filter_relative.get_state()
                d_phi_pred, d_phi_succ = d_phis['pred'], d_phis['succ']
                
                # --- B. CONTROL ---
                # 3. Compute Correction Gain
                # Goal: Make gaps equal (sum of signed errors = 0)
                # phase_error = (d_phi_pred + d_phi_succ)
                phase_error = 1/(d_phi_pred + 1e-6) + 1/(d_phi_succ + 1e-6)
                u_correction = self.k_phi * phase_error

                # 4. INTEGRATE Virtual Phase (The "Motion" Step)
                # Initialize alpha if first run
                if self.alpha is None:
                    # Start at current angle relative to car
                    curr_pos = self.T_curr[0:2, 3] 
                    self.alpha = np.arctan2(curr_pos[1] - center_pos[1], 
                                            curr_pos[0] - center_pos[0])

                # Update the angle: Nominal Orbit + Feedback Correction
                # This replaces the "Rc @ exp(omega)" logic from the old filter
                rotation_speed = self.omega_nominal + u_correction
                self.alpha += rotation_speed * self.timer_period

                # Wrap to [0, 2*pi]
                self.alpha = wrap_to_2pi(self.alpha)

                # Publishing omega
                omega_d = Float32()
                omega_d.data = rotation_speed
                self.omega_pub.publish(omega_d)

                # --- C. TRAJECTORY GENERATION ---
                # 5. Polar -> Cartesian (Global Frame)
                # This generates the moving setpoint on the circle
                x_des = center_pos[0] + self.radius_nominal * np.cos(self.alpha)
                y_des = center_pos[1] + self.radius_nominal * np.sin(self.alpha)
                z_des = center_z      + self.hover_height
                p_des = np.array([x_des, y_des, z_des])

                # Desired position in the Vicon frame
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

    def _update_callback(self, msg: NamedPoseArray):
        ''' Callback to update the filter with new GPS measurements. '''
        # Extract measurement for the robot
        for pose in msg.poses:
            if 'limo' in pose.name.lower():
                # Relative pose of LIMO in the current robot frame
                self.LIMO_pose = pose
            if pose.name == self.follower:
                # Relative pose of the follower in the current robot frame
                self.FOLLOWER_pose = pose
            if pose.name == self.leader:
                # Relative pose of the leader in the current robot frame
                self.LEADER_pose = pose

        # Getting measurements
        if self.LIMO_pose is not None and \
            self.FOLLOWER_pose is not None and \
            self.LEADER_pose is not None:

            measurement_limo_noise = np.random.multivariate_normal(np.zeros(3), np.diag(np.square(self.V_list[0:3])))
            measurement_limo = np.array([self.LIMO_pose.pose.position.x, self.LIMO_pose.pose.position.y, self.LIMO_pose.pose.position.z])
            measurement_limo += measurement_limo_noise
            
            measurement_pred_noise = np.random.multivariate_normal(np.zeros(3), np.diag(np.square(self.V_rel_list)))
            measurement_pred = np.array([self.FOLLOWER_pose.pose.position.x, self.FOLLOWER_pose.pose.position.y, self.FOLLOWER_pose.pose.position.z])
            measurement_pred += measurement_pred_noise
            
            measurement_succ_noise = np.random.multivariate_normal(np.zeros(3), np.diag(np.square(self.V_rel_list)))
            measurement_succ = np.array([self.LEADER_pose.pose.position.x, self.LEADER_pose.pose.position.y, self.LEADER_pose.pose.position.z])
            measurement_succ += measurement_succ_noise

            # Updating filters
            if self.state == 1:
                # Current pose in the initial frame
                T_curr_robot = np.linalg.inv(self.T_init) @ self.T_curr
                self.filter_relative.update(measurement_limo, measurement_pred, measurement_succ, T_curr_robot[0:3, 0:3], T_curr_robot[0:3, 3])

            elif self.state == 1 or self.state == 2:
                # Current pose in the initial frame
                T_curr_robot = np.linalg.inv(self.T_init) @ self.T_curr
                self.filter_unicycle.update(measurement_limo, T_curr_robot[0:3, 0:3], T_curr_robot[0:3, 3])

                # Current pose in the initial frame
                T_curr_robot = np.linalg.inv(self.T_init) @ self.T_curr
                self.filter_relative.update(measurement_limo, measurement_pred, measurement_succ, T_curr_robot[0:3, 0:3], T_curr_robot[0:3, 3])             

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
                    # Difference between current and initial positions (ignoring orientation)
                    position_diff = np.linalg.norm(self.T_init[0:3, 3] - self.T_curr[0:3, 3])
                    # self.info(f'Position difference: {position_diff:.3f} m')

                    if position_diff < self.hover_height * 1.05:  # Threshold of 0.1 meters
                        self.T_final = self.T_curr.copy()
                        self.info("Landing...")
                        self.landing_traj(3)
                        self.has_final = True
                    else:
                        self.send_position(np.array([self.T_init[0, 3], self.T_init[1, 3], self.T_init[2, 3] + self.hover_height]))

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
        # self.info(f'Order received - Leader ({self.leader}) | Follower ({self.follower})')

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
            self.t_landing = np.arange(self.T_final[2, 3], self.T_init[2, 3], -0.01)
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
        self.encirclement_stage = 0  # Start with approach phase
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
    
    def _check_approach_complete(self, center_pos: np.ndarray, dist_to_limo: float) -> bool:
        ''' Check if approach phase is complete and ready for full encirclement. '''
        # Condition 1: Distance to limo center within tolerance
        radius_error = abs(dist_to_limo - self.radius_nominal)
        at_nominal_radius = radius_error < self.approach_tolerance
        
        # Condition 2 & 3: Check distances to neighbors
        safe_from_neighbors = True
        
        if self.FOLLOWER_pose is not None and self.LEADER_pose is not None:
            # Distance to follower in current frame
            follower_dist = np.linalg.norm([
                self.FOLLOWER_pose.pose.position.x,
                self.FOLLOWER_pose.pose.position.y
            ])
            
            # Distance to leader in current frame
            leader_dist = np.linalg.norm([
                self.LEADER_pose.pose.position.x,
                self.LEADER_pose.pose.position.y
            ])
            
            # Both neighbors must be at safe distance
            safe_from_neighbors = (follower_dist > self.min_neighbor_distance and 
                                   leader_dist > self.min_neighbor_distance)
        
        return at_nominal_radius and safe_from_neighbors


def main():
    rclpy.init()
    follower = FollowUnicycleEncirclement()
    rclpy.spin(follower)
    follower.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
