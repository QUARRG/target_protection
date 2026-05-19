import rclpy
import time
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from crazyflie_interfaces.msg import StringArray, Position
from motion_capture_tracking_interfaces.msg import NamedPoseArray
from crazyflie_interfaces.srv import Arm
from std_msgs.msg import Bool, String
from std_srvs.srv import Empty
from std_msgs.msg import Float32
from crazy_encirclement.filters import (
    build_Re,
    build_Rc,
    get_phase,
    FilterGPS,
    FilterRelative,
    wrap_to_2pi,
    wrap_to_pi,
    phase_controller
)
from crazy_encirclement_interfaces.msg import Metadata


class CircleDistortion(Node):
    def __init__(self):
        """
            Node that sends the crazyflie to a desired position
            The desired position comes from the distortion of a circle
            
            Strategy: Use GPS + Relative filters with communication outage simulation
            - 0-20s: Use GPS filter + subscribe to leader/follower phases
            - 20-40s: Communication outage - use relative filters only
            - 40-60s: Resume GPS filter + subscribe to leader/follower phases
        """
        super().__init__('circle_distortion')
        self.info = self.get_logger().info
        self.info('Circle distortion node (combined filters) has been started.')

        # Parameters
        self.declare_parameter('robot', 'C20')
        self.declare_parameter('number_of_agents', 4)
        self.declare_parameter('radius_nominal', 1.0)
        self.declare_parameter('omega_nominal', 0.8)
        self.declare_parameter('k_phi', 8.0)
        self.declare_parameter('embedding_fn_name', 'modelB')
        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('hover_height', 0.9)

        self.robot    = str(self.get_parameter('robot').value)
        self.n_agents = int(self.get_parameter('number_of_agents').value)
        self.k_phi    = float(self.get_parameter('k_phi').value)
        self.radius_nominal = float(self.get_parameter('radius_nominal').value)
        self.omega_nominal  = float(self.get_parameter('omega_nominal').value)
        self.embedding_fn_name = str(self.get_parameter('embedding_fn_name').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.hover_height = float(self.get_parameter('hover_height').value)

        # Filter parameters
        self.declare_parameter('P_ego', [0.0001, 0.0001, 0.25, 0.0001])
        self.declare_parameter('Q_ego', [0.0, 0.0, 0.001, 0.0001])
        self.declare_parameter('V_ego', [0.1, 0.1, 0.1])
        self.declare_parameter('P_rel', [0.01, 0.01, 0.25])
        self.declare_parameter('Q_rel', [0.0, 0.0, 0.01])
        self.declare_parameter('V_rel', [0.1, 0.1, 0.1])
        self.declare_parameter('predict_hz', 50.0)
        self.declare_parameter('update_hz', 10.0)
        self.declare_parameter('seed', 42)

        # Time parameters for communication outage simulation
        self.declare_parameter('outage_start_time', 60.0)  # Start communication outage at 20s
        self.declare_parameter('outage_duration', 60.0)    # Outage lasts 20s (until 40s)

        # Get filter parameters
        self.P_ego_list = self.get_parameter('P_ego').value
        self.Q_ego_list = self.get_parameter('Q_ego').value
        self.V_ego_list = self.get_parameter('V_ego').value
        self.P_rel_list = self.get_parameter('P_rel').value
        self.Q_rel_list = self.get_parameter('Q_rel').value
        self.V_rel_list = self.get_parameter('V_rel').value
        self.predict_hz = self.get_parameter('predict_hz').value
        self.update_hz  = self.get_parameter('update_hz').value
        self.outage_start_time = self.get_parameter('outage_start_time').value
        self.outage_duration = self.get_parameter('outage_duration').value

        # Set random seed
        seed = self.get_parameter('seed').value
        np.random.seed(seed)

        # Reboot client
        self.reboot_client = self.create_client(Empty, self.robot + '/reboot')

        # Flags and variables
        self.timer_period = 1.0 / self.predict_hz
        self.initial_phase = 0.0
        self.initial_radius = 1.0
        self.initial_pose = np.zeros(3)
        self.order = []

        self.has_initial_pose = False
        self.has_final = False
        self.land_flag = False
        self.has_order = False
        self.has_phase_leader = False
        self.has_phase_follower = False

        self.final_pose   = np.zeros(3)
        self.current_pose = np.zeros(3)
        self.previous_pose = np.zeros(3)
        self.previous_pose_time = 0.0
        
        self.leader   = None
        self.follower = None     

        self.i_landing = 0
        self.i_takeoff = 0

        self.phases = np.zeros(3)  # [leader, ego, follower]
        self.encirclement_start_time = None  # Track when encirclement started
        self.in_communication_outage = False  # Track outage state
        self.outage_started = False  # Flag to detect outage transition
        self.outage_ended = False    # Flag to detect outage ending transition
        self.relative_color = '0x480091'
        self.gps_color = '0x0AAC00'

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

        # Subscription to Vicon positions of the robot
        self.create_subscription(
            PoseStamped, f'/{self.robot}/vicon_position',
            self._poses_changed,
            10
        )

        # Subscription to agents order
        self.create_subscription(
            StringArray, '/agents_order',
            self._order_callback,
            10)    
        self.publish_estimated_omega_x = self.create_publisher(Float32, f'/{self.robot}/filtered/measured/omega/x', 10)
        self.publish_estimated_omega_y = self.create_publisher(Float32, f'/{self.robot}/filtered/measured/omega/y', 10)
        self.publish_estimated_omega_z = self.create_publisher(Float32, f'/{self.robot}/filtered/measured/omega/z', 10)
        # Wait until order is received
        while (not self.has_order):
            rclpy.spin_once(self, timeout_sec=0.1)

        # Wait until initial pose is received from Vicon
        while (not self.has_initial_pose):
            rclpy.spin_once(self, timeout_sec=0.1)
        
        self.info(f"Initial pose received: phase={self.initial_phase:.3f}, radius={self.initial_radius:.3f}")

        # Create GPS filter instance
        self.filter_gps_params = {
            'P': self.P_ego_list,
            'Q': self.Q_ego_list,
            'V': self.V_ego_list,
            'radius_nominal': self.radius_nominal,
            'radius_guess': self.initial_radius + np.random.normal(0, 0.15),
            'phase_guess': self.initial_phase + np.random.normal(0, 0.1),
            'frame_id': self.frame_id
        }
        self.filter_gps = FilterGPS(self.robot, self.embedding_fn_name, self.filter_gps_params, self)

        # Subscribe to leader and follower vicon positions for relative filter initialization
        self.has_leader_initial_pose = False
        self.has_follower_initial_pose = False
        self.leader_initial_phase = 0.0
        self.follower_initial_phase = 0.0
        self.leader_initial_radius = 0.0
        self.follower_initial_radius = 0.0
        
        self.create_subscription(
            PoseStamped, f'/{self.leader}/vicon_position',
            self._leader_vicon_callback,
            10
        )
        
        self.create_subscription(
            PoseStamped, f'/{self.follower}/vicon_position',
            self._follower_vicon_callback,
            10
        )
        # Wait until initial poses are received from all drones
        while (not self.has_leader_initial_pose or not self.has_follower_initial_pose):
            rclpy.spin_once(self, timeout_sec=0.1)

        # Create relative filter instances
        self.filter_relative_leader_params = {
            'P': self.P_rel_list,
            'Q': self.Q_rel_list,
            'V': self.V_rel_list,
            'radius_nominal': self.radius_nominal,
            'radius_guess': self.leader_initial_radius + np.random.normal(0, 0.15),
            'phase_guess': self.leader_initial_phase + np.random.normal(0, 0.1),
            'frame_id': self.frame_id
        }
        self.filter_relative_leader = FilterRelative(
            self.robot, self.embedding_fn_name, self.filter_relative_leader_params, self
        )

        self.filter_relative_follower_params = {
            'P': self.P_rel_list,
            'Q': self.Q_rel_list,
            'V': self.V_rel_list,
            'radius_nominal': self.radius_nominal,
            'radius_guess': self.follower_initial_radius + np.random.normal(0, 0.15),
            'phase_guess': self.follower_initial_phase + np.random.normal(0, 0.1),
            'frame_id': self.frame_id
        }
        self.filter_relative_follower = FilterRelative(
            self.robot, self.embedding_fn_name, self.filter_relative_follower_params, self
        )
        
        # Initialize phases with initial values
        self.phases = np.array([self.leader_initial_phase, self.initial_phase, self.follower_initial_phase])

        # Create subscriber for GPS measurements
        self.create_subscription(
            PoseStamped,
            f'/{self.robot}/gps_position',
            self.update_callback_gps,
            10
        )

        # Create subscriber for all drones' positions (for relative filter updates)
        self.create_subscription(
            NamedPoseArray,
            f'/{self.robot}/gps_scanner_position',
            self.update_callback_relative,
            10
        )

        # Crazyflie position command publisher
        self.position_pub = self.create_publisher(Position, f'/{self.robot}/cmd_position', 10)

        # Publishers for phase differences and diagnostics
        self.publish_omega = self.create_publisher(Float32, f'/{self.robot}/filtered/omega', 10)
        self.publish_gain  = self.create_publisher(Float32, f'/{self.robot}/filtered/controller_gain', 10)
        self.publish_phase_diff_leader = self.create_publisher(Float32, f'/{self.robot}/filtered/phase_diff/leader', 10)
        self.publish_phase_diff_follower = self.create_publisher(Float32, f'/{self.robot}/filtered/phase_diff/follower', 10)
        self.color_pub = self.create_publisher(String,'/'+ self.robot + '/color_led', 10)

        # Metadata publisher
        self.metadata_pub = self.create_publisher(Metadata, f'/{self.robot}/metadata', 10)
        self.metadata_timer = self.create_timer(10.0, self.publish_metadata)
        
        # Create subscribers for leader/follower phases (will use only when not in outage)
        self.create_subscription(Float32, f'/{self.leader}/filtered/phase',   self._phase_callback_leader, 1)
        self.create_subscription(Float32, f'/{self.follower}/filtered/phase', self._phase_callback_follower, 1)
        
        # Arming all drones
        self.arm_client = self.create_client(Arm, self.robot + '/arm')
        # Wait until the service is available
        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.arm()
        time.sleep(2)

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):
        ''' Timer callback to send position commands to the crazyflie based on the current state. '''
        try:
            # Take-off state
            if self.state == 0:
                if self.has_initial_pose:
                    self.phases[1] = self.initial_phase
                    self.takeoff()
                    self.color_pub.publish(String(data=self.gps_color))
                    self.filter_gps.pub_phase.publish(Float32(data=self.phases[1]))
                    self.publish_phase_differences()

            # Hover state
            elif self.state == 1:
                self.hover()
                self.filter_gps.pub_phase.publish(Float32(data=self.phases[1]))
                self.publish_phase_differences()
            
            # Encirclement state
            elif self.state == 2:
                # Track encirclement time and determine if in communication outage
                if self.encirclement_start_time is None:
                    self.encirclement_start_time = time.time()
                
                time_since_encirclement_start = time.time() - self.encirclement_start_time
                outage_end_time = self.outage_start_time + self.outage_duration
                was_in_outage = self.in_communication_outage
                self.in_communication_outage = (self.outage_start_time <= time_since_encirclement_start <= outage_end_time)
                
                # Handle transition INTO outage (GPS -> Relative)
                if self.in_communication_outage and not was_in_outage and not self.outage_started:
                    self.info("=== COMMUNICATION OUTAGE STARTED ===")
                    self.info("Synchronizing relative filters with recent states")
                    self._sync_relative_filters()
                    self.color_pub.publish(String(data=self.relative_color))
                    self.outage_started = True
                    self.outage_ended = False
                
                # Handle transition OUT of outage (Relative -> GPS)
                if not self.in_communication_outage and was_in_outage and not self.outage_ended:
                    self.info("=== COMMUNICATION OUTAGE ENDED ===")
                    self.color_pub.publish(String(data=self.gps_color))
                    self.outage_ended = True
                    self.outage_started = False
                
                # Use appropriate phases based on outage status
                if not self.in_communication_outage:
                    # Normal operation: use leader/follower phases if available
                    if not self.has_phase_leader or not self.has_phase_follower:
                        self.info("Waiting for leader/follower phase messages...")
                        return
                
                # Compute desired omega based on current phases
                omega, gain = phase_controller(self.phases[1], self.phases[0], self.phases[2], self.omega_nominal, self.k_phi)
                omega_msg = Float32()
                omega_msg.data = omega
                self.publish_omega.publish(omega_msg)

                gain_msg = Float32()
                gain_msg.data = gain
                self.publish_gain.publish(gain_msg)

                # Propagate GPS filter (prediction step)
                phase, current_position, desired_position = self.filter_gps.predict(omega, self.timer_period)
                self.phases[1] = phase.data

                if self.in_communication_outage:
                    phase_leader = self.filter_relative_leader.predict(self.omega_nominal, self.timer_period)
                    phase_follower = self.filter_relative_follower.predict(self.omega_nominal, self.timer_period)
                    self.phases = [phase_leader.data, phase.data, phase_follower.data]

                self.publish_phase_differences()

                # Send position command
                target_r = np.array([desired_position.pose.position.x,
                                     desired_position.pose.position.y,
                                     desired_position.pose.position.z + self.hover_height])
                self.send_position(target_r)
            
            # Landing state
            elif self.state == 3:
                if self.has_final:
                    self.landing()
                    if self.i_landing < len(self.t_landing)-1:
                        self.i_landing += 1
                    else:
                        self.reboot()
                        self.info('Exiting circle node')  
                        self.destroy_node()
                        rclpy.shutdown()   

        except KeyboardInterrupt:
            self.info('Exiting circle node')
    
    def update_callback_gps(self, gps_pose: PoseStamped):
        ''' Callback to update GPS filter with measurements. '''
        if self.state != 2 or self.in_communication_outage:
            return  # Only update during normal encirclement (not in outage)
        
        y = np.array([gps_pose.pose.position.x,
                      gps_pose.pose.position.y,
                      gps_pose.pose.position.z]).reshape((3, 1))
        phase, current_position, desired_position = self.filter_gps.update(y)

    def update_callback_relative(self, gps_scanner_poses: NamedPoseArray):
        ''' Callback to update relative filters with measurements. '''
        if self.state != 2 or not self.in_communication_outage:
            return  # Only use relative filters during communication outage
        
        y_ego_gps = None
        y_rel_leader = None
        y_rel_follower = None

        # Extract measurements for leader and follower from scanner
        for pose in gps_scanner_poses.poses:
            if pose.name == self.robot:
                y_ego_gps = np.array([pose.pose.position.x,
                                  pose.pose.position.y,
                                  pose.pose.position.z]).reshape((3, 1))
            elif pose.name == self.leader:
                y_rel_leader = np.array([pose.pose.position.x,
                                        pose.pose.position.y,
                                        pose.pose.position.z]).reshape((3, 1))
            elif pose.name == self.follower:
                y_rel_follower = np.array([pose.pose.position.x,
                                          pose.pose.position.y,
                                          pose.pose.position.z]).reshape((3, 1))
        
        # Updating GPS filter
        phase_ego, current_position, desired_position = self.filter_gps.update(y_ego_gps)
        
        # Get current ego position from GPS filter
        qi = np.asarray([current_position.pose.pose.position.x,
                         current_position.pose.pose.position.y,
                         current_position.pose.pose.position.z]).reshape((3, 1))
        Rei = build_Re(self.filter_gps.embedding_fn, self.phases[1])
        Rci = self.filter_gps.Rc

        # Update relative filter for leader
        rel_noise = np.random.multivariate_normal(np.zeros(3), np.diag(np.square(self.V_rel_list))).reshape((3, 1))
        y_leader = (Rci.T @ Rei.T @ y_rel_leader) + rel_noise
        phase_leader = self.filter_relative_leader.update(y_leader, Rei, Rci, qi)
        self.phases[0] = phase_leader.data

        # Update relative filter for follower
        rel_noise = np.random.multivariate_normal(np.zeros(3), np.diag(np.square(self.V_rel_list))).reshape((3, 1))
        y_follower = (Rci.T @ Rei.T @ y_rel_follower) + rel_noise
        phase_follower = self.filter_relative_follower.update(y_follower, Rei, Rci, qi)
        self.phases[2] = phase_follower.data

    def _sync_relative_filters(self):
        """
        Synchronize relative filters with recent phases when entering communication outage.
        """
        try:
            P_ego = self.filter_gps.P[:3,:3]
            # Copy GPS filter's rotation matrix (Rc) to relative filters
            self.filter_relative_leader.Rc   = build_Rc(wrap_to_2pi(self.phases[0]))
            self.filter_relative_leader.P    = P_ego.copy()
            self.filter_relative_follower.Rc = build_Rc(wrap_to_2pi(self.phases[2]))
            self.filter_relative_follower.P  = P_ego.copy()
        except Exception as e:
            self.info(f"Error synchronizing relative filters: {e}")

    def _poses_changed(self, robot_pose: PoseStamped):
        """ Callback for robot's Vicon position. """
        # Initialize the initial pose and phase if not already set
        if not self.has_initial_pose:      
            self.initial_pose[0] = robot_pose.pose.position.x
            self.initial_pose[1] = robot_pose.pose.position.y
            self.initial_pose[2] = robot_pose.pose.position.z   
            self.initial_phase = wrap_to_2pi(np.arctan2(self.initial_pose[1], self.initial_pose[0]))   
            self.initial_radius = np.sqrt(self.initial_pose[0]**2 + self.initial_pose[1]**2)
            self.previous_pose = self.initial_pose.copy()
            self.previous_pose_time = robot_pose.header.stamp.sec + robot_pose.header.stamp.nanosec * 1e-9

            self.takeoff_traj(4)
            self.has_initial_pose = True    
            
        # Update current pose if not landing
        elif not self.land_flag:
            self.current_pose[0] = robot_pose.pose.position.x
            self.current_pose[1] = robot_pose.pose.position.y
            self.current_pose[2] = robot_pose.pose.position.z

            # Estimate 3D omega using delta pose and velocity cross product
            delta_pose = self.current_pose - self.previous_pose
            delta_velocity = delta_pose / ((robot_pose.header.stamp.sec + robot_pose.header.stamp.nanosec * 1e-9) - self.previous_pose_time)
            omega_3D = np.cross(self.previous_pose, delta_velocity) / (np.linalg.norm(self.previous_pose)**2 + 1e-6)
            self.previous_pose = self.current_pose.copy()
            self.previous_pose_time = robot_pose.header.stamp.sec + robot_pose.header.stamp.nanosec * 1e-9
            self.publish_estimated_omega_x.publish(Float32(data=omega_3D[0]))
            self.publish_estimated_omega_y.publish(Float32(data=omega_3D[1]))
            self.publish_estimated_omega_z.publish(Float32(data=omega_3D[2]))

        # Set final pose when landing is commanded
        elif (self.has_final == False) and (self.land_flag == True):
            self.final_pose = np.zeros(3)
            self.info("Landing...")
            self.final_pose[0] = robot_pose.pose.position.x
            self.final_pose[1] = robot_pose.pose.position.y
            self.final_pose[2] = robot_pose.pose.position.z
            self.landing_traj(3)
            self.has_final = True

    def _leader_vicon_callback(self, msg: PoseStamped):
        ''' Callback to receive leader's initial Vicon position. '''
        if not self.has_leader_initial_pose:
            leader_pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            self.leader_initial_phase = wrap_to_2pi(np.arctan2(leader_pose[1], leader_pose[0]))
            self.leader_initial_radius = np.sqrt(leader_pose[0]**2 + leader_pose[1]**2)
            self.has_leader_initial_pose = True
            self.info(f"Leader initial phase: {self.leader_initial_phase:.3f}, radius: {self.leader_initial_radius:.3f}")

    def _follower_vicon_callback(self, msg: PoseStamped):
        ''' Callback to receive follower's initial Vicon position. '''
        if not self.has_follower_initial_pose:
            follower_pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            self.follower_initial_phase = wrap_to_2pi(np.arctan2(follower_pose[1], follower_pose[0]))
            self.follower_initial_radius = np.sqrt(follower_pose[0]**2 + follower_pose[1]**2)
            self.has_follower_initial_pose = True
            self.info(f"Follower initial phase: {self.follower_initial_phase:.3f}, radius: {self.follower_initial_radius:.3f}")

    def _phase_callback_leader(self, msg: Float32):
        ''' Callback to receive filtered phase of the leader agent. '''
        self.has_phase_leader = True
        self.phases[0] = msg.data

    def _phase_callback_follower(self, msg: Float32):
        ''' Callback to receive filtered phase of the follower agent. '''
        self.has_phase_follower = True
        self.phases[2] = msg.data

    def _order_callback(self, msg: StringArray):
        ''' Callback to receive the order of agents. '''
        if not self.has_order:
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

    def publish_phase_differences(self):
        ''' Publish phase differences to leader and follower. '''
        diff_leader = wrap_to_pi(self.phases[0] - self.phases[1])
        diff_follower = wrap_to_pi(self.phases[1] - self.phases[2])
        self.publish_phase_diff_leader.publish(Float32(data=diff_leader))
        self.publish_phase_diff_follower.publish(Float32(data=diff_follower))

    def takeoff(self):
        ''' Take-off procedure to reach hover height. '''
        self.send_position(self.r_takeoff[:, self.i_takeoff])
        if self.i_takeoff < len(self.t_takeoff)-1:
            self.i_takeoff += 1
        else:
            self.state = 1

    def takeoff_traj(self, t_max: float):
        ''' Take-off trajectory generation. '''
        self.t_takeoff = np.arange(0, t_max, self.timer_period)
        self.r_takeoff = np.zeros((3, len(self.t_takeoff))) 
        self.r_takeoff[0,:] += self.initial_pose[0] * np.ones(len(self.t_takeoff))
        self.r_takeoff[1,:] += self.initial_pose[1] * np.ones(len(self.t_takeoff))
        self.r_takeoff[2,:] = self.hover_height * (self.t_takeoff / t_max)

    def landing_traj(self, t_max: float):
        ''' Landing trajectory generation. '''
        self.t_landing = np.arange(t_max, 0.1, -self.timer_period)
        self.i_landing = 0
        self.r_landing = np.zeros((3, len(self.t_landing)))
        self.r_landing[0,:] += self.final_pose[0] * np.ones(len(self.t_landing))
        self.r_landing[1,:] += self.final_pose[1] * np.ones(len(self.t_landing))
        self.r_landing[2,:] = self.final_pose[2] * (self.t_landing / t_max)

    def publish_metadata(self):
        """Publish experiment metadata every 10 seconds."""
        metadata = Metadata()
        metadata.group_name = 'combined'
        metadata.drone_id = self.robot
        metadata.seed = int(self.get_parameter('seed').value)
        metadata.filter_type = 'FilterCombined'
        metadata.embedding_fn_name = self.embedding_fn_name
        metadata.frame_id = self.frame_id
        metadata.radius_nominal = self.radius_nominal
        metadata.omega_nominal = self.omega_nominal
        metadata.hover_height = self.hover_height
        metadata.k_phi = self.k_phi
        metadata.p_ego = [float(x) for x in self.get_parameter('P_ego').value]
        metadata.q_ego = [float(x) for x in self.get_parameter('Q_ego').value]
        metadata.v_ego = [float(x) for x in self.get_parameter('V_ego').value]
        metadata.p_rel = [float(x) for x in self.get_parameter('P_rel').value]
        metadata.q_rel = [float(x) for x in self.get_parameter('Q_rel').value]
        metadata.v_rel = [float(x) for x in self.get_parameter('V_rel').value]
        metadata.predict_hz = self.predict_hz
        metadata.update_hz = self.update_hz
        metadata.phase_guess = self.initial_phase
        metadata.radius_guess = self.initial_radius
        metadata.stamp = self.get_clock().now().to_msg()
        self.metadata_pub.publish(metadata)

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
            
    def _landing_callback(self, msg):
        ''' Callback to initiate landing procedure. '''
        self.land_flag = msg.data
        self.state = 3

    def _encircle_callback(self, msg):
        ''' Callback to initiate encirclement procedure. '''
        self.state = 2
        self.encirclement_start_time = time.time()

    def hover(self):
        ''' Hovering procedure at hover height. '''
        msg = Position()
        msg.x = self.initial_pose[0]
        msg.y = self.initial_pose[1]
        msg.z = self.hover_height
        self.position_pub.publish(msg)

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


def main():
    rclpy.init()
    encirclement = CircleDistortion()
    rclpy.spin(encirclement)
    encirclement.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
