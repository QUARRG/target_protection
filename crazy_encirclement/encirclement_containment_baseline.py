import rclpy
import time
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from crazyflie_interfaces.msg import StringArray, Position, VelocityWorld
from std_msgs.msg import Bool
from std_srvs.srv import Empty
from std_msgs.msg import Float32
from crazy_encirclement.filters import BaselineFilter, wrap_to_2pi, wrap_to_pi
from crazy_encirclement.bearing_formation_control import bearing_based_formation_control
from crazy_encirclement_interfaces.msg import Metadata
from rclpy.qos import QoSPresetProfiles
from motion_capture_tracking_interfaces.msg import NamedPoseArray
from scipy.spatial.transform import Rotation as R

class Encirclement_Containment(Node):
    def __init__(self):
        """
            Node that sends the crazyflie to a desired position
            The desired position comes from the distortion of a circle
        """
        super().__init__('circle_distortion')
        self.info = self.get_logger().info
        self.info('Circle distortion node has been started.')

        # Parameters
        self.declare_parameter('robot', 'C20')
        self.declare_parameter('evader', 'C25')
        self.declare_parameter('number_of_agents', 4)
        self.declare_parameter('initial_phase', '0.0')
        self.declare_parameter('radius_nominal', 1.0)
        self.declare_parameter('omega_nominal', 0.8)
        self.declare_parameter('k_phi', 8.0)
        self.declare_parameter('embedding_fn_name', 'modelB')
        self.declare_parameter('hover_height', 0.9)
        self.declare_parameter('frame_id', 'world')

        self.robot    = str(self.get_parameter('robot').value)
        self.evader    = str(self.get_parameter('evader').value)
        self.n_agents = int(self.get_parameter('number_of_agents').value)
        self.k_phi    = float(self.get_parameter('k_phi').value)
        self.radius_nominal = float(self.get_parameter('radius_nominal').value)
        self.omega_nominal  = float(self.get_parameter('omega_nominal').value)
        self.embedding_fn_name = str(self.get_parameter('embedding_fn_name').value)
        self.hover_height = float(self.get_parameter('hover_height').value)
        self.frame_id = str(self.get_parameter('frame_id').value)

        # Desired phase difference
        self.desired_phase_diff = 2.0 * np.pi / self.n_agents
        self.initial_radius = self.radius_nominal
        self.swarm_poses = np.array((3,self.n_agents-1))
        self.evader_pos = np.array(3)
        self.R_dw = None

        # Filter parameters
        self.declare_parameter('predict_hz', 50.0)
        self.declare_parameter('update_hz', 10.0)
        self.predict_hz = self.get_parameter('predict_hz').value
        self.update_hz  = self.get_parameter('update_hz').value

        # Reboot client
        self.reboot_client = self.create_client(Empty, self.robot + '/reboot')

        # Flags and variables
        self.timer_period = 1.0 / self.predict_hz
        self.initial_phase = 0.0
        self.initial_pose = np.zeros(3)
        self.order = []

        self.has_initial_pose = False
        self.has_final = False
        self.land_flag = False
        self.has_order = False
        self.has_phase_follower = False
        self.has_phase_leader = False

        self.final_pose   = np.zeros(3)
        self.current_pose = np.zeros(3)
        self.previous_pose = np.zeros(3)
        self.previous_pose_time = 0.0
        self.initial_pose = np.zeros(3)
        self.distances = None
        self.order = None
        
        self.leader   = None
        self.follower = None     

        self.i_landing = 0
        self.i_takeoff = 0

        self.phases = np.zeros(self.n_agents)

        self.state = 0
        # 0-take-off, 1-hover, 2-encirclement, 3-navigating to pursuer, 4-landing

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

        #subscribing to bearing measurements
        qos_profile = QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(
            NamedPoseArray, f'{self.robot}/distances',
            self._bearing_callback,
            qos_profile
        )

        # Subscription to Vicon positions of the robot that are coming from the gps node
        self.create_subscription(
            PoseStamped, f'/{self.robot}/pose',
            self._poses_changed,
            10
        )

        # Subscription to agents order
        self.create_subscription(
            StringArray, '/agents_order',
            self._order_callback,
            10)    
        
        # Subscription to evader detection
        self.create_subscription(
            Bool, '/evader detection',
            self._evader_detection_callback,
            10) 
        self.publish_estimated_omega_x   = self.create_publisher(Float32, f'/{self.robot}/baseline/measured/omega/x', 10)
        self.publish_estimated_omega_y   = self.create_publisher(Float32, f'/{self.robot}/baseline/measured/omega/y', 10)
        self.publish_estimated_omega_z   = self.create_publisher(Float32, f'/{self.robot}/baseline/measured/omega/z', 10)

        # Wait until order is received
        while (not self.order):
            rclpy.spin_once(self, timeout_sec=0.1)

        # Create filter instance
        self.params = {
            'k_phi': self.k_phi,
            'embedding_fn_name': self.embedding_fn_name,
            'omega_nominal': self.omega_nominal,
            'radius_nominal': self.radius_nominal,
            'radius_guess': self.initial_radius,
            'phase_guess': self.initial_phase,
            'frame_id': self.frame_id,
            'dt': self.timer_period
        }
        self.baseline = BaselineFilter(self.robot, self.embedding_fn_name, self.params, self)

        # Create subscribers for the other agents' filtered phases
        self.create_subscription(Float32, f'/{self.leader}/baseline/phase',   self._phase_callback_leader, 1)
        self.create_subscription(Float32, f'/{self.follower}/baseline/phase', self._phase_callback_follower, 1)

        # Crazyflie position command publisher
        self.position_pub = self.create_publisher(Position, f'/{self.robot}/cmd_position', 10)
        # Crazyflie velocidade command publisher
        self.velocity_pub = self.create_publisher(VelocityWorld, f'/{self.robot}/cmd_velocity_world', 10)

        # Publishers for phase differences
        self.publish_phase_diff_leader   = self.create_publisher(Float32, f'/{self.robot}/baseline/phase_diff/leader', 10)
        self.publish_phase_diff_follower = self.create_publisher(Float32, f'/{self.robot}/baseline/phase_diff/follower', 10)

        # Metadata publisher
        self.metadata_pub = self.create_publisher(Metadata, f'/{self.robot}/metadata', 10)
        self.metadata_timer = self.create_timer(10.0, self.publish_metadata)
        
        # input("Press Enter to takeoff")
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):
        ''' Timer callback to send position commands to the crazyflie based on the current state. '''
        try:
            # Take-off state
            if self.state == 0:
                if self.has_initial_pose:
                    self.phases[1] = self.initial_phase
                    self.takeoff()
                    self.baseline.pub_phase.publish(Float32(data=self.phases[1]))
                    self.publish_phase_differences()

            # Hover state
            elif self.state == 1:
                self.hover()
                self.baseline.pub_phase.publish(Float32(data=self.phases[1]))
                self.publish_phase_differences()
            
            # Encirclement state
            elif self.state == 2: 
                if self.has_phase_follower and self.has_phase_leader:                  
                    # Propagating the system
                    current_pose = self.current_pose.copy()
                    current_pose[2] -= self.hover_height  # Remove height offset
                    phase, position = self.baseline.predict(current_pose, self.phases)
                    # Updating internal parameters
                    self.phases[1] = phase.data
                    self.publish_phase_differences()

                    # Sending position command
                    target_r = np.array([position.pose.position.x,
                                         position.pose.position.y,
                                         position.pose.position.z + self.hover_height])
                    self.send_position(target_r)

                else:
                    self.state = 1  # Return to hover if phases are not available
                    self.info("Lost phase information, returning to hover.")
            
            elif self.state == 3:
                v = bearing_based_formation_control(self.swarm_poses, self.current_pose, self.evader_pos, 1 ,self.radius_nominal)
                vel_world = VelocityWorld()
                v = self.R_dw*v #transforming velocity in drone frame to world frame
                vel_world.vel.x = v[0]
                vel_world.vel.y = v[1]
                vel_world.vel.z = 0

                if np.linalg.norm(self.current_pose[0:2]-self.evader_pos[0:2])< self.radius_nominal:
                    self.land_flag == True
                    vel_world = VelocityWorld()
                    self.velocity_pub.publish(vel_world)
            # Landing state
            elif self.state == 4:
                self.landing()
                if self.i_landing < len(self.t_landing)-1:
                    self.i_landing += 1
                else:
                    self.reboot()
                    self.info('Exiting circle node')  
                    self.destroy_node()
                    rclpy.shutdown()  

        except KeyboardInterrupt:
            self.info('Exiting open loop command node')

    def _bearing_callback(self, msg: NamedPoseArray):
        for pose in msg.poses:
            if pose.name in self.order and pose.name != self.robot:
                i = self.order.index(pose.name)
                self.swarm_poses[0,i] = pose.pose.position.x
                self.swarm_poses[1,i] = pose.pose.position.y
                self.swarm_poses[2,i] = pose.pose.position.z
            elif pose.name == self.evader:
                self.evader_pos[0] = pose.pose.position.x
                self.evader_pos[1] = pose.pose.position.y 
                self.evader_pos[2] = pose.pose.position.z  
            

    def _poses_changed(self, robot_pose: PoseStamped):
        """ Topic update callback to the motion capture lib's
            poses topic to send through the external position
            to the crazyflie. All steps based on the Vicon position.
        """
        self.R_dw = R.from_quat([robot_pose.pose.orientation.x, robot_pose.pose.orientation.y, robot_pose.pose.orientation.z, robot_pose.pose.orientation.w])
        # Initialize the initial pose and phase if not already set using vicon data
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

            # Estimate 3D omega using the delta pose and velocity cross product
            delta_pose = self.current_pose - self.previous_pose
            delta_velocity = delta_pose / ( (robot_pose.header.stamp.sec + robot_pose.header.stamp.nanosec * 1e-9) - self.previous_pose_time )
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
            self.state = 4

    def _phase_callback_leader(self, msg: Float32):
        ''' Callback to receive the filtered phase of the leader agent. '''
        self.has_phase_leader = True
        if msg.data:
            self.phases[0] = msg.data

    def _phase_callback_follower(self, msg: Float32):
        ''' Callback to receive the filtered phase of the follower agent. '''
        self.has_phase_follower = True
        if msg.data:
            self.phases[2] = msg.data

    def _order_callback(self, msg: StringArray):
        ''' Callback to receive the order of agents. '''
        # self.info(f"Phase received: {msg.data}")
        self.order = msg.data
        for robot in self.order:
            if robot == self.robot:
                i = self.order.index(robot)
                if i == 0:
                    self.leader = self.order[self.n_agents-1]
                    self.follower = self.order[i+1]
                elif i == (self.n_agents-1):
                    self.leader = self.order[i-1]
                    self.follower = self.order[0]
                else:
                    self.leader = self.order[i-1]
                    self.follower = self.order[i+1]
            # self.info(f"Leader: {self.leader}, Follower: {self.follower}")

    def _evader_detection_callback(self, msg: Bool):
        if msg.data == True:
            self.state == 3
        else:
            self.state == 1 #hover
            
    def publish_phase_differences(self):
        ''' Publish phase differences to leader and follower. '''
        diff_leader = wrap_to_pi(self.phases[0] - self.phases[1])
        diff_follower = wrap_to_pi(self.phases[1] - self.phases[2])
        self.publish_phase_diff_leader.publish(Float32(data=diff_leader))
        self.publish_phase_diff_follower.publish(Float32(data=diff_follower))

    def takeoff(self):
        ''' Take-off procedure to reach the hover height. '''
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
        metadata.group_name = 'baseline'
        metadata.drone_id = self.robot
        metadata.seed = 0  # Baseline doesn't use seed
        metadata.filter_type = 'Baseline'
        metadata.embedding_fn_name = self.embedding_fn_name
        metadata.frame_id = self.frame_id
        metadata.radius_nominal = self.radius_nominal
        metadata.omega_nominal = self.omega_nominal
        metadata.hover_height = self.hover_height
        metadata.k_phi = self.k_phi
        metadata.p_ego = []
        metadata.q_ego = []
        metadata.v_ego = []
        metadata.p_rel = []
        metadata.q_rel = []
        metadata.v_rel = []
        metadata.predict_hz = self.predict_hz
        metadata.update_hz = self.update_hz
        metadata.phase_guess = self.initial_phase
        metadata.radius_guess = self.initial_radius
        metadata.stamp = self.get_clock().now().to_msg()
        self.metadata_pub.publish(metadata)

    def _landing_callback(self, msg):
        ''' Callback to initiate landing procedure. '''
        self.land_flag = msg.data

    def _encircle_callback(self, msg):
        ''' Callback to initiate encirclement procedure. '''
        self.state = 2

    def hover(self):
        ''' Hovering procedure at the hover height. '''
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