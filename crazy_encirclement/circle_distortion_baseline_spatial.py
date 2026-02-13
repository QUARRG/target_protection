#!/usr/bin/env python3
"""
Circle distortion baseline experiment with spatial filter.
This node implements a baseline controller using the SpatialBaselineFilter
for 3D circle formation with predefined adjoint rotation.
"""
import time
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from std_srvs.srv import Empty
from geometry_msgs.msg import PoseStamped
from crazyflie_interfaces.msg import Position, StringArray
from crazy_encirclement.filters import FilterSpatialBaseline
from crazy_encirclement.filters import wrap_to_pi, wrap_to_2pi
from crazy_encirclement_interfaces.msg import Metadata
from crazyflie_py import Crazyswarm


class CircleDistortionSpatialBaseline(Node):
    def __init__(self, swarm=None):
        super().__init__("circle_distortion_baseline_spatial")
        
        # Declare and get parameters
        self.declare_parameter('robot', 'C04')
        self.declare_parameter('n_agents', 3)
        self.declare_parameter('k_phi_z', 1.0)
        self.declare_parameter('radius_nominal', 1.0)
        self.declare_parameter('omega_z_nominal', 0.5)
        self.declare_parameter('adjoint_angle', 0.0)    # Predefined adjoint angle in radians
        self.declare_parameter('omega_y_nominal', 0.0)  # Spatial angular velocity
        self.declare_parameter('k_phi_y', 0.0)          # Spatial gain
        self.declare_parameter('hover_height', 1.0)
        self.declare_parameter('predict_hz', 100.0)
        self.declare_parameter('update_hz', 100.0)
        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('adjoint_direction', 0.0)  # Direction of adjoint rotation: 1.0 for positive, -1.0 for negative

        # Get parameters
        self.robot           = self.get_parameter('robot').get_parameter_value().string_value
        self.n_agents        = self.get_parameter('n_agents').get_parameter_value().integer_value
        self.k_phi_z         = self.get_parameter('k_phi_z').get_parameter_value().double_value
        self.radius_nominal  = self.get_parameter('radius_nominal').get_parameter_value().double_value
        self.omega_z_nominal = self.get_parameter('omega_z_nominal').get_parameter_value().double_value
        self.adjoint_angle   = self.get_parameter('adjoint_angle').get_parameter_value().double_value
        self.omega_y_nominal = self.get_parameter('omega_y_nominal').get_parameter_value().double_value
        self.k_phi_y         = self.get_parameter('k_phi_y').get_parameter_value().double_value
        self.hover_height    = self.get_parameter('hover_height').get_parameter_value().double_value
        self.predict_hz      = self.get_parameter('predict_hz').get_parameter_value().double_value
        self.update_hz       = self.get_parameter('update_hz').get_parameter_value().double_value
        self.frame_id        = self.get_parameter('frame_id').get_parameter_value().string_value
        self.adjoint_direction = self.get_parameter('adjoint_direction').get_parameter_value().double_value
        
        # State variables
        self.state = 0  # 0: takeoff, 1: hover, 2: encirclement, 3: landing
        self.has_initial_pose = False
        self.has_final = False
        self.land_flag = False
        self.timer_period = 1.0 / self.predict_hz  # seconds
        
        # Initial pose and phase
        self.initial_pose = np.zeros(3)
        self.current_pose = np.zeros(3)
        self.previous_pose = np.zeros(3)
        self.previous_pose_time = 0.0
        self.final_pose = np.zeros(3)
        self.initial_phase = 0.0
        self.initial_radius = 0.0
        
        # Takeoff and landing trajectories
        self.t_takeoff = []
        self.r_takeoff = []
        self.i_takeoff = 0
        self.t_landing = []
        self.r_landing = []
        self.i_landing = 0

        # Phase tracking for leader and follower
        self.has_phase_leader = False
        self.has_phase_follower = False
        self.has_order = False
        self.leader = None
        self.follower = None
        self.phases = np.zeros(3)  # [leader, ego, follower]

        # Publishers for estimated 3D omega
        self.publish_estimated_omega_x = self.create_publisher(Float32, f'/{self.robot}/spatial_baseline/measured/omega/x', 10)
        self.publish_estimated_omega_y = self.create_publisher(Float32, f'/{self.robot}/spatial_baseline/measured/omega/y', 10)
        self.publish_estimated_omega_z = self.create_publisher(Float32, f'/{self.robot}/spatial_baseline/measured/omega/z', 10)
        
        # Subscribers
        self.create_subscription(
            PoseStamped,
            f'/{self.robot}/vicon_position',
            self._poses_changed,
            10)
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
        
        # Subscribe to command center updates
        self.adjoint_angle_sub = self.create_subscription(
            Float32, '/command_center/adjoint_angle', self._adjoint_angle_callback, 10)
        self.radius_nominal_sub = self.create_subscription(
            Float32, '/command_center/radius_nominal', self._radius_nominal_callback, 10)
        
        # Wait until order is received
        while (not self.has_order):
            rclpy.spin_once(self, timeout_sec=0.1)

        self.info(f"Drone {self.robot} received agent order. Leader: {self.leader}, Follower: {self.follower}")
        
        # Initialize filter with parameters
        params = {
            'k_phi_z': self.k_phi_z,
            'radius_nominal': self.radius_nominal,
            'omega_z_nominal': self.omega_z_nominal,
            'adjoint_angle': self.adjoint_angle,
            'omega_y_nominal': self.omega_y_nominal,
            'k_phi_y': self.k_phi_y,
            'dt': self.timer_period
        }
        self.spatial_baseline_filter = FilterSpatialBaseline(self.robot, params, self)

        # Subscribe to leader and follower phases
        self.phase_leader_sub = self.create_subscription(
                    Float32, f'/{self.leader}/spatial_baseline/phase', self._phase_callback_leader, 10)
        self.phase_follower_sub = self.create_subscription(
            Float32, f'/{self.follower}/spatial_baseline/phase', self._phase_callback_follower, 10)

        # Publishers for baseline filter outputs
        self.position_pub    = self.create_publisher(Position, f'/{self.robot}/cmd_position', 10)
        self.publish_phase_diff_leader   = self.create_publisher(Float32, f'/{self.robot}/spatial_baseline/phase_diff/leader', 10)
        self.publish_phase_diff_follower = self.create_publisher(Float32, f'/{self.robot}/spatial_baseline/phase_diff/follower', 10)

        # Service client for reboot
        self.reboot_client = self.create_client(Empty, f'/{self.robot}/reboot')

        # Metadata publisher
        self.metadata_pub = self.create_publisher(Metadata, f'/{self.robot}/spatial_baseline/metadata', 10)

        # Main timer
        self.timer_period = 1.0 / self.predict_hz  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        # Metadata timer (every 10 seconds)
        self.metadata_timer = self.create_timer(10.0, self.publish_metadata)

        # Arming drones
        self.swarm = swarm
        if self.swarm:
            self.timeHelper = self.swarm.timeHelper
            self.allcfs = self.swarm.allcfs
            # arm (one by one)
            for cf in self.allcfs.crazyflies:
                cf.arm(True)
                self.timeHelper.sleep(1.0)

        self.info(f"Circle distortion spatial baseline node initialized for {self.robot}")
        self.info(f"Adjoint angle: {self.adjoint_angle:.3f} rad ({np.degrees(self.adjoint_angle):.1f}°)")

    def info(self, msg: str):
        """Log info message."""
        self.get_logger().info(msg)

    def timer_callback(self):
        """Main timer callback for state machine."""
        try:
            # State machine
            if self.state == 0:  # Takeoff
                if self.has_initial_pose:
                    self.phases[1] = self.initial_phase
                    self.takeoff()
                    self.spatial_baseline_filter.pub_phase.publish(Float32(data=self.phases[1]))
                    self.publish_phase_differences()

            elif self.state == 1:  # Hover
                self.hover()
                self.spatial_baseline_filter.pub_phase.publish(Float32(data=self.phases[1]))
                self.publish_phase_differences()

            elif self.state == 2:  # Encirclement with spatial baseline filter
                if self.has_phase_leader and self.has_phase_follower:
                    # Propagating the system
                    current_pose = self.current_pose.copy()
                    current_pose[2] -= self.hover_height  # Remove height offset
                    phase, position = self.spatial_baseline_filter.predict(current_pose, self.phases)
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

            elif self.state == 3:  # Landing
                self.landing()
                if self.i_landing < len(self.t_landing) - 1:
                    self.i_landing += 1
                else:
                    self.reboot()
                    self.info('Exiting spatial baseline node')
                    self.destroy_node()
                    rclpy.shutdown()

        except KeyboardInterrupt:
            self.info('Exiting spatial baseline node')

    def _poses_changed(self, robot_pose: PoseStamped):
        """Topic update callback for motion capture poses."""
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

            # Estimate 3D omega using the delta pose and velocity cross product
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
            self.state = 3

    def _phase_callback_leader(self, msg: Float32):
        """Callback to receive the phase of the leader agent."""
        self.has_phase_leader = True
        if msg.data:
            self.phases[0] = msg.data

    def _phase_callback_follower(self, msg: Float32):
        """Callback to receive the phase of the follower agent."""
        self.has_phase_follower = True
        if msg.data:
            self.phases[2] = msg.data

    def _order_callback(self, msg: StringArray):
        """Callback to receive the order of agents."""
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
    
    def _adjoint_angle_callback(self, msg: Float32):
        """Callback to receive adjoint angle updates from command center."""
        self.adjoint_angle = msg.data
        # Update filter parameter
        self.spatial_baseline_filter.adjoint_angle = self.adjoint_direction * msg.data
        self.info(f"Adjoint angle updated: {np.degrees(self.adjoint_angle):.1f}° ({self.adjoint_angle:.4f} rad)")
    
    def _radius_nominal_callback(self, msg: Float32):
        """Callback to receive radius nominal updates from command center."""
        self.radius_nominal = msg.data
        # Update filter parameter
        self.spatial_baseline_filter.radius_nominal = msg.data
        self.info(f"Nominal radius updated: {self.radius_nominal:.2f} m")

    def publish_phase_differences(self):
        """Publish phase differences to leader and follower."""
        diff_leader = wrap_to_pi(self.phases[0] - self.phases[1])
        diff_follower = wrap_to_pi(self.phases[1] - self.phases[2])
        self.publish_phase_diff_leader.publish(Float32(data=diff_leader))
        self.publish_phase_diff_follower.publish(Float32(data=diff_follower))

    def takeoff(self):
        """Take-off procedure to reach the hover height."""
        self.send_position(self.r_takeoff[:, self.i_takeoff])
        
        # Increment take-off index or switch to hover state
        if self.i_takeoff < len(self.t_takeoff)-1:
            self.i_takeoff += 1
        else:
            self.state = 1

    def takeoff_traj(self, t_max: float):
        """Take-off trajectory generation."""
        self.t_takeoff = np.arange(0, t_max, self.timer_period)
        self.r_takeoff = np.zeros((3, len(self.t_takeoff)))
        self.r_takeoff[0, :] += self.initial_pose[0] * np.ones(len(self.t_takeoff))
        self.r_takeoff[1, :] += self.initial_pose[1] * np.ones(len(self.t_takeoff))
        self.r_takeoff[2, :] = self.hover_height * (self.t_takeoff / t_max)

    def landing_traj(self, t_max: float):
        """Landing trajectory generation."""
        self.t_landing = np.arange(t_max, 0.1, -self.timer_period)
        self.i_landing = 0
        self.r_landing = np.zeros((3, len(self.t_landing)))
        self.r_landing[0, :] += self.final_pose[0] * np.ones(len(self.t_landing))
        self.r_landing[1, :] += self.final_pose[1] * np.ones(len(self.t_landing))
        self.r_landing[2, :] = self.final_pose[2] * (self.t_landing / t_max)

    def publish_metadata(self):
        """Publish experiment metadata every 10 seconds."""
        metadata = Metadata()
        metadata.group_name = 'spatial_baseline'
        metadata.drone_id = self.robot
        metadata.seed = 0  # Baseline doesn't use seed
        metadata.filter_type = 'SpatialBaseline'
        metadata.embedding_fn_name = 'modelE'  # Always identity for spatial
        metadata.frame_id = self.frame_id
        metadata.radius_nominal = self.radius_nominal
        metadata.omega_nominal = self.omega_z_nominal
        metadata.hover_height = self.hover_height
        metadata.k_phi = self.k_phi_z
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
        """Callback to initiate landing procedure."""
        self.land_flag = msg.data

    def _encircle_callback(self, msg):
        """Callback to initiate encirclement procedure."""
        self.state = 2

    def hover(self):
        """Hovering procedure at the hover height."""
        msg = Position()
        msg.x = self.initial_pose[0]
        msg.y = self.initial_pose[1]
        msg.z = self.hover_height
        self.position_pub.publish(msg)

    def landing(self):
        """Landing procedure to reach the ground."""
        self.send_position(self.r_landing[:, self.i_landing])

    def reboot(self):
        """Reboot the system."""
        req = Empty.Request()
        self.reboot_client.call_async(req)
        time.sleep(1.0)

    def send_position(self, r):
        """Send position command to the crazyflie."""
        msg = Position()
        msg.x = float(r[0])
        msg.y = float(r[1])
        msg.z = float(r[2])
        self.position_pub.publish(msg)


def main():
    swarm = Crazyswarm()
    if not rclpy.ok():
        rclpy.init()
    encirclement = CircleDistortionSpatialBaseline(swarm)
    rclpy.spin(encirclement)
    encirclement.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
