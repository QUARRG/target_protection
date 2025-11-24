import rclpy
import time
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from crazyflie_interfaces.msg import StringArray, Position
from std_msgs.msg import Bool
from std_srvs.srv import Empty
from std_msgs.msg import Float32
from crazy_encirclement.filters import FilterGPS, wrap_to_pi, phase_controller


class CircleDistortion(Node):
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
        self.declare_parameter('number_of_agents', 4)
        self.declare_parameter('initial_phase', '0.0')
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
        self.declare_parameter('P', [0.0001, 0.0001, 0.25, 0.0001])
        self.declare_parameter('Q', [0.0, 0.0, 0.001, 0.0001])
        self.declare_parameter('V', [0.1, 0.1, 0.1])
        self.declare_parameter('predict_hz', 50.0)
        self.declare_parameter('update_hz', 10.0)

         # Get filter parameters
        P_list = self.get_parameter('P').value
        Q_list = self.get_parameter('Q').value
        V_list = self.get_parameter('V').value
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
        self.initial_pose = np.zeros(3)
        
        self.leader   = None
        self.follower = None     

        self.i_landing = 0
        self.i_takeoff = 0

        self.phases = np.zeros(self.n_agents)

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

        # Subscription to Vicon positions of the robot that are coming from the gps node
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

        # Wait until order is received
        while (not self.has_order):
            rclpy.spin_once(self, timeout_sec=0.1)

        # Create filter instance
        self.filter_params = {
            'P': P_list,
            'Q': Q_list,
            'V': V_list,
            'radius_guess': self.radius_nominal + np.random.normal(0, 0.15),
            'phase_guess': self.initial_phase + np.random.normal(0, 0.1),
            'frame_id': self.frame_id
        }
        self.filter = FilterGPS(self.robot, self.embedding_fn_name, self.filter_params, self)

        # Create subscribers for the other agents' filtered phases
        self.create_subscription(Float32, f'/{self.leader}/filtered/phase',   self._phase_callback_leader, 1)
        self.create_subscription(Float32, f'/{self.follower}/filtered/phase', self._phase_callback_follower, 1)

        # Create subscriber for filter updates using the GPS measurements
        self.create_subscription(
            PoseStamped,
            f'/{self.robot}/gps_position',
            self.update_callback,
            10
        )

        # Crazyflie position command publisher
        self.position_pub = self.create_publisher(Position, f'/{self.robot}/cmd_position', 10)

        # Publishers for phase differences
        self.publish_omega = self.create_publisher(Float32, f'/{self.robot}/filtered/omega', 10)
        self.publish_phase_diff_leader = self.create_publisher(Float32, f'/{self.robot}/filtered/phase_diff/leader', 10)
        self.publish_phase_diff_follower = self.create_publisher(Float32, f'/{self.robot}/filtered/phase_diff/follower', 10)

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
                    self.filter.pub_phase.publish(Float32(data=self.phases[1]))
                    self.publish_phase_differences()

            # Hover state
            elif self.state == 1:
                self.hover()
                self.filter.pub_phase.publish(Float32(data=self.phases[1]))
                self.publish_phase_differences()
            
            # Encirclement state
            elif self.state == 2: 
                if self.has_phase_follower and self.has_phase_leader:
                    # Computing desired phi_dot based on the leader and follower phases
                    omega = phase_controller(self.phases[1], self.phases[0], self.phases[2], self.omega_nominal, self.k_phi)
                    omega_msg = Float32()
                    omega_msg.data = omega
                    self.publish_omega.publish(omega_msg)

                    # Propagating the filter
                    phase, position = self.filter.predict(omega, self.timer_period)

                    # Updating internal parameters
                    self.phases[1] = phase
                    self.publish_phase_differences()

                    target_r = np.array([position.pose.position.x,
                                         position.pose.position.y,
                                         position.pose.position.z + self.hover_height])

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
            self.info('Exiting open loop command node')
    
    def update_callback(self, gps_pose: PoseStamped):
        ''' Callback to update the filter with GPS measurements. '''
        if self.state != 2:
            return  # Only update during encirclement state
        y = np.array([gps_pose.pose.position.x,
                      gps_pose.pose.position.y,
                      gps_pose.pose.position.z]).reshape((3, 1))
        phase, position = self.filter.update(y)

        # Updating internal parameters
        self.phases[1] = phase

        # Checking phase differences
        self.publish_phase_differences()

        target_r = np.array([position.pose.position.x,
                             position.pose.position.y,
                             position.pose.position.z + self.hover_height])
        self.send_position(target_r)

    def _poses_changed(self, robot_pose: PoseStamped):
        """ Topic update callback to the motion capture lib's
            poses topic to send through the external position
            to the crazyflie. All steps based on the Vicon position.
        """
        # Initialize the initial pose and phase if not already set using vicon data
        if not self.has_initial_pose:      
            self.initial_pose[0] = robot_pose.pose.position.x
            self.initial_pose[1] = robot_pose.pose.position.y
            self.initial_pose[2] = robot_pose.pose.position.z   
            self.initial_phase = wrap_to_pi(np.arctan2(self.initial_pose[1], self.initial_pose[0]))   
            self.takeoff_traj(4)
            self.has_initial_pose = True    
            
        # Update current pose if not landing
        elif not self.land_flag:
            self.current_pose[0] = robot_pose.pose.position.x
            self.current_pose[1] = robot_pose.pose.position.y
            self.current_pose[2] = robot_pose.pose.position.z

        # Set final pose when landing is commanded
        elif (self.has_final == False) and (self.land_flag == True):
            self.final_pose = np.zeros(3)
            self.info("Landing...")
            self.final_pose[0] = robot_pose.pose.position.x
            self.final_pose[1] = robot_pose.pose.position.y
            self.final_pose[2] = robot_pose.pose.position.z
            self.landing_traj(3)
            self.has_final = True

    def _phase_callback_leader(self, msg):
        ''' Callback to receive the filtered phase of the leader agent. '''
        self.has_phase_leader = True
        if msg.data:
            self.phases[0] = msg.data

    def _phase_callback_follower(self, msg):
        ''' Callback to receive the filtered phase of the follower agent. '''
        self.has_phase_follower = True
        if msg.data:
            self.phases[2] = msg.data

    def _order_callback(self, msg):
        ''' Callback to receive the order of agents. '''
        if not self.has_order:
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

    def publish_phase_differences(self):
        ''' Publish phase differences to leader and follower. '''
        diff_leader = wrap_to_pi(self.phases[0] - self.phases[1])
        diff_follower = wrap_to_pi(self.phases[2] - self.phases[1])
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
        self.r_landing[2,:] = self.hover_height * (self.t_landing / t_max)

    def _landing_callback(self, msg):
        ''' Callback to initiate landing procedure. '''
        self.land_flag = msg.data
        self.state = 3

    def _encircle_callback(self, msg):
        ''' Callback to initiate encirclement procedure. '''
        self.state = 2

    def hover(self):
        ''' Hovering procedure at the hover height. '''
        # self.phase_pub.publish(self.phi_cur)
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
