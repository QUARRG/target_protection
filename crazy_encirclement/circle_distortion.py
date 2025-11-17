import rclpy
import time
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy,QoSDurabilityPolicy
from motion_capture_tracking_interfaces.msg import NamedPoseArray
from crazyflie_interfaces.msg import FullState, StringArray, Position
from std_msgs.msg import Bool
from rclpy.duration import Duration
from crazy_encirclement.embedding_SO3_ros import Embedding
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, Float32
from geometry_msgs.msg import Pose, Twist, PoseStamped
from crazy_encirclement.utils2 import  trajectory, R3_so3, so3_R3
from scipy.linalg import expm, logm
from scipy.spatial.transform import Rotation as R

from filters import FilterGPS


class Circle_distortion(Node):
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
        self.declare_parameter('number_of_agents', '4')
        self.declare_parameter('r', '1.')
        self.declare_parameter('phi_dot', '0.8')

        self.robot = self.get_parameter('robot').value
        self.n_agents  = int(self.get_parameter('number_of_agents').value)
        self.r  = float(self.get_parameter('r').value)
        self.k_phi = 8  #float(self.get_parameter('k_phi').value)
        self.phi_dot = float(self.get_parameter('phi_dot').value)

        # Filter parameters
        self.declare_parameter('P', [0.0001, 0.0001, 0.25, 0.0001])
        self.declare_parameter('Q', [0.0, 0.0, 0.001, 0.0001])
        self.declare_parameter('V', [0.1, 0.1, 0.1])
        self.declare_parameter('predict_hz', 50.0)
        self.declare_parameter('update_hz', 10.0)
        self.declare_parameter('embedding_fn_name', 'modelB')

         # Get filter parameters
        P_list = self.get_parameter('P').value
        Q_list = self.get_parameter('Q').value
        V_list = self.get_parameter('V').value
        self.predict_hz = self.get_parameter('predict_hz').value
        self.update_hz  = self.get_parameter('update_hz').value
        self.embedding_fn_name = self.get_parameter('embedding_fn_name').value

        self.filter_params = {
            'P': P_list,
            'Q': Q_list,
            'V': V_list,
            'radius_guess': self.r + np.random.normal(0, 0.15),
            'phase_guess': (2*np.pi)/self.n_agents + np.random.normal(0, 0.1),
        }

        # Reboot client
        self.reboot_client = self.create_client(Empty, self.robot + '/reboot')

        # Flags and variables
        self.hover_height = 0.9  # meters

        self.order = []
        self.has_initial_pose = False
        self.has_final = False
        self.land_flag = False
        self.has_order = False
        self.has_phase_follower = False
        self.has_phase_leader = False

        self.final_pose = np.zeros(3)
        self.current_pos = np.zeros(3)
        self.initial_pose = np.zeros(3)
        
        self.leader = None
        self.follower = None
        self.Rot_des = np.eye(3)
        self.target_r = np.zeros(3)
        self.timer_period = 0.1

        self.i_landing = 0
        self.i_takeoff = 0
        # self.info(f"Number of agents: {self.n_agents}")

        self.phases = np.zeros(self.n_agents)
        # self.phi_cur = Float32()
        # self.phase_pub = self.create_publisher(Float32,'/'+ self.robot + '/phase', 1)

        self.state = 0
        # 0-take-off, 1-hover, 2-encirclement, 3-landing

        # Subscribers
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

        qos_profile = QoSProfile(reliability =QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            deadline=Duration(seconds=0, nanoseconds=0))

        self.create_subscription(
            NamedPoseArray, "/poses",
            self._poses_changed, qos_profile
        )

        self.create_subscription(
            StringArray, '/agents_order',
            self._order_callback,
            10)
        
        # Subscription to Vicon positions of the robot that are coming from the gps node
        self.create_subscription(
            Position, "/" + self.robot + "/vicon_position",
            self._poses_changed,
            10
        )        

        # Wait until order is received
        while (not self.has_order):
            rclpy.spin_once(self, timeout_sec=0.1)

        # Create filter instance
        self.filter = FilterGPS(self.robot, self.embedding_fn_name, self.filter_params, self)

        # Create subscribers for the other agents' filtered phases
        self.create_subscription(Float32, '/' + self.leader + '/filtered/phase', self._phase_callback_leader, 1)
        self.create_subscription(Float32, '/' + self.follower + '/filtered/phase', self._phase_callback_follower, 1)
        
    
        # while (self.phases[0] == 0):
        #     self.phase_pub.publish(self.phi_cur)
        #     rclpy.spin_once(self, timeout_sec=0.1)

        # while (self.phases[2] == 0):
        #     self.phase_pub.publish(self.phi_cur)
        #     rclpy.spin_once(self, timeout_sec=0.1)

        # self.info(f"agents phases: {self.phases}")
        self.wd = Float32()
        self.phi_diff = Float32()
        
        self.position_pub = self.create_publisher(Position,'/'+ self.robot + '/cmd_position', 10)
        self.publisher_w = self.create_publisher(Float32,'/'+ self.robot + '/omega_d', 10)
        self.publish_phi_diff = self.create_publisher(Float32,'/'+ self.robot + '/phi_diff', 10)

        self.publisher_w.publish(self.wd)
        #initiating some variables

        self.embedding = Embedding(self.r, self.phi_dot,self.k_phi, self.n_agents,self.initial_pose,self.hover_height,self.timer_period,self.phase_pub)

        # input("Press Enter to takeoff")
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):

        try:
            if self.state == 0:
                if self.has_initial_pose:
                    self.phi_cur.data = float(self.initial_phase)
                    self.phases[1] = self.initial_phase
                    # self.phase_pub.publish(self.phi_cur)
                    self.takeoff()
                    phi_k = self.phases[0]
                    phi_i = self.initial_phase
                    unit_i = np.array([np.cos(phi_i), np.sin(phi_i), 0])
                    unit_k = np.array([np.cos(phi_k), np.sin(phi_k), 0])
                    self.phi_diff.data = np.arccos(np.dot(unit_i,unit_k))
                    self.publish_phi_diff.publish(self.phi_diff)

            elif self.state == 1:
                self.hover() 
            
            elif self.state == 2: 
                if self.has_phase_follower and self.has_phase_leader:
                    phi, target_r, wd, phi_diff = self.embedding.targets(self.current_pos,self.phases)
                    self.phi_diff.data = phi_diff
                    self.publish_phi_diff.publish(self.phi_diff)
                    self.phi_cur.data = float(phi)
                    self.phase_pub.publish(self.phi_cur)
                    self.wd.data = wd
                    self.publisher_w.publish(self.wd)
                    self.send_position(target_r)
            
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

            #self.publishers[i].publish(msg)
    
    def _poses_changed(self, msg):
        """ Topic update callback to the motion capture lib's
            poses topic to send through the external position
            to the crazyflie 
        """
        Topic update callback to the motion capture lib's
           poses topic to send through the external position
           to the crazyflie 
        """
        for pose in msg.poses:
            if pose.name == self.robot:
                robot_pose = pose.pose
                break
        if not self.has_initial_pose:      
            self.initial_pose[0] = robot_pose.position.x
            self.initial_pose[1] = robot_pose.position.y
            self.initial_pose[2] = robot_pose.position.z   
            self.initial_phase = np.mod(np.arctan2(self.initial_pose[1], self.initial_pose[0]),2*np.pi)   
            self.takeoff_traj(4)
            self.has_initial_pose = True    
            
        elif not self.land_flag :

            self.current_pos[0] = robot_pose.position.x
            self.current_pos[1] = robot_pose.position.y
            self.current_pos[2] = robot_pose.position.z

        elif (self.has_final == False) and (self.land_flag == True):
            
            self.final_pose = np.zeros(3)
            self.info("Landing...")
            self.final_pose[0] = robot_pose.position.x
            self.final_pose[1] = robot_pose.position.y
            self.final_pose[2] = robot_pose.position.z
            self.landing_traj(2)
            self.has_final = True

    def _phase_callback_leader(self, msg):
        self.has_phase_leader = True
        if msg.data:
            self.phases[0] = msg.data

    def _phase_callback_follower(self, msg):
        self.has_phase_follower = True
        if msg.data:
            self.phases[2] = msg.data

    def _order_callback(self, msg):
        if not self.has_order:
            self.info(f"Phase received: {msg.data}")
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
            self.info(f"Leader: {self.leader}, Follower: {self.follower}")

    def takeoff(self):
        self.send_position(self.r_takeoff[:,self.i_takeoff])
        self.get_logger().info(f"publishing phase {self.phi_cur.data}")
        self.phase_pub.publish(self.phi_cur)
        #self.info(f"Publishing to {msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z}")
        if self.i_takeoff < len(self.t_takeoff)-1:
            self.i_takeoff += 1
        else:
            self.state = 1

    def takeoff_traj(self,t_max):
        #takeoff trajectory
        self.t_takeoff = np.arange(0,t_max,self.timer_period)
        self.r_takeoff = np.zeros((3,len(self.t_takeoff))) 
        self.r_takeoff[0,:] += self.initial_pose[0]*np.ones(len(self.t_takeoff))
        self.r_takeoff[1,:] += self.initial_pose[1]*np.ones(len(self.t_takeoff))
        self.r_takeoff[2,:] = self.hover_height*(self.t_takeoff/t_max)

    def landing_traj(self,t_max):
        #landing trajectory
        self.t_landing = np.arange(t_max,0.1,-self.timer_period)
        self.i_landing = 0
        self.r_landing = np.zeros((3,len(self.t_landing)))
        self.r_landing[0,:] += self.final_pose[0]*np.ones(len(self.t_landing))
        self.r_landing[1,:] += self.final_pose[1]*np.ones(len(self.t_landing))
        self.r_landing[2,:] = self.hover_height*(self.t_landing/t_max)
    
    def _landing_callback(self, msg):
        self.land_flag = msg.data
        self.state = 3

    def _encircle_callback(self, msg):
        self.state = 2

    def hover(self):
        self.phase_pub.publish(self.phi_cur)
        msg = Position()
        msg.x = self.initial_pose[0]
        msg.y = self.initial_pose[1]
        msg.z = self.hover_height
        self.position_pub.publish(msg)


    def landing(self):
        self.send_position(self.r_landing[:,self.i_landing])

    def reboot(self):
        req = Empty.Request()
        self.reboot_client.call_async(req)
        time.sleep(1.0)    

    def send_position(self,r):
        msg = Position()
        msg.x = float(r[0])
        msg.y = float(r[1])
        msg.z = float(r[2])

        self.position_pub.publish(msg)

def main():
    rclpy.init()
    encirclement = Circle_distortion()
    rclpy.spin(encirclement)
    encirclement.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
