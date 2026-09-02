import rclpy
import time
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from crazyflie_interfaces.msg import StringArray, Position
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSPresetProfiles
from rclpy.duration import Duration
from motion_capture_tracking_interfaces.msg import NamedPoseArray
from std_msgs.msg import Bool, String
from std_srvs.srv import Empty
from target_protection.utils2 import R3_so3
from scipy.linalg import expm
from crazyflie_interfaces.srv import Arm
from std_msgs.msg import Float32

class Evader(Node):
    def __init__(self):
        """
            Node that sends the crazyflie to a desired position
            The desired position comes from the distortion of a circle
        """
        super().__init__('circle_distortion')
        self.info = self.get_logger().info
        

        # Parameters
        self.declare_parameter('robot', 'C23')
        self.declare_parameter('controls.hover_height', 0.3)
        self.declare_parameter('others.frame_id', 'world')
        self.declare_parameter('target', 'LIMO')
        self.declare_parameter('trajectory','follow_limo')
        self.declare_parameter('use_sim', False)

        self.robot    = str(self.get_parameter('robot').value)
        self.hover_height = 2.0 
        # self.hover_height = float(self.get_parameter('controls.hover_height').value)   
        self.frame_id = str(self.get_parameter('others.frame_id').value)
        self.target = str(self.get_parameter('target').value)
        self.trajectory = str(self.get_parameter('trajectory').value)
        self.use_sim = bool(self.get_parameter('use_sim').value)
        self.color = '0x8E1F20' #red color for evader

        # Reboot client
        self.reboot_client = self.create_client(Empty, self.robot + '/reboot')
        self.noise = np.zeros(3)
        # Flags and variables
        self.timer_period = 0.1
        if self.trajectory == 'eight':
            A = 0.9      # amplitude
            w = 0.5      # frequency
            t_end = 20   # total time
            dt = 0.05    # time step

            # Time vector
            self.t = np.arange(0, 2*np.pi/w, dt)
            self.x = A*np.cos(w*self.t)
            self.y = A*np.sin(w*self.t)*np.cos(w*self.t)
            self.index = 0
        elif self.trajectory == 'back_forth':
            self.x = np.arange(-1, 1, 0.01)
            self.y = np.zeros_like(self.x)
            self.index = 0
            self.increment = 1
        elif self.trajectory == 'embedding':#hover_height must be 1m for this trajectory
            self.radius = 0.8
            self.w_z = 0.4
            self.s = 0.7
        elif self.trajectory == 'follow_limo':
            w = 5   # frequency
            dt = 0.05    # time step
            self.t = np.arange(0, 2*np.pi, 0.2)
            # self.t = np.linspace(0,2*np.pi,0.05)
            self.z = 0.8*np.sin(self.t)
            self.index = 0
        

        self.initial_pose = np.zeros(3)
        self.order = []

        self.has_initial_pose = False
        self.has_final = False
        self.land_flag = False
        self.takeoff_flag = not self.use_sim

        self.final_pose   = np.zeros(3)
        self.current_pose = np.zeros(3)
        self.initial_pose = np.zeros(3)
        self.target_pos = np.zeros(3)
        
        self.i_landing = 0
        self.i_takeoff = 0

        self.state = 0
        # 0-take-off, 1-hover, 2-pursuing, 3-landing

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
            '/evade',
            self._evade_callback,
            10)

        self.create_subscription(
            Bool,
            '/evader_takeoff',
            self._takeoff_callback,
            10)

        poses_qos_deadline = 100.0  # example Hz

        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            deadline=Duration(nanoseconds=int(1e9 / poses_qos_deadline))
        )
        # Subscription to Vicon positions of the robot that are coming from the gps node
        self.create_subscription(
            NamedPoseArray, '/poses',
            self._poses_changed,
            qos_profile
        )
                # Wait until order is received
        while (not self.has_initial_pose):
            rclpy.spin_once(self, timeout_sec=0.1)
        # Arm the CrazySim firmware through the Crazyswarm2 server.
        self.arm_client = self.create_client(Arm, self.robot + '/arm')
        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Arm service not available, waiting again...')
        self.arm()
        time.sleep(2)

        # Crazyflie position command publisher
        self.position_pub = self.create_publisher(Position, f'/{self.robot}/cmd_position', 10)
        self.detection_pub = self.create_publisher(Bool, '/evader_detection', 10)
        self.color_pub = self.create_publisher(String,'/'+ self.robot + '/color_led', 10)

        # input("Press Enter to takeoff")
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.info('Evader node has been started.')

    def timer_callback(self):
        ''' Timer callback to send position commands to the crazyflie based on the current state. '''
        try:

            # Take-off state
            if self.state == 0:
                if self.has_initial_pose and self.takeoff_flag:
                    self.takeoff()
                    self.color_pub.publish(String(data=self.color))

            # Hover state
            elif self.state == 1:
                self.hover()
            
            # Encirclement state
            elif self.state == 2:
                if self.trajectory == 'follow_limo': 
                    if np.linalg.norm(self.target_pos - self.current_pose) > 0.6:                  
                        if self.index >= int(0.5*len(self.z)):
                            self.noise = np.zeros(3)
                            
                        if self.index >= len(self.z):
                            self.index = 0
                            noise_xy = np.random.uniform(-0.15,0.15)
                            noise_z = np.random.uniform(0.0,0.3)
                            self.noise = np.array([0, 0, noise_z])
                        # self.info(f'noise {self.noise}')
                        # if self.index > 0:
                        #     self.current_pose[2] -= self.z[self.index-1]
                        next_pos = self.current_pose + 0.1*(self.target_pos - self.current_pose) + self.noise
                        # next_pos[2] +=self.z[self.index]   
                    else:
                        next_pos = self.current_pose + 0.1*(self.target_pos - self.current_pose)

                    next_pos[2] = np.clip(next_pos[2],0.4,2.0)
                    next_pos[0] = np.clip(next_pos[0],-2.0,2.0)
                    next_pos[1] = np.clip(next_pos[1],-1.0,1.0)

                    self.send_position(next_pos)
                    self.index += 1
                elif self.trajectory == 'eight':
                    if self.index > (len(self.t)-1):
                        self.index = 0
                    self.send_position(np.array([self.x[self.index], self.y[self.index], self.hover_height]))
                    self.index +=1
                elif self.trajectory == 'back_forth':
                    if self.index >= (len(self.x)-1):
                        self.increment = -1
                    if self.index <= 0:
                        self.increment = 1
                    self.send_position(np.array([self.x[self.index], self.y[self.index], self.hover_height]))
                    self.index +=self.increment
                elif self.trajectory == 'embedding':
                    next_position = self.embedding_traj(self.current_pose,self.radius,self.w_z)
                    self.send_position(next_position)
                else:
                    self.hover()
            
            # Landing state
            elif self.state == 3:
                self.landing()
                if self.i_landing < len(self.t_landing)-1:
                    self.i_landing += 1
                else:
                    # time.sleep(2)
                    self.reboot()
                    self.info('Exiting circle node')
                    self.timer.cancel()
                    rclpy.shutdown()

        except KeyboardInterrupt:
            self.info('Exiting open loop command node')

    def _poses_changed(self, msg: NamedPoseArray):
        """ Topic update callback to the motion capture lib's
            poses topic to send through the external position
            to the crazyflie. All steps based on the Vicon position.
        """
        for robot_pose in msg.poses:
            if robot_pose.name == self.robot:
                # Initialize the initial pose and phase if not already set using vicon data
                if not self.has_initial_pose:     
                    self.initial_pose[0] = robot_pose.pose.position.x
                    self.initial_pose[1] = robot_pose.pose.position.y
                    self.initial_pose[2] = robot_pose.pose.position.z   
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
                    self.state = 3
            if robot_pose.name == self.target:
                self.target_pos[0] = robot_pose.pose.position.x
                self.target_pos[1] = robot_pose.pose.position.y
                self.target_pos[2] = robot_pose.pose.position.z

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
        self.t_landing = np.arange(t_max, 0.0, -self.timer_period)
        self.i_landing = 0
        self.r_landing = np.zeros((3, len(self.t_landing)))
        self.r_landing[0,:] += self.final_pose[0] * np.ones(len(self.t_landing))
        self.r_landing[1,:] += self.final_pose[1] * np.ones(len(self.t_landing))
        self.r_landing[2,:] = self.final_pose[2] * (self.t_landing / t_max)

    def _landing_callback(self, msg):
        ''' Callback to initiate landing procedure. '''
        self.land_flag = msg.data

    def _evade_callback(self, msg):
        ''' Callback to initiate encirclement procedure. '''
        if msg.data == True:
            self.state = 2
        else:
            self.info('Evade command received, but data is False. No action taken.')
            self.detection_pub.publish(Bool(data=False))
            self.land_flag = True

    def _takeoff_callback(self, msg):
        '''Enable the evader's takeoff sequence.'''
        self.takeoff_flag = msg.data

    def arm(self):
        ''' Reboot the system. '''
        req = Arm.Request()
        req.arm = True
        future = self.arm_client.call_async(req)
        # Wait for the result and handle the response
        rclpy.spin_until_future_complete(self, future)
    def hover(self):
        ''' Hovering procedure at the hover height. '''
        self.send_position(np.array([self.initial_pose[0], self.initial_pose[1], self.hover_height]))

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

    def embedding_traj(self, position, r,w_z):
        position -= np.array([0,0,self.hover_height]) 
        phi = np.arctan2(position[1],position[0])
        phi = np.mod(phi,2*np.pi)
        desired_pos_xy = np.array([r*np.cos(phi),r*np.sin(phi),0])
        Rot_z = expm(R3_so3(np.array([[0],[0],[w_z]])))
        desired_pos_xy_next = Rot_z@desired_pos_xy
        phi_next = np.arctan2(desired_pos_xy_next[1],desired_pos_xy_next[0])
        phi_next = np.mod(phi_next,2*np.pi)
        w_x = self.s*np.cos(phi_next)*np.sin(phi_next)
        w_y = self.s*np.sin(phi_next)
        Rot_xy = expm(R3_so3(np.array([w_x,w_y,0]).reshape(3,1)))
        desired_3d = Rot_xy@desired_pos_xy_next
        desired_3d += np.array([0,0,self.hover_height])
        return desired_3d

def main():
    rclpy.init()
    evader = Evader()
    try:
        rclpy.spin(evader)
    except KeyboardInterrupt:
        pass
    finally:
        evader.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
