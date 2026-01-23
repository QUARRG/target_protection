import rclpy
import time
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from crazyflie_interfaces.msg import StringArray, Position
from std_msgs.msg import Bool
from std_srvs.srv import Empty
from std_msgs.msg import Float32
from crazy_encirclement.filters import BaselineFilter, wrap_to_2pi, wrap_to_pi
from crazy_encirclement_interfaces.msg import Metadata
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSPresetProfiles
from motion_capture_tracking_interfaces.msg import NamedPoseArray
from rclpy.duration import Duration

class Follow_Limo(Node):
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
        self.declare_parameter('hover_height', 0.5)

        self.robot    = str(self.get_parameter('robot').value)
        self.hover_height = float(self.get_parameter('hover_height').value)


        # Filter parameters

        self.hz = 10

        # Reboot client
        self.reboot_client = self.create_client(Empty, self.robot + '/reboot')

        # Flags and variables
        self.timer_period = 1.0 / self.hz

        self.has_initial_pose = False
        self.has_final = False
        self.land_flag = False

        self.final_pose   = np.zeros(3)
        self.current_pose = np.zeros(3)
        self.previous_pose = np.zeros(3)
        self.previous_pose_time = 0.0
        self.initial_pose = np.zeros(3)
        self.set_point = np.array([-0.8,0,self.hover_height])

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
        
        qos_profile = QoSProfile(reliability =QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            deadline=Duration(seconds=0, nanoseconds=0))

        self.create_subscription(
            NamedPoseArray, "poses_relative",
            self._poses_changed, qos_profile
        )

        # Wait until order is received
        while (not self.has_initial_pose):
            rclpy.spin_once(self, timeout_sec=0.1)

        # Crazyflie position command publisher
        self.position_pub = self.create_publisher(Position, f'/{self.robot}/cmd_position', 10)
        
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

            # Encirclement state
            elif self.state == 2: 

                self.send_position(self.set_point)

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

    def _poses_changed(self, msg):
        """ Topic update callback to the motion capture lib's
            poses topic to send through the external position
            to the crazyflie. All steps based on the Vicon position.
        """
        # Initialize the initial pose and phase if not already set using vicon data
        for robot_pose in msg.poses:
            if robot_pose.name == self.robot:
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
                    self.landing_traj(2)
                    self.has_final = True

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
        self.r_takeoff[0,:] = self.initial_pose[0] * np.ones(len(self.t_takeoff))
        self.r_takeoff[1,:] = self.initial_pose[1] * np.ones(len(self.t_takeoff))
        self.r_takeoff[2,:] = self.hover_height * (self.t_takeoff / t_max)

    def landing_traj(self, t_max: float):
        ''' Landing trajectory generation. '''
        self.t_landing = np.arange(t_max, 0.1, -self.timer_period)
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
        self.state = 2

    def hover(self):
        ''' Hovering procedure at the hover height. '''
        msg = Position()
        msg.x = self.initial_pose[0]
        msg.y = self.initial_pose[1]
        msg.z = self.hover_height
        self.send_position(np.array([self.initial_pose[0],self.initial_pose[1],self.hover_height]))
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


def main():
    rclpy.init()
    follower = Follow_Limo()
    rclpy.spin(follower)
    follower.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
