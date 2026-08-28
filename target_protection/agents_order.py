import rclpy
import numpy as np

from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from motion_capture_tracking_interfaces.msg import NamedPoseArray
from crazyflie_interfaces.msg import StringArray
from std_msgs.msg import Bool
from rclpy.duration import Duration
from std_srvs.srv import Empty
from std_msgs.msg import Float32MultiArray, Float32
from functools import partial
from scipy.spatial.transform import Rotation as R


class AgentsOrder(Node):
    """
    Subscribes to the poses topic and publishes the agents' order
    """
    def __init__(self):
        super().__init__('encirclement')
        self.info = self.get_logger().info
        self.info('Agents order node has been started.')
        self.declare_parameter('robot_data', ['C04', 'C13', 'C05','C14','C20']) 

        self.robots = self.get_parameter('robot_data').value
        self.n_agents = len(self.robots)
        self.phases = np.zeros(self.n_agents)

        self.order = StringArray()
        self.has_order = False  # np.zeros((3,self.n_agents))
        self.initial_phases = {}    
        self.positions = np.zeros((3, self.n_agents))    
        self.distances = np.zeros(self.n_agents)

        qos_profile = QoSProfile(reliability =QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            deadline=Duration(seconds=0, nanoseconds=0))
        
        self.create_subscription(
            NamedPoseArray, "/poses",
            self._poses_changed, qos_profile
        )
        while self.has_order == False:
            rclpy.spin_once(self, timeout_sec=0.1)
        # self.order.data = ['C04', 'C20', 'C05']

        self.order_publisher = self.create_publisher(StringArray, '/agents_order', 10)
        self.distances_pubs = []
        for i in range(self.n_agents):
            self.distances_pubs.append(self.create_publisher(Float32, f'/{self.robots[i]}/distance_to_leader', 10))

        # for robot in self.order.data:
        #     self.phase_pub.append(self.create_publisher(Float32MultiArray,'/'+ robot + '/phases', 1))

        # for robot in self.order.data:
        #     self.create_subscription(Float32, '/'+ robot + '/phase', partial(self._phase_callback,robot), 1)
        
        self.timer_period = 0.01
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):
        try:
            self.order_publisher.publish(self.order)   
            for i in range(self.n_agents):
                distance_msg = Float32()
                distance_msg.data = float(self.distances[i])
                self.distances_pubs[i].publish(distance_msg)
                # self.info(f"phases agent {i+1}: {phases_this_robot}")

        except KeyboardInterrupt:
            self.info('Exiting node...')
            self.destroy_node()
            rclpy.shutdown()

    def _poses_changed(self, msg):
        """ Topic update callback to the motion capture lib's
            poses topic to send through the external position
            to the crazyflie 
        """
        if not self.has_order:
            self.initial_pose = np.zeros((3,self.n_agents))
            for pose in msg.poses:
                if pose.name in self.robots:
                    self.initial_phases[str(pose.name)] = np.mod(np.arctan2(pose.pose.position.y, pose.pose.position.x), 2*np.pi)
            
            self.order.data = sorted(self.initial_phases, key=lambda x: self.initial_phases[x], reverse=True)
            self.get_logger().info(f'Phases of agents: {self.initial_phases}')
            self.get_logger().info(f'Order of agents: {self.order.data}')
            self.has_order = True         
            self.info(f'Order of agents: {self.order}, robots: {self.robots}')    
        else:
            for pose in msg.poses:
                if pose.name in self.robots:
                    self.positions[:, self.order.data.index(pose.name)] = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
            for i in range(self.n_agents):              
                if i == 0:
                    k = self.n_agents -1
                else:
                    k = i -1
                self.distances[i] = np.linalg.norm(self.positions[:,i] - self.positions[:,k])


def main():
    rclpy.init()
    order = AgentsOrder()
    rclpy.spin(order)
    order.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
