import rclpy
from rclpy.node import Node
import numpy as np

# Incoming message types (from your existing network)
from motion_capture_tracking_interfaces.msg import NamedPoseArray
from crazyflie_interfaces.msg import Position
from std_msgs.msg import Float32MultiArray, Float64, Float64MultiArray, Float32
from rclpy.qos import qos_profile_sensor_data
# Outgoing message types (for Foxglove)
from visualization_msgs.msg import Marker, MarkerArray

from geometry_msgs.msg import Pose, Point, PoseStamped


class FoxgloveVisualizerNode(Node):
    def __init__(self):
        super().__init__('foxglove_visualizer_node')
        self.get_logger().info('Foxglove Visualizer Node started.')

        # Parameters (Match these to your main node)
        self.declare_parameter('pursuers', ['C26','C23','C24'])
        self.all_agents = self.get_parameter('pursuers').value
        self.agents = ['C26', 'C25', 'C24']  # your active drones
        self.evader = ['C23']
        # self.obstacles = ['Qdrone1', 'Qdrone4', 'Qdrone3']  # drones treated as obstacles
        self.target = ['LIMO']

        self.all_agents = self.agents  # only active agents
        self.get_logger().info(f'evader pos{self.all_agents}')


        # Internal State Variables
        self.target_center = np.zeros((len(self.all_agents), 3))
        self.target_radius = np.zeros(len(self.all_agents))

        self.sensing_range = 0.0

        self.trajectory_history = {
            name: [] for name in self.all_agents
        }
        self.max_history = 500  # limit memory

        

        self.agent_positions = {name: np.zeros(3) for name in self.all_agents}

        # ==========================================
        # SUBSCRIBERS (Listening to your main node)
        # ==========================================
        self.create_subscription(NamedPoseArray, '/poses', self.poses_callback, qos_profile_sensor_data)
        self.create_subscription(Position, "/", self.target_position,10)
        self.create_subscription(Float32, "/sensing_range", self.sensing_callback, 10)

        # ==========================================
        # PUBLISHERS (Broadcasting to Foxglove)
        # ==========================================
        self.marker_pub = self.create_publisher(MarkerArray, '/foxglove/markers', 10)
        self.target_marker_pub = self.create_publisher(Marker, '/foxglove/target_area', 10)
        self.sensing_marker_pub = self.create_publisher(Marker, '/foxglove/sensing_radius', 10)
        self.obstacle_marker_pub = self.create_publisher(Marker, '/foxglove/obstacle_area', 10)
        
        self.distances_pub = self.create_publisher(Float64MultiArray, '/foxglove/pursuer_distances', 10)

        # Timer to run the visualizer at 10Hz
        self.timer = self.create_timer(0.1, self.publish_visuals)


    # --- Callbacks to update internal state ---

    def sensing_callback(self, msg):
        self.sensing_range = float(msg.data)

    def target_position(self,msg):

        self.target_center = np.array(msg.data)

    # def targets_callback(self,msg):
    #     self.get_logger().info(f'target pos{msg.data}')
    #     for i in range(3):
    #         self.target_center[i] = np.array([msg.data[4*i],msg.data[4*i+1], msg.data[4*i+2]])
    #         self.target_radius[i] = float(msg.data[4*i+3])

    # def obstacles_callback(self,msg):
    #     for i in range(3):
    #         self.obstacles_center[i] = np.array([msg.data[4*i],msg.data[4*i+1], msg.data[4*i+2]])
    #         self.obstacles_radius[i] = float(msg.data[4*i+3])

    # def payoff_callback(self, msg):
    #     self.payoff = float(msg.data)


    # def pursuer_cp_callback(self, msg):
    #     cp = np.array([msg.x, msg.y, msg.z])
        
        # Record the very first capture point
        # if not self.first_cp_recorded and np.linalg.norm(cp) > 0:
        #     self.first_cp = cp
        #     self.first_cp_recorded = True

        # # Publish the constantly updating capture point
        # marker = self.create_sphere_marker("capture_points", 301, cp, [0.08, 0.08, 0.08], [1.0, 1.0, 0.0, 1.0]) # Yellow
        # self.current_cp_pub.publish(marker)


    def poses_callback(self, msg):
        # Update current positions from motion capture
        # self.get_logger().info(f'agent posititons pos{msg.poses}')
        for pose in msg.poses:

            if pose.name not in self.all_agents:
                continue
            # self.get_logger().info(f'agent posititons pos{pose}')
            if pose.name in self.agent_positions:
                self.agent_positions[pose.name][0] = pose.pose.position.x
                self.agent_positions[pose.name][1] = pose.pose.position.y
                self.agent_positions[pose.name][2] = pose.pose.position.z

            # store trajectory
            self.trajectory_history[pose.name].append(self.agent_positions[pose.name].copy())

            # limit size
            if len(self.trajectory_history[pose.name]) > self.max_history:
                self.trajectory_history[pose.name].pop(0)

    def capture_radius_callback(self, msg):
        for i, data_ in enumerate(msg.data):
            self.capture_radius[i] = float(data_)

    def speeds_callback(self, msg):
        self.real_evader_speed = msg.data[0]
        self.noisy_evader_speed = msg.data[1]
        self.estimated_evader_speed = msg.data[2]
        self.real_pursuer_speed = msg.data[3:3+len(self.pursuers)]


    # --- Main Visualization Loop ---
    def publish_visuals(self):
        timestamp = self.get_clock().now().to_msg()
        frame_id = "map"

        # 1. Publish Agents and Capture Ranges
        marker_array = MarkerArray()
        
        for i, agent_name in enumerate(self.all_agents):
            pos = self.agent_positions[agent_name]
            self.get_logger().info(f'agent{agent_name} pos{pos}')
            self.get_logger().info(f'agent{agent_name} sensing_range:{self.sensing_range}')
           
            # self.get_logger().info(f'agent posititons pos{self.agent_positions}')
            # if is_evader:
            #     self.get_logger().info(f'evader pos{pos}')
            # else:
            #     self.get_logger().info(f'pursuer pos{pos}')

            # Agent Marker (Cube)
            # color = [1.0, 0.0, 0.0, 1.0] if is_evader else [0.0, 0.0, 1.0, 1.0] # Red evader, Blue pursuer

            scale = [self.sensing_range*2, self.sensing_range*2, 0.1] #self.sensing_range*2]
            sensing_marker = self.create_sphere_marker(f"sensing_range{i}", 200, pos, scale, [0.5, 0.0, 0.5, 0.15])

            self.sensing_marker_pub.publish(sensing_marker)











            # Capture Range (Transparent Sphere for

            ## line to ground
            # created_line = self.create_line(i + 500, pos, color)
            # marker_array.markers.append(created_line)

            history = self.trajectory_history[agent_name]

            if len(history) < 2:
                continue

            traj_marker = Marker()
            traj_marker.header.frame_id = "world"
            traj_marker.header.stamp = self.get_clock().now().to_msg()

            traj_marker.ns = f"trajectory_{agent_name}"
            traj_marker.id = i + 1000

            traj_marker.type = Marker.LINE_STRIP
            traj_marker.action = Marker.ADD

            traj_marker.scale.x = 0.05  # line thickness

            # # color: evader red, pursuer blue
            # if is_evader:
            #     traj_marker.color.r = 1.0
            #     traj_marker.color.g = 0.0
            #     traj_marker.color.b = 0.0
            # else:
            traj_marker.color.r = 0.0
            traj_marker.color.g = 0.0
            traj_marker.color.b = 1.0

            traj_marker.color.a = 1.0

            for p in history:
                pt = Point()
                pt.x = float(p[0])
                pt.y = float(p[1])
                pt.z = float(p[2])
                traj_marker.points.append(pt)

            marker_array.markers.append(traj_marker)
        self.marker_pub.publish(marker_array)

        # 2. Publish Target Area
        for i in range(len(self.target_radius)):
            scale = [self.target_radius[0]*2, self.target_radius[0]*2, self.target_radius[0]*2]
            # self.get_logger().info(f'evader pos{self.target_center}')
            if i==0:
                color = [0.854,0.647,0.125,1]
            elif i==1:
                color = [0.482,0.407,0.93,1]
            else:
                color = [0.10,0.37,0.24,1]
            target_marker = self.create_sphere_marker(f"target_{i}", 200, self.target_center[i], scale, color)
    
            self.target_marker_pub.publish(target_marker)

        for i in range(len(self.obstacles_radius)):
            scale = [self.obstacles_radius[0]*2, self.obstacles_radius[0]*2, 0.5]
            obstacle_marker = self.create_cylinder_marker(f"target_{i}", 200, self.obstacles_center[i], scale, [1.0, 1.0, 1.0, 0.2])
    
            self.obstacle_marker_pub.publish(obstacle_marker)

            
        
        # # 4. Calculate and Publish Plot Metrics
        # # evader_pos_global = self.agent_positions[self.evader]
        # evader_pos_local = evader_pos_global - self.target_center

        # Calculate Distances
        # distances = []
        # for pur in self.pursuers:
        #     dist = np.linalg.norm(self.target_center - self.agent_positions[pur])
        #     distances.append(float(dist))
        
        # dist_msg = Float64MultiArray()
        # dist_msg.data = distances
        # self.distances_pub.publish(dist_msg)


    # --- Helper functions to keep code clean ---
    def create_cube_marker(self, ns, id, pos, scale, color):
        return self._build_marker(Marker.CUBE, ns, id, pos, scale, color)
    
    def create_line(self, id, pos, color):

        marker = Marker()

        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "ground_lines"
        marker.id = id

        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = 0.01

        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])

        p1 = Point()
        p1.x = float(pos[0])
        p1.y = float(pos[1])
        p1.z = float(pos[2])

        p2 = Point()
        p2.x = float(pos[0])
        p2.y = float(pos[1])
        p2.z = 0.0

        marker.points.append(p1)
        marker.points.append(p2)

        return marker

    def create_sphere_marker(self, ns, id, pos, scale, color):
        return self._build_marker(Marker.SPHERE, ns, id, pos, scale, color)

    def _build_marker(self, m_type, ns, id, pos, scale, color):
        m = Marker()
        m.header.frame_id = "world"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = id
        m.type = m_type
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(pos[0]), float(pos[1]), float(pos[2])
        m.scale.x, m.scale.y, m.scale.z = float(scale[0]), float(scale[1]), float(scale[2])
        m.color.r, m.color.g, m.color.b, m.color.a = float(color[0]), float(color[1]), float(color[2]), float(color[3])
        return m



    def create_cylinder_marker(self, ns, id, pos, scale, color):
        return self._build_marker(Marker.CYLINDER, ns, id, pos, scale, color)

    # marker = self.create_cylinder_marker(
    #     ns="obstacles",
    #     id=0,
    #     pos=[1.0, 2.0, 0.5],
    #     scale=[0.3, 0.3, 1.0],   # diameter x, diameter y, height z
    #     color=[1.0, 0.0, 0.0, 1.0]
    # )

def main(args=None):
    rclpy.init()
    node = FoxgloveVisualizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



