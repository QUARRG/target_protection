"""Publish a bounded point-mass target for simulation experiments."""

import math
import random
import socket
import struct

from geometry_msgs.msg import PoseStamped, TransformStamped

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile

from std_msgs.msg import Bool

from tf2_ros import TransformBroadcaster


class SimPointMass(Node):
    """Simulate a planar point mass and publish its pose and TF transform."""

    def __init__(self):
        """Initialize the point mass at a random nonzero position."""
        super().__init__('sim_point_mass')

        self.declare_parameter('name', 'LIMO')
        self.declare_parameter('reference_frame', 'world')
        self.declare_parameter('half_extent', 1.0)
        self.declare_parameter('minimum_origin_distance', 0.1)
        self.declare_parameter('randomize_initial_position', True)
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('height', 0.0)
        self.declare_parameter('velocity_x', 0.0)
        self.declare_parameter('velocity_y', 0.0)
        self.declare_parameter('update_hz', 20.0)
        self.declare_parameter('random_seed', -1)
        self.declare_parameter('circle_radius', 2.0)
        self.declare_parameter('angular_velocity', 0.2)
        self.declare_parameter('mujoco_pose_enabled', False)
        self.declare_parameter('mujoco_host', '127.0.0.1')
        self.declare_parameter('mujoco_pose_port', 19849)

        self.target_name = str(self.get_parameter('name').value)
        self.reference_frame = str(
            self.get_parameter('reference_frame').value).lstrip('/')
        self.half_extent = float(self.get_parameter('half_extent').value)
        minimum_distance = float(
            self.get_parameter('minimum_origin_distance').value)
        randomize_position = bool(
            self.get_parameter('randomize_initial_position').value)
        self.height = float(self.get_parameter('height').value)
        self.velocity = [
            float(self.get_parameter('velocity_x').value),
            float(self.get_parameter('velocity_y').value),
        ]
        update_hz = float(self.get_parameter('update_hz').value)
        random_seed = int(self.get_parameter('random_seed').value)
        self.circle_radius = float(
            self.get_parameter('circle_radius').value)
        self.angular_velocity = float(
            self.get_parameter('angular_velocity').value)
        self.mujoco_pose_enabled = bool(
            self.get_parameter('mujoco_pose_enabled').value)
        mujoco_host = str(self.get_parameter('mujoco_host').value)
        mujoco_pose_port = int(
            self.get_parameter('mujoco_pose_port').value)

        if self.half_extent <= 0.0:
            raise ValueError('half_extent must be positive.')
        if not 0.0 < minimum_distance < self.half_extent:
            raise ValueError(
                'minimum_origin_distance must be positive and less than half_extent.')
        if update_hz <= 0.0:
            raise ValueError('update_hz must be positive.')
        if self.circle_radius <= 0.0:
            raise ValueError('circle_radius must be positive.')

        if randomize_position:
            generator = random.Random(None if random_seed < 0 else random_seed)
            while True:
                x = generator.uniform(-self.half_extent, self.half_extent)
                y = generator.uniform(-self.half_extent, self.half_extent)
                if math.hypot(x, y) >= minimum_distance:
                    self.position = [x, y]
                    break
        else:
            self.position = [
                float(self.get_parameter('initial_x').value),
                float(self.get_parameter('initial_y').value),
            ]
            if any(abs(value) > self.half_extent for value in self.position):
                raise ValueError('The configured initial position is outside the square.')
            if math.hypot(*self.position) < minimum_distance:
                raise ValueError('The configured initial position is too close to the origin.')

        self.time_step = 1.0 / update_hz
        pose_qos = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pose_publisher = self.create_publisher(
            PoseStamped, f'/{self.target_name}/pose', pose_qos)
        self.start_limo_subscription = self.create_subscription(
            Bool, '/start_limo', self._start_limo_callback, 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(self.time_step, self._timer_callback)
        self._mujoco_socket = None
        self._mujoco_destination = None
        if self.mujoco_pose_enabled:
            self._mujoco_socket = socket.socket(
                socket.AF_INET, socket.SOCK_DGRAM)
            self._mujoco_destination = (mujoco_host, mujoco_pose_port)

        self.circle_active = False
        self.circle_center = [0.0, 0.0]
        self.circle_start_time = None

        self.get_logger().info(
            f'{self.target_name} point mass initialized at '
            f'({self.position[0]:.3f}, {self.position[1]:.3f}, {self.height:.3f})')

    def _start_limo_callback(self, msg):
        """Start circular target motion on the first true start_limo flag."""
        if not msg.data or self.circle_active:
            return

        # Make the current position the phase-zero point so activation does not
        # introduce a discontinuity in the target pose.
        self.circle_center = [
            self.position[0] - self.circle_radius,
            self.position[1],
        ]
        self.circle_start_time = self.get_clock().now()
        self.circle_active = True
        self.get_logger().info(
            f'{self.target_name} starting circular motion: radius '
            f'{self.circle_radius:.3f} m, angular velocity '
            f'{self.angular_velocity:.3f} rad/s')

    def _timer_callback(self):
        if self.circle_active:
            elapsed = (
                self.get_clock().now() - self.circle_start_time
            ).nanoseconds * 1e-9
            phase = self.angular_velocity * elapsed
            self.position[0] = self.circle_center[0] + (
                self.circle_radius * math.cos(phase))
            self.position[1] = self.circle_center[1] + (
                self.circle_radius * math.sin(phase))
            direction = 1.0 if self.angular_velocity >= 0.0 else -1.0
            yaw = phase + direction * math.pi / 2.0
        else:
            for axis in range(2):
                self.position[axis] += self.velocity[axis] * self.time_step
                if self.position[axis] > self.half_extent:
                    self.position[axis] = self.half_extent
                    self.velocity[axis] *= -1.0
                elif self.position[axis] < -self.half_extent:
                    self.position[axis] = -self.half_extent
                    self.velocity[axis] *= -1.0

            yaw = 0.0
            if math.hypot(*self.velocity) > 0.0:
                yaw = math.atan2(self.velocity[1], self.velocity[0])

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.reference_frame
        pose.pose.position.x = self.position[0]
        pose.pose.position.y = self.position[1]
        pose.pose.position.z = self.height
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        self.pose_publisher.publish(pose)

        transform = TransformStamped()
        transform.header = pose.header
        transform.child_frame_id = self.target_name
        transform.transform.translation.x = pose.pose.position.x
        transform.transform.translation.y = pose.pose.position.y
        transform.transform.translation.z = pose.pose.position.z
        transform.transform.rotation = pose.pose.orientation
        self.tf_broadcaster.sendTransform(transform)

        if self._mujoco_socket is not None:
            packet = struct.pack(
                '<7d',
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
                pose.pose.orientation.w,
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
            )
            try:
                self._mujoco_socket.sendto(
                    packet, self._mujoco_destination)
            except OSError as error:
                self.get_logger().warning(
                    f'Could not send LIMO pose to MuJoCo: {error}',
                    throttle_duration_sec=5.0)

    def destroy_node(self):
        """Close the optional MuJoCo transport before destroying the node."""
        if self._mujoco_socket is not None:
            self._mujoco_socket.close()
            self._mujoco_socket = None
        return super().destroy_node()


def main(args=None):
    """Run the simulation point mass."""
    rclpy.init(args=args)
    node = SimPointMass()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
