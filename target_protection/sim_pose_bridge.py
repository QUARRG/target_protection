"""Convert Crazyflie simulation transforms to motion-capture poses."""

from motion_capture_tracking_interfaces.msg import NamedPose, NamedPoseArray

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

from tf2_msgs.msg import TFMessage


class SimPoseBridge(Node):
    """Publish simulator TF poses using the motion-capture message format."""

    def __init__(self):
        """Create the TF subscriber and motion-capture pose publisher."""
        super().__init__('sim_pose_bridge')

        self.declare_parameter('input_topic', '/tf')
        self.declare_parameter('output_topic', '/poses')
        self.declare_parameter('reference_frame', 'world')
        self.declare_parameter(
            'robot_names', rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter('poses_deadline', 0.01)

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.reference_frame = self._normalized_frame(
            self.get_parameter('reference_frame').value)
        self.pose_names = list(self.get_parameter('robot_names').value)
        self.robot_names = set(self.pose_names)
        self.latest_poses = {}
        poses_deadline = self.get_parameter('poses_deadline').value

        tf_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        poses_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            deadline=Duration(seconds=float(poses_deadline)),
        )

        self.publisher = self.create_publisher(
            NamedPoseArray, output_topic, poses_qos)
        self.subscription = self.create_subscription(
            TFMessage, input_topic, self._tf_callback, tf_qos)

        robots = ', '.join(sorted(self.robot_names)) or 'all direct child frames'
        self.get_logger().info(
            f'Bridging {input_topic} to {output_topic} in frame '
            f'{self.reference_frame} for {robots}')

    @staticmethod
    def _normalized_frame(frame_id):
        """Normalize an ROS frame ID for comparisons."""
        return frame_id.lstrip('/')

    def _tf_callback(self, message):
        latest_stamp = None

        for transform in message.transforms:
            parent = self._normalized_frame(transform.header.frame_id)
            name = self._normalized_frame(transform.child_frame_id)
            if parent != self.reference_frame:
                continue
            if self.robot_names and name not in self.robot_names:
                continue

            pose = NamedPose()
            pose.name = name
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            pose.pose.orientation = transform.transform.rotation
            self.latest_poses[name] = pose
            latest_stamp = transform.header.stamp

        if latest_stamp is None:
            return
        if self.robot_names and not self.robot_names.issubset(self.latest_poses):
            return

        output = NamedPoseArray()
        output.header.stamp = latest_stamp
        output.header.frame_id = self.reference_frame
        names = self.pose_names or sorted(self.latest_poses)
        output.poses = [self.latest_poses[name] for name in names]
        self.publisher.publish(output)


def main(args=None):
    """Run the simulation pose bridge."""
    rclpy.init(args=args)
    node = SimPoseBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
