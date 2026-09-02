"""Publish timed experiment-control flags for simulated runs."""

import time

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool


class SimulationExperimentController(Node):
    """Publish a configured sequence of experiment flags for simulation."""

    def __init__(self):
        """Create publishers and load the configured event schedule."""
        super().__init__('simulation_experiment_controller')

        self.declare_parameter('experiment_type', 'complete')
        self.declare_parameter('defenders_takeoff_time', 2.0)
        self.declare_parameter('evader_takeoff_time', 2.0)
        self.declare_parameter('encirclement_time', 8.0)
        self.declare_parameter('evade_time', 15.0)
        self.declare_parameter('evader_desengage', 22.0)
        self.declare_parameter('land_time', 30.0)
        self.declare_parameter('start_limo_time', 18.0)

        self.experiment_type = str(
            self.get_parameter('experiment_type').value).strip().lower()
        self.defenders_takeoff_time = float(
            self.get_parameter('defenders_takeoff_time').value)
        self.evader_takeoff_time = float(
            self.get_parameter('evader_takeoff_time').value)
        self.encirclement_time = float(
            self.get_parameter('encirclement_time').value)
        self.evade_time = float(self.get_parameter('evade_time').value)
        self.evader_desengage = float(
            self.get_parameter('evader_desengage').value)
        self.land_time = float(self.get_parameter('land_time').value)
        self.start_limo_time = float(self.get_parameter('start_limo_time').value)
        self._validate_schedule()

        self.defenders_takeoff_pub = self.create_publisher(
            Bool, '/defenders_takeoff', 10)
        self.evader_takeoff_pub = self.create_publisher(
            Bool, '/evader_takeoff', 10)
        self.encircle_pub = self.create_publisher(Bool, '/encircle', 10)
        self.start_limo_pub = self.create_publisher(Bool, '/start_limo', 10)
        self.evade_pub = self.create_publisher(Bool, '/evade', 10)
        self.landing_pub = self.create_publisher(Bool, '/landing', 10)

        self.start_time = time.monotonic()
        self.published_events = set()
        self.timer = self.create_timer(0.05, self._timer_callback)

        self.get_logger().info(
            f'Started simulation experiment schedule: {self.experiment_type}')

    def _validate_schedule(self):
        times = {
            'defenders_takeoff_time': self.defenders_takeoff_time,
            'evader_takeoff_time': self.evader_takeoff_time,
            'encirclement_time': self.encirclement_time,
            'evade_time': self.evade_time,
            'evader_desengage': self.evader_desengage,
            'land_time': self.land_time,
            'start_limo_time': self.start_limo_time,
        }
        if any(value < 0.0 for value in times.values()):
            raise ValueError('Simulation experiment times cannot be negative.')
        if self.evade_time <= self.encirclement_time:
            raise ValueError('evade_time must be greater than encirclement_time.')
        if self.land_time <= self.evade_time:
            raise ValueError('land_time must be greater than evade_time.')
        if (self.experiment_type == 'give up' and
                not self.evade_time < self.evader_desengage < self.land_time):
            raise ValueError(
                'For a give up experiment, evader_desengage must be after '
                'evade_time and before land_time.')

    def _publish_once(self, event, event_time, publisher, value):
        elapsed = time.monotonic() - self.start_time
        if event not in self.published_events and elapsed >= event_time:
            publisher.publish(Bool(data=value))
            self.published_events.add(event)
            self.get_logger().info(
                f'Published {event}={value} at {elapsed:.2f} seconds.')

    def _timer_callback(self):
        self._publish_once(
            'defenders_takeoff', self.defenders_takeoff_time,
            self.defenders_takeoff_pub, True)
        self._publish_once(
            'evader_takeoff', self.evader_takeoff_time,
            self.evader_takeoff_pub, True)
        self._publish_once(
            'encircle', self.encirclement_time, self.encircle_pub, True)
        self._publish_once(
            'evade', self.evade_time, self.evade_pub, True)
        self._publish_once(
            'start_limo', self.start_limo_time, self.start_limo_pub, True)

        if self.experiment_type == 'give up':
            self._publish_once(
                'evader_disengage', self.evader_desengage,
                self.evade_pub, False)

        self._publish_once(
            'landing', self.land_time, self.landing_pub, True)


def main(args=None):
    """Run the simulation experiment controller."""
    rclpy.init(args=args)
    node = SimulationExperimentController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
