#!/usr/bin/env python3
"""
Command Center Node for controlling SpatialBaseline parameters.
Provides keyboard interface for adjusting adjoint angle and nominal radius in real-time.
"""
import sys
import termios
import tty
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


class CommandCenter(Node):
    def __init__(self):
        super().__init__("command_center")
        
        # Declare and get parameters
        self.declare_parameter('adjoint_angle_initial', 0.0)
        self.declare_parameter('radius_nominal_initial', 1.0)
        self.declare_parameter('adjoint_angle_increment', 5.0)  # degrees
        self.declare_parameter('radius_increment', 0.1)  # meters
        
        # Get initial values
        self.adjoint_angle = self.get_parameter('adjoint_angle_initial').get_parameter_value().double_value
        self.radius_nominal = self.get_parameter('radius_nominal_initial').get_parameter_value().double_value
        self.adjoint_angle_increment = self.get_parameter('adjoint_angle_increment').get_parameter_value().double_value
        self.radius_increment = self.get_parameter('radius_increment').get_parameter_value().double_value
        
        # Convert angle increment to radians
        self.adjoint_angle_increment_rad = np.radians(self.adjoint_angle_increment)
        
        # Publishers for command signals
        self.adjoint_angle_pub = self.create_publisher(Float32, '/command_center/adjoint_angle', 10)
        self.radius_nominal_pub = self.create_publisher(Float32, '/command_center/radius_nominal', 10)
        
        # Publish initial values
        self.publish_commands()
        
        # Terminal settings for non-blocking input
        self.settings = None
        
        # Start keyboard input thread
        self.running = True
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_thread.start()
        
        self.info("Command Center initialized")
        self.print_help()
    
    def info(self, msg: str):
        """Log info message."""
        self.get_logger().info(msg)
    
    def print_help(self):
        """Print help message with available commands."""
        print("\n" + "="*60)
        print("Command Center - Keyboard Controls")
        print("="*60)
        print(f"  '+' : Increment adjoint angle by {self.adjoint_angle_increment}°")
        print(f"  '-' : Decrement adjoint angle by {self.adjoint_angle_increment}°")
        print(f"  ']' : Increment nominal radius by {self.radius_increment} m")
        print(f"  '[' : Decrement nominal radius by {self.radius_increment} m")
        print("  'h' : Show this help message")
        print("  'q' : Quit")
        print("="*60)
        print(f"\nCurrent values:")
        print(f"  Adjoint angle: {np.degrees(self.adjoint_angle):.1f}° ({self.adjoint_angle:.4f} rad)")
        print(f"  Nominal radius: {self.radius_nominal:.2f} m")
        print("="*60 + "\n")
    
    def get_key(self):
        """Get a single keypress from terminal."""
        if self.settings is None:
            self.settings = termios.tcgetattr(sys.stdin)
        
        try:
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        
        return key
    
    def keyboard_listener(self):
        """Listen for keyboard input in separate thread."""
        while self.running and rclpy.ok():
            try:
                key = self.get_key()
                self.process_key(key)
            except Exception as e:
                self.get_logger().error(f"Keyboard error: {e}")
                break
    
    def process_key(self, key: str):
        """Process keyboard input and update parameters."""
        updated = False
        
        if key == '+':
            self.adjoint_angle += self.adjoint_angle_increment_rad
            updated = True
            print(f"[+] Adjoint angle: {np.degrees(self.adjoint_angle):.1f}° ({self.adjoint_angle:.4f} rad)")
        
        elif key == '-':
            self.adjoint_angle -= self.adjoint_angle_increment_rad
            updated = True
            print(f"[-] Adjoint angle: {np.degrees(self.adjoint_angle):.1f}° ({self.adjoint_angle:.4f} rad)")
        
        elif key == ']':
            self.radius_nominal += self.radius_increment
            updated = True
            print(f"[]] Nominal radius: {self.radius_nominal:.2f} m")
        
        elif key == '[':
            self.radius_nominal = max(0.1, self.radius_nominal - self.radius_increment)
            updated = True
            print(f"[[] Nominal radius: {self.radius_nominal:.2f} m")
        
        elif key == 'h' or key == 'H':
            self.print_help()
        
        elif key == 'q' or key == 'Q':
            print("\nShutting down Command Center...")
            self.running = False
            self.cleanup()
            rclpy.shutdown()
        
        # Publish updated values
        if updated:
            self.publish_commands()
    
    def publish_commands(self):
        """Publish current command values."""
        self.adjoint_angle_pub.publish(Float32(data=self.adjoint_angle))
        self.radius_nominal_pub.publish(Float32(data=self.radius_nominal))
    
    def cleanup(self):
        """Restore terminal settings."""
        if self.settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)


def main():
    rclpy.init()
    command_center = CommandCenter()
    
    try:
        rclpy.spin(command_center)
    except KeyboardInterrupt:
        pass
    finally:
        command_center.cleanup()
        command_center.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
