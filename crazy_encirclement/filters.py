import numpy as np
from rclpy.node import Node
from typing import Callable
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from scipy.linalg import expm


# ----------------------------------------------------------------------
# Embedding Functions
# ----------------------------------------------------------------------
def omega_func_modelA(theta: float) -> np.ndarray:
    return np.asarray([0.3 * np.sin(6 * theta) * np.cos(6 * theta), 0.3, 0.])

def omega_func_modelB(theta: float) -> np.ndarray:
    return np.asarray([np.sin(theta)*np.cos(theta), 0., 0.])

def omega_func_modelC(theta: float) -> np.ndarray:
    return np.asarray([0.6 * np.cos(2 * theta), 0.6 * np.cos(theta)**2, 0.])

def omega_func_modelD(theta: float) -> np.ndarray:
    return np.asarray([0.9 * np.cos(3 * theta) * np.sin(theta), 0.5 * 0.9, 0.])

REGISTRED_OMEGA_FUNCTIONS = {
    'modelA': omega_func_modelA,
    'modelB': omega_func_modelB,
    'modelC': omega_func_modelC,
    'modelD': omega_func_modelD
}
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def wrap_to_pi(angle):
    """Wrap angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def skew(v: np.ndarray) -> np.ndarray:
    ''' Skew-symmetric matrix for SO(3)
        v: 1x3, 3x1 or 3, vector
        Returns: 3x3 skew-symmetric matrix
    '''
    v = v.flatten()
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def exp_SO3(omega: np.ndarray) -> np.ndarray:
    ''' Exponential map for SO(3) using Rodrigues' formula.
        omega: 3x1 vector
        Returns: 3x3 rotation matrix
    '''
    return expm(skew(omega))


def orthonormalize(R: np.ndarray) -> np.ndarray:
    ''' Orthonormalizes a given square matrix using Singular Value Decomposition (SVD).
        R: 3x3 rotation matrix
        Returns: 3x3 rotation matrix
    '''
    U, S, Vt = np.linalg.svd(R)
    return U @ Vt


def phase_controller(phase_ego, phase_leader, phase_follower, phi_dot_nominal, k_p=1.0):
    """
    A simple PD controller to adjust the phase angle of the ego vehicle
    based on the angles of the vehicles ahead and behind.
    """
    error_ahead  = phase_ego - phase_leader
    error_behind = phase_ego - phase_follower

    # Normalize errors to the range [-pi, pi]
    error_ahead  = wrap_to_pi(error_ahead)
    error_behind = wrap_to_pi(error_behind)

    eps = 1e-6  # small constant to avoid division by zero

    control_signal = phi_dot_nominal + k_p * (1/(error_ahead + eps) + 1/(error_behind + eps))

    return control_signal
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Filters
# ----------------------------------------------------------------------
class BaseFilter:
    ''' Base LIEKF for encirclement tasks with customizable embedding functions.
    '''
    def __init__(self, name: str, embedding_fn_name: str, params: dict, node: Node):
        self.name = name
        self.params = params
        self.node: Node = node

        # Initialize filter parameters 
        self.P: np.ndarray = np.diag(np.square(self.params.get('P', np.zeros(4))))
        self.Q: np.ndarray = np.diag(np.square(self.params.get('Q', np.zeros(4))))
        self.V: np.ndarray = np.diag(np.square(self.params.get('V', np.zeros(3))))
        self.Rc: np.ndarray = self.build_Rc(wrap_to_pi(self.params.get('phase_guess', 0.0)))
        self.radius: float = self.params.get('radius_guess', 2.0)
        self.s: float = np.log(self.radius)
        self.e_x: np.ndarray = np.asarray([[1.], [0.], [0.]])

        # Checking embedding function
        if embedding_fn_name not in REGISTRED_OMEGA_FUNCTIONS:
            raise ValueError(f"Embedding function '{embedding_fn_name}' is not allowed. Choose from: {list(REGISTRED_OMEGA_FUNCTIONS.keys())}")
        self.embedding_fn: Callable = REGISTRED_OMEGA_FUNCTIONS[embedding_fn_name]

        # Publishers
        self.frame_id: str = self.params.get('frame_id', 'world')
        self.pub_pose  = self.node.create_publisher(PoseStamped, f'/{self.name}/filtered/pose', 10)
        self.pub_phase = self.node.create_publisher(Float32, f'/{self.name}/filtered/phase', 10)
        self.node.info(f'Filter for agent {self.name} initialized with embedding function {embedding_fn_name}.')

        # Initialize by publishing initial pose and phase
        pose_msg, phase_msg = self.build_pose_phase_msgs()
        self.pub_pose.publish(pose_msg)
        self.pub_phase.publish(phase_msg)

    def build_Rc(self, phase: float) -> np.ndarray:
        # wrap phase to [-pi, pi]
        phase = wrap_to_pi(phase)
        c, s = np.cos(phase), np.sin(phase)
        return np.array([[c, -s, 0],
                         [s,  c, 0],
                         [0,  0, 1]])

    def build_Re(self, embedding_func: Callable[[float], np.ndarray], phase: float) -> np.ndarray:
        return exp_SO3(embedding_func(phase))
    
    def get_phase(self, Rc: np.ndarray) -> float:
        return wrap_to_pi(np.arctan2(Rc[1,0], Rc[0,0]))

    def build_pose_phase_msgs(self) -> list[PoseStamped, Float32]:
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = self.frame_id
        pose_msg.header.stamp = self.node.get_clock().now().to_msg()
        phase: float = self.get_phase(self.Rc)
        Re: np.ndarray = self.build_Re(self.embedding_fn, phase)
        Rc: np.ndarray = self.build_Rc(phase)
        radius: float = np.exp(self.s)
        q: np.ndarray = (Re @ Rc @ (self.e_x * radius)).flatten()
        pose_msg.pose.position = Point(x=q[0], y=q[1], z=q[2])
        pose_msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        phase_msg = Float32()
        phase_msg.data = phase
        return pose_msg, phase_msg
    
    def predict(self, omega_z: float, dt: float):
        # Update theta based on omega_z and time step
        self.Rc = exp_SO3(np.asarray([0., 0., omega_z * dt])) @ self.Rc
        self.Rc = orthonormalize(self.Rc)
        # self.r = self.r  # constant radius
        
        # Predict the next covariance
        F = np.eye(4)
        F[0:3, 0:3] = self.build_Rc(omega_z * dt)
        Q = self.Q.copy()
        Q[3, 3] = ((Q[3, 3]**0.5) / np.exp(self.s)) ** 2
        self.P = F @ self.P @ F.T + Q * dt
        self.P = (self.P + self.P.T) / 2  # Ensure symmetry

        # Publish predicted pose and phase
        pose_msg, phase_msg = self.build_pose_phase_msgs()
        self.pub_pose.publish(pose_msg)
        self.pub_phase.publish(phase_msg)
        # self.node.get_logger().info(f'Published pose for agent {self.name}')

        return phase_msg.data, pose_msg.pose.position


class FilterGPS(BaseFilter):
    ''' LIEKF for encirclement tasks using GPS-like measurements.
    '''
    def __init__(self, name: str, embedding_fn_name: str, params: dict, node: Node):
        super().__init__(name, embedding_fn_name, params, node)

    def update(self, y: np.ndarray):
        # Measurement Jacobian
        radius: float = np.exp(self.s)
        Re: np.ndarray = self.build_Re(self.embedding_fn, self.get_theta(self.Rc))
        H_theta: np.ndarray = -self.Rc.T @ (Re @ self.Rc @ skew(self.e_x)) * radius    # body frame
        H_r: np.ndarray = (self.Rc.T @ (Re @ self.Rc @ self.e_x )) * radius            # inertial frame
        H: np.ndarray = np.hstack((H_theta, H_r))

        # Kalman Gain
        V: np.ndarray = self.Rc.T @ self.V @ self.Rc
        S: np.ndarray = H @ self.P @ H.T + V
        IdS: np.ndarray = np.eye(S.shape[0])
        S = 0.5 * np.add(S, S.T) + IdS * 1e-8
        S_inv = np.linalg.inv(S)
        K: np.ndarray = self.P @ H.T @ S_inv

        # Update state
        y_hat: np.ndarray = Re @ self.Rc @ self.e_x * radius
        z: np.ndarray = self.Rc.T @ (y - y_hat)
        # print(f"NIS: {np.squeeze(z.T @ S_inv @ z).item():.3f}")
        delta = K @ z.flatten()                   # Correction vector in the algebra
        theta_correction = exp_SO3(delta[0:3])    # Exponential map to group element
        self.Rc = theta_correction @ self.Rc      
        self.Rc = orthonormalize(self.Rc)
        self.s += delta[3]
        self.radius = float(np.exp(self.s)) 

        # Update covariance
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ V @ K.T
        self.P = 0.5 * np.add(self.P, self.P.T)   

        # Publish updated pose and phase
        pose_msg, phase_msg = self.build_pose_phase_msgs()
        self.pub_pose.publish(pose_msg)
        self.pub_phase.publish(phase_msg)

        return phase_msg.data, pose_msg.pose.position
# ----------------------------------------------------------------------
