import numpy as np
from rclpy.node import Node
from typing import Callable
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped, Point, Quaternion, PoseWithCovarianceStamped, PoseWithCovariance
from scipy.linalg import expm


# ----------------------------------------------------------------------
# Embedding Functions
# ----------------------------------------------------------------------
def omega_func_modelA(theta: float) -> np.ndarray:
    return np.asarray([0.3 * np.sin(6 * theta) * np.cos(6 * theta), 0.3, 0.])

def omega_func_modelB(theta: float) -> np.ndarray:
    return np.asarray([0.9 * np.sin(theta)*np.cos(theta), 0., 0.])

def omega_func_modelC(theta: float) -> np.ndarray:
    return np.asarray([0.6 * np.cos(2 * theta), 0.6 * np.cos(theta)**2, 0.])

def omega_func_modelD(theta: float) -> np.ndarray:
    return np.asarray([0.2 * np.cos(3 * theta) * np.sin(theta), 0.5 * 0.9, 0.])

def omega_func_modelE(theta: float) -> np.ndarray:
    return np.asarray([0.0, 0.0, 0.])
REGISTRED_OMEGA_FUNCTIONS = {
    'modelA': omega_func_modelA,
    'modelB': omega_func_modelB,
    'modelC': omega_func_modelC,
    'modelD': omega_func_modelD,
    'modelE': omega_func_modelE,
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


def build_Rc(phase: float) -> np.ndarray:
    """Build rotation matrix from phase angle.
    
    Args:
        phase: Phase angle in radians
    
    Returns:
        3x3 rotation matrix
    """
    phase = wrap_to_pi(phase)
    c, s = np.cos(phase), np.sin(phase)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])


def build_Re(embedding_func: Callable[[float], np.ndarray], phase: float) -> np.ndarray:
    """Build embedding rotation matrix.
    
    Args:
        embedding_func: Embedding function that takes phase and returns omega vector
        phase: Phase angle in radians
    
    Returns:
        3x3 rotation matrix
    """
    return exp_SO3(embedding_func(phase))


def get_phase(Rc: np.ndarray) -> float:
    """Extract phase angle from rotation matrix.
    
    Args:
        Rc: 3x3 rotation matrix
    
    Returns:
        Phase angle in radians, wrapped to [-pi, pi]
    """
    return wrap_to_pi(np.arctan2(Rc[1,0], Rc[0,0]))


def phase_controller(phase_ego, phase_leader, phase_follower, omega_nominal, k_p=1.0):
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

    control_signal = omega_nominal + k_p * (1/(error_ahead + eps) + 1/(error_behind + eps))

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
        self.P: np.ndarray  = np.diag(np.square(self.params.get('P', np.zeros(4))))
        self.Q: np.ndarray  = np.diag(np.square(self.params.get('Q', np.zeros(4))))
        self.V: np.ndarray  = np.diag(np.square(self.params.get('V', np.zeros(3))))
        self.Rc: np.ndarray = build_Rc(wrap_to_pi(self.params.get('phase_guess', 0.0)))
        self.radius: float  = self.params.get('radius_guess', 2.0)
        self.radius_nominal: float = self.params.get('radius_nominal', 2.0)
        self.e_x: np.ndarray = np.asarray([[1.], [0.], [0.]])

        # Checking embedding function
        if embedding_fn_name not in REGISTRED_OMEGA_FUNCTIONS:
            raise ValueError(f"Embedding function '{embedding_fn_name}' is not allowed. Choose from: {list(REGISTRED_OMEGA_FUNCTIONS.keys())}")
        self.embedding_fn: Callable = REGISTRED_OMEGA_FUNCTIONS[embedding_fn_name]

        # Publishers
        self.frame_id: str = self.params.get('frame_id', 'world')
        self.pub_pose: Node.Publisher   = self.node.create_publisher(PoseWithCovarianceStamped, f'/{self.name}/filtered/pose', 10)
        self.pub_phase: Node.Publisher  = self.node.create_publisher(Float32, f'/{self.name}/filtered/phase', 10)
        self.pub_radius: Node.Publisher = self.node.create_publisher(Float32, f'/{self.name}/filtered/radius', 10)
        self.node.info(f'Filter for agent {self.name} initialized with embedding function {embedding_fn_name}.')

    def build_pose_phase_msgs(self) -> list[PoseWithCovarianceStamped, PoseStamped, Float32, Float32]:
        # Build PoseStamped and Float32 messages for current pose, phase and radius
        current_pose_msg = PoseWithCovarianceStamped()
        current_pose_msg.header.frame_id = self.frame_id
        current_pose_msg.header.stamp = self.node.get_clock().now().to_msg()
        phase: float = get_phase(self.Rc)
        Re: np.ndarray = build_Re(self.embedding_fn, phase)
        Rc: np.ndarray = build_Rc(phase)
        radius: float = self.radius
        q: np.ndarray = (Re @ Rc @ (self.e_x * radius)).flatten()
        current_pose_msg.pose.pose.position = Point(x=q[0], y=q[1], z=q[2])
        current_pose_msg.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        # Fill covariance veryfing the size of P diagonal
        if hasattr(self, 'P'):
            if len(self.P.diagonal()) == 4:
                cov = np.zeros((6,6))
                cov[0, 0] = self.P[3, 3]
                cov[3:6, 0] = self.P[0:3, 3]
                cov[0, 3:6] = self.P[3, 0:3]
                cov[3:6, 3:6] = self.P[0:3, 0:3]
                current_pose_msg.pose.covariance = cov.flatten().tolist()
            elif len(self.P.diagonal()) == 3:
                cov = np.zeros((6,6))
                cov[3:6, 3:6] = self.P[0:3, 0:3]
                current_pose_msg.pose.covariance = cov.flatten().tolist()
        else:
            current_pose_msg.pose.covariance = np.zeros(36).tolist()
        phase_msg = Float32()
        phase_msg.data = phase
        radius_msg = Float32()
        radius_msg.data = radius

        # Building desired pose message with nominal radius
        desired_pose_msg = PoseStamped()
        desired_pose_msg.header.frame_id = self.frame_id
        desired_pose_msg.header.stamp = self.node.get_clock().now().to_msg()
        q_desired: np.ndarray = (Re @ Rc @ (self.e_x * self.radius_nominal)).flatten()
        desired_pose_msg.pose.position = Point(x=q_desired[0], y=q_desired[1], z=q_desired[2])
        desired_pose_msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        return current_pose_msg, desired_pose_msg, phase_msg, radius_msg
    
    def predict(self, omega_z: float, dt: float):
        # Update theta based on omega_z and time step
        self.Rc = exp_SO3(np.asarray([0., 0., omega_z * dt])) @ self.Rc
        self.Rc = orthonormalize(self.Rc)
        # self.r = self.r  # constant radius
        
        # Predict the next covariance
        F = np.eye(4)
        F[0:3, 0:3] = build_Rc(omega_z * dt)
        Q = self.Q.copy()
        # Q[3, 3] = ((Q[3, 3]**0.5) / np.exp(self.s)) ** 2
        self.P = F @ self.P @ F.T + Q * dt
        self.P = (self.P + self.P.T) / 2  # Ensure symmetry

        # Publish predicted pose and phase
        current_pose_msg, desired_pose_msg, phase_msg, radius_msg = self.build_pose_phase_msgs()
        self.pub_pose.publish(current_pose_msg)
        self.pub_phase.publish(phase_msg)
        self.pub_radius.publish(radius_msg)
        # self.node.get_logger().info(f'Published pose for agent {self.name}')

        return phase_msg, current_pose_msg, desired_pose_msg


class FilterGPS(BaseFilter):
    ''' LIEKF for encirclement tasks using GPS-like measurements.
    '''
    def __init__(self, name: str, embedding_fn_name: str, params: dict, node: Node):
        super().__init__(name, embedding_fn_name, params, node)

    def update(self, y: np.ndarray):
        # Measurement Jacobian
        radius: float = self.radius
        Re: np.ndarray = build_Re(self.embedding_fn, get_phase(self.Rc))
        H_theta: np.ndarray = -self.Rc.T @ (Re @ self.Rc @ skew(self.e_x * radius ))   # body frame
        H_r: np.ndarray = (self.Rc.T @ (Re @ self.Rc @ self.e_x ))                     # inertial frame
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
        self.radius += delta[3]

        # Update covariance
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ V @ K.T
        self.P = 0.5 * np.add(self.P, self.P.T)   

        # Publish updated pose and phase
        current_pose_msg, desired_pose_msg, phase_msg, radius_msg = self.build_pose_phase_msgs()
        # self.pub_pose.publish(current_pose_msg)
        # self.pub_phase.publish(phase_msg)
        # self.pub_radius.publish(radius_msg)

        return phase_msg, current_pose_msg, desired_pose_msg


class FilterRelative(BaseFilter):
    ''' LIEKF for encirclement tasks using GPS-Scanner-like measurements.
    '''
    def __init__(self, name: str, embedding_fn_name: str, params: dict, node: Node):
        self.name = name
        self.embedding_fn_name = embedding_fn_name
        self.params = params
        self.node: Node = node

        # Checking embedding function
        if embedding_fn_name not in REGISTRED_OMEGA_FUNCTIONS:
            raise ValueError(f"Embedding function '{embedding_fn_name}' is not allowed. Choose from: {list(REGISTRED_OMEGA_FUNCTIONS.keys())}")
        self.embedding_fn: Callable = REGISTRED_OMEGA_FUNCTIONS[embedding_fn_name]

        # Initialize filter parameters         
        self.P: np.ndarray  = np.diag(np.square(self.params.get('P', np.zeros(3))))
        self.Q: np.ndarray  = np.diag(np.square(self.params.get('Q', np.zeros(3))))
        self.V: np.ndarray  = np.diag(np.square(self.params.get('V', np.zeros(3))))
        self.Rc: np.ndarray = build_Rc(wrap_to_pi(self.params.get('phase_guess', 0.0)))
        self.radius: float  = self.params.get('radius_guess', 2.0)
        self.radius_nominal: float = self.params.get('radius_nominal', 2.0)
        self.s: float = np.log(self.radius)
        self.e_x: np.ndarray = np.asarray([[1.], [0.], [0.]])

    def predict(self, omega_z: float, dt: float) -> float:
        # Update theta based on omega_z and time step
        self.Rc = exp_SO3(np.asarray([0., 0., omega_z * dt])) @ self.Rc
        self.Rc = orthonormalize(self.Rc)
        # self.r = self.r  # constant radius
        
        # Predict the next covariance
        F = build_Rc(omega_z * dt)
        Q = self.Q.copy()
        self.P = F @ self.P @ F.T + Q * dt
        self.P = (self.P + self.P.T) / 2  # Ensure symmetry

        return Float32(data=get_phase(self.Rc))

    def update(self, y: np.ndarray, Rei: np.ndarray, Rci: np.ndarray, qi: np.ndarray):
        # Measurement Jacobian
        Rck = self.Rc.copy()
        Rek = build_Re(self.embedding_fn, get_phase(Rck))
        H = -Rck.T @ Rci.T @ Rei.T @ Rek @ Rck @ skew(self.e_x * self.radius_nominal)

        # Kalman Gain
        V = Rck.T @ self.V @ Rck
        S = H @ self.P @ H.T + V
        IdS = np.eye(S.shape[0])
        S = 0.5 * np.add(S, S.T) + IdS * 1e-8
        S_inv = np.linalg.inv(S)
        K = self.P @ H.T @ S_inv

        # Update state
        y_hat = Rci.T @ Rei.T @ (Rek @ Rck @ (self.e_x * self.radius_nominal) - qi)
        z = Rck.T @ (y - y_hat)
        # print(f"NIS: {np.squeeze(z.T @ S_inv @ z).item():.3f}")

        delta = K @ z.flatten()                   # Correction vector in the algebra
        theta_correction = exp_SO3(delta)         # Exponential map to group element
        Rck = theta_correction @ Rck      
        Rck = orthonormalize(Rck)
        self.Rc = Rck

        # Update covariance
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ V @ K.T
        self.P = 0.5 * np.add(self.P, self.P.T)
        
        return Float32(data=get_phase(self.Rc))


class BaselineFilter(BaseFilter):
    ''' Baseline filter without state estimation for encirclement tasks.
    '''
    def __init__(self, name: str, embedding_fn_name: str, params: dict, node: Node):
        self.name = name
        self.embedding_fn_name = embedding_fn_name
        self.params = params
        self.node: Node = node

        # Checking embedding function
        if embedding_fn_name not in REGISTRED_OMEGA_FUNCTIONS:
            raise ValueError(f"Embedding function '{embedding_fn_name}' is not allowed. Choose from: {list(REGISTRED_OMEGA_FUNCTIONS.keys())}")
        self.embedding_fn: Callable = REGISTRED_OMEGA_FUNCTIONS[embedding_fn_name]

        # Initialize parameters         
        self.k_phi: float = self.params.get('k_phi', 0.5)
        self.omega_nominal: float = self.params.get('omega_nominal', 0.5)
        self.frame_id: str = self.params.get('frame_id', 'world')
        self.radius: float = self.params.get('radius_guess', 2.0)
        self.radius_nominal: float = self.params.get('radius_nominal', 2.0)

        self.s: float = np.log(self.radius)
        self.Rc: np.ndarray = build_Rc(wrap_to_pi(self.params.get('phase_guess', 0.0)))
        self.dt : float = self.params.get('dt', 0.1)
        self.e_x: np.ndarray = np.asarray([[1.], [0.], [0.]])

        # Publishers
        self.pub_omega  = self.node.create_publisher(Float32, f'/{self.name}/baseline/omega', 10)
        self.pub_pose   = self.node.create_publisher(PoseWithCovarianceStamped, f'/{self.name}/baseline/pose', 10)
        self.pub_phase  = self.node.create_publisher(Float32, f'/{self.name}/baseline/phase', 10)
        self.pub_radius = self.node.create_publisher(Float32, f'/{self.name}/baseline/radius', 10)
        self.node.info(f'Baseline filter for agent {self.name} initialized.')
    
    def predict(self, current_pose: np.ndarray, phases: list[float]):
        prev_leader_phase, prev_ego_phase, prev_follower_phase = phases
        Re = build_Re(self.embedding_fn, prev_ego_phase)
        Rc = build_Rc(prev_ego_phase)
        p = Rc.T @ Re.T @ current_pose
        # curr_ego_phase = np.arctan2(p[1], p[0])
        self.radius  = np.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
        omega = phase_controller(prev_ego_phase, prev_leader_phase, prev_follower_phase, self.omega_nominal, self.k_phi)
        # Update phase
        self.Rc = exp_SO3(np.asarray([0., 0., omega * self.dt])) @ self.Rc
        self.Rc = orthonormalize(self.Rc)
        
        # Publish predicted pose and phase
        current_pose_msg, desired_pose_msg, phase_msg, radius_msg = self.build_pose_phase_msgs()
        self.pub_pose.publish(current_pose_msg)
        self.pub_phase.publish(phase_msg)
        self.pub_radius.publish(radius_msg)

        omega_msg = Float32()
        omega_msg.data = omega
        self.pub_omega.publish(omega_msg)

        return phase_msg, desired_pose_msg
# ----------------------------------------------------------------------