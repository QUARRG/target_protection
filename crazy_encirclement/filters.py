import numpy as np
from rclpy.node import Node
from typing import Callable
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped, Point, Quaternion, PoseWithCovarianceStamped
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as R
from crazy_encirclement_interfaces.msg import FilterUnicycleState

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

def omega_func_modelF(theta: float) -> np.ndarray:
    return np.asarray([0.4 * (np.cos(theta) * np.sin(theta)-np.sin(theta)**3), 0.4*np.cos(theta)**2*np.sin(-theta), 0.])


REGISTRED_OMEGA_FUNCTIONS = {
    'modelA': omega_func_modelA,
    'modelB': omega_func_modelB,
    'modelC': omega_func_modelC,
    'modelD': omega_func_modelD,
    'modelE': omega_func_modelE,
    'modelF': omega_func_modelF,
}
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def wrap_to_pi(angle):
    """Wrap angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def wrap_to_2pi(angle):
    """Wrap angle to [0, 2pi]."""
    return angle % (2 * np.pi)


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
        Phase angle in radians, wrapped to [0, 2pi]
    """
    return wrap_to_2pi(np.arctan2(Rc[1,0], Rc[0,0]))


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
    gain =  k_p * (1/(error_ahead + eps) + 1/(error_behind + eps))
    control_signal = omega_nominal + gain
    # control_signal = np.clip(control_signal, -2*omega_nominal,2*omega_nominal)

    return control_signal, gain

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
        self.Rc: np.ndarray = build_Rc(wrap_to_2pi(self.params.get('phase_guess', 0.0)))
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
        q_desired: np.ndarray = (Re @ self.Rc @ (self.e_x * self.radius_nominal)).flatten()
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
        #still have to publish xz phase
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
        self.Rc: np.ndarray = build_Rc(wrap_to_2pi(self.params.get('phase_guess', 0.0)))
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
        self.Rc: np.ndarray = build_Rc(wrap_to_2pi(self.params.get('phase_guess', 0.0)))
        self.dt : float = self.params.get('dt', 0.1)
        self.e_x: np.ndarray = np.asarray([[1.], [0.], [0.]])

        # Publishers
        self.pub_omega  = self.node.create_publisher(Float32, f'/{self.name}/baseline/omega', 10)
        self.pub_gain   = self.node.create_publisher(Float32, f'/{self.name}/baseline/controller_gain', 10)
        self.pub_pose   = self.node.create_publisher(PoseWithCovarianceStamped, f'/{self.name}/baseline/pose', 10)
        self.pub_phase  = self.node.create_publisher(Float32, f'/{self.name}/baseline/phase', 10)
        self.pub_radius = self.node.create_publisher(Float32, f'/{self.name}/baseline/radius', 10)
        self.node.info(f'Baseline filter for agent {self.name} initialized.')
    
    def predict(self, current_pose: np.ndarray, phases: list[float]):
        prev_leader_phase, prev_ego_phase, prev_follower_phase = phases
        Re = build_Re(self.embedding_fn, prev_ego_phase)
        Rc = build_Rc(prev_ego_phase)
        p = Rc.T @ Re.T @ current_pose
        self.radius  = np.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
        pose = Re.T @ current_pose
        current_ego_phase = wrap_to_2pi(np.arctan2(pose[1], pose[0]))
        # Rc = build_Rc(current_ego_phase)
        # curr_ego_phase = np.arctan2(p[1], p[0])

        omega, gain = phase_controller(current_ego_phase, prev_leader_phase, prev_follower_phase, self.omega_nominal, self.k_phi)
        # Update phase
        des_ego_pose_2D = np.array([self.radius_nominal*np.cos(current_ego_phase),self.radius_nominal*np.sin(current_ego_phase), 0])
        desired_ego_pose = exp_SO3(np.asarray([0., 0., omega *0.6])) @ des_ego_pose_2D
        desired_ego_phase = wrap_to_2pi(np.arctan2(desired_ego_pose[1], desired_ego_pose[0]))
        des_Re = build_Re(self.embedding_fn, desired_ego_phase)
        desired_ego_pose_3D = des_Re@desired_ego_pose
        
        # Publish predicted pose, phase and controller gain
        current_pose_msg, desired_pose_msg, phase_msg, radius_msg = self.build_pose_phase_msgs()
        phase_msg_test = Float32()
        phase_msg_test.data = current_ego_phase
        # Building desired pose message with nominal radius
        desired_pose_msg = PoseStamped()
        desired_pose_msg.header.frame_id = self.frame_id
        desired_pose_msg.header.stamp = self.node.get_clock().now().to_msg()
        desired_pose_msg.pose.position = Point(x=desired_ego_pose_3D[0], y=desired_ego_pose_3D[1], z=desired_ego_pose_3D[2])
        desired_pose_msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        radius_msg = Float32()
        radius_msg.data = self.radius
        self.pub_pose.publish(current_pose_msg)
        self.pub_phase.publish(phase_msg_test)
        self.pub_radius.publish(radius_msg)

        omega_msg = Float32()
        omega_msg.data = omega
        self.pub_omega.publish(omega_msg)

        gain_msg = Float32()
        gain_msg.data = gain
        self.pub_gain.publish(gain_msg)

        return phase_msg_test, desired_pose_msg


class FilterUnicycle:
    ''' LIEKF for encirclement tasks with 3D ground plane estimation.
        State: [x, y, theta, omega, v, z_ground]
    '''
    def __init__(self, name: str, params: dict, node: Node):
        self.name = name
        self.params = params
        self.node = node

        # Constants
        self.dim_state: int = 6  # Augmented state
        self.dim_meas: int = 3   # 3D Measurement

        # Initialize Covariances
        # P: State Covariance (6x6)
        self.P: np.ndarray = np.diag(np.square(self.params.get('P', np.ones(self.dim_state) * 0.1)))
        
        # Q: Process Noise (6x6) - Add noise for z_ground (usually very small/zero for static)
        self.Q: np.ndarray = np.diag(np.square(self.params.get('Q', np.ones(self.dim_state) * 0.01)))
        
        # V: Measurement Noise (3x3) - Now represents full 3D sensor noise
        self.V: np.ndarray = np.diag(np.square(self.params.get('V', np.ones(self.dim_meas) * 0.1)))

        # Initial state
        self.p: np.ndarray = np.array(self.params.get('position_guess', [0.0, 0.0]), dtype=float) # [x, y]
        self.z_ground: float = float(self.params.get('z_ground_guess', 0.0))                      # z_g
        self.theta: float = float(self.params.get('heading_guess', 0.0))
        self.R: np.ndarray = self._rotm(self.theta)
        
        # Kinematic Parameters
        self.linear_speed: float = float(self.params.get('linear_speed_guess', 0.0))
        self.angular_speed: float = float(self.params.get('angular_speed_guess', 0.0))
        self.zupt_threshold: float = float(self.params.get('zupt_threshold', 0.05))

        # State publisher
        self.pub_pose: Node.Publisher = self.node.create_publisher(FilterUnicycleState, f'/{self.name}/filtered', 10)

        self.I = np.eye(self.dim_state)

    def predict(self, dt: float) -> dict:
        ''' Propagates the state and covariance forward in time. '''

        # --- ZUPT LOGIC START ---
        is_stopped = abs(self.linear_speed) < self.zupt_threshold

        if is_stopped:
            # 1. Force velocity to zero
            self.linear_speed = 0.0
            self.angular_speed = 0.0
            
            # 2. Freeze Covariance
            # We must treat the kinematic parameters as "Fixed Constants" (Zero Noise)
            current_Q = np.zeros_like(self.Q)
            
            # Allow small position noise for GPS corrections
            current_Q[0,0] = self.Q[0,0] # x
            current_Q[1,1] = self.Q[1,1] # y
            current_Q[5,5] = self.Q[5,5] # z_ground
            
            # IMPORTANT: theta(2), omega(3), v(4) must have 0.0 noise
        else:
            current_Q = self.Q
        # --- ZUPT LOGIC END ---
        
        # 1. State Propagation
        # Rotation: R_next = R * Exp(omega * dt)
        R_delta = self._rotm(self.angular_speed * dt)
        self.R = self.R @ R_delta
        self.theta = np.arctan2(self.R[1,0], self.R[0,0])

        # Position: p_next = p + R * [v, 0]' * dt
        # z_ground is constant, so it does not change in prediction
        vel_body = np.array([self.linear_speed, 0.0])
        self.p = self.p + (self.R @ vel_body) * dt

        # 2. Jacobian Calculation (A_t)
        # Order: [xi_x, xi_y, xi_theta, xi_omega, xi_v, xi_z]
        A = np.zeros((self.dim_state, self.dim_state))
        
        # Row 0 (x-error dynamics)
        # x error grows if we have y error while turning (Coriolis)
        # x error grows if we have linear velocity error (xi_v at index 4)
        A[0, 1] = self.angular_speed  # +omega * y
        A[0, 4] = 1.0                 # x_err grows with v_err
        
        # Row 1 (y-error dynamics)
        # y error grows if we have x error while turning (Coriolis)
        # y error grows if we have heading error (xi_theta at index 2) while moving
        A[1, 0] = -self.angular_speed # -omega * x
        A[1, 2] = self.linear_speed   # +v * theta  <-- Critical for lateral drift
        
        # Row 2 (theta-error dynamics)
        # theta error grows if we have angular velocity error (xi_omega at index 3)
        A[2, 3] = 1.0                 # theta_err grows with omega_err
        
        # Row 3 (omega), 4 (v), 5 (z) are 0 (Random Walk / Constant)

        # 3. Covariance Propagation
        F = self.I + A * dt
        self.P = F @ self.P @ F.T + current_Q * dt

        # ---- Publish predicted state ----
        state_msg = FilterUnicycleState()
        state_msg.x = self.p[0]
        state_msg.y = self.p[1]
        state_msg.z_ground = self.z_ground
        state_msg.theta = self.theta
        state_msg.linear_speed = self.linear_speed
        state_msg.angular_speed = self.angular_speed
        state_msg.covariance = self.P.flatten().tolist()
        self.pub_pose.publish(state_msg)
        # ---------------------------------
        
        return self.get_state()

    def update(self,
               y_rel: np.ndarray,
               R_drone: np.ndarray,
               p_drone: np.ndarray) -> None:
        """
        Updates the state using a relative 3D measurement from the drone.
        """
        # 1. Virtual Global Measurement Construction
        # Transform relative 3D vector to Global 3D: y_global = R_drone * y_rel + p_drone
        # Shape: (3,)
        y_global_3d = R_drone @ y_rel + p_drone

        # 2. Innovation Calculation
        # We handle Planar (SE2) and Vertical (R1) separately in the innovation vector
        
        # Planar Innovation: z_xy = R^T * (y_xy - p_est)
        # Projects global position error into the vehicle's body frame
        diff_xy = y_global_3d[:2] - self.p
        innov_xy = self.R.T @ diff_xy
        
        # Vertical Innovation: z_z = y_z - z_ground
        innov_z = y_global_3d[2] - self.z_ground
        
        # Full Innovation Vector (3,)
        z = np.hstack([innov_xy, innov_z])

        # 3. Measurement Jacobian (H)
        # Maps error state [xi_x, xi_y, xi_th, xi_w, xi_v, xi_z] -> [z_x, z_y, z_z]
        H = np.zeros((self.dim_meas, self.dim_state))
        
        # Planar part: Identity block for xi_x, xi_y
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        
        # Vertical part: Identity for xi_z
        H[2, 5] = 1.0

        # 4. Effective Noise Covariance (N_t)
        # Rotate the 3D sensor noise into the Global Frame
        # N_t = R_drone * V_sensor * R_drone^T
        # Note: We do NOT project to 2D anymore; we keep the full 3D noise ellipsoid
        N_t = R_drone @ self.V @ R_drone.T

        # 5. Kalman Gain Calculation
        # S = H P H^T + N_t
        S = H @ self.P @ H.T + N_t
        
        # Numerical stability
        S = 0.5 * (S + S.T) + np.eye(S.shape[0]) * 1e-8
        
        try:
            # Use cholesky solve or lstsq for better stability than inv
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Fallback if S is singular (shouldn't happen with regularization)
            return

        # 6. State Update (Left-Invariant Update)
        xi = K @ z 

        # Decompose correction vector
        xi_p = xi[0:2]       # [x, y] correction (Body Frame)
        xi_theta = xi[2]     # Heading correction
        xi_omega = xi[3]     # Angular speed correction
        xi_v = xi[4]         # Linear speed correction
        xi_z = xi[5]         # Ground height correction

        # --- Geometric Update X <- X * exp(xi) ---
        # Create the Lie algebra element se(2)
        xi_hat = np.array([
            [0, -xi_theta, xi_p[0]],
            [xi_theta, 0, xi_p[1]],
            [0, 0, 0]
        ])
        
        # Compute matrix exponential
        exp_xi = expm(xi_hat)
        
        # Current state as SE(2) matrix
        X = np.array([
            [self.R[0,0], self.R[0,1], self.p[0]],
            [self.R[1,0], self.R[1,1], self.p[1]],
            [0, 0, 1]
        ])
        
        # Apply geometric update: X <- X * exp(xi)
        X_new = X @ exp_xi
        
        # Extract updated position and rotation
        self.p = X_new[0:2, 2]
        self.R = X_new[0:2, 0:2]
        self.theta = np.arctan2(self.R[1,0], self.R[0,0])

        # --- Parameter Update (Additive) ---
        self.angular_speed += xi_omega
        self.linear_speed  += xi_v
        self.z_ground      += xi_z

        # 7. Covariance Update (Josephson Form)
        I_KH = self.I - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ N_t @ K.T
        self.P = 0.5 * (self.P + self.P.T) 

    def get_state(self) -> dict:
        return {
            'position': self.p,
            'z_ground': self.z_ground,
            'heading': self.theta,
            'linear_speed': self.linear_speed,
            'angular_speed': self.angular_speed
        }

    @staticmethod
    def _rotm(theta):
        """Create 2x2 rotation matrix."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])


class FilterSpatial(BaseFilter):
    ''' LIEKF for 3D encirclement using adjoint transformations with y-axis rotation.
        Always uses modelE (identity embedding) and applies spatial rotation via adjoint.
        State: planar phase (Rc around z), spatial rotation (Ry around y), radius
    '''
    def __init__(self, name: str, params: dict, node: Node):
        # Initialize with ModelE (no planar embedding, identity rotation)
        super().__init__(name, 'modelE', params, node)
        
        # Additional state for y-axis rotation (adjoint transformation)
        self.Ry: np.ndarray = self._rotation_y(self.params.get('adjoint_angle_guess', 0.0))
        
        # Extend covariance matrix to include adjoint angle uncertainty
        # Original P is 4x4: [theta_z (3x3), radius (1x1)]
        # New P is 7x7: [theta_z (3x3), theta_y (3x3), radius (1x1)]
        P_adjoint = np.square(self.params.get('P_adjoint', 0.1))
        self.P_spatial = np.zeros((7, 7))
        self.P_spatial[0:3, 0:3] = self.P[0:3, 0:3]  # theta_z covariance
        self.P_spatial[3:6, 3:6] = np.eye(3) * P_adjoint  # theta_y covariance
        self.P_spatial[6, 6] = self.P[3, 3]  # radius covariance
        
        # Extend process noise
        Q_adjoint = np.square(self.params.get('Q_adjoint', 0.01))
        self.Q_spatial = np.zeros((7, 7))
        self.Q_spatial[0:3, 0:3] = self.Q[0:3, 0:3]  # theta_z process noise
        self.Q_spatial[3:6, 3:6] = np.eye(3) * Q_adjoint  # theta_y process noise
        self.Q_spatial[6, 6] = self.Q[3, 3]  # radius process noise
        
        self.node.info(f'Spatial filter for agent {self.name} initialized with adjoint transformation.')

    def _rotation_y(self, angle: float) -> np.ndarray:
        """Create rotation matrix around y-axis for adjoint transformation"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])
    
    def _get_adjoint_angle(self) -> float:
        """Extract rotation angle around y-axis from Ry matrix"""
        return np.arctan2(self.Ry[0, 2], self.Ry[0, 0])

    def predict(self, omega_z: float, omega_y: float, dt: float):
        """Predict state with both planar (z-axis) and spatial (y-axis) angular velocities"""
        
        # Update planar rotation (around z-axis)
        self.Rc = exp_SO3(np.array([0., 0., omega_z * dt])) @ self.Rc
        self.Rc = orthonormalize(self.Rc)
        
        # Update spatial rotation (around y-axis) for adjoint transformation
        self.Ry = exp_SO3(np.array([0., omega_y * dt, 0.])) @ self.Ry
        self.Ry = orthonormalize(self.Ry)
        
        # Predict covariance
        # F has blocks for Rc, Ry, and radius
        F = np.eye(7)
        F[0:3, 0:3] = build_Rc(omega_z * dt)  # Rc dynamics
        F[3:6, 3:6] = self._rotation_y(omega_y * dt)  # Ry dynamics (simplified)
        # radius block remains identity
        
        self.P_spatial = F @ self.P_spatial @ F.T + self.Q_spatial * dt
        self.P_spatial = 0.5 * (self.P_spatial + self.P_spatial.T)  # Ensure symmetry
        
        # Update base P for compatibility with parent methods
        self.P[0:3, 0:3] = self.P_spatial[0:3, 0:3]
        self.P[3, 3] = self.P_spatial[6, 6]
        
        # Publish predicted pose
        current_pose_msg, desired_pose_msg, phase_msg, radius_msg = self.build_pose_phase_msgs()
        self.pub_pose.publish(current_pose_msg)
        self.pub_phase.publish(phase_msg)
        self.pub_radius.publish(radius_msg)
        
        return phase_msg, current_pose_msg, desired_pose_msg

    def update(self, y: np.ndarray):
        """Update state using 3D GPS-like measurement
        
        Args:
            y: 3D measurement in global frame (3x1)
        """
        # Get current state
        phase_z = get_phase(self.Rc)
        Re = build_Re(self.embedding_fn, phase_z)  # ModelE -> identity
        radius = self.radius
        
        # Predicted measurement: Apply adjoint transformation
        # y_hat = Ry @ (Re @ Rc @ e_x * radius)
        # Since Re is identity for modelE: y_hat = Ry @ (Rc @ e_x * radius)
        planar_point = self.Rc @ self.e_x * radius  # Point on planar circle
        y_hat = self.Ry @ planar_point  # Apply adjoint (spatial rotation)
        
        # Measurement Jacobians
        # H = [H_theta_z, H_theta_y, H_r]
        # H_theta_z: derivative w.r.t. planar rotation
        H_theta_z = -self.Ry @ skew(planar_point)  # 3x3
        
        # H_theta_y: derivative w.r.t. spatial rotation (adjoint angle)
        H_theta_y = -skew(self.Ry @ planar_point)  # 3x3
        
        # H_r: derivative w.r.t. radius
        H_r = self.Ry @ (self.Rc @ self.e_x)  # 3x1
        
        # Build full Jacobian (3x7)
        H = np.zeros((3, 7))
        H[0:3, 0:3] = H_theta_z
        H[0:3, 3:6] = H_theta_y
        H[0:3, 6] = H_r.flatten()
        
        # Measurement noise covariance
        V = self.V.copy()
        
        # Kalman gain
        S = H @ self.P_spatial @ H.T + V
        S = 0.5 * (S + S.T) + np.eye(3) * 1e-8
        S_inv = np.linalg.inv(S)
        K = self.P_spatial @ H.T @ S_inv
        
        # Innovation
        z = y - y_hat
        
        # State update
        delta = K @ z.flatten()
        
        # Update Rc (planar rotation)
        theta_z_correction = exp_SO3(delta[0:3])
        self.Rc = theta_z_correction @ self.Rc
        self.Rc = orthonormalize(self.Rc)
        
        # Update Ry (spatial rotation)
        theta_y_correction = exp_SO3(delta[3:6])
        self.Ry = theta_y_correction @ self.Ry
        self.Ry = orthonormalize(self.Ry)
        
        # Update radius
        self.radius += delta[6]
        
        # Covariance update
        I_KH = np.eye(7) - K @ H
        self.P_spatial = I_KH @ self.P_spatial @ I_KH.T + K @ V @ K.T
        self.P_spatial = 0.5 * (self.P_spatial + self.P_spatial.T)
        
        # Update base P for compatibility
        self.P[0:3, 0:3] = self.P_spatial[0:3, 0:3]
        self.P[3, 3] = self.P_spatial[6, 6]
        
        # Publish updated pose
        current_pose_msg, desired_pose_msg, phase_msg, radius_msg = self.build_pose_phase_msgs()
        
        return phase_msg, current_pose_msg, desired_pose_msg

    def build_pose_phase_msgs(self):
        """Override to include adjoint transformation in pose calculation"""
        current_pose_msg = PoseWithCovarianceStamped()
        current_pose_msg.header.frame_id = self.frame_id
        current_pose_msg.header.stamp = self.node.get_clock().now().to_msg()
        
        phase_z = get_phase(self.Rc)
        Re = build_Re(self.embedding_fn, phase_z)  # ModelE -> identity
        radius = self.radius
        
        # Apply adjoint transformation: Ry @ (Re @ Rc @ e_x * radius)
        planar_point = self.Rc @ (self.e_x * radius)
        q = (self.Ry @ planar_point).flatten()
        
        current_pose_msg.pose.pose.position = Point(x=q[0], y=q[1], z=q[2])
        current_pose_msg.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        
        # Fill covariance (spatial: 7x7 -> 6x6 pose covariance)
        cov = np.zeros((6, 6))
        # Map spatial covariance to pose covariance
        # Position covariance comes from radius uncertainty
        cov[0, 0] = self.P_spatial[6, 6]  # x from radius
        # Rotation covariances
        cov[3:6, 3:6] = self.P_spatial[0:3, 0:3]  # Rc covariance
        current_pose_msg.pose.covariance = cov.flatten().tolist()
        
        phase_msg = Float32()
        phase_msg.data = phase_z
        
        radius_msg = Float32()
        radius_msg.data = radius
        
        # Desired pose with nominal radius
        desired_pose_msg = PoseStamped()
        desired_pose_msg.header.frame_id = self.frame_id
        desired_pose_msg.header.stamp = self.node.get_clock().now().to_msg()
        planar_point_nominal = self.Rc @ (self.e_x * self.radius_nominal)
        q_desired = (self.Ry @ planar_point_nominal).flatten()
        desired_pose_msg.pose.position = Point(x=q_desired[0], y=q_desired[1], z=q_desired[2])
        desired_pose_msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        
        return current_pose_msg, desired_pose_msg, phase_msg, radius_msg

        
class FilterSpatialBaseline:
    ''' Baseline filter for 3D encirclement using adjoint transformations without state estimation.
        Always uses modelE (identity embedding). The adjoint rotation (Ry) is predefined.
        Computes planar phase directly from measurements and applies adjoint transformation.
    '''
    def __init__(self, name: str, params: dict, node: Node):
        self.name = name
        self.params = params
        self.node: Node = node

        # Initialize parameters
        self.k_phi_z: float = self.params.get('k_phi_z', 0.5)
        self.k_phi_y: float = self.params.get('k_phi_y', 0.5)
        self.omega_z_nominal: float = self.params.get('omega_z_nominal', 0.5)
        self.omega_y_nominal: float = self.params.get('omega_y_nominal', 0.0)
        self.frame_id: str = self.params.get('frame_id', 'world')
        self.radius: float = self.params.get('radius_guess', 2.0)
        self.radius_nominal: float = self.params.get('radius_nominal', 2.0)

        # Predefined y-axis rotation for adjoint transformation
        self.adjoint_angle: float = self.params.get('adjoint_angle', 0.0)
        self.Ry: np.ndarray = self._rotation_y(self.adjoint_angle)
        
        # Planar rotation matrix (will be updated based on measured phase)
        self.Rc: np.ndarray = build_Rc(wrap_to_2pi(self.params.get('phase_guess', 0.0)))
        self.dt: float = self.params.get('dt', 0.1)
        self.e_x: np.ndarray = np.asarray([[1.], [0.], [0.]])

        # Publishers
        self.pub_omega_z = self.node.create_publisher(Float32, f'/{self.name}/spatial_baseline/omega_z', 10)
        self.pub_omega_y = self.node.create_publisher(Float32, f'/{self.name}/spatial_baseline/omega_y', 10)
        self.pub_k_phi_z = self.node.create_publisher(Float32, f'/{self.name}/spatial_baseline/k_phi_z', 10)
        self.pub_k_phi_y = self.node.create_publisher(Float32, f'/{self.name}/spatial_baseline/k_phi_y', 10)
        self.pub_pose    = self.node.create_publisher(PoseWithCovarianceStamped, f'/{self.name}/spatial_baseline/pose', 10)
        self.pub_phase   = self.node.create_publisher(Float32, f'/{self.name}/spatial_baseline/phase', 10)
        self.pub_adjoint_angle = self.node.create_publisher(Float32, f'/{self.name}/spatial_baseline/adjoint_angle', 10)
        self.pub_radius  = self.node.create_publisher(Float32, f'/{self.name}/spatial_baseline/radius', 10)
        self.node.info(f'Spatial baseline filter for agent {self.name} initialized with predefined adjoint angle={self.adjoint_angle:.3f} rad.')

    def _rotation_y(self, angle: float) -> np.ndarray:
        """Create rotation matrix around y-axis for adjoint transformation"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])

    def predict(self, current_pose: np.ndarray, phases: list[float]):
        """Compute desired position using direct measurements (no filtering)
        
        Args:
            current_pose: Current 3D position measurement (3,)
            phases: [leader_phase, ego_phase, follower_phase] for planar controller
        
        Returns:
            phase_msg: Current planar phase
            desired_pose_msg: Desired position with adjoint transformation
        """
        prev_leader_phase, prev_ego_phase, prev_follower_phase = phases

        # Updating adjoint rotation
        self.Ry = self._rotation_y(self.adjoint_angle)
        
        # Compute current state from measurement
        # First, invert the adjoint transformation to get planar position
        # current_pose = Ry @ (Rc @ e_x * radius)
        # Ry^T @ current_pose = Rc @ e_x * radius
        planar_pose = self.Ry.T @ current_pose
        
        # Compute radius from planar pose
        self.radius = np.linalg.norm(planar_pose)
        
        # Compute planar phase from projected position
        current_ego_phase = wrap_to_2pi(np.arctan2(planar_pose[1], planar_pose[0]))
        
        # Planar phase controller (around z-axis)
        omega_z, gain_z = phase_controller(current_ego_phase, prev_leader_phase, prev_follower_phase, 
                                           self.omega_z_nominal, self.k_phi_z)
        
        # Spatial rotation control (around y-axis) - can be extended for 3D coordination
        # omega_y = self.omega_y_nominal
        
        # Update planar rotation for desired pose
        # Rc_desired = build_Rc(current_ego_phase)
        
        # Compute desired pose in planar coordinates
        des_ego_pose_2D = np.array([
            self.radius_nominal * np.cos(current_ego_phase),
            self.radius_nominal * np.sin(current_ego_phase),
            0.0
        ])
        
        # Apply planar rotation control
        desired_planar_pose = exp_SO3(np.array([0., 0., omega_z])) @ des_ego_pose_2D
        
        # Apply adjoint transformation to get 3D desired pose
        desired_ego_pose_3D = self.Ry @ desired_planar_pose
        
        # Build messages
        current_pose_msg, desired_pose_msg, phase_msg, radius_msg = self.build_pose_phase_msgs(
            current_pose, current_ego_phase)
        
        phase_msg_current = Float32()
        phase_msg_current.data = current_ego_phase
        
        # Build desired pose message
        desired_pose_msg = PoseStamped()
        desired_pose_msg.header.frame_id = self.frame_id
        desired_pose_msg.header.stamp = self.node.get_clock().now().to_msg()
        desired_pose_msg.pose.position = Point(
            x=desired_ego_pose_3D[0],
            y=desired_ego_pose_3D[1],
            z=desired_ego_pose_3D[2]
        )
        desired_pose_msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        
        # Publish all topics
        self.pub_pose.publish(current_pose_msg)
        self.pub_phase.publish(phase_msg_current)
        self.pub_adjoint_angle.publish(Float32(data=self.adjoint_angle))
        self.pub_radius.publish(Float32(data=self.radius))
        self.pub_omega_z.publish(Float32(data=omega_z))
        self.pub_k_phi_z.publish(Float32(data=gain_z))
        
        return phase_msg_current, desired_pose_msg

    def build_pose_phase_msgs(self, current_pose: np.ndarray, current_phase: float):
        """Build pose and phase messages for publishing
        
        Args:
            current_pose: Current 3D position (3,)
            current_phase: Current planar phase angle
        
        Returns:
            current_pose_msg, desired_pose_msg, phase_msg, radius_msg
        """
        # Current pose message
        current_pose_msg = PoseWithCovarianceStamped()
        current_pose_msg.header.frame_id = self.frame_id
        current_pose_msg.header.stamp = self.node.get_clock().now().to_msg()
        current_pose_msg.pose.pose.position = Point(
            x=current_pose[0],
            y=current_pose[1],
            z=current_pose[2]
        )
        current_pose_msg.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        current_pose_msg.pose.covariance = np.zeros(36).tolist()
        
        # Phase message
        phase_msg = Float32()
        phase_msg.data = current_phase
        
        # Radius message
        radius_msg = Float32()
        radius_msg.data = self.radius
        
        # Desired pose (computed in predict, placeholder here)
        desired_pose_msg = PoseStamped()
        desired_pose_msg.header.frame_id = self.frame_id
        desired_pose_msg.header.stamp = self.node.get_clock().now().to_msg()
        
        return current_pose_msg, desired_pose_msg, phase_msg, radius_msg


# ----------------------------------------------------------------------

class Baseline3DFilter(BaseFilter):
    ''' Baseline filter without state estimation for encirclement tasks.
    '''
    def __init__(self, index, name: str, embedding_fn_name: str, params: dict, node: Node):
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
        self.Rc: np.ndarray = build_Rc(wrap_to_2pi(self.params.get('phase_guess', 0.0)))
        self.Re = R.from_euler('y',index*2*np.pi/3).as_matrix()
        self.normal = self.Re@np.array([0., 0., 1.])
        self.dt : float = self.params.get('dt', 0.1)
        self.e_x: np.ndarray = np.asarray([[1.], [0.], [0.]])

        # Publishers
        self.pub_omega  = self.node.create_publisher(Float32, f'/{self.name}/baseline/omega', 10)
        self.pub_gain   = self.node.create_publisher(Float32, f'/{self.name}/baseline/controller_gain', 10)
        self.pub_pose   = self.node.create_publisher(PoseWithCovarianceStamped, f'/{self.name}/baseline/pose', 10)
        self.pub_phase  = self.node.create_publisher(Float32, f'/{self.name}/baseline/phase', 10)
        self.pub_phase_normal  = self.node.create_publisher(Float32, f'/{self.name}/baseline/phase_normal', 10)
        self.pub_radius = self.node.create_publisher(Float32, f'/{self.name}/baseline/radius', 10)
        self.node.info(f'Baseline filter for agent {self.name} initialized.')
    
    def predict(self, current_pose: np.ndarray, phases: list[float]):
        prev_leader_phase, prev_ego_phase, prev_follower_phase = phases
        # Rc = build_Rc(prev_ego_phase)
        # p = Rc.T @ Re.T @ current_pose
        # self.radius  = np.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
        # self.radius  = np.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
     
        pose = self.Re.T @ current_pose
        current_ego_phase = wrap_to_2pi(np.arctan2(pose[1], pose[0]))
        current_ego_phase_normal = wrap_to_2pi(np.arctan2(self.normal[2],self.normal[0]))
        # Rc = build_Rc(current_ego_phase)
        # curr_ego_phase = np.arctan2(p[1], p[0])

        omega_z, gain_z = phase_controller(current_ego_phase, prev_leader_phase, prev_follower_phase, self.omega_nominal, self.k_phi)

        # Update phase
        des_ego_pose_2D = np.array([self.radius_nominal*np.cos(current_ego_phase),self.radius_nominal*np.sin(current_ego_phase), 0])
        desired_ego_pose = exp_SO3(np.asarray([0., 0., omega_z *0.6])) @ des_ego_pose_2D
        # Re = build_Re(self.embedding_fn, current_ego_phase)
        # desired_ego_phase = wrap_to_2pi(np.arctan2(desired_ego_pose[1], desired_ego_pose[0]))
        # self.Re = exp_SO3(np.asarray([0., omega_y*0.6, 0.])) @ self.Re
        desired_ego_pose_3D = self.Re@desired_ego_pose
        
        # Publish predicted pose, phase and controller gain
        current_pose_msg, desired_pose_msg, phase_msg, radius_msg = self.build_pose_phase_msgs()
        phase_msg_test = Float32()
        phase_msg_test.data = current_ego_phase
        msg_normal = Float32()
        msg_normal.data = current_ego_phase_normal

        # Building desired pose message with nominal radius
        desired_pose_msg = PoseStamped()
        desired_pose_msg.header.frame_id = self.frame_id
        desired_pose_msg.header.stamp = self.node.get_clock().now().to_msg()
        desired_pose_msg.pose.position = Point(x=desired_ego_pose_3D[0], y=desired_ego_pose_3D[1], z=desired_ego_pose_3D[2])
        desired_pose_msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        radius_msg = Float32()
        radius_msg.data = self.radius
        self.pub_pose.publish(current_pose_msg)
        self.pub_phase.publish(phase_msg_test)
        self.pub_phase_normal.publish(msg_normal)
        self.pub_radius.publish(radius_msg)

        omega_msg = Float32()
        omega_msg.data = omega_z
        self.pub_omega.publish(omega_msg)

        gain_msg = Float32()
        gain_msg.data = gain_z
        self.pub_gain.publish(gain_msg)

        return phase_msg_test, desired_pose_msg