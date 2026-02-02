import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm

D = np.zeros((3,3))
mb= 1
g = 9.81
I3 = np.array([0,0,1]).T.reshape((3,1))
w_r = 0 #reference yaw
ca_1 = np.array([np.cos(w_r),np.sin(w_r),0]).T #auxiliar vector 

def generate_reference(prev_states, curr_pos,dt):

    va_r = (curr_pos-prev_states[0:3])/dt
    va_r = np.clip(va_r, -1,1)
    va_r_dot = (va_r-prev_states[3:6])/dt
    Ca_r = R.from_quat([prev_states[9:13]])
    fa_r = mb*va_r_dot +mb*g*I3 #+ Ca_r@D@Ca_r.T@va_r
    # f_T_r = I3.T@Ca_r.inv().apply(fa_r)
    if np.linalg.norm(fa_r) != 0:
        r3 = fa_r.reshape(3,1)/np.linalg.norm(fa_r)
    else:
        r3 = np.zeros((3,1))

    aux = R3_so3(r3)@ca_1;
    if np.linalg.norm(aux) != 0:
        r2 = aux.reshape(3,1)/np.linalg.norm(aux);
    else:
        r2 = np.zeros((3,1))

    r1 = (R3_so3(r2)@r3).reshape(3,1);
    Ca_r_new = np.hstack((r1, r2, r3))
    if np.linalg.det(Ca_r) != 0:
        q_r = R.from_matrix(Ca_r_new)
        Wr_r = so3_R3(np.linalg.inv(Ca_r)@Ca_r_new)/dt
    else:
        q_r = np.array([0,0,0,1])
        Wr_r = np.zeros((3,1))

    # angles = R.from_matrix(Ca_r_new).as_euler('zyx', degrees=False)
    q_r = R.from_matrix(Ca_r_new)
    #Wr_r = Ca_b.T@Ca_r@Wr_r
    return va_r, va_r_dot,q_r, Wr_r

def R3_so3(w):
    v3 = w[2,0]
    v2 = w[1,0]
    v1 = w[0,0]
    so3 = np.array([[ 0 , -v3,  v2],
          [v3,   0, -v1],
          [-v2,  v1,   0]])

    return so3

def so3_R3(log_R):

    #log_R = logm(Rot)
    w1 = log_R[2,1]
    w2 = log_R[0,2]
    w3 = log_R[1,0]
    w = np.array([w1,w2,w3]).T
    return w

def trajectory(r,dt):
    v = np.zeros_like(r)
    v_dot = np.zeros_like(v)
    v[:,0:-1] = np.diff(r,axis=1)/dt
    v_dot[:,0:-2] = np.diff(v[:,:-1],axis=1)/dt
    return v,v_dot



        # fa_r = self.mb*va_r_dot +self.mb*self.g*self.I3 #+ Ca_r@D@Ca_r.T@va_r
        # f_T_r = self.I3.T@self.Ca_r.T@fa_r
        # if np.linalg.norm(fa_r) != 0:
        #     r3 = fa_r.reshape(3,1)/np.linalg.norm(fa_r)
        #     aux = R3_so3(r3)@self.ca_1;
        #     r2 = aux.reshape(3,1)/np.linalg.norm(aux)
        #     r1 = (R3_so3(r2)@r3).reshape(3,1)
        #     Ca_r_new = np.hstack((r1, r2, r3))
        #     Wr_r = so3_R3(np.linalg.inv(self.Ca_r)@Ca_r_new)/dt
        #     Wr_r = so3_R3(np.linalg.inv(self.Ca_r)@Ca_r_new)/dt
        # else:
        #     r3 = np.zeros((3,1))
        #     r2 = np.zeros((3,1))
        #     Wr_r = np.zeros((3,1))
        #     r1 = (R3_so3(r2)@r3).reshape(3,1)
        #     Ca_r_new = np.hstack((r1, r2, r3))
        #     Wr_r = np.zeros((3,1))