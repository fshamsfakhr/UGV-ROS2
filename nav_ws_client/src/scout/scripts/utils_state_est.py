import numpy as np
import math

import numpy as np
from scipy.spatial.distance import cdist

def traj_optimization(path, s_hat, prms):
    subpath = get_subpath(path, s_hat, prms['r'])
    if subpath.size > 0:
        m = len(prms['InputMat'])
        input_cost = np.zeros(m)
        for i in range(m):
            in_val = prms['InputMat'][i, :]
            input_cost[i] = calc_input_cost(s_hat, in_val, subpath, prms)
        
        cost = np.min(input_cost)
        nmn = np.argmin(input_cost)
        best_input = synth_inputs(prms['InputMat'][nmn, :], prms['n'])
    else:
        best_input = np.array([])  # No valid input
        cost = float('inf')
    
    best_s = np.zeros((prms['n']+1, s_hat.shape[0]))  # n+1 rows, same columns as s0
    best_s[0] = s_hat
    for i in range(prms['n']):
        best_s[i+1] = dd_model(best_s[i], best_input[i], prms['dk'])
  
    return best_s, best_input, cost
    
    
def calc_input_cost(s_hat, u_in, subpath, prms):
    # Generate control inputs using synth_inputs
    u = synth_inputs(u_in, prms['n'])
    
    # Initialize s with s_hat
    s = np.zeros((prms['n'] + 1, len(s_hat)))  # s will be an array with prms.n+1 rows and same number of columns as s_hat
    s[0, :] = s_hat
    
    # Compute the new state using the dynamic model
    for i in range(prms['n']):
        s[i + 1, :] = dd_model(s[i, :], u[i, :], prms['dk'])
    
    # Subsample s according to prms.step
    s = s[::prms['step'], :]

    # Compute input_cost1 as the distance between the final state and the end of the subpath
    input_cost1 = np.linalg.norm(s[-1, :2] - subpath[-1, :2])

    # Calculate the path angle
    dlt = np.diff(subpath[-2:, :], axis=0)
    th_path = np.arctan2(dlt[0, 1], dlt[0, 0])
    
    # Get the heading from the second-to-last state
    th_hat = s[-2, 2]
    
    # Calculate angular difference
    input_cost2 = angdiff(th_path, th_hat)

    # Check if the angular cost exceeds the threshold
    if input_cost2 > 0.3:
        input_cost = float('inf')
    else:
        input_cost = input_cost1
    
    return input_cost

def angdiff(th1, th2):
    return (th2 - th1 + np.pi) % (2 * np.pi) - np.pi
    
def get_subpath(currentpath, s_hat, r):
    # Calculate the Euclidean distance between s_hat's first two elements and each point in currentpath
    distances = np.sqrt(np.sum((s_hat[:2] - currentpath)**2, axis=1))
    
    # Find indices where the distance is less than or equal to r
    idx = np.where(distances <= r)[0]
    
    if len(idx) > 0:
        # Find interruptions in indices
        id_intrupt = np.where(np.diff(idx) > 1)[0]
        
        if len(id_intrupt) > 0:
            # Return subpath from the first index to the first interruption
            subpath = currentpath[idx[0]:idx[id_intrupt[0]] + 1, :]
        else:
            # Return the entire valid subpath
            subpath = currentpath[idx, :]
    else:
        subpath = np.array([])

    return subpath

def synth_inputs(in_val, n):
    # Create an array v of size n with all elements equal to in_val[0]
    v = np.ones(n) * in_val[0]
    
    # Check if in_val[2] (in MATLAB it's in(3)) is true or non-zero
    if in_val[2] != 0:
        # Create w as a linearly spaced array from 0 to in_val[1] of size n
        w = np.linspace(0, in_val[1], n)
    else:
        # Create w as an array of size n with all elements equal to in_val[1]
        w = np.ones(n) * in_val[1]
    
    # Combine v and w into a 2D array and transpose it
    u = np.column_stack((v, w))
    
    return u

def dd_model(s, u, dk):
    v, w = u
    if abs(w) < 0.0009:
        A_k = v * dk
        S_k = np.sin(s[2])
        C_k = np.cos(s[2])
    else:
        A_k = 2 * (v / w) * np.sin((w * dk) / 2)
        S_k = np.sin(s[2] + (w * dk) / 2)
        C_k = np.cos(s[2] + (w * dk) / 2)
        
    s_new = np.copy(s)
    s_new[0] = s[0] + A_k * C_k
    s_new[1] = s[1] + A_k * S_k
    s_new[2] = s[2] + w * dk
    
    return s_new 

def read_points(file_path):
    # Adjust this function to read your points from the file
    data = np.loadtxt(file_path)
    x_min, x_max, y_min, y_max = data[:, 0].min(), data[:, 0].max(), data[:, 1].min(), data[:, 1].max()
    return data[:, :2], x_min, x_max, y_min, y_max
    
def ekf_fun(ss_k, z_k1, v, omega, P, Q, R, dk):
    ss_k = dd_model(ss_k, [v, omega], dk)
    A = dg_dstate([v, omega], ss_k, dk)
    B = dg_dcontrol([v, omega], ss_k, dk)
    P = A @ P @ A.T + B @ Q @ B.T
    ss_k1, P = correction(ss_k, P, z_k1, R)
    return ss_k1, P

def dg_dstate(u, s, dk):
    v = u[0]
    w = u[1]
    if abs(w) < 0.0009:
        a13 = -v * dk * np.sin(s[2])
        a23 = v * dk * np.cos(s[2])
    else:
        a13 = -2 * (v / w) * np.sin(w * dk / 2) * np.sin(s[2] + w * dk / 2)
        a23 = 2 * (v / w) * np.sin(w * dk / 2) * np.cos(s[2] + w * dk / 2)
    
    Jfx = np.array([[1, 0, a13],
                    [0, 1, a23],
                    [0, 0, 1]])
    return Jfx

def dg_dcontrol(u, s, dk):
    v = u[0]
    w = u[1]
    if abs(w) < 0.0009:
        b11 = dk * np.cos(s[2])
        b12 = 0
        b21 = dk * np.sin(s[2])
        b22 = 0
    else:
        b11 = (2 / w) * np.sin(w * dk / 2) * np.cos(s[2] + w * dk / 2)
        b12 = ((((dk / 2) * np.cos(w * dk / 2) * np.cos(s[2] + w * dk / 2) -
                 (dk / 2) * np.sin(s[2] + w * dk / 2) * np.sin(w * dk / 2)) * w -
                 np.sin(w * dk / 2) * np.cos(s[2] + w * dk / 2)) / (w ** 2)) * 2 * v
        b21 = (2 / w) * np.sin(w * dk / 2) * np.sin(s[2] + w * dk / 2)
        b22 = ((((dk / 2) * np.cos(w * dk / 2) * np.sin(s[2] + w * dk / 2) +
                 (dk / 2) * np.cos(s[2] + w * dk / 2) * np.sin(w * dk / 2)) * w -
                 np.sin(w * dk / 2) * np.sin(s[2] + w * dk / 2)) / (w ** 2)) * 2 * v
    
    Jfw = np.array([[b11, b12],
                    [b21, b22],
                    [0, dk]])
    return Jfw

def correction(s, P, z, R):
    H = np.array([[1, 0, 0],
                  [0, 1, 0]])
    
    z_hat = s[:2]
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    innovation = z - z_hat
    s = s + (K @ innovation).T
    I = np.eye(3)
    P = (I - K @ H) @ P
    
    return s, P



# Wrap function
def wrap_to_2pi(angle):
    return angle % (2 * np.pi)
