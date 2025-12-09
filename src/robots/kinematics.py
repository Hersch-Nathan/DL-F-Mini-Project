# Kinematics Functions
# EE599 - Deep Learning Fundamentals
# Hersch Nathan 
# Last Modified 12/07/25

import numpy as np

def forward_kinematics(dh, angle, return_all=False):
    n = angle.size
    robot = dh.copy()
    robot[:, 3] = angle[:]
    
    ctheta = np.cos(robot[:, 3])
    stheta = np.sin(robot[:, 3])
    calpha = np.cos(robot[:, 1])
    salpha = np.sin(robot[:, 1])
    a = robot[:, 0]
    d = robot[:, 2]
    
    A = np.zeros((4, 4, n))
    T = np.zeros((4, 4, n))
    
    for i in range(n):
        A[:, :, i] = np.array([
            [ctheta[i], -stheta[i]*calpha[i], stheta[i]*salpha[i], a[i]*ctheta[i]],
            [stheta[i], ctheta[i]*calpha[i], -ctheta[i]*salpha[i], a[i]*stheta[i]],
            [0, salpha[i], calpha[i], d[i]],
            [0, 0, 0, 1]
        ])
    
    T[:, :, 0] = A[:, :, 0]
    for i in range(1, n):
        T[:, :, i] = T[:, :, i-1] @ A[:, :, i]
    
    if return_all:
        return T[:, :, n-1], T, A
    return T[:, :, n-1]

def inverse_kinematics_3dof_rrr(dh, homo):
    a1 = dh[0, 0]
    a2 = dh[1, 0]
    d3 = dh[2, 2]
    
    px = homo[0, 3]
    py = homo[1, 3]
    pz = homo[2, 3]
    
    r13 = homo[0, 2]
    r23 = homo[1, 2]
    r33 = homo[2, 2]
    
    wx = px - d3 * r13
    wy = py - d3 * r23
    wz = pz - d3 * r33
    
    r = np.sqrt(wx**2 + wy**2)
    s = wz
    
    denominator = 2 * a1 * a2
    if np.abs(denominator) < 1e-10:
        return None
    
    D = (r**2 + s**2 - a1**2 - a2**2) / denominator
    
    if np.abs(D) > 1.0:
        return None
    
    theta2_elbow_down = np.arctan2(np.sqrt(1 - D**2), D)
    theta2_elbow_up = np.arctan2(-np.sqrt(1 - D**2), D)
    
    solutions = []
    
    for theta2 in [theta2_elbow_down, theta2_elbow_up]:
        k1 = a1 + a2 * np.cos(theta2)
        k2 = a2 * np.sin(theta2)
        
        theta1 = np.arctan2(wy, wx) - np.arctan2(k2, k1)
        
        c1 = np.cos(theta1)
        s1 = np.sin(theta1)
        c2 = np.cos(theta2)
        s2 = np.sin(theta2)
        c12 = np.cos(theta1 + theta2)
        s12 = np.sin(theta1 + theta2)
        
        r11 = homo[0, 0]
        r21 = homo[1, 0]
        
        theta3 = np.arctan2(r21 * c12 - r11 * s12, r11 * c12 + r21 * s12)
        
        solutions.append(np.array([theta1, theta2, theta3]))
    
    return solutions

def compute_jacobian(dh, angle):
    n = angle.size
    _, T, A = forward_kinematics(dh, angle, return_all=True)
    
    z = np.zeros((3, n))
    o = np.zeros((3, n))
    
    for i in range(n):
        z[:, i] = T[0:3, 2, i]
        o[:, i] = T[0:3, 3, i]
    
    o0 = np.array([0, 0, 0])
    z0 = z[:, 0]
    on = o[:, n-1]
    
    J = np.zeros((6, n))
    J[:, 0] = np.concatenate([np.cross(z0, (on - o0)), z0])
    
    for i in range(1, n):
        z_curr = z[:, i-1]
        o_curr = o[:, i-1]
        J[:, i] = np.concatenate([np.cross(z_curr, (on - o_curr)), z_curr])
    
    return J

def inverse_kinematics_dls(dh, desired_pos, lamda=0.1, epsilon=1e-6, max_iter=1000):
    angle = np.zeros(dh.shape[0])
    
    for iteration in range(max_iter):
        T = forward_kinematics(dh, angle)
        x_current = T[0:2, 3]
        
        dx = desired_pos - x_current
        
        J = compute_jacobian(dh, angle)
        J_reduced = J[0:2, :]
        
        J_inter = np.linalg.inv(J_reduced @ J_reduced.T + (lamda**2) * np.eye(2))
        dth = J_reduced.T @ J_inter @ dx
        
        if np.linalg.norm(dth) <= epsilon:
            break
        
        angle = angle + dth
    
    return angle

def inverse_kinematics_dls_6dof(dh, target_homo, lamda=0.01, epsilon=1e-6, max_iter=1000, initial_angles=None):
    """6-DOF Damped Least Squares IK solver.
    
    Args:
        dh: DH parameters (nx4)
        target_homo: Target 4x4 homogeneous transformation matrix
        lamda: Damping factor (smaller = less damping, faster convergence)
        epsilon: Convergence threshold
        max_iter: Maximum iterations
        initial_angles: Initial guess for joint angles (if None, uses zeros)
    
    Returns:
        Joint angles that achieve the target pose
    """
    n_joints = dh.shape[0]
    angle = initial_angles if initial_angles is not None else np.random.uniform(-0.1, 0.1, n_joints)
    
    target_pos = target_homo[0:3, 3]
    target_rot = target_homo[0:3, 0:3]
    
    for iteration in range(max_iter):
        T = forward_kinematics(dh, angle)
        current_pos = T[0:3, 3]
        current_rot = T[0:3, 0:3]
        
        # Position error
        pos_error = target_pos - current_pos
        
        # Orientation error (axis-angle representation)
        rot_error_matrix = target_rot @ current_rot.T
        # Extract rotation vector from rotation matrix
        angle_axis = np.arccos(np.clip((np.trace(rot_error_matrix) - 1) / 2, -1, 1))
        if angle_axis > 1e-10:
            axis = np.array([
                rot_error_matrix[2, 1] - rot_error_matrix[1, 2],
                rot_error_matrix[0, 2] - rot_error_matrix[2, 0],
                rot_error_matrix[1, 0] - rot_error_matrix[0, 1]
            ]) / (2 * np.sin(angle_axis))
            ori_error = angle_axis * axis
        else:
            ori_error = np.zeros(3)
        
        # Combined error vector (6D: position + orientation)
        error = np.concatenate([pos_error, ori_error])
        
        # Check convergence
        if np.linalg.norm(error) <= epsilon:
            break
        
        # Compute Jacobian
        J = compute_jacobian(dh, angle)
        
        # Damped least squares update
        J_inter = np.linalg.inv(J.T @ J + (lamda**2) * np.eye(n_joints))
        dth = J_inter @ J.T @ error
        
        # Adaptive step size for stability
        step_size = 0.5 if iteration < 10 else 0.3 if iteration < 50 else 0.1
        angle = angle + step_size * dth
    
    return angle
