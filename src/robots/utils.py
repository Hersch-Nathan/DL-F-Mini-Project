# Utility Functions
# EE599 - Deep Learning Fundamentals
# Hersch Nathan 
# Last Modified 12/07/25

import numpy as np

def deg_to_rad_dh(dh):
    dh_rad = dh.copy()
    dh_rad[:, 1] = np.deg2rad(dh[:, 1])
    dh_rad[:, 3] = np.where(np.isnan(dh[:, 3]), np.nan, np.deg2rad(dh[:, 3]))  
    return dh_rad

def deg_to_rad_angle(angle):
    angle_rad = angle.copy()
    angle_rad[:] = np.deg2rad(angle[:])
    return angle_rad

def homo_to_rpy(homo):
    x = homo[0, 3]
    y = homo[1, 3]
    z = homo[2, 3]
    
    r11 = homo[0, 0]
    r21 = homo[1, 0]
    r31 = homo[2, 0]
    r32 = homo[2, 1]
    r33 = homo[2, 2]
    
    pitch = np.arctan2(-r31, np.sqrt(r11**2 + r21**2))
    
    if np.abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(r32, r33)
        yaw = np.arctan2(r21, r11)
    else:
        roll = np.arctan2(-homo[1, 2], homo[1, 1])
        yaw = 0.0
    
    return np.array([x, y, z, roll, pitch, yaw])

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
