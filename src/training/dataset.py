# Dataset Generation
# EE599 - Deep Learning Fundamentals
# Hersch Nathan 
# Last Modified 12/07/25

import numpy as np
from ..robots.utils import deg_to_rad_dh, homo_to_rpy
from ..robots.kinematics import forward_kinematics

def generate_dataset(dh, num_samples=10000, angle_min=-180, angle_max=180, seed=42):
    np.random.seed(seed)
    
    dh_rad = deg_to_rad_dh(dh)
    n_joints = dh.shape[0]
    
    joint_angles = np.random.uniform(angle_min, angle_max, (num_samples, n_joints))
    joint_angles_rad = np.deg2rad(joint_angles)
    
    positions = np.zeros((num_samples, 3))
    orientations = np.zeros((num_samples, 3))
    
    for i in range(num_samples):
        T = forward_kinematics(dh_rad, joint_angles_rad[i])
        pose = homo_to_rpy(T)
        positions[i] = pose[0:3]
        orientations[i] = pose[3:6]
    
    return joint_angles_rad, positions, orientations

def save_dataset(filename, joint_angles, positions, orientations):
    np.savez(filename, 
             joint_angles=joint_angles, 
             positions=positions, 
             orientations=orientations)

def load_dataset(filename):
    data = np.load(filename)
    return data['joint_angles'], data['positions'], data['orientations']

def generate_consistent_dataset(dh, num_samples=10000, angle_min=-90, angle_max=90, seed=42):
    """Generate dataset with random angles but consistent IK solutions.
    
    Uses random sampling of joint angles (not a grid) to create diverse data,
    while maintaining a one-to-one mapping between poses and joint angles.
    This avoids the multiple IK solution problem while keeping the data realistic.
    """
    np.random.seed(seed)
    
    dh_rad = deg_to_rad_dh(dh)
    n_joints = dh.shape[0]
    
    angle_min_rad = np.deg2rad(angle_min)
    angle_max_rad = np.deg2rad(angle_max)
    
    # Generate random joint angles (not grid-based)
    joint_angles_rad = np.random.uniform(angle_min_rad, angle_max_rad, (num_samples, n_joints))
    
    positions = np.zeros((num_samples, 3))
    orientations = np.zeros((num_samples, 3))
    
    # Compute forward kinematics for each random configuration
    for i in range(num_samples):
        T = forward_kinematics(dh_rad, joint_angles_rad[i])
        pose = homo_to_rpy(T)
        positions[i] = pose[0:3]
        orientations[i] = pose[3:6]
    
    return joint_angles_rad, positions, orientations
