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
    np.random.seed(seed)
    
    dh_rad = deg_to_rad_dh(dh)
    n_joints = dh.shape[0]
    
    joint_angles_list = []
    positions_list = []
    orientations_list = []
    
    angle_min_rad = np.deg2rad(angle_min)
    angle_max_rad = np.deg2rad(angle_max)
    
    num_per_dim = int(np.ceil(num_samples ** (1/n_joints)))
    angle_ranges = [np.linspace(angle_min_rad, angle_max_rad, num_per_dim) for _ in range(n_joints)]
    
    count = 0
    
    def generate_combinations(depth, current_angles):
        nonlocal count
        if count >= num_samples:
            return
        
        if depth == n_joints:
            angles = np.array(current_angles)
            T = forward_kinematics(dh_rad, angles)
            pose = homo_to_rpy(T)
            
            joint_angles_list.append(angles)
            positions_list.append(pose[0:3])
            orientations_list.append(pose[3:6])
            count += 1
            return
        
        for angle in angle_ranges[depth]:
            if count >= num_samples:
                return
            generate_combinations(depth + 1, current_angles + [angle])
    
    generate_combinations(0, [])
    
    return np.array(joint_angles_list), np.array(positions_list), np.array(orientations_list)
