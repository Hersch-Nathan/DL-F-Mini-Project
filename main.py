# Main File
# EE599 - Deep Learning Fundamentals
# Hersch Nathan 
# Last Modified 12/07/25

import numpy as np
import torch
import time
from torch.utils.data import TensorDataset, DataLoader, random_split

from config import RRR_dh, RRRRRR_dh, NUM_SAMPLES, ANGLE_MIN, ANGLE_MAX, RRR_SEED, RRRRRR_SEED, TEST_SPLIT, GENERATE_RRR_DATASET, GENERATE_RRRRRR_DATASET
from utils import deg_to_rad_dh, deg_to_rad_angle, homo_to_rpy
from kinematics import forward_kinematics, inverse_kinematics_3dof_rrr, compute_jacobian, inverse_kinematics_dls
from dataset import generate_dataset, save_dataset, load_dataset
from models import RRR_Linear, TejomurtKak_Model, train_model, train_tejomurt_model, evaluate_model

def main():
    if GENERATE_RRR_DATASET:
        print("Generating RRR dataset")
        rrr_angles, rrr_pos, rrr_orient = generate_dataset(
            RRR_dh, 
            num_samples=NUM_SAMPLES, 
            angle_min=ANGLE_MIN, 
            angle_max=ANGLE_MAX, 
            seed=RRR_SEED
        )
        save_dataset('data/rrr_dataset.npz', rrr_angles, rrr_pos, rrr_orient)
        print(f"RRR dataset saved: {NUM_SAMPLES} samples")
    
    if GENERATE_RRRRRR_DATASET:
        print("Generating RRRRRR dataset")
        rrrrrr_angles, rrrrrr_pos, rrrrrr_orient = generate_dataset(
            RRRRRR_dh, 
            num_samples=NUM_SAMPLES, 
            angle_min=ANGLE_MIN, 
            angle_max=ANGLE_MAX, 
            seed=RRRRRR_SEED
        )
        save_dataset('data/rrrrrr_dataset.npz', rrrrrr_angles, rrrrrr_pos, rrrrrr_orient)
        print(f"RRRRRR dataset saved: {NUM_SAMPLES} samples")
    
    print("\nLoading datasets with torch...")
    
    rrr_angles_load, rrr_pos_load, rrr_orient_load = load_dataset('data/rrr_dataset.npz')
    
    dh_rad = deg_to_rad_dh(RRR_dh)
    dh_params = dh_rad[:, :3].flatten()
    dh_params_repeated = np.tile(dh_params, (len(rrr_pos_load), 1))
    
    rrr_inputs_raw = np.concatenate([rrr_pos_load, rrr_orient_load, dh_params_repeated], axis=1)
    
    input_mean = rrr_inputs_raw.mean(axis=0)
    input_std = rrr_inputs_raw.std(axis=0) + 1e-8
    rrr_inputs_norm = (rrr_inputs_raw - input_mean) / input_std
    
    output_mean = rrr_angles_load.mean(axis=0)
    output_std = rrr_angles_load.std(axis=0) + 1e-8
    rrr_outputs_norm = (rrr_angles_load - output_mean) / output_std
    
    rrr_inputs = torch.tensor(rrr_inputs_norm, dtype=torch.float32)
    rrr_outputs = torch.tensor(rrr_outputs_norm, dtype=torch.float32)
    rrr_dataset = TensorDataset(rrr_inputs, rrr_outputs)
    
    test_size = int(TEST_SPLIT * len(rrr_dataset))
    train_size = len(rrr_dataset) - test_size
    rrr_train, rrr_test = random_split(rrr_dataset, [train_size, test_size])
    print(f"RRR: {len(rrr_train)} train, {len(rrr_test)} test")
    
    rrrrrr_angles_load, rrrrrr_pos_load, rrrrrr_orient_load = load_dataset('data/rrrrrr_dataset.npz')
    rrrrrr_inputs = torch.tensor(np.concatenate([rrrrrr_pos_load, rrrrrr_orient_load], axis=1), dtype=torch.float32)
    rrrrrr_outputs = torch.tensor(rrrrrr_angles_load, dtype=torch.float32)
    rrrrrr_dataset = TensorDataset(rrrrrr_inputs, rrrrrr_outputs)
    
    test_size = int(TEST_SPLIT * len(rrrrrr_dataset))
    train_size = len(rrrrrr_dataset) - test_size
    rrrrrr_train, rrrrrr_test = random_split(rrrrrr_dataset, [train_size, test_size])
    print(f"RRRRRR: {len(rrrrrr_train)} train, {len(rrrrrr_test)} test")
    
    print("\nDataset generation complete.")
    
    train_loader = DataLoader(rrr_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(rrr_test, batch_size=64, shuffle=False)
    
    print("\n" + "="*60)
    print("Training RRR_Linear model (Deep Network with Huber Loss)")
    print("="*60)
    model1 = RRR_Linear(input_size=15)
    model1 = train_model(model1, train_loader, test_loader, output_mean, output_std, epochs=100, lr=0.001)
    
    print("\nEvaluating RRR_Linear model...")
    evaluate_model(model1, test_loader, output_mean, output_std)
    
    print("\n" + "="*60)
    print("Training TejomurtKak_Model (Structured Network from Paper)")
    print("="*60)
    model2 = TejomurtKak_Model(input_size=15)
    model2 = train_tejomurt_model(model2, train_loader, test_loader, output_mean, output_std, epochs=100, lr=0.01)
    
    print("\nEvaluating TejomurtKak_Model...")
    evaluate_model(model2, test_loader, output_mean, output_std)
    
    print("\n" + "="*60)
    print("Speed Comparison")
    print("="*60)
    dh_rad = deg_to_rad_dh(RRR_dh)
    
    test_sample = rrr_test[0]
    test_pose = test_sample[0][:6].numpy()
    test_angles = test_sample[1].numpy()
    
    homo = forward_kinematics(dh_rad, test_angles)
    
    start_geom = time.time()
    for _ in range(100):
        angles_geom = inverse_kinematics_3dof_rrr(dh_rad, homo)
    time_geom = (time.time() - start_geom) / 100
    
    model1.eval()
    with torch.no_grad():
        test_input = test_sample[0].unsqueeze(0)
        start_nn1 = time.time()
        for _ in range(100):
            angles_nn1 = model1(test_input)
        time_nn1 = (time.time() - start_nn1) / 100
    
    model2.eval()
    with torch.no_grad():
        test_input = test_sample[0].unsqueeze(0)
        start_nn2 = time.time()
        for _ in range(100):
            angles_nn2 = model2(test_input)
        time_nn2 = (time.time() - start_nn2) / 100
    
    print(f"\nAverage over 100 runs:")
    print(f"Geometric IK: {time_geom*1000:.4f} ms")
    print(f"RRR_Linear (Deep): {time_nn1*1000:.4f} ms (Speedup: {time_geom/time_nn1:.2f}x)")
    print(f"TejomurtKak (Structured): {time_nn2*1000:.4f} ms (Speedup: {time_geom/time_nn2:.2f}x)")

if __name__ == "__main__":
    main()