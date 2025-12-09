# Main File
# EE599 - Deep Learning Fundamentals
# Hersch Nathan 
# Last Modified 12/07/25

import numpy as np
import torch
import time
from torch.utils.data import TensorDataset, DataLoader, random_split

from src.config import RRR_dh, RRRRRR_dh, NUM_SAMPLES, RRR_SEED, RRRRRR_SEED, TEST_SPLIT, GENERATE_RRR_DATASET, GENERATE_RRRRRR_DATASET
from src.robots import forward_kinematics, inverse_kinematics_3dof_rrr, deg_to_rad_dh
from src.training import generate_consistent_dataset, save_dataset, load_dataset, train_model
from src.models import Simple4Layer
from src.evaluation import evaluate_model

def main():
    if GENERATE_RRR_DATASET:
        print("Generating RRR dataset with consistent IK solutions")
        rrr_angles, rrr_pos, rrr_orient = generate_consistent_dataset(
            RRR_dh, 
            num_samples=NUM_SAMPLES, 
            angle_min=-90, 
            angle_max=90, 
            seed=RRR_SEED
        )
        save_dataset('data/rrr_dataset.npz', rrr_angles, rrr_pos, rrr_orient)
        print(f"RRR dataset saved: {len(rrr_angles)} samples with consistent IK")
    
    if GENERATE_RRRRRR_DATASET:
        print("Generating RRRRRR dataset with consistent solutions")
        rrrrrr_angles, rrrrrr_pos, rrrrrr_orient = generate_consistent_dataset(
            RRRRRR_dh, 
            num_samples=NUM_SAMPLES, 
            angle_min=-90, 
            angle_max=90, 
            seed=RRRRRR_SEED
        )
        save_dataset('data/rrrrrr_dataset.npz', rrrrrr_angles, rrrrrr_pos, rrrrrr_orient)
        print(f"RRRRRR dataset saved: {len(rrrrrr_angles)} samples with consistent solutions")
    
    print("\nLoading datasets with torch...")
    
    rrr_angles_load, rrr_pos_load, rrr_orient_load = load_dataset('data/rrr_dataset.npz')
    
    dh_rad = deg_to_rad_dh(RRR_dh)
    dh_params = dh_rad[:, :3].flatten()
    dh_params_repeated = np.tile(dh_params, (len(rrr_pos_load), 1))
    
    rrr_inputs_raw = np.concatenate([rrr_pos_load, rrr_orient_load, dh_params_repeated], axis=1)
    
    # No normalization - test if model can still learn
    rrr_inputs = torch.tensor(rrr_inputs_raw, dtype=torch.float32)
    rrr_outputs = torch.tensor(rrr_angles_load, dtype=torch.float32)
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
    print("Training Simple4Layer Model for RRR Robot (No Normalization)")
    print("="*60)
    model = Simple4Layer(input_size=15)
    model = train_model(model, train_loader, test_loader, epochs=100, lr=0.001)
    
    print("\nEvaluating Simple4Layer model...")
    evaluate_model(model, test_loader)
    
    print("\n" + "="*60)
    print("Speed Comparison: Classical IK vs Neural Network")
    print("="*60)
    dh_rad = deg_to_rad_dh(RRR_dh)
    
    test_sample = rrr_test[0]
    test_pose = test_sample[0][:6].numpy()
    test_angles = test_sample[1].numpy()
    
    homo = forward_kinematics(dh_rad, test_angles)
    
    # Classical geometric IK
    start_geom = time.time()
    for _ in range(100):
        angles_geom = inverse_kinematics_3dof_rrr(dh_rad, homo)
    time_geom = (time.time() - start_geom) / 100
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Neural network IK
    model.eval()
    with torch.no_grad():
        test_input = test_sample[0].unsqueeze(0).to(device)
        start_nn = time.time()
        for _ in range(100):
            angles_nn = model(test_input)
        time_nn = (time.time() - start_nn) / 100
    
    print(f"\nAverage over 100 runs:")
    print(f"Classical Geometric IK: {time_geom*1000:.4f} ms")
    print(f"Neural Network IK: {time_nn*1000:.4f} ms")
    print(f"Speedup: {time_geom/time_nn:.2f}x faster")

if __name__ == "__main__":
    main()