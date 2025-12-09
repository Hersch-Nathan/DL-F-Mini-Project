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
from src.training import generate_dataset, generate_consistent_dataset, save_dataset, load_dataset, train_model
from src.models import Simple4Layer
from src.evaluation import evaluate_model

def main():
    if GENERATE_RRR_DATASET:
        print("Generating RRR random dataset (multiple IK solutions)")
        rrr_angles_random, rrr_pos_random, rrr_orient_random = generate_dataset(
            RRR_dh, 
            num_samples=NUM_SAMPLES, 
            angle_min=-90, 
            angle_max=90, 
            seed=RRR_SEED
        )
        save_dataset('data/rrr_dataset_random.npz', rrr_angles_random, rrr_pos_random, rrr_orient_random)
        print(f"RRR random dataset saved: {len(rrr_angles_random)} samples")
        
        print("\nGenerating RRR consistent dataset (unique IK solutions)")
        rrr_angles, rrr_pos, rrr_orient = generate_consistent_dataset(
            RRR_dh, 
            num_samples=NUM_SAMPLES, 
            angle_min=-90, 
            angle_max=90, 
            seed=RRR_SEED
        )
        save_dataset('data/rrr_dataset.npz', rrr_angles, rrr_pos, rrr_orient)
        print(f"RRR consistent dataset saved: {len(rrr_angles)} samples with consistent IK")
    
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
    
    # Load random dataset (for demonstrating the problem)
    print("\n--- Random Dataset (Multiple IK Solutions) ---")
    rrr_angles_random, rrr_pos_random, rrr_orient_random = load_dataset('data/rrr_dataset_random.npz')
    
    dh_rad = deg_to_rad_dh(RRR_dh)
    dh_params = dh_rad[:, :3].flatten()
    dh_params_repeated_random = np.tile(dh_params, (len(rrr_pos_random), 1))
    
    rrr_inputs_raw_random = np.concatenate([rrr_pos_random, rrr_orient_random, dh_params_repeated_random], axis=1)
    rrr_inputs_random = torch.tensor(rrr_inputs_raw_random, dtype=torch.float32)
    rrr_outputs_random = torch.tensor(rrr_angles_random, dtype=torch.float32)
    rrr_dataset_random = TensorDataset(rrr_inputs_random, rrr_outputs_random)
    
    test_size = int(TEST_SPLIT * len(rrr_dataset_random))
    train_size = len(rrr_dataset_random) - test_size
    rrr_train_random, rrr_test_random = random_split(rrr_dataset_random, [train_size, test_size])
    print(f"RRR Random: {len(rrr_train_random)} train, {len(rrr_test_random)} test")
    
    # Load consistent dataset (for demonstrating the solution)
    print("\n--- Consistent Dataset (Unique IK Solutions) ---")
    rrr_angles_load, rrr_pos_load, rrr_orient_load = load_dataset('data/rrr_dataset.npz')
    
    dh_params_repeated = np.tile(dh_params, (len(rrr_pos_load), 1))
    
    rrr_inputs_raw = np.concatenate([rrr_pos_load, rrr_orient_load, dh_params_repeated], axis=1)
    
    # No normalization - test if model can still learn
    rrr_inputs = torch.tensor(rrr_inputs_raw, dtype=torch.float32)
    rrr_outputs = torch.tensor(rrr_angles_load, dtype=torch.float32)
    rrr_dataset = TensorDataset(rrr_inputs, rrr_outputs)
    
    test_size = int(TEST_SPLIT * len(rrr_dataset))
    train_size = len(rrr_dataset) - test_size
    rrr_train, rrr_test = random_split(rrr_dataset, [train_size, test_size])
    print(f"RRR Consistent: {len(rrr_train)} train, {len(rrr_test)} test")
    
    rrrrrr_angles_load, rrrrrr_pos_load, rrrrrr_orient_load = load_dataset('data/rrrrrr_dataset.npz')
    rrrrrr_inputs = torch.tensor(np.concatenate([rrrrrr_pos_load, rrrrrr_orient_load], axis=1), dtype=torch.float32)
    rrrrrr_outputs = torch.tensor(rrrrrr_angles_load, dtype=torch.float32)
    rrrrrr_dataset = TensorDataset(rrrrrr_inputs, rrrrrr_outputs)
    
    test_size = int(TEST_SPLIT * len(rrrrrr_dataset))
    train_size = len(rrrrrr_dataset) - test_size
    rrrrrr_train, rrrrrr_test = random_split(rrrrrr_dataset, [train_size, test_size])
    print(f"RRRRRR: {len(rrrrrr_train)} train, {len(rrrrrr_test)} test")
    
    print("\nDataset generation complete.")
    
    # Train on random dataset first (to show the problem)
    print("\n" + "="*60)
    print("PROBLEM: Training on Random Dataset (Multiple IK Solutions)")
    print("="*60)
    train_loader_random = DataLoader(rrr_train_random, batch_size=64, shuffle=True)
    test_loader_random = DataLoader(rrr_test_random, batch_size=64, shuffle=False)
    
    model_random = Simple4Layer(input_size=15)
    model_random = train_model(model_random, train_loader_random, test_loader_random, epochs=100, lr=0.001)
    
    print("\nEvaluating model trained on random dataset...")
    evaluate_model(model_random, test_loader_random)
    
    # Train on consistent dataset (to show the solution)
    print("\n" + "="*60)
    print("SOLUTION: Training on Consistent Dataset (Unique IK Solutions)")
    print("="*60)
    train_loader = DataLoader(rrr_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(rrr_test, batch_size=64, shuffle=False)
    
    model = Simple4Layer(input_size=15)
    model = train_model(model, train_loader, test_loader, epochs=100, lr=0.001)
    
    print("\nEvaluating model trained on consistent dataset...")
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