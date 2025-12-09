# Main training and evaluation script
# EE599 - Deep Learning Fundamentals
# Hersch Nathan

import numpy as np
import torch
import time
import os
from torch.utils.data import TensorDataset, DataLoader, random_split

from src.config import RRR_dh, RRRRRR_dh, NUM_SAMPLES, RRR_SEED, RRRRRR_SEED, TEST_SPLIT, GENERATE_RRR_DATASET, GENERATE_RRRRRR_DATASET
from src.robots import forward_kinematics, inverse_kinematics_3dof_rrr, deg_to_rad_dh
from src.training import generate_dataset, generate_consistent_dataset, save_dataset, load_dataset, train_model
from src.models import Simple4Layer, SimpleCNN, Simple4Layer6DOF, SimpleCNN6DOF
from src.evaluation import evaluate_model

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

def prepare_dataset(angles, pos, orient, dh_params):
    """Prepare dataset for training/testing."""
    dh_repeat = np.tile(dh_params, (len(pos), 1))
    inputs = torch.tensor(np.concatenate([pos, orient, dh_repeat], axis=1), dtype=torch.float32)
    outputs = torch.tensor(angles, dtype=torch.float32)
    dataset = TensorDataset(inputs, outputs)
    train_idx = int(0.8 * len(dataset))
    train, test = random_split(dataset, [train_idx, len(dataset) - train_idx])
    return (DataLoader(train, batch_size=64, shuffle=True),
            DataLoader(test, batch_size=64, shuffle=False))

def main():
    """Main training and evaluation pipeline."""
    # Generate datasets if needed
    if GENERATE_RRR_DATASET:
        print("Generating RRR datasets...")
        angles_rand, pos_rand, orient_rand = generate_dataset(
            RRR_dh, num_samples=NUM_SAMPLES, angle_min=-180, angle_max=180, seed=RRR_SEED
        )
        save_dataset('data/rrr_dataset_random.npz', angles_rand, pos_rand, orient_rand)
        
        angles, pos, orient = generate_dataset(
            RRR_dh, num_samples=NUM_SAMPLES, angle_min=-90, angle_max=90, seed=RRR_SEED
        )
        save_dataset('data/rrr_dataset.npz', angles, pos, orient)
    
    if GENERATE_RRRRRR_DATASET:
        print("Generating 6-DOF dataset...")
        angles, pos, orient = generate_consistent_dataset(
            RRRRRR_dh, num_samples=NUM_SAMPLES, angle_min=-90, angle_max=90, seed=RRRRRR_SEED
        )
        save_dataset('data/rrrrrr_dataset.npz', angles, pos, orient)
    
    print("Loading datasets...\n")
    
    # Setup 3-DOF RRR
    print("="*70)
    print("3-DOF RRR ROBOT")
    print("="*70)
    
    dh_rad_rrr = deg_to_rad_dh(RRR_dh)
    dh_params_rrr = dh_rad_rrr[:, :3].flatten()
    
    # Load random dataset (multiple IK solutions)
    angles_rand, pos_rand, orient_rand = load_dataset('data/rrr_dataset_random.npz')
    loader_train_rand, loader_test_rand = prepare_dataset(angles_rand, pos_rand, orient_rand, dh_params_rrr)
    
    # Load consistent dataset (unique IK solutions)
    angles_rrr, pos_rrr, orient_rrr = load_dataset('data/rrr_dataset.npz')
    loader_train_rrr, loader_test_rrr = prepare_dataset(angles_rrr, pos_rrr, orient_rrr, dh_params_rrr)
    
    # Train on random dataset at 0.5 rad precision
    print("\n1. Training Simple4Layer on RANDOM dataset (0.5 rad precision)")
    model_rrr_fc_rand = Simple4Layer(input_size=15)
    model_rrr_fc_rand = train_model(model_rrr_fc_rand, loader_train_rand, loader_test_rand, 
                                     epochs=100, lr=0.001, accuracy_threshold=0.5)
    torch.save(model_rrr_fc_rand.state_dict(), 'models/rrr_fc_random.pth')
    print("Model saved: models/rrr_fc_random.pth")
    
    # Train on consistent dataset at 0.5 rad precision
    print("\n2. Training Simple4Layer on CONSISTENT dataset (0.5 rad precision)")
    model_rrr_fc = Simple4Layer(input_size=15)
    model_rrr_fc = train_model(model_rrr_fc, loader_train_rrr, loader_test_rrr, 
                                epochs=100, lr=0.001, accuracy_threshold=0.5)
    torch.save(model_rrr_fc.state_dict(), 'models/rrr_fc.pth')
    print("Model saved: models/rrr_fc.pth")
    
    # Evaluate at 0.01 rad precision
    print("\n3. Evaluating Simple4Layer at 0.01 rad precision")
    evaluate_model(model_rrr_fc, loader_test_rrr, accuracy_threshold=0.01)
    
    # Train CNN at 0.01 rad precision
    print("\n4. Training SimpleCNN on CONSISTENT dataset (0.01 rad precision)")
    model_rrr_cnn = SimpleCNN(input_size=15)
    model_rrr_cnn = train_model(model_rrr_cnn, loader_train_rrr, loader_test_rrr, 
                                 epochs=100, lr=0.001, accuracy_threshold=0.01)
    torch.save(model_rrr_cnn.state_dict(), 'models/rrr_cnn.pth')
    print("Model saved: models/rrr_cnn.pth")
    
    # Evaluate CNN at 0.01 rad precision
    print("\n4b. Evaluating SimpleCNN at 0.01 rad precision")
    evaluate_model(model_rrr_cnn, loader_test_rrr, accuracy_threshold=0.01)
    
    # Setup 6-DOF RRRRRR
    print("\n" + "="*70)
    print("6-DOF RRRRRR ROBOT")
    print("="*70)
    
    dh_rad_6d = deg_to_rad_dh(RRRRRR_dh)
    dh_params_6d = dh_rad_6d[:, :3].flatten()
    
    # Load consistent dataset for 6-DOF
    angles_6d, pos_6d, orient_6d = load_dataset('data/rrrrrr_dataset.npz')
    loader_train_6d, loader_test_6d = prepare_dataset(angles_6d, pos_6d, orient_6d, dh_params_6d)
    
    # Train on 6-DOF dataset at 0.5 rad precision
    print("\n1. Training Simple4Layer6DOF on CONSISTENT dataset (0.5 rad precision)")
    model_6d_fc = Simple4Layer6DOF(input_size=21)
    model_6d_fc = train_model(model_6d_fc, loader_train_6d, loader_test_6d, 
                               epochs=100, lr=0.001, accuracy_threshold=0.5)
    torch.save(model_6d_fc.state_dict(), 'models/6dof_fc.pth')
    print("Model saved: models/6dof_fc.pth")
    
    # Evaluate at 0.01 rad precision
    print("\n2. Evaluating Simple4Layer6DOF at 0.01 rad precision")
    evaluate_model(model_6d_fc, loader_test_6d, accuracy_threshold=0.01)
    
    # Train CNN at 0.01 rad precision
    print("\n3. Training SimpleCNN6DOF on CONSISTENT dataset (0.01 rad precision)")
    model_6d_cnn = SimpleCNN6DOF(input_size=21)
    model_6d_cnn = train_model(model_6d_cnn, loader_train_6d, loader_test_6d, 
                                epochs=100, lr=0.001, accuracy_threshold=0.01)
    torch.save(model_6d_cnn.state_dict(), 'models/6dof_cnn.pth')
    print("Model saved: models/6dof_cnn.pth")
    
    # Evaluate CNN at 0.01 rad precision
    print("\n3b. Evaluating SimpleCNN6DOF at 0.01 rad precision")
    evaluate_model(model_6d_cnn, loader_test_6d, accuracy_threshold=0.01)
    
    # Compare DLS vs Neural Networks for 6-DOF
    print("\n" + "="*70)
    print("DLS vs NEURAL NETWORK COMPARISON (6-DOF)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get test sample from 6-DOF
    test_sample = loader_test_6d.dataset[0]
    angles_test = test_sample[1].numpy()
    
    # Calculate actual homogeneous transformation
    homo = forward_kinematics(dh_rad_6d, angles_test)
    
    print("\nAccuracy Comparison (on test sample):")
    print("-" * 70)
    
    # Classical DLS solver - accuracy and speed
    print("\nClassical DLS Solver:")
    dls_angles = inverse_kinematics_3dof_rrr(dh_rad_6d, homo)
    dls_error = np.abs(angles_test - dls_angles)
    dls_acc_01 = 100 * np.all(dls_error < 0.01)
    dls_acc_05 = 100 * np.all(dls_error < 0.5)
    print(f"  Error (radians): {np.max(dls_error):.6e}")
    print(f"  Accuracy (0.01 rad): {dls_acc_01:.0f}%")
    print(f"  Accuracy (0.5 rad): {dls_acc_05:.0f}%")
    
    # DLS Speed benchmark
    start = time.time()
    for _ in range(100):
        _ = inverse_kinematics_3dof_rrr(dh_rad_6d, homo)
    time_dls = (time.time() - start) / 100
    
    # Neural Network comparisons
    model_6d_fc.eval()
    model_6d_cnn.eval()
    
    with torch.no_grad():
        nn_input = test_sample[0].unsqueeze(0).to(device)
        
        # FC Network
        print("\nSimple4Layer6DOF (FC Network):")
        nn_angles_fc = model_6d_fc(nn_input).cpu().numpy()[0]
        nn_error_fc = np.abs(angles_test - nn_angles_fc)
        nn_acc_fc_01 = 100 * np.all(nn_error_fc < 0.01)
        nn_acc_fc_05 = 100 * np.all(nn_error_fc < 0.5)
        print(f"  Error (radians): {np.max(nn_error_fc):.6e}")
        print(f"  Accuracy (0.01 rad): {nn_acc_fc_01:.0f}%")
        print(f"  Accuracy (0.5 rad): {nn_acc_fc_05:.0f}%")
        
        # CNN Network
        print("\nSimpleCNN6DOF (CNN Network):")
        nn_angles_cnn = model_6d_cnn(nn_input).cpu().numpy()[0]
        nn_error_cnn = np.abs(angles_test - nn_angles_cnn)
        nn_acc_cnn_01 = 100 * np.all(nn_error_cnn < 0.01)
        nn_acc_cnn_05 = 100 * np.all(nn_error_cnn < 0.5)
        print(f"  Error (radians): {np.max(nn_error_cnn):.6e}")
        print(f"  Accuracy (0.01 rad): {nn_acc_cnn_01:.0f}%")
        print(f"  Accuracy (0.5 rad): {nn_acc_cnn_05:.0f}%")
    
    # Speed benchmarks
    print("\nSpeed Comparison (average over 100 runs):")
    print("-" * 70)
    print(f"DLS Solver:       {time_dls*1000:.4f} ms")
    
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model_6d_fc(nn_input)
    time_fc = (time.time() - start) / 100
    print(f"Simple4Layer6DOF: {time_fc*1000:.4f} ms ({time_dls/time_fc:.2f}x speedup)")
    
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model_6d_cnn(nn_input)
    time_cnn = (time.time() - start) / 100
    print(f"SimpleCNN6DOF:    {time_cnn*1000:.4f} ms ({time_dls/time_cnn:.2f}x speedup)")

if __name__ == "__main__":
    main()