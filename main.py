# Main training and evaluation script
# EE599 - Deep Learning Fundamentals
# Hersch Nathan

import numpy as np
import torch
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split

from src.config import RRR_dh, RRRRRR_dh, NUM_SAMPLES, RRR_SEED, RRRRRR_SEED, TEST_SPLIT, GENERATE_RRR_DATASET, GENERATE_RRRRRR_DATASET, EVALUATION_ONLY
from src.robots import forward_kinematics, inverse_kinematics_3dof_rrr, inverse_kinematics_dls_6dof, deg_to_rad_dh
from src.training import generate_dataset, generate_consistent_dataset, save_dataset, load_dataset, train_model
from src.models import Simple4Layer, SimpleCNN, Simple4Layer6DOF, SimpleCNN6DOF, DeepFC3DOF, DeepFC6DOF
from src.evaluation import evaluate_model

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

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

def plot_training_curves(all_stats, model_names):
    """Plot training curves for all models."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Statistics Across All Models', fontsize=16)
    
    for stats, name in zip(all_stats, model_names):
        if not stats:
            continue
        df = pd.DataFrame(stats)
        
        # Plot training loss
        axes[0, 0].plot(df['epoch'], df['train_loss'], label=name, alpha=0.7)
        
        # Plot test loss
        axes[0, 1].plot(df['epoch'], df['test_loss'], label=name, alpha=0.7)
        
        # Plot training accuracy
        axes[1, 0].plot(df['epoch'], df['train_accuracy'], label=name, alpha=0.7)
        
        # Plot test accuracy
        axes[1, 1].plot(df['epoch'], df['test_accuracy'], label=name, alpha=0.7)
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Test Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Test Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('logs/training_curves.png', dpi=300, bbox_inches='tight')
    print("\nTraining curves saved to logs/training_curves.png")
    plt.show()

def main():
    """Main training and evaluation pipeline."""
    # Track all training statistics
    all_training_stats = []
    model_names = []
    
    # Generate datasets if needed
    if GENERATE_RRR_DATASET and not EVALUATION_ONLY:
        print("Generating RRR datasets...")
        angles_rand, pos_rand, orient_rand = generate_dataset(
            RRR_dh, num_samples=NUM_SAMPLES, angle_min=-180, angle_max=180, seed=RRR_SEED
        )
        save_dataset('data/rrr_dataset_random.npz', angles_rand, pos_rand, orient_rand)
        
        angles, pos, orient = generate_dataset(
            RRR_dh, num_samples=NUM_SAMPLES, angle_min=-90, angle_max=90, seed=RRR_SEED
        )
        save_dataset('data/rrr_dataset.npz', angles, pos, orient)
    
    if GENERATE_RRRRRR_DATASET and not EVALUATION_ONLY:
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
    if EVALUATION_ONLY:
        print("  Loading pre-trained model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_rrr_fc_rand.load_state_dict(torch.load('models/rrr_fc_random.pth', map_location=device))
        model_rrr_fc_rand = model_rrr_fc_rand.to(device)
        stats_rand = []
    else:
        model_rrr_fc_rand, stats_rand = train_model(model_rrr_fc_rand, loader_train_rand, loader_test_rand, 
                                         epochs=100, lr=0.001, accuracy_threshold=0.5,
                                         log_file='logs/rrr_fc_random.csv')
        torch.save(model_rrr_fc_rand.state_dict(), 'models/rrr_fc_random.pth')
        print("Model saved: models/rrr_fc_random.pth")
    all_training_stats.append(stats_rand)
    model_names.append('RRR-FC-Random')
    
    # Train on consistent dataset at 0.5 rad precision
    print("\n2. Training Simple4Layer on CONSISTENT dataset (0.5 rad precision)")
    model_rrr_fc = Simple4Layer(input_size=15)
    if EVALUATION_ONLY:
        print("  Loading pre-trained model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_rrr_fc.load_state_dict(torch.load('models/rrr_fc.pth', map_location=device))
        model_rrr_fc = model_rrr_fc.to(device)
        stats_fc = []
    else:
        model_rrr_fc, stats_fc = train_model(model_rrr_fc, loader_train_rrr, loader_test_rrr, 
                                    epochs=100, lr=0.001, accuracy_threshold=0.5,
                                    log_file='logs/rrr_fc.csv')
        torch.save(model_rrr_fc.state_dict(), 'models/rrr_fc.pth')
        print("Model saved: models/rrr_fc.pth")
    all_training_stats.append(stats_fc)
    model_names.append('RRR-FC')
    
    # Evaluate at 0.01 rad precision
    print("\n3. Evaluating Simple4Layer at 0.01 rad precision")
    evaluate_model(model_rrr_fc, loader_test_rrr, accuracy_threshold=0.01)
    
    # Train large deep network for DLS-level accuracy
    print("\n4. Training DeepFC3DOF for DLS-level accuracy (1e-6 rad precision)")
    model_rrr_deep = DeepFC3DOF(input_size=15)
    if EVALUATION_ONLY:
        print("  Loading pre-trained model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_rrr_deep.load_state_dict(torch.load('models/rrr_deep.pth', map_location=device))
        model_rrr_deep = model_rrr_deep.to(device)
        stats_deep = []
    else:
        model_rrr_deep, stats_deep = train_model(model_rrr_deep, loader_train_rrr, loader_test_rrr, 
                                      epochs=200, lr=0.0005, accuracy_threshold=1e-6,
                                      log_file='logs/rrr_deep.csv')
        torch.save(model_rrr_deep.state_dict(), 'models/rrr_deep.pth')
        print("Model saved: models/rrr_deep.pth")
    all_training_stats.append(stats_deep)
    model_names.append('RRR-DeepFC')
    
    # Evaluate deep network at DLS precision
    print("\n4b. Evaluating DeepFC3DOF at 1e-6 rad precision")
    evaluate_model(model_rrr_deep, loader_test_rrr, accuracy_threshold=1e-6)
    
    # CNN models disabled for now
    # print("\n5. Training SimpleCNN on CONSISTENT dataset (0.01 rad precision)")
    # model_rrr_cnn = SimpleCNN(input_size=15)
    # model_rrr_cnn = train_model(model_rrr_cnn, loader_train_rrr, loader_test_rrr, 
    #                              epochs=100, lr=0.001, accuracy_threshold=0.01)
    # torch.save(model_rrr_cnn.state_dict(), 'models/rrr_cnn.pth')
    # print("Model saved: models/rrr_cnn.pth")
    
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
    model_6d_fc = Simple4Layer6DOF(input_size=24)
    if EVALUATION_ONLY:
        print("  Loading pre-trained model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_6d_fc.load_state_dict(torch.load('models/6dof_fc.pth', map_location=device))
        model_6d_fc = model_6d_fc.to(device)
        stats_6d_fc = []
    else:
        model_6d_fc, stats_6d_fc = train_model(model_6d_fc, loader_train_6d, loader_test_6d, 
                                   epochs=100, lr=0.001, accuracy_threshold=0.5,
                                   log_file='logs/6dof_fc.csv')
        torch.save(model_6d_fc.state_dict(), 'models/6dof_fc.pth')
        print("Model saved: models/6dof_fc.pth")
    all_training_stats.append(stats_6d_fc)
    model_names.append('6DOF-FC')
    
    # Evaluate at 0.01 rad precision
    print("\n2. Evaluating Simple4Layer6DOF at 0.01 rad precision")
    evaluate_model(model_6d_fc, loader_test_6d, accuracy_threshold=0.01)
    
    # Train large deep network for DLS-level accuracy
    print("\n3. Training DeepFC6DOF for DLS-level accuracy (1e-6 rad precision)")
    model_6d_deep = DeepFC6DOF(input_size=24)
    if EVALUATION_ONLY:
        print("  Loading pre-trained model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_6d_deep.load_state_dict(torch.load('models/6dof_deep.pth', map_location=device))
        model_6d_deep = model_6d_deep.to(device)
        stats_6d_deep = []
    else:
        model_6d_deep, stats_6d_deep = train_model(model_6d_deep, loader_train_6d, loader_test_6d, 
                                     epochs=200, lr=0.0005, accuracy_threshold=1e-6,
                                     log_file='logs/6dof_deep.csv')
        torch.save(model_6d_deep.state_dict(), 'models/6dof_deep.pth')
        print("Model saved: models/6dof_deep.pth")
    all_training_stats.append(stats_6d_deep)
    model_names.append('6DOF-DeepFC')
    
    # Evaluate deep network at DLS precision
    print("\n3b. Evaluating DeepFC6DOF at 1e-6 rad precision")
    evaluate_model(model_6d_deep, loader_test_6d, accuracy_threshold=1e-6)
    
    # CNN models disabled for now
    # print("\n4. Training SimpleCNN6DOF on CONSISTENT dataset (0.01 rad precision)")
    # model_6d_cnn = SimpleCNN6DOF(input_size=24)
    # model_6d_cnn = train_model(model_6d_cnn, loader_train_6d, loader_test_6d, 
    #                             epochs=100, lr=0.001, accuracy_threshold=0.01)
    # torch.save(model_6d_cnn.state_dict(), 'models/6dof_cnn.pth')
    # print("Model saved: models/6dof_cnn.pth")
    
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
    print("\nClassical DLS Solver (6-DOF):")
    dls_angles = inverse_kinematics_dls_6dof(dh_rad_6d, homo, lamda=0.01, epsilon=1e-6, max_iter=500)
    dls_error = np.abs(angles_test - dls_angles)
    dls_acc_1e6 = 100 * np.all(dls_error < 1e-6)
    dls_acc_01 = 100 * np.all(dls_error < 0.01)
    dls_acc_05 = 100 * np.all(dls_error < 0.5)
    print(f"  Error (radians): {np.max(dls_error):.6e}")
    print(f"  Accuracy (1e-6 rad): {dls_acc_1e6:.0f}%")
    print(f"  Accuracy (0.01 rad): {dls_acc_01:.0f}%")
    print(f"  Accuracy (0.5 rad): {dls_acc_05:.0f}%")
    
    # DLS Speed benchmark
    start = time.time()
    for _ in range(100):
        _ = inverse_kinematics_dls_6dof(dh_rad_6d, homo, lamda=0.01, epsilon=1e-6, max_iter=500)
    time_dls = (time.time() - start) / 100
    
    # Neural Network comparisons
    model_6d_fc.eval()
    model_6d_deep.eval()
    
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
        
        # Deep Network
        print("\nDeepFC6DOF (Large Deep Network):")
        nn_angles_deep = model_6d_deep(nn_input).cpu().numpy()[0]
        nn_error_deep = np.abs(angles_test - nn_angles_deep)
        nn_acc_deep_1e6 = 100 * np.all(nn_error_deep < 1e-6)
        nn_acc_deep_01 = 100 * np.all(nn_error_deep < 0.01)
        print(f"  Error (radians): {np.max(nn_error_deep):.6e}")
        print(f"  Accuracy (1e-6 rad): {nn_acc_deep_1e6:.0f}%")
        print(f"  Accuracy (0.01 rad): {nn_acc_deep_01:.0f}%")
    
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
            _ = model_6d_deep(nn_input)
    time_deep = (time.time() - start) / 100
    print(f"DeepFC6DOF:       {time_deep*1000:.4f} ms ({time_dls/time_deep:.2f}x speedup)")
    
    # Plot training curves if any training was done
    if not EVALUATION_ONLY:
        plot_training_curves(all_training_stats, model_names)

if __name__ == "__main__":
    main()