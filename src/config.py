# Configuration for neural network inverse kinematics
# EE599 - Deep Learning Fundamentals

import numpy as np

# Robot DH Parameters: [a, alpha, d, theta]
# a: link length (mm), alpha: link twist (deg), d: link offset (mm), theta: joint angle (deg, np.nan = variable)
RRR_dh = np.array([
    [0, -90, 0, np.nan],
    [0, 90, 0, np.nan],
    [0, 0, 50, np.nan]
])

RRRRRR_dh = np.array([
    [0, 90, 10, np.nan],
    [40, 0, 0, np.nan],
    [40, 0, 0, np.nan],
    [0, 90, 10, np.nan],
    [0, -90, 10, np.nan],
    [0, 0, 10, np.nan]
])

# Dataset and Training Configuration
NUM_SAMPLES = 50000
TEST_SPLIT = 0.2
RRR_SEED = 42
RRRRRR_SEED = 123

# Accuracy Threshold matching DLS convergence precision (1e-6 radians ≈ 0.0000573 degrees)
ACCURACY_THRESHOLD = 1e-6

# Dataset Generation Flags
GENERATE_RRR_DATASET = False
GENERATE_RRRRRR_DATASET = False
