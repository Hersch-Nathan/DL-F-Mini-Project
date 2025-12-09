# Robot Configuration
# EE599 - Deep Learning Fundamentals
# Hersch Nathan 

import numpy as np

# DH Parameters: [a, alpha, d, theta]
# a: link length (mm)
# alpha: link twist (deg)
# d: link offset (mm)
# theta: joint angle (deg, np.nan = variable)

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

# Dataset Configuration
NUM_SAMPLES = 50000
TEST_SPLIT = 0.2
RRR_SEED = 42
RRRRRR_SEED = 123

# Accuracy Threshold (from DLS precision level, epsilon=1e-6)
ACCURACY_THRESHOLD = 1e-6

# Dataset Generation Flags
GENERATE_RRR_DATASET = False
GENERATE_RRRRRR_DATASET = False
