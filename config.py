# Robot Configuration
# EE599 - Deep Learning Fundamentals
# Hersch Nathan 
# Last Modified 12/07/25

import numpy as np

# ai alphai di thetai
# link length (mm), link twist(deg), link offset (mm), joint angle (deg)
# nan represents the variable

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

RRR_angle = np.array([45, 32, -10])
RRRRRR_angle = np.array([45, 32, -10, 12, 42, -31])

NUM_SAMPLES = 50000
ANGLE_MIN = -180
ANGLE_MAX = 180
RRR_SEED = 42
RRRRRR_SEED = 123

DLS_LAMBDA = 0.1
DLS_EPSILON = 1e-6
DLS_MAX_ITER = 1000

TEST_SPLIT = 0.2

GENERATE_RRR_DATASET = True
GENERATE_RRRRRR_DATASET = True
