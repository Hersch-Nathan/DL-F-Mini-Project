import torch
from torch import nn

class Simple4Layer(nn.Module):
    """Simple 4-layer fully connected network for RRR inverse kinematics.
    
    Architecture: 15 -> 128 -> 64 -> 32 -> 3
    - Input: [position(3), orientation(3), DH_params(9)]
    - Output: [theta1, theta2, theta3] joint angles
    """
    def __init__(self, input_size=15):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 3)
        self.relu = nn.ReLU()
        
        # Xavier initialization for better training
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
