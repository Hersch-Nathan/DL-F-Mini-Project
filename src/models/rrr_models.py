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

class SimpleCNN(nn.Module):
    """Simple CNN-based network for RRR inverse kinematics.
    
    Treats input as a 1D signal and applies convolutions to extract features.
    Architecture: Conv1D layers -> Flatten -> FC layers -> 3 outputs
    - Input: [position(3), orientation(3), DH_params(9)] reshaped to [15, 1]
    - Output: [theta1, theta2, theta3] joint angles
    """
    def __init__(self, input_size=15):
        super().__init__()
        # Conv layers treat input as 1D sequence
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)
        
        # Calculate flattened size after convolutions and pooling
        # Input size: 15 -> pool -> 7 -> pool -> 3
        self.flattened_size = 128 * 3
        
        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)
        self.relu = nn.ReLU()
        
        # Xavier initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Reshape from [batch, 15] to [batch, 1, 15] for Conv1d
        x = x.unsqueeze(1)
        
        # Convolutional layers with pooling
        x = self.relu(self.conv1(x))  # [batch, 32, 15]
        x = self.pool(x)               # [batch, 32, 7]
        
        x = self.relu(self.conv2(x))  # [batch, 64, 7]
        x = self.pool(x)               # [batch, 64, 3]
        
        x = self.relu(self.conv3(x))  # [batch, 128, 3]
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)     # [batch, 128*3]
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Simple4Layer6DOF(nn.Module):
    """Simple 4-layer fully connected network for 6-DOF RRRRRR inverse kinematics.
    
    Architecture: 24 -> 256 -> 128 -> 64 -> 6
    - Input: [position(3), orientation(3), DH_params(18)]
    - Output: [theta1, theta2, theta3, theta4, theta5, theta6] joint angles
    """
    def __init__(self, input_size=24):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 6)
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

class SimpleCNN6DOF(nn.Module):
    """Simple CNN-based network for 6-DOF RRRRRR inverse kinematics.
    
    Treats input as a 1D signal and applies convolutions to extract features.
    Architecture: Conv1D layers -> Flatten -> FC layers -> 6 outputs
    - Input: [position(3), orientation(3), DH_params(18)] reshaped to [24, 1]
    - Output: [theta1-theta6] joint angles
    """
    def __init__(self, input_size=24):
        super().__init__()
        # Conv layers treat input as 1D sequence
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)
        
        # Calculate flattened size after convolutions and pooling
        # Input size: 24 -> pool -> 12 -> pool -> 6
        self.flattened_size = 256 * 6
        
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)
        self.relu = nn.ReLU()
        
        # Xavier initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Reshape from [batch, 24] to [batch, 1, 24] for Conv1d
        x = x.unsqueeze(1)
        
        # Convolutional layers with pooling
        x = self.relu(self.conv1(x))  # [batch, 32, 24]
        x = self.pool(x)               # [batch, 32, 12]
        
        x = self.relu(self.conv2(x))  # [batch, 64, 12]
        x = self.pool(x)               # [batch, 64, 6]
        
        x = self.relu(self.conv3(x))  # [batch, 128, 6]
        x = self.relu(self.conv4(x))  # [batch, 256, 6]
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)     # [batch, 256*6]
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DeepFC3DOF(nn.Module):
    """Large deep network for 3-DOF RRR to achieve DLS-level accuracy.
    
    Architecture: 15 -> 512 -> 1024 -> 1024 -> 512 -> 256 -> 128 -> 3
    - Input: [position(3), orientation(3), DH_params(9)]
    - Output: [theta1, theta2, theta3] joint angles
    """
    def __init__(self, input_size=15):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc5 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc6 = nn.Linear(256, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.fc7 = nn.Linear(128, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.relu(self.bn5(self.fc5(x)))
        x = self.relu(self.bn6(self.fc6(x)))
        x = self.fc7(x)
        return x

class DeepFC6DOF(nn.Module):
    """Large deep network for 6-DOF RRRRRR to achieve DLS-level accuracy.
    
    Architecture: 24 -> 512 -> 1024 -> 2048 -> 1024 -> 512 -> 256 -> 128 -> 6
    - Input: [position(3), orientation(3), DH_params(18)]
    - Output: [theta1, theta2, theta3, theta4, theta5, theta6] joint angles
    """
    def __init__(self, input_size=24):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.fc4 = nn.Linear(2048, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.fc5 = nn.Linear(1024, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc6 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.fc7 = nn.Linear(256, 128)
        self.bn7 = nn.BatchNorm1d(128)
        self.fc8 = nn.Linear(128, 6)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.relu(self.bn5(self.fc5(x)))
        x = self.dropout(x)
        x = self.relu(self.bn6(self.fc6(x)))
        x = self.relu(self.bn7(self.fc7(x)))
        x = self.fc8(x)
        return x
