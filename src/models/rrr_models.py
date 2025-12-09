import torch
from torch import nn

class RRR_Linear(nn.Module):
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
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    
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
        
        x = self.fc7(x)
        return x

class TejomurtKak_Model(nn.Module):
    def __init__(self, input_size=15):
        super().__init__()
        self.input_size = input_size
        
        self.subnet1 = nn.Sequential(
            nn.Linear(input_size, 30),
            nn.Sigmoid(),
            nn.Linear(30, 20),
            nn.Sigmoid()
        )
        
        self.subnet2 = nn.Sequential(
            nn.Linear(20, 15),
            nn.Sigmoid(),
            nn.Linear(15, 10),
            nn.Sigmoid()
        )
        
        self.subnet3 = nn.Sequential(
            nn.Linear(10, 8),
            nn.Sigmoid(),
            nn.Linear(8, 3)
        )
    
    def forward(self, x):
        x = self.subnet1(x)
        x = self.subnet2(x)
        x = self.subnet3(x)
        return x

class Simple4Layer(nn.Module):
    def __init__(self, input_size=15):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 3)
        self.relu = nn.ReLU()
        
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
