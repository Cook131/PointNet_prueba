import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    """
    T-Net: Transformation Network for input and feature transforms
    """
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        
        # Shared MLPs
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Apply shared MLPs
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        # Fully connected layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Initialize as identity matrix
        identity = torch.eye(self.k, dtype=torch.float32, device=x.device)
        identity = identity.view(1, self.k*self.k).repeat(batch_size, 1)
        
        x = x + identity
        x = x.view(-1, self.k, self.k)
        
        return x

class STN3d(TNet):
    """3D Spatial Transformer Network"""
    def __init__(self):
        super(STN3d, self).__init__(k=3)

class STNkd(TNet):
    """Feature Spatial Transformer Network"""
    def __init__(self, k=64):
        super(STNkd, self).__init__(k=k)