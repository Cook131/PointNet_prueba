import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transform_net import STN3d, STNkd

class PointNetPose(nn.Module):
    """
    PointNet adaptado para estimación de pose del oil pan
    Predice 6 DOF: 3 rotaciones (euler angles) + 3 traslaciones
    """
    
    def __init__(self, num_points=1024, pose_dim=6, dropout=0.3, feature_transform=True):
        super(PointNetPose, self).__init__()
        self.num_points = num_points
        self.pose_dim = pose_dim
        self.feature_transform = feature_transform
        
        # Spatial transformer networks
        self.stn3d = STN3d()
        if self.feature_transform:
            self.stnkd = STNkd(k=64)
        
        # Shared MLPs
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # Pose estimation head
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, pose_dim)
        
        # Batch normalization for FC layers
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: input point cloud [B, 3, N]
        Returns:
            pose: predicted pose [B, 6] (rx, ry, rz, tx, ty, tz)
            feature_transform: transformation matrix for regularization
        """
        batch_size = x.size(0)
        num_points = x.size(2)
        
        # Input transform
        input_transform = self.stn3d(x)  # [B, 3, 3]
        x = x.transpose(2, 1)  # [B, N, 3]
        x = torch.bmm(x, input_transform)  # Apply transformation
        x = x.transpose(2, 1)  # [B, 3, N]
        
        # First layer
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, N]
        
        # Feature transform
        if self.feature_transform:
            feature_transform = self.stnkd(x)  # [B, 64, 64]
            x = x.transpose(2, 1)  # [B, N, 64]
            x = torch.bmm(x, feature_transform)  # Apply transformation
            x = x.transpose(2, 1)  # [B, 64, N]
        else:
            feature_transform = None
        
        # Shared MLPs
        point_features = F.relu(self.bn2(self.conv2(x)))  # [B, 128, N]
        x = self.bn3(self.conv3(point_features))  # [B, 1024, N]
        
        # Global feature
        global_feature = torch.max(x, 2, keepdim=True)[0]  # [B, 1024, 1]
        global_feature = global_feature.view(-1, 1024)  # [B, 1024]
        
        # Pose estimation
        x = F.relu(self.bn4(self.fc1(global_feature)))  # [B, 512]
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))  # [B, 256]
        x = self.dropout(x)
        pose = self.fc3(x)  # [B, 6]
        
        return pose, feature_transform

class PointNetPoseClassification(nn.Module):
    """
    Variante que trata la estimación de pose como clasificación discreta
    Útil para poses limitadas a un conjunto específico
    """
    
    def __init__(self, num_points=1024, num_pose_classes=24, dropout=0.3, feature_transform=True):
        super(PointNetPoseClassification, self).__init__()
        self.num_points = num_points
        self.num_pose_classes = num_pose_classes
        self.feature_transform = feature_transform
        
        # Usar la misma estructura base
        self.stn3d = STN3d()
        if self.feature_transform:
            self.stnkd = STNkd(k=64)
        
        # Shared MLPs
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # Classification head
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_pose_classes)
        
        # Batch normalization for FC layers
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        """
        Forward pass for classification
        Args:
            x: input point cloud [B, 3, N]
        Returns:
            logits: class logits [B, num_pose_classes]
            feature_transform: transformation matrix for regularization
        """
        batch_size = x.size(0)
        
        # Input transform
        input_transform = self.stn3d(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, input_transform)
        x = x.transpose(2, 1)
        
        # First layer
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Feature transform
        if self.feature_transform:
            feature_transform = self.stnkd(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, feature_transform)
            x = x.transpose(2, 1)
        else:
            feature_transform = None
        
        # Shared MLPs
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        # Global feature
        global_feature = torch.max(x, 2, keepdim=True)[0]
        global_feature = global_feature.view(-1, 1024)
        
        # Classification
        x = F.relu(self.bn4(self.fc1(global_feature)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        logits = self.fc3(x)
        
        return logits, feature_transform

class PointNetPoseHybrid(nn.Module):
    """
    Modelo híbrido que predice tanto pose continua como clasificación discreta
    Útil para combinar precisión continua con robustez de clasificación
    """
    
    def __init__(self, num_points=1024, pose_dim=6, num_pose_classes=24, dropout=0.3):
        super(PointNetPoseHybrid, self).__init__()
        self.num_points = num_points
        self.pose_dim = pose_dim
        self.num_pose_classes = num_pose_classes
        
        # Feature extractor compartido
        self.stn3d = STN3d()
        self.stnkd = STNkd(k=64)
        
        # Shared MLPs
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # Capas compartidas
        self.fc_shared = nn.Linear(1024, 512)
        self.bn_shared = nn.BatchNorm1d(512)
        
        # Head de regresión (pose continua)
        self.fc_reg1 = nn.Linear(512, 256)
        self.fc_reg2 = nn.Linear(256, pose_dim)
        self.bn_reg = nn.BatchNorm1d(256)
        
        # Head de clasificación (pose discreta)
        self.fc_cls1 = nn.Linear(512, 256)
        self.fc_cls2 = nn.Linear(256, num_pose_classes)
        self.bn_cls = nn.BatchNorm1d(256)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        """
        Forward pass híbrido
        Returns:
            pose_regression: pose continua [B, 6]
            pose_classification: logits de clasificación [B, num_classes]
            feature_transform: para regularización
        """
        batch_size = x.size(0)
        
        # Feature extraction (igual que PointNet estándar)
        input_transform = self.stn3d(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, input_transform)
        x = x.transpose(2, 1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        
        feature_transform = self.stnkd(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, feature_transform)
        x = x.transpose(2, 1)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        # Global feature
        global_feature = torch.max(x, 2, keepdim=True)[0]
        global_feature = global_feature.view(-1, 1024)
        
        # Capa compartida
        shared_features = F.relu(self.bn_shared(self.fc_shared(global_feature)))
        shared_features = self.dropout(shared_features)
        
        # Head de regresión
        reg_features = F.relu(self.bn_reg(self.fc_reg1(shared_features)))
        reg_features = self.dropout(reg_features)
        pose_regression = self.fc_reg2(reg_features)
        
        # Head de clasificación
        cls_features = F.relu(self.bn_cls(self.fc_cls1(shared_features)))
        cls_features = self.dropout(cls_features)
        pose_classification = self.fc_cls2(cls_features)
        
        return pose_regression, pose_classification, feature_transform

# Función de utilidad para crear el modelo según configuración
def create_pose_model(config, model_type='regression'):
    """
    Crear modelo según configuración
    
    Args:
        config: configuración del sistema
        model_type: 'regression', 'classification', o 'hybrid'
    """
    model_config = config['model']
    
    if model_type == 'regression':
        model = PointNetPose(
            num_points=model_config.num_points,
            pose_dim=model_config.pose_dim,
            dropout=model_config.dropout,
            feature_transform=True
        )
    elif model_type == 'classification':
        model = PointNetPoseClassification(
            num_points=model_config.num_points,
            num_pose_classes=model_config.num_pose_classes,
            dropout=model_config.dropout,
            feature_transform=True
        )
    elif model_type == 'hybrid':
        model = PointNetPoseHybrid(
            num_points=model_config.num_points,
            pose_dim=model_config.pose_dim,
            num_pose_classes=model_config.num_pose_classes,
            dropout=model_config.dropout
        )
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")
    
    return model