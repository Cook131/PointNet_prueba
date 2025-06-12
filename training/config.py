import os
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ModelConfig:
    """Configuración del modelo PointNet"""
    input_dim: int = 3  # x, y, z coordinates
    num_points: int = 1024  # Número de puntos por nube
    feature_dim: int = 64  # Dimensión de características
    dropout: float = 0.3
    use_batch_norm: bool = True
    
    # Para estimación de pose
    pose_dim: int = 6  # 3 para rotación (euler angles) + 3 para traslación
    num_pose_classes: int = 24  # Número de poses discretas (cada 15 grados)

@dataclass  
class TrainingConfig:
    """Configuración de entrenamiento"""
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 200
    weight_decay: float = 0.0005
    
    # Scheduler
    step_size: int = 20
    gamma: float = 0.7
    
    # Regularización
    feature_transform_reg_weight: float = 0.001
    
    # Early stopping
    patience: int = 20
    min_delta: float = 0.001

@dataclass
class DataConfig:
    """Configuración de datos"""
    data_dir: str = "./data"
    processed_dir: str = "./data/processed"
    augmented_dir: str = "./data/augmented"
    
    # Rutas específicas para oil pan
    oil_pan_ply: str = "./OilPan/CenteredPointClouds/oil_pan_full_pc_10000.ply"
    
    # Augmentación
    rotation_range: float = 360.0  # grados
    noise_std: float = 0.01
    scale_range: Tuple[float, float] = (0.8, 1.2)
    
    # Validación
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

@dataclass
class PoseConfig:
    """Configuración específica para estimación de pose"""
    # Definir rangos de pose para oil pan
    rotation_x_range: Tuple[float, float] = (-45, 45)  # pitch
    rotation_y_range: Tuple[float, float] = (-45, 45)  # yaw  
    rotation_z_range: Tuple[float, float] = (-180, 180)  # roll
    
    # Pasos de discretización
    rotation_step: float = 15.0  # grados
    
    # Para generar vistas sintéticas
    camera_distance: float = 2.0
    camera_positions: List[Tuple[float, float, float]] = None
    
    def __post_init__(self):
        if self.camera_positions is None:
            # Generar posiciones de cámara esféricas
            import numpy as np
            positions = []
            for theta in np.arange(0, 360, 30):  # azimuth
                for phi in np.arange(30, 150, 30):  # elevation
                    x = self.camera_distance * np.sin(np.radians(phi)) * np.cos(np.radians(theta))
                    y = self.camera_distance * np.cos(np.radians(phi))
                    z = self.camera_distance * np.sin(np.radians(phi)) * np.sin(np.radians(theta))
                    positions.append((x, y, z))
            self.camera_positions = positions

@dataclass
class SystemConfig:
    """Configuración general del sistema"""
    # Rutas de salida
    model_save_dir: str = "./models"
    log_dir: str = "./logs"
    results_dir: str = "./results"
    checkpoints_dir: str = "./checkpoints"
    
    # Device
    device: str = "cuda"  # "cuda" o "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Logging
    log_interval: int = 10
    save_interval: int = 50
    
    # Visualización
    vis_interval: int = 100
    num_vis_samples: int = 8
    
    def __post_init__(self):
        # Crear directorios si no existen
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

# Configuración global del sistema
model_config = ModelConfig()
training_config = TrainingConfig()
data_config = DataConfig()
pose_config = PoseConfig()
system_config = SystemConfig()

# Configuraciones específicas para diferentes tareas
POSE_ESTIMATION_CONFIG = {
    'model': model_config,
    'training': training_config,
    'data': data_config,
    'pose': pose_config,
    'system': system_config
}

def get_config():
    """Obtener configuración completa del sistema"""
    return POSE_ESTIMATION_CONFIG

def update_config(**kwargs):
    """Actualizar configuración con nuevos valores"""
    config = get_config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config