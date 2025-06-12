import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import os
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from training.config import get_config
from models.pointnet_utils import pc_normalize, rotate_point_cloud, jitter_point_cloud

class OilPanPoseDataset(Dataset):
    """
    Dataset para entrenamiento de estimación de pose del oil pan
    Genera vistas sintéticas desde diferentes ángulos
    """
    
    def __init__(self, ply_file_path, num_samples=5000, num_points=1024, 
                 augment=True, split='train'):
        self.ply_file_path = ply_file_path
        self.num_samples = num_samples
        self.num_points = num_points
        self.augment = augment
        self.split = split
        
        # Cargar configuración
        self.config = get_config()
        
        # Cargar el mesh original
        self.original_mesh = self.load_original_mesh()
        
        # Generar dataset sintético
        self.generate_synthetic_dataset()
        
    def load_original_mesh(self):
        """Cargar el mesh original del oil pan"""
        if self.ply_file_path.endswith('.ply'):
            mesh = o3d.io.read_point_cloud(self.ply_file_path)
        else:
            raise ValueError("Solo se soportan archivos .ply")
            
        # Normalizar a esfera unitaria
        points = np.asarray(mesh.points)
        points = pc_normalize(points)
        
        mesh.points = o3d.utility.Vector3dVector(points)
        return mesh
    
    def generate_synthetic_dataset(self):
        """
        Generar dataset sintético con diferentes poses
        """
        self.point_clouds = []
        self.poses = []
        
        print(f"Generando {self.num_samples} muestras sintéticas...")
        
        for i in tqdm(range(self.num_samples)):
            # Generar pose aleatoria
            pose = self.generate_random_pose()
            
            # Aplicar transformación al objeto
            transformed_points = self.apply_pose_to_points(
                np.asarray(self.original_mesh.points), pose
            )
            
            # Simular vista desde cámara (sampling parcial)
            visible_points = self.simulate_camera_view(transformed_points)
            
            # Submuestrear a num_points
            if len(visible_points) > self.num_points:
                indices = np.random.choice(len(visible_points), self.num_points, replace=False)
                visible_points = visible_points[indices]
            elif len(visible_points) < self.num_points:
                # Pad con puntos aleatorios si hay muy pocos
                padding_needed = self.num_points - len(visible_points)
                padding_indices = np.random.choice(len(visible_points), padding_needed, replace=True)
                padding_points = visible_points[padding_indices]
                visible_points = np.concatenate([visible_points, padding_points])
            
            self.point_clouds.append(visible_points)
            self.poses.append(pose)
        
        # Convertir a arrays
        self.point_clouds = np.array(self.point_clouds)
        self.poses = np.array(self.poses)
        
        print(f"Dataset generado: {len(self.point_clouds)} muestras")
    
    def generate_random_pose(self):
        """
        Generar pose aleatoria dentro de rangos realistas
        """
        pose_config = self.config['pose']
        
        # Rotaciones (en radianes)
        rx = np.random.uniform(
            np.radians(pose_config.rotation_x_range[0]),
            np.radians(pose_config.rotation_x_range[1])
        )
        ry = np.random.uniform(
            np.radians(pose_config.rotation_y_range[0]),
            np.radians(pose_config.rotation_y_range[1])
        )
        rz = np.random.uniform(
            np.radians(pose_config.rotation_z_range[0]),
            np.radians(pose_config.rotation_z_range[1])
        )
        
        # Traslaciones pequeñas
        tx = np.random.uniform(-0.2, 0.2)
        ty = np.random.uniform(-0.2, 0.2)
        tz = np.random.uniform(-0.2, 0.2)
        
        return np.array([rx, ry, rz, tx, ty, tz])
    
    def apply_pose_to_points(self, points, pose):
        """
        Aplicar transformación de pose a los puntos
        """
        # Separar rotación y traslación
        rotation = pose[:3]  # ángulos de Euler
        translation = pose[3:6]
        
        # Crear matriz de rotación
        R_matrix = R.from_euler('xyz', rotation).as_matrix()
        
        # Aplicar transformación
        transformed_points = (R_matrix @ points.T).T + translation
        
        return transformed_points
    
    def simulate_camera_view(self, points):
        """
        Simular vista parcial desde una cámara
        Elimina puntos que no serían visibles
        """
        # Simular oclusión simple: eliminar puntos detrás de un plano
        camera_position = np.array([0, 0, 2])  # Cámara en Z=2
        camera_direction = np.array([0, 0, -1])  # Mirando hacia -Z
        
        # Calcular vectores desde cámara a puntos
        to_points = points - camera_position
        
        # Mantener solo puntos en el hemisferio visible
        dot_products = np.dot(to_points, camera_direction)
        visible_mask = dot_products > 0
        
        # Agregar algo de ruido en la visibilidad
        noise = np.random.random(len(visible_mask)) > 0.1  # 90% de puntos visibles son mantenidos
        visible_mask = visible_mask & noise
        
        return points[visible_mask]
    
    def __len__(self):
        return len(self.point_clouds)
    
    def __getitem__(self, idx):
        points = self.point_clouds[idx].copy()
        pose = self.poses[idx].copy()
        
        # Augmentación durante entrenamiento
        if self.augment and self.split == 'train':
            # Agregar ruido
            points = jitter_point_cloud(points[None, ...], sigma=0.01, clip=0.05)[0]
            
            # Rotación aleatoria pequeña
            if np.random.random() > 0.5:
                points = rotate_point_cloud(points[None, ...])[0]
        
        # Convertir a tensores
        points = torch.from_numpy(points).float()
        pose = torch.from_numpy(pose).float()
        
        return points.transpose(1, 0), pose  # [3, N] para PointNet

def create_oil_pan_datasets(ply_file_path, config):
    """
    Crear datasets de entrenamiento, validación y prueba
    """
    total_samples = 10000  # Aumentar para mejor rendimiento
    
    # Calcular tamaños de splits
    train_size = int(total_samples * config['data'].train_split)
    val_size = int(total_samples * config['data'].val_split)
    test_size = total_samples - train_size - val_size
    
    # Crear datasets
    train_dataset = OilPanPoseDataset(
        ply_file_path, num_samples=train_size, 
        num_points=config['model'].num_points, 
        augment=True, split='train'
    )
    
    val_dataset = OilPanPoseDataset(
        ply_file_path, num_samples=val_size,
        num_points=config['model'].num_points,
        augment=False, split='val'
    )
    
    test_dataset = OilPanPoseDataset(
        ply_file_path, num_samples=test_size,
        num_points=config['model'].num_points,
        augment=False, split='test'
    )
    
    return train_dataset, val_dataset, test_dataset

def create_data_loaders(ply_file_path, config):
    """
    Crear data loaders para entrenamiento
    """
    train_dataset, val_dataset, test_dataset = create_oil_pan_datasets(ply_file_path, config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training'].batch_size,
        shuffle=True,
        num_workers=config['system'].num_workers,
        pin_memory=config['system'].pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training'].batch_size,
        shuffle=False,
        num_workers=config['system'].num_workers,
        pin_memory=config['system'].pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training'].batch_size,
        shuffle=False,
        num_workers=config['system'].num_workers,
        pin_memory=config['system'].pin_memory
    )
    
    return train_loader, val_loader, test_loader

# Función para validar el dataset
def validate_dataset(ply_file_path):
    """
    Validar que el archivo PLY se puede cargar correctamente
    """
    try:
        # Intentar cargar el archivo
        point_cloud = o3d.io.read_point_cloud(ply_file_path)
        points = np.asarray(point_cloud.points)
        
        print(f"✓ Archivo PLY cargado exitosamente")
        print(f"✓ Número de puntos: {len(points)}")
        print(f"✓ Rango X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
        print(f"✓ Rango Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
        print(f"✓ Rango Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        
        # Verificar si tiene colores
        if point_cloud.has_colors():
            print(f"✓ El archivo tiene información de color")
        else:
            print(f"✓ El archivo no tiene información de color (solo geometría)")
            
        # Verificar si tiene normales
        if point_cloud.has_normals():
            print(f"✓ El archivo tiene normales")
        else:
            print(f"✓ El archivo no tiene normales")
            
        return True
        
    except Exception as e:
        print(f"✗ Error al cargar archivo PLY: {e}")
        return False

if __name__ == "__main__":
    # Ejemplo de uso
    ply_path = "./OilPan/CenteredPointClouds/oil_pan_full_pc_10000.ply"
    
    # Validar dataset
    if validate_dataset(ply_path):
        print("Creando datasets...")
        config = get_config()
        train_loader, val_loader, test_loader = create_data_loaders(ply_path, config)
        
        print(f"Dataset creado exitosamente!")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Mostrar ejemplo
        for points, poses in train_loader:
            print(f"Batch shape - Points: {points.shape}, Poses: {poses.shape}")
            break