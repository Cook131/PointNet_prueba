import numpy as np
import h5py
import os
from sklearn.model_selection import train_test_split

class PoseDataset:
    def __init__(self, data_path, num_points=1024):
        self.data_path = data_path
        self.num_points = num_points
        
    def load_modelnet_for_pose(self):
        """
        Adaptar ModelNet40 para pose estimation
        Cada objeto será rotado/trasladado con poses conocidas
        """
        # Cargar ModelNet40 base
        train_data, train_labels = self.load_h5_files('train')
        test_data, test_labels = self.load_h5_files('test')
        
        # Generar poses sintéticas
        train_poses = self.generate_synthetic_poses(len(train_data))
        test_poses = self.generate_synthetic_poses(len(test_data))
        
        # Aplicar transformaciones a las nubes de puntos
        train_transformed = self.apply_poses_to_pointclouds(train_data, train_poses)
        test_transformed = self.apply_poses_to_pointclouds(test_data, test_poses)
        
        return (train_transformed, train_poses), (test_transformed, test_poses)
    
    def generate_synthetic_poses(self, num_samples):
        """Generar poses sintéticas para entrenamiento"""
        poses = []
        
        for i in range(num_samples):
            # Rotación aleatoria (euler angles en radianes)
            rx = np.random.uniform(-np.pi, np.pi)
            ry = np.random.uniform(-np.pi, np.pi) 
            rz = np.random.uniform(-np.pi, np.pi)
            
            # Translación aleatoria (normalizada)
            tx = np.random.uniform(-0.5, 0.5)
            ty = np.random.uniform(-0.5, 0.5)
            tz = np.random.uniform(-0.5, 0.5)
            
            pose = np.array([tx, ty, tz, rx, ry, rz])
            poses.append(pose)
            
        return np.array(poses)
    
    def apply_poses_to_pointclouds(self, pointclouds, poses):
        """Aplicar transformaciones de pose a las nubes de puntos"""
        transformed_pcs = []
        
        for pc, pose in zip(pointclouds, poses):
            # Extraer parámetros
            tx, ty, tz, rx, ry, rz = pose
            
            # Crear matriz de rotación
            R = self.euler_to_rotation_matrix(rx, ry, rz)
            t = np.array([tx, ty, tz])
            
            # Aplicar transformación: pc_new = R * pc + t
            pc_transformed = np.dot(pc, R.T) + t
            transformed_pcs.append(pc_transformed)
            
        return np.array(transformed_pcs)
    
    def euler_to_rotation_matrix(self, rx, ry, rz):
        """Convertir ángulos de Euler a matriz de rotación"""
        # Matrices de rotación individuales
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx), np.cos(rx)]])
        
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz), np.cos(rz), 0],
                       [0, 0, 1]])
        
        # Combinación: R = Rz * Ry * Rx
        R = np.dot(Rz, np.dot(Ry, Rx))
        return R
    
    def save_pose_dataset(self, train_data, test_data, output_path):
        """Guardar dataset de pose en formato HDF5"""
        with h5py.File(os.path.join(output_path, 'pose_train.h5'), 'w') as f:
            f.create_dataset('data', data=train_data[0])
            f.create_dataset('poses', data=train_data[1])
            
        with h5py.File(os.path.join(output_path, 'pose_test.h5'), 'w') as f:
            f.create_dataset('data', data=test_data[0])
            f.create_dataset('poses', data=test_data[1])

# Uso del dataset
dataset = PoseDataset('./data')
train_data, test_data = dataset.load_modelnet_for_pose()
dataset.save_pose_dataset(train_data, test_data, './data/processed/')