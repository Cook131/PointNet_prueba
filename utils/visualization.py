import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def visualize_pose_prediction(pointcloud, pred_pose, gt_pose=None):
    """Visualizar nube de puntos con pose predicha y ground truth"""
    
    # Crear nube de puntos
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gris
    
    # Crear ejes de coordenadas para pose predicha (rojo)
    pred_frame = create_coordinate_frame(pred_pose, size=0.3, color=[1, 0, 0])
    
    geometries = [pcd, pred_frame]
    
    # Si hay ground truth, mostrar en verde
    if gt_pose is not None:
        gt_frame = create_coordinate_frame(gt_pose, size=0.3, color=[0, 1, 0])
        geometries.append(gt_frame)
    
    # Visualizar
    o3d.visualization.draw_geometries(geometries)

def create_coordinate_frame(pose, size=0.3, color=[1, 0, 0]):
    """Crear marco de coordenadas a partir de pose"""
    tx, ty, tz, rx, ry, rz = pose
    
    # Crear matriz de transformación
    R = euler_to_rotation_matrix(rx, ry, rz)
    t = np.array([tx, ty, tz])
    
    # Crear ejes
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    
    # Aplicar transformación
    frame.rotate(R, center=(0, 0, 0))
    frame.translate(t)
    
    return frame

def euler_to_rotation_matrix(rx, ry, rz):
    """Convierte ángulos de Euler (en radianes) a matriz de rotación 3x3 (orden ZYX)"""
    # Rotación alrededor de Z
    cz = np.cos(rz)
    sz = np.sin(rz)
    Rz = np.array([
        [cz, -sz, 0],
        [sz,  cz, 0],
        [ 0,   0, 1]
    ])
    # Rotación alrededor de Y
    cy = np.cos(ry)
    sy = np.sin(ry)
    Ry = np.array([
        [cy, 0, sy],
        [ 0, 1,  0],
        [-sy, 0, cy]
    ])
    # Rotación alrededor de X
    cx = np.cos(rx)
    sx = np.sin(rx)
    Rx = np.array([
        [1,  0,   0],
        [0, cx, -sx],
        [0, sx,  cx]
    ])
    # Orden ZYX: R = Rz @ Ry @ Rx
    return Rz @ Ry @ Rx

def plot_training_curves(losses, errors):
    """Graficar curvas de entrenamiento"""
    epochs = range(len(losses))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Pérdidas
    ax1.plot(epochs, losses['train'], label='Train Loss')
    ax1.plot(epochs, losses['val'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Curves - Loss')
    ax1.legend()
    
    # Errores
    ax2.plot(epochs, errors['translation'], label='Translation Error')
    ax2.plot(epochs, errors['rotation'], label='Rotation Error')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Error')
    ax2.set_title('Training Curves - Errors')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()