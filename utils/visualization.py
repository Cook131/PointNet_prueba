import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import open3d as o3d
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots



class PointCloudVisualizer:
    """Clase para visualizar nubes de puntos y resultados de pose"""
    
    def __init__(self):
        self.fig_size = (12, 8)
        
    def plot_point_cloud_matplotlib(self, points, colors=None, title="Point Cloud", save_path=None):
        """
        Visualizar nube de puntos con matplotlib
        """
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.add_subplot(111, projection='3d')
        
        if colors is not None:
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                               c=colors, cmap='viridis', s=1)
            plt.colorbar(scatter)
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # Hacer que los ejes tengan la misma escala
        max_range = np.array([points[:,0].max()-points[:,0].min(),
                             points[:,1].max()-points[:,1].min(),
                             points[:,2].max()-points[:,2].min()]).max() / 2.0
        mid_x = (points[:,0].max()+points[:,0].min()) * 0.5
        mid_y = (points[:,1].max()+points[:,1].min()) * 0.5
        mid_z = (points[:,2].max()+points[:,2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_point_cloud_plotly(self, points, colors=None, title="Point Cloud"):
        """
        Visualizar nube de puntos con plotly (interactivo)
        """
        if colors is not None:
            fig = go.Figure(data=[go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1], 
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=colors,
                    colorscale='Viridis',
                    showscale=True
                )
            )])
        else:
            fig = go.Figure(data=[go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(size=2, color='blue')
            )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            )
        )
        
        return fig
    
    def visualize_pose_estimation(self, points, true_pose, pred_pose, save_path=None):
        """
        Visualizar comparación entre pose verdadera y predicha
        """
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=('True Pose', 'Predicted Pose')
        )
        
        # Aplicar transformaciones de pose
        true_transformed = self.apply_pose_transform(points, true_pose)
        pred_transformed = self.apply_pose_transform(points, pred_pose)
        
        # Plot true pose
        fig.add_trace(
            go.Scatter3d(
                x=true_transformed[:, 0],
                y=true_transformed[:, 1],
                z=true_transformed[:, 2],
                mode='markers',
                marker=dict(size=2, color='green'),
                name='True'
            ),
            row=1, col=1
        )
        
        # Plot predicted pose
        fig.add_trace(
            go.Scatter3d(
                x=pred_transformed[:, 0],
                y=pred_transformed[:, 1],
                z=pred_transformed[:, 2],
                mode='markers',
                marker=dict(size=2, color='red'),
                name='Predicted'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Pose Estimation Results",
            scene=dict(aspectmode='cube'),
            scene2=dict(aspectmode='cube')
        )
        
        return fig
    
    def apply_pose_transform(self, points, pose):
        """
        Aplicar transformación de pose a los puntos
        pose: [rx, ry, rz, tx, ty, tz] (euler angles + translation)
        """
        # Convertir a numpy si es tensor
        if torch.is_tensor(points):
            points = points.cpu().numpy()
        if torch.is_tensor(pose):
            pose = pose.cpu().numpy()
            
        # Separar rotación y traslación
        rotation = pose[:3]  # euler angles en radianes
        translation = pose[3:6]
        
        # Crear matriz de rotación desde ángulos de Euler
        rx, ry, rz = rotation
        
        # Matriz de rotación X
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        # Matriz de rotación Y  
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        # Matriz de rotación Z
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        # Rotación combinada
        R = Rz @ Ry @ Rx
        
        # Aplicar transformación
        transformed_points = (R @ points.T).T + translation
        
        return transformed_points
    
    def plot_training_curves(self, train_losses, val_losses, train_accs=None, val_accs=None, save_path=None):
        """
        Visualizar curvas de entrenamiento
        """
        epochs = range(1, len(train_losses) + 1)
        
        if train_accs is not None and val_accs is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Loss
            ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
            ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Accuracy
            ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')  
            ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
            ax2.set_title('Training and Validation Accuracy')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
            ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
            ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pose_distribution(self, poses, title="Pose Distribution", save_path=None):
        """
        Visualizar distribución de poses
        """
        if torch.is_tensor(poses):
            poses = poses.cpu().numpy()
            
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(title)
        
        pose_names = ['Rotation X', 'Rotation Y', 'Rotation Z', 
                     'Translation X', 'Translation Y', 'Translation Z']
        
        for i, (ax, name) in enumerate(zip(axes.flat, pose_names)):
            ax.hist(poses[:, i], bins=30, alpha=0.7, edgecolor='black')
            ax.set_title(name)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_feature_maps(self, features, title="Feature Maps", save_path=None):
        """
        Visualizar mapas de características
        """
        if torch.is_tensor(features):
            features = features.cpu().numpy()
        
        # Tomar las primeras 16 características para visualizar
        num_features = min(16, features.shape[1])
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle(title)
        
        for i in range(num_features):
            ax = axes[i // 4, i % 4]
            im = ax.imshow(features[0, i].reshape(-1, 1), cmap='viridis', aspect='auto')
            ax.set_title(f'Feature {i+1}')
            ax.axis('off')
            
        # Ocultar axes vacíos
        for i in range(num_features, 16):
            axes[i // 4, i % 4].axis('off')
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Funciones de utilidad
def visualize_batch_predictions(points_batch, true_poses, pred_poses, num_samples=4):
    """
    Visualizar predicciones de un batch
    """
    visualizer = PointCloudVisualizer()
    
    fig = make_subplots(
        rows=num_samples, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}] for _ in range(num_samples)],
        subplot_titles=[f'Sample {i+1} - True' if j == 0 else f'Sample {i+1} - Pred' 
                       for i in range(num_samples) for j in range(2)]
    )
    
    for i in range(min(num_samples, len(points_batch))):
        points = points_batch[i]
        true_pose = true_poses[i]
        pred_pose = pred_poses[i]
        
        # Transformar puntos
        true_transformed = visualizer.apply_pose_transform(points, true_pose)
        pred_transformed = visualizer.apply_pose_transform(points, pred_pose)
        
        # Agregar trazas
        fig.add_trace(
            go.Scatter3d(
                x=true_transformed[:, 0],
                y=true_transformed[:, 1],
                z=true_transformed[:, 2],
                mode='markers',
                marker=dict(size=2, color='green'),
                name=f'True {i+1}',
                showlegend=False
            ),
            row=i+1, col=1
        )
        
        fig.add_trace(
            go.Scatter3d(
                x=pred_transformed[:, 0],
                y=pred_transformed[:, 1],
                z=pred_transformed[:, 2],
                mode='markers',
                marker=dict(size=2, color='red'),
                name=f'Pred {i+1}',
                showlegend=False
            ),
            row=i+1, col=2
        )
    
    fig.update_layout(
        title="Batch Predictions Comparison",
        height=200 * num_samples
    )
    
    return fig

def save_visualization_report(visualizer, results_dict, save_dir):
    """
    Guardar reporte completo de visualizaciones
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Guardar curvas de entrenamiento
    if 'train_losses' in results_dict:
        visualizer.plot_training_curves(
            results_dict['train_losses'],
            results_dict['val_losses'],
            results_dict.get('train_accs'),
            results_dict.get('val_accs'),
            save_path=os.path.join(save_dir, 'training_curves.png')
        )
    
    # Guardar distribución de poses
    if 'poses' in results_dict:
        visualizer.plot_pose_distribution(
            results_dict['poses'],
            save_path=os.path.join(save_dir, 'pose_distribution.png')
        )
    
    # Guardar ejemplos de predicciones
    if 'predictions' in results_dict:
        fig = visualize_batch_predictions(
            results_dict['points'],
            results_dict['true_poses'],
            results_dict['predictions']
        )
        fig.write_html(os.path.join(save_dir, 'predictions_comparison.html'))
    
    print(f"Visualizations saved to {save_dir}")