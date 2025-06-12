#!/usr/bin/env python3
"""
Script de inferencia para usar el modelo PointNet entrenado
Ubicación: inference/predict_pose.py
"""

import torch
import numpy as np
import open3d as o3d
import argparse
import os
import sys
from scipy.spatial.transform import Rotation as R

# Agregar el directorio padre al path para imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports corregidos para tu estructura
from models.pointnet_pose import PointNetPose
from models.pointnet_utils import pc_normalize
from utils.visualization import PointCloudVisualizer
from training.config import get_config

class PosePredictor:
    """Clase para hacer inferencia con el modelo entrenado"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.config = get_config()
        
        # Cargar modelo
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Inicializar visualizador
        self.visualizer = PointCloudVisualizer()
        
    def load_model(self, model_path):
        """Cargar modelo entrenado"""
        # Ajustar ruta relativa desde inference/
        if not os.path.isabs(model_path):
            model_path = os.path.join('..', model_path)
            
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Crear modelo
        model = PointNetPose(
            num_points=self.config['model'].num_points,
            pose_dim=self.config['model'].pose_dim,
            dropout=self.config['model'].dropout
        ).to(self.device)
        
        # Cargar pesos
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✓ Modelo cargado desde: {model_path}")
        print(f"✓ Época del modelo: {checkpoint['epoch']}")
        print(f"✓ Pérdida de validación: {checkpoint['val_loss']:.6f}")
        
        return model
    
    def preprocess_point_cloud(self, point_cloud_path):
        """Preprocesar nube de puntos para inferencia"""
        # Ajustar ruta relativa desde inference/
        if not os.path.isabs(point_cloud_path):
            point_cloud_path = os.path.join('..', point_cloud_path)
            
        # Cargar nube de puntos
        if point_cloud_path.endswith('.ply'):
            pcd = o3d.io.read_point_cloud(point_cloud_path)
            points = np.asarray(pcd.points)
        else:
            raise ValueError("Solo se soportan archivos .ply")
        
        print(f"✓ Cargados {len(points)} puntos desde {os.path.basename(point_cloud_path)}")
        
        # Normalizar
        points = pc_normalize(points)
        
        # Submuestrear o padding al número correcto de puntos
        target_points = self.config['model'].num_points
        
        if len(points) > target_points:
            indices = np.random.choice(len(points), target_points, replace=False)
            points = points[indices]
            print(f"✓ Submuestreado a {target_points} puntos")
        elif len(points) < target_points:
            padding_needed = target_points - len(points)
            padding_indices = np.random.choice(len(points), padding_needed, replace=True)
            padding_points = points[padding_indices]
            points = np.concatenate([points, padding_points])
            print(f"✓ Expandido a {target_points} puntos con padding")
        
        # Convertir a tensor y ajustar dimensiones
        points_tensor = torch.from_numpy(points).float()
        points_tensor = points_tensor.transpose(1, 0).unsqueeze(0)  # [1, 3, N]
        
        return points_tensor.to(self.device), points
    
    def predict_pose(self, point_cloud_path, visualize=True, save_results=None):
        """Predecir pose de una nube de puntos"""
        print(f"\n🔍 Prediciendo pose para: {point_cloud_path}")
        
        # Preprocesar
        points_tensor, original_points = self.preprocess_point_cloud(point_cloud_path)
        
        # Inferencia
        with torch.no_grad():
            pose_pred, _ = self.model(points_tensor)
            pose_pred = pose_pred.cpu().numpy()[0]  # [6]
        
        # Convertir a formato legible
        pose_readable = self.format_pose_output(pose_pred)
        
        # Mostrar resultados
        self.print_results(point_cloud_path, pose_readable)
        
        # Visualizar si se solicita
        if visualize:
            self.visualize_result(original_points, pose_pred, save_results)
        
        return pose_pred, pose_readable
    
    def format_pose_output(self, pose):
        """Formatear salida de pose para lectura humana"""
        rotation_rad = pose[:3]
        translation = pose[3:6]
        
        # Convertir rotación a grados
        rotation_deg = np.degrees(rotation_rad)
        
        # Convertir a matriz de rotación
        rotation_matrix = R.from_euler('xyz', rotation_rad).as_matrix()
        
        return {
            'rotation_rad': rotation_rad,
            'rotation_deg': rotation_deg,
            'translation': translation,
            'rotation_matrix': rotation_matrix,
            'pose_vector': pose
        }
    
    def print_results(self, file_path, pose_readable):
        """Imprimir resultados formateados"""
        print(f"\n📊 RESULTADOS DE ESTIMACIÓN DE POSE")
        print(f"{'='*50}")
        print(f"Archivo: {os.path.basename(file_path)}")
        print(f"\n🔄 ROTACIÓN (grados):")
        print(f"  • Pitch (X): {pose_readable['rotation_deg'][0]:+7.1f}°")
        print(f"  • Yaw   (Y): {pose_readable['rotation_deg'][1]:+7.1f}°") 
        print(f"  • Roll  (Z): {pose_readable['rotation_deg'][2]:+7.1f}°")
        print(f"\n📍 TRASLACIÓN:")
        print(f"  • X: {pose_readable['translation'][0]:+7.3f}")
        print(f"  • Y: {pose_readable['translation'][1]:+7.3f}")
        print(f"  • Z: {pose_readable['translation'][2]:+7.3f}")
        print(f"{'='*50}")
    
    def visualize_result(self, points, pose, save_path=None):
        """Visualizar resultado de estimación de pose"""
        print(f"\n🎨 Generando visualización...")
        
        # Crear visualización con pose aplicada
        transformed_points = self.visualizer.apply_pose_transform(points, pose)
        
        # Plotear comparación
        fig = self.visualizer.plot_point_cloud_plotly(
            transformed_points, 
            title="🔧 Oil Pan - Pose Estimada"
        )
        
        if save_path:
            # Ajustar ruta de salida relativa
            if not os.path.isabs(save_path):
                save_path = os.path.join('..', save_path)
            html_path = f"{save_path}_pose_result.html"
            fig.write_html(html_path)
            print(f"✓ Visualización guardada en: {html_path}")
        else:
            fig.show()
            print(f"✓ Visualización mostrada en navegador")

def main():
    """Función principal del script de inferencia"""
    parser = argparse.ArgumentParser(
        description='🤖 Inferencia de pose con PointNet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso (desde la carpeta inference/):

  # Archivo individual
  python predict_pose.py --model ../models/best_model.pth --input ../OilPan/CenteredPointClouds/oil_pan_full_pc_10000.ply

  # Con visualización guardada
  python predict_pose.py --model ../models/best_model.pth --input ../test.ply --output ../results/test

  # Sin visualización
  python predict_pose.py --model ../models/best_model.pth --input ../test.ply --no-visualize
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                        help='Ruta al modelo entrenado (.pth)')
    parser.add_argument('--input', type=str, required=True,
                        help='Archivo .ply de entrada')
    parser.add_argument('--output', type=str, default=None,
                        help='Ruta base para guardar resultados')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Desactivar visualizaciones')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    print(f"🚀 Iniciando inferencia de pose PointNet")
    print(f"{'='*50}")
    
    # Verificar que el modelo existe
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.join('..', model_path)
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Modelo no encontrado en {model_path}")
        return
    
    # Crear predictor
    try:
        predictor = PosePredictor(args.model, args.device)
    except Exception as e:
        print(f"❌ Error al cargar modelo: {e}")
        return
    
    # Verificar archivo de entrada
    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join('..', input_path)
        
    if not os.path.exists(input_path):
        print(f"❌ Error: Archivo no encontrado en {input_path}")
        return
    
    try:
        predictor.predict_pose(
            args.input,  # Usar ruta original para manejo interno
            visualize=not args.no_visualize,
            save_results=args.output
        )
        print(f"\n✅ ¡Inferencia completada exitosamente!")
        
    except Exception as e:
        print(f"❌ Error durante la inferencia: {e}")
        return

if __name__ == "__main__":
    main()