#!/usr/bin/env python3
"""
Script principal para entrenar el modelo PointNet para estimación de pose del oil pan
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import sys
import os
import time
import json
from tqdm import tqdm
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports locales
from models.pointnet_pose import PointNetPose
from training.config import get_config
from utils.data_prep_pose import create_data_loaders, validate_dataset
from models.pointnet_utils import feature_transform_regularizer
from utils.visualization import PointCloudVisualizer
from training.evaluate_pose import PoseEvaluator

class PoseTrainer:
    """Clase principal para entrenar el modelo de estimación de pose"""
    
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        
        # Crear directorios necesarios
        os.makedirs(config['system'].model_save_dir, exist_ok=True)
        os.makedirs(config['system'].log_dir, exist_ok=True)
        os.makedirs(config['system'].checkpoints_dir, exist_ok=True)
        
        # Inicializar modelo
        self.model = PointNetPose(
            num_points=config['model'].num_points,
            pose_dim=config['model'].pose_dim,
            dropout=config['model'].dropout
        ).to(device)
        
        # Inicializar optimizador
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training'].learning_rate,
            weight_decay=config['training'].weight_decay
        )
        
        # Scheduler
        self.scheduler = StepLR(
            self.optimizer,
            step_size=config['training'].step_size,
            gamma=config['training'].gamma
        )
        
        # Criterio de pérdida
        self.criterion = nn.MSELoss()
        
        # Métricas de entrenamiento
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Inicializar visualizador y evaluador
        self.visualizer = PointCloudVisualizer()
        self.evaluator = PoseEvaluator(device)
        
    def train_epoch(self, train_loader):
        """Entrenar una época"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (points, poses) in enumerate(progress_bar):
            points = points.to(self.device)  # [B, 3, N]
            poses = poses.to(self.device)    # [B, 6]
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs, feature_transform = self.model(points)
            
            # Calcular pérdida principal
            pose_loss = self.criterion(outputs, poses)
            
            # Pérdida de regularización para la transformación de características
            reg_loss = feature_transform_regularizer(feature_transform)
            
            # Pérdida total
            total_loss_batch = pose_loss + self.config['training'].feature_transform_reg_weight * reg_loss
            
            # Backward pass
            total_loss_batch.backward()
            self.optimizer.step()
            
            # Acumular estadísticas
            total_loss += total_loss_batch.item() * points.size(0)
            total_samples += points.size(0)
            
            # Actualizar progress bar
            progress_bar.set_postfix({
                'Loss': f"{total_loss_batch.item():.4f}",
                'Pose Loss': f"{pose_loss.item():.4f}",
                'Reg Loss': f"{reg_loss.item():.4f}"
            })
            
            # Log cada cierto número de batches
            if batch_idx % self.config['system'].log_interval == 0:
                self.log_training_step(batch_idx, len(train_loader), total_loss_batch.item())
        
        return total_loss / total_samples
    
    def validate_epoch(self, val_loader):
        """Validar una época"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        all_predictions = []
        all_ground_truth = []
        
        with torch.no_grad():
            for points, poses in tqdm(val_loader, desc="Validation"):
                points = points.to(self.device)
                poses = poses.to(self.device)
                
                # Forward pass
                outputs, feature_transform = self.model(points)
                
                # Calcular pérdida
                pose_loss = self.criterion(outputs, poses)
                reg_loss = feature_transform_regularizer(feature_transform)
                total_loss_batch = pose_loss + self.config['training'].feature_transform_reg_weight * reg_loss
                
                # Acumular estadísticas
                total_loss += total_loss_batch.item() * points.size(0)
                total_samples += points.size(0)
                
                # Guardar predicciones para evaluación
                all_predictions.append(outputs.cpu())
                all_ground_truth.append(poses.cpu())
        
        # Calcular métricas de pose
        all_predictions = torch.cat(all_predictions, dim=0)
        all_ground_truth = torch.cat(all_ground_truth, dim=0)
        
        pose_metrics = self.evaluator.evaluate_pose_accuracy(all_predictions, all_ground_truth)
        
        return total_loss / total_samples, pose_metrics
    
    def train(self, train_loader, val_loader, num_epochs=None):
        """Bucle principal de entrenamiento"""
        if num_epochs is None:
            num_epochs = self.config['training'].num_epochs
            
        print(f"Iniciando entrenamiento por {num_epochs} épocas...")
        print(f"Modelo: {self.model.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Parámetros del modelo: {sum(p.numel() for p in self.model.parameters())}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Entrenar
            train_loss = self.train_epoch(train_loader)
            
            # Validar
            val_loss, pose_metrics = self.validate_epoch(val_loader)
            
            # Actualizar scheduler
            self.scheduler.step()
            
            # Guardar métricas
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            epoch_time = time.time() - epoch_start_time
            
            # Imprimir progreso
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            print(f"Rotation Accuracy: {pose_metrics['rotation_accuracy']:.4f}")
            print(f"Translation Accuracy: {pose_metrics['translation_accuracy']:.4f}")
            print(f"Combined Accuracy: {pose_metrics['combined_accuracy']:.4f}")
            print(f"Mean Rotation Error: {pose_metrics['mean_rotation_error_deg']:.2f}°")
            print(f"Epoch Time: {epoch_time:.2f}s")
            print("-" * 50)
            
            # Guardar mejor modelo
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss, pose_metrics, is_best=True)
                print(f"✓ Nuevo mejor modelo guardado (val_loss: {val_loss:.6f})")
            else:
                self.patience_counter += 1
            
            # Guardar checkpoint periódico
            if (epoch + 1) % self.config['system'].save_interval == 0:
                self.save_checkpoint(epoch, val_loss, pose_metrics, is_best=False)
            
            # Early stopping
            if self.patience_counter >= self.config['training'].patience:
                print(f"Early stopping activado después de {epoch+1} épocas")
                break
            
            # Visualización periódica
            if (epoch + 1) % self.config['system'].vis_interval == 0:
                self.visualize_training_progress(epoch)
        
        total_time = time.time() - start_time
        print(f"\nEntrenamiento completado en {total_time/3600:.2f} horas")
        
        # Guardar métricas finales
        self.save_training_metrics()
        
    def save_checkpoint(self, epoch, val_loss, pose_metrics, is_best=False):
        """Guardar checkpoint del modelo"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'pose_metrics': pose_metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        if is_best:
            filepath = os.path.join(self.config['system'].model_save_dir, 'best_model.pth')
        else:
            filepath = os.path.join(self.config['system'].checkpoints_dir, f'checkpoint_epoch_{epoch+1}.pth')
        
        torch.save(checkpoint, filepath)
        
    def load_checkpoint(self, filepath):
        """Cargar checkpoint del modelo"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        return checkpoint['epoch'], checkpoint['val_loss']
    
    def save_training_metrics(self):
        """Guardar métricas de entrenamiento"""
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        filepath = os.path.join(self.config['system'].log_dir, 'training_metrics.json')
        with open(filepath, 'w') as f:
            # Convertir config a dict serializable
            serializable_config = {}
            for key, value in self.config.items():
                if hasattr(value, '__dict__'):
                    serializable_config[key] = value.__dict__
                else:
                    serializable_config[key] = value
            
            metrics['config'] = serializable_config
            json.dump(metrics, f, indent=2)
    
    def visualize_training_progress(self, epoch):
        """Visualizar progreso del entrenamiento"""
        self.visualizer.plot_training_curves(
            self.train_losses,
            self.val_losses,
            save_path=os.path.join(self.config['system'].log_dir, f'training_curves_epoch_{epoch+1}.png')
        )
    
    def log_training_step(self, batch_idx, total_batches, loss):
        """Log de paso de entrenamiento"""
        if batch_idx % self.config['system'].log_interval == 0:
            progress = 100.0 * batch_idx / total_batches
            print(f'Batch {batch_idx}/{total_batches} ({progress:.1f}%) - Loss: {loss:.6f}')

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Entrenar PointNet para estimación de pose')
    parser.add_argument('--ply_path', type=str, required=True,
                        help='Ruta al archivo .ply del oil pan')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Número de épocas de entrenamiento')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Tamaño del batch')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Ruta a checkpoint para resumir entrenamiento')
    
    args = parser.parse_args()
    
    # Validar archivo PLY
    if not validate_dataset(args.ply_path):
        print("Error: No se puede cargar el archivo PLY especificado")
        return
    
    # Configurar device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Usando device: {device}")
    
    # Cargar configuración
    config = get_config()
    
    # Actualizar configuración con argumentos
    config['training'].num_epochs = args.epochs
    config['training'].batch_size = args.batch_size
    config['training'].learning_rate = args.lr
    config['system'].device = device
    
    # Crear data loaders
    print("Creando datasets...")
    train_loader, val_loader, test_loader = create_data_loaders(args.ply_path, config)
    
    # Inicializar trainer
    trainer = PoseTrainer(config, device)
    
    # Resumir entrenamiento si se especifica
    start_epoch = 0
    if args.resume:
        print(f"Resumiendo entrenamiento desde: {args.resume}")
        start_epoch, _ = trainer.load_checkpoint(args.resume)
        start_epoch += 1
    
    # Entrenar
    trainer.train(train_loader, val_loader, args.epochs - start_epoch)
    
    # Evaluación final
    print("\nEvaluando modelo final...")
    from evaluate_pose import evaluate_model_complete
    evaluate_model_complete(trainer.model, test_loader, device, './results/final_evaluation')
    
    print("¡Entrenamiento completado!")

if __name__ == "__main__":
    main()