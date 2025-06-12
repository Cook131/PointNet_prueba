import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R

class PoseEvaluator:
    """Clase para evaluar resultados de estimación de pose"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
    def evaluate_pose_accuracy(self, predictions, ground_truth, threshold_rotation=15.0, threshold_translation=0.1):
        """
        Evaluar precisión de estimación de pose
        
        Args:
            predictions: tensor [N, 6] - poses predichas [rx, ry, rz, tx, ty, tz]
            ground_truth: tensor [N, 6] - poses verdaderas
            threshold_rotation: umbral en grados para considerarse correcto
            threshold_translation: umbral en unidades para considerarse correcto
        """
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(ground_truth):
            ground_truth = ground_truth.cpu().numpy()
            
        # Separar rotación y traslación
        pred_rot = predictions[:, :3]  # radianes
        pred_trans = predictions[:, 3:6]
        true_rot = ground_truth[:, :3]
        true_trans = ground_truth[:, 3:6]
        
        # Evaluar rotación
        rotation_errors = self.compute_rotation_error(pred_rot, true_rot)
        rotation_accuracy = np.mean(rotation_errors < np.radians(threshold_rotation))
        
        # Evaluar traslación
        translation_errors = np.linalg.norm(pred_trans - true_trans, axis=1)
        translation_accuracy = np.mean(translation_errors < threshold_translation)
        
        # Precisión combinada
        combined_accuracy = np.mean(
            (rotation_errors < np.radians(threshold_rotation)) & 
            (translation_errors < threshold_translation)
        )
        
        results = {
            'rotation_accuracy': rotation_accuracy,
            'translation_accuracy': translation_accuracy,
            'combined_accuracy': combined_accuracy,
            'rotation_errors_deg': np.degrees(rotation_errors),
            'translation_errors': translation_errors,
            'mean_rotation_error_deg': np.degrees(np.mean(rotation_errors)),
            'mean_translation_error': np.mean(translation_errors),
            'median_rotation_error_deg': np.degrees(np.median(rotation_errors)),
            'median_translation_error': np.median(translation_errors)
        }
        
        return results
    
    def compute_rotation_error(self, pred_rot, true_rot):
        """
        Calcular error de rotación usando la métrica geodésica en SO(3)
        """
        errors = []
        
        for i in range(len(pred_rot)):
            # Convertir ángulos de Euler a matrices de rotación
            R_pred = R.from_euler('xyz', pred_rot[i]).as_matrix()
            R_true = R.from_euler('xyz', true_rot[i]).as_matrix()
            
            # Calcular error relativo
            R_rel = R_pred.T @ R_true
            
            # Extraer ángulo de rotación
            trace = np.trace(R_rel)
            angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            errors.append(angle)
            
        return np.array(errors)
    
    def evaluate_classification_accuracy(self, predictions, ground_truth):
        """
        Evaluar precisión si tratamos la pose como clasificación discreta
        """
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(ground_truth):
            ground_truth = ground_truth.cpu().numpy()
            
        # Si las predicciones son probabilidades, tomar argmax
        if predictions.ndim == 2 and predictions.shape[1] > 6:
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = ground_truth
        else:
            # Discretizar poses continuas
            pred_classes = self.discretize_poses(predictions)
            true_classes = self.discretize_poses(ground_truth)
        
        accuracy = accuracy_score(true_classes, pred_classes)
        cm = confusion_matrix(true_classes, pred_classes)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'num_classes': len(np.unique(true_classes))
        }
    
    def discretize_poses(self, poses, rotation_bins=24, translation_bins=10):
        """
        Discretizar poses continuas en clases
        """
        # Simplificación: usar solo rotación Z para discretización
        rotation_z = poses[:, 2]  # radianes
        
        # Convertir a grados y discretizar
        rotation_deg = np.degrees(rotation_z)
        rotation_deg = (rotation_deg + 180) % 360 - 180  # Normalizar a [-180, 180]
        
        # Asignar a bins
        bins = np.linspace(-180, 180, rotation_bins + 1)
        discretized = np.digitize(rotation_deg, bins) - 1
        discretized = np.clip(discretized, 0, rotation_bins - 1)
        
        return discretized
    
    def compute_pose_metrics(self, model, data_loader):
        """
        Calcular métricas completas en un dataset
        """
        model.eval()
        all_predictions = []
        all_ground_truth = []
        
        with torch.no_grad():
            for batch_idx, (points, poses) in enumerate(data_loader):
                points = points.to(self.device)
                poses = poses.to(self.device)
                
                # Predicción
                outputs = model(points)
                
                # Si el modelo devuelve múltiples salidas (logits, features, etc.)
                if isinstance(outputs, tuple):
                    predictions = outputs[0]  # Tomar la primera salida como predicción
                else:
                    predictions = outputs
                
                all_predictions.append(predictions.cpu())
                all_ground_truth.append(poses.cpu())
        
        # Concatenar todos los resultados
        all_predictions = torch.cat(all_predictions, dim=0)
        all_ground_truth = torch.cat(all_ground_truth, dim=0)
        
        # Calcular métricas
        pose_metrics = self.evaluate_pose_accuracy(all_predictions, all_ground_truth)
        
        # Si el modelo hace clasificación, también evaluar eso
        if all_predictions.shape[1] > 6:  # Probablemente clasificación
            classification_metrics = self.evaluate_classification_accuracy(
                all_predictions, all_ground_truth
            )
            pose_metrics.update(classification_metrics)
        
        return pose_metrics, all_predictions, all_ground_truth
    
    def plot_error_distribution(self, rotation_errors, translation_errors, save_path=None):
        """
        Visualizar distribución de errores
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Error de rotación
        ax1.hist(rotation_errors, bins=30, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Rotation Error (degrees)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Rotation Error Distribution')
        ax1.axvline(np.mean(rotation_errors), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(rotation_errors):.2f}°')
        ax1.axvline(np.median(rotation_errors), color='green', linestyle='--', 
                   label=f'Median: {np.median(rotation_errors):.2f}°')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Error de traslación
        ax2.hist(translation_errors, bins=30, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Translation Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Translation Error Distribution')
        ax2.axvline(np.mean(translation_errors), color='red', linestyle='--',
                   label=f'Mean: {np.mean(translation_errors):.4f}')
        ax2.axvline(np.median(translation_errors), color='green', linestyle='--',
                   label=f'Median: {np.median(translation_errors):.4f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm, class_names=None, save_path=None):
        """
        Visualizar matriz de confusión
        """
        plt.figure(figsize=(10, 8))
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(cm.shape[0])]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_evaluation_report(self, metrics, save_path=None):
        """
        Generar reporte completo de evaluación
        """
        report = f"""
POSE ESTIMATION EVALUATION REPORT
================================

ACCURACY METRICS:
- Rotation Accuracy: {metrics['rotation_accuracy']:.4f}
- Translation Accuracy: {metrics['translation_accuracy']:.4f}
- Combined Accuracy: {metrics['combined_accuracy']:.4f}

ERROR STATISTICS:
Rotation Errors:
- Mean: {metrics['mean_rotation_error_deg']:.2f}°
- Median: {metrics['median_rotation_error_deg']:.2f}°
- Std: {np.std(metrics['rotation_errors_deg']):.2f}°

Translation Errors:
- Mean: {metrics['mean_translation_error']:.4f}
- Median: {metrics['median_translation_error']:.4f}
- Std: {np.std(metrics['translation_errors']):.4f}

"""
        
        # Agregar métricas de clasificación si están disponibles
        if 'accuracy' in metrics:
            report += f"""
CLASSIFICATION METRICS:
- Classification Accuracy: {metrics['accuracy']:.4f}
- Number of Classes: {metrics['num_classes']}
"""
        
        print(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report

def evaluate_model_complete(model, test_loader, device='cuda', save_dir='./results'):
    """
    Función principal para evaluación completa del modelo
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    evaluator = PoseEvaluator(device)
    
    print("Evaluating model...")
    metrics, predictions, ground_truth = evaluator.compute_pose_metrics(model, test_loader)
    
    # Generar reporte
    evaluator.generate_evaluation_report(metrics, 
                                       save_path=os.path.join(save_dir, 'evaluation_report.txt'))
    
    # Visualizar distribución de errores
    evaluator.plot_error_distribution(
        metrics['rotation_errors_deg'],
        metrics['translation_errors'],
        save_path=os.path.join(save_dir, 'error_distribution.png')
    )
    
    # Matriz de confusión si es clasificación
    if 'confusion_matrix' in metrics:
        evaluator.plot_confusion_matrix(
            metrics['confusion_matrix'],
            save_path=os.path.join(save_dir, 'confusion_matrix.png')
        )
    
    return metrics, predictions, ground_truth