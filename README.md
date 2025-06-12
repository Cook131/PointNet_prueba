# PointNet para Estimación de Pose de Oil Pan

Sistema de deep learning basado en PointNet para estimar la pose (posición y orientación) de un oil pan a partir de nubes de puntos 3D.

## 🎯 Objetivo

Determinar la pose de un objeto (oil pan) basándose en cuánto de cada superficie o cara se está viendo desde un punto dado, utilizando la arquitectura PointNet adaptada para regresión de pose 6DOF (6 grados de libertad: 3 rotaciones + 3 traslaciones).

## 📁 Estructura del Proyecto

```
point_pose/
├── data/                           # Datos del proyecto
│   ├── augmented/                  # Datos aumentados
│   ├── pose_dataset/              # Dataset de poses
│   └── processed/                 # Datos procesados
├── checkpoints/                   # Checkpoints de entrenamiento
├── logs/                         # Logs de entrenamiento  
├── results/                      # Resultados de evaluación
├── inference/                    # Módulo de inferencia
│   ├── __init__.py
│   └── predict_pose.py           # Script de inferencia principal
├── models/                       # Modelos y arquitecturas
│   ├── __init__.py
│   ├── pointnet_pose.py          # Modelo principal PointNet
│   ├── pointnet_utils.py         # Utilidades de PointNet
│   └── transform_net.py          # Redes de transformación T-Net
├── training/                     # Módulo de entrenamiento
│   ├── __init__.py
│   ├── config.py                 # Configuración del sistema
│   ├── evaluate_pose.py          # Evaluación del modelo
│   ├── pose_loss.py             # Funciones de pérdida
│   └── train_pose.py            # Script de entrenamiento principal
├── utils/                        # Utilidades generales
│   ├── __init__.py
│   ├── augmentation.py          # Augmentación de datos
│   ├── data_prep_pose.py        # Preparación de datos
│   ├── metrics.py               # Métricas de evaluación
│   └── visualization.py         # Herramientas de visualización
├── OilPan/CenteredPointClouds/  # Datos del oil pan
│   └── oil_pan_full_pc_10000.ply # Nube de puntos del oil pan
└── README.md                     # Este archivo
```

## 🛠️ Instalación

### Prerrequisitos
- Python 3.8+
- CUDA 11.0+ (recomendado para GPU)

### Dependencias
```bash
pip install torch torchvision torchaudio
pip install open3d matplotlib plotly seaborn scipy scikit-learn tqdm numpy pandas
```

### Configuración del entorno
```bash
# Clonar repositorio
git clone https://github.com/Cook131/PointNet_prueba.git
cd PointNet_prueba

# Crear archivos __init__.py necesarios
touch models/__init__.py
touch training/__init__.py
touch utils/__init__.py
touch inference/__init__.py

# Verificar estructura
ls -la models/ training/ utils/ inference/
```

## 🚀 Uso Rápido

### 1. Entrenamiento Básico
```bash
# Desde la raíz del proyecto
python training/train_pose.py --ply_path ./OilPan/CenteredPointClouds/oil_pan_full_pc_10000.ply
```

### 2. Entrenamiento Personalizado
```bash
python training/train_pose.py \
    --ply_path ./OilPan/CenteredPointClouds/oil_pan_full_pc_10000.ply \
    --epochs 300 \
    --batch_size 16 \
    --lr 0.0005 \
    --device cuda
```

### 3. Inferencia - Opción 1 (desde raíz)
```bash
python inference/predict_pose.py \
    --model models/best_model.pth \
    --input OilPan/CenteredPointClouds/oil_pan_full_pc_10000.ply
```

### 4. Inferencia - Opción 2 (desde carpeta inference)
```bash
cd inference/
python predict_pose.py \
    --model ../models/best_model.pth \
    --input ../OilPan/CenteredPointClouds/oil_pan_full_pc_10000.ply
```

### 5. Inferencia sin visualización
```bash
python inference/predict_pose.py \
    --model models/best_model.pth \
    --input test_oil_pan.ply \
    --no-visualize
```

### 6. Guardar resultados de inferencia
```bash
python inference/predict_pose.py \
    --model models/best_model.pth \
    --input test_oil_pan.ply \
    --output results/mi_test
```

## 📊 Configuración

El archivo `training/config.py` contiene todas las configuraciones del sistema:

### Configuración del Modelo (`ModelConfig`)
```python
input_dim: 3              # Coordenadas x, y, z
num_points: 1024          # Puntos por nube
pose_dim: 6               # 3 rotaciones + 3 traslaciones
dropout: 0.3              # Dropout rate
feature_dim: 64           # Dimensión de características
num_pose_classes: 24      # Clases discretas de pose
```

### Configuración de Entrenamiento (`TrainingConfig`)
```python
batch_size: 32            # Tamaño del batch
learning_rate: 0.001      # Learning rate inicial
num_epochs: 200           # Épocas de entrenamiento
weight_decay: 0.0005      # Regularización L2
step_size: 20            # Pasos para scheduler
gamma: 0.7               # Factor de reducción de LR
patience: 20             # Early stopping patience
```

### Configuración de Datos (`DataConfig`)
```python
train_split: 0.8          # 80% para entrenamiento
val_split: 0.1           # 10% para validación
test_split: 0.1          # 10% para prueba
noise_std: 0.01          # Desviación estándar del ruido
scale_range: (0.8, 1.2)  # Rango de escalado
```

### Configuración de Pose (`PoseConfig`)
```python
rotation_x_range: (-45, 45)    # Rango pitch en grados
rotation_y_range: (-45, 45)    # Rango yaw en grados  
rotation_z_range: (-180, 180)  # Rango roll en grados
rotation_step: 15.0            # Paso de discretización
camera_distance: 2.0           # Distancia de cámara
```

## 🏗️ Arquitectura del Modelo

### PointNet Base (`models/pointnet_pose.py`)
```
Input [B, 3, N] → STN3d → [B, 3, N]
                ↓
          Conv1d(3→64) → BN → ReLU
                ↓
          STNkd → [B, 64, N] 
                ↓
          Conv1d(64→128) → BN → ReLU
                ↓
          Conv1d(128→1024) → BN
                ↓
          MaxPool → [B, 1024]
                ↓
          FC(1024→512) → BN → ReLU → Dropout
                ↓
          FC(512→256) → BN → ReLU → Dropout
                ↓
          FC(256→6) → [B, 6] (pose output)
```

### Componentes Clave

1. **Input Transform (STN3d)**: Alineación espacial de entrada
2. **Feature Transform (STNkd)**: Transformación de características 64D
3. **Shared MLPs**: Extracción de características por punto
4. **Global Feature**: Max pooling para característica global
5. **Pose Head**: Regresión a 6DOF

### Variantes Disponibles
- **PointNetPose**: Regresión continua de pose 6DOF
- **PointNetPoseClassification**: Clasificación discreta de poses
- **PointNetPoseHybrid**: Combinación de regresión y clasificación

## 📈 Métricas de Evaluación

### Métricas de Precisión
- **Rotation Accuracy**: % de rotaciones dentro del umbral (15°)
- **Translation Accuracy**: % de traslaciones dentro del umbral (0.1 unidades)
- **Combined Accuracy**: Precisión combinada de rotación y traslación

### Métricas de Error
- **Mean Rotation Error**: Error promedio de rotación en grados
- **Mean Translation Error**: Error promedio de traslación
- **Median Rotation Error**: Error mediano de rotación (más robusto)
- **Median Translation Error**: Error mediano de traslación

### Evaluación Completa
```bash
# Evaluar modelo entrenado
python -c "
from training.evaluate_pose import evaluate_model_complete
from models.pointnet_pose import PointNetPose
import torch

# Cargar y evaluar modelo
model = PointNetPose()
checkpoint = torch.load('models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Ejecutar evaluación completa
evaluate_model_complete(model, test_loader, save_dir='results/evaluation')
"
```

## 🔬 Generación de Datos Sintéticos

El sistema genera automáticamente datos de entrenamiento mediante:

### 1. **Generación de Poses Aleatorias**
- Rotaciones dentro de rangos realistas para cada eje
- Traslaciones pequeñas para simular variaciones de posición
- Distribución uniforme en espacios de rotación

### 2. **Transformación 3D**
- Aplicación de matrices de rotación Euler XYZ
- Traslación en coordenadas cartesianas
- Preservación de la geometría del objeto

### 3. **Simulación de Vista de Cámara**
- Oclusión parcial para simular vistas realistas
- Eliminación de puntos no visibles desde la cámara
- Adición de ruido para robustez

### 4. **Augmentación de Datos**
- Ruido gaussiano aleatorio (σ = 0.01)
- Rotaciones aleatorias menores
- Escalado aleatorio (0.8x - 1.2x)
- Jittering de puntos

## 📊 Visualización y Monitoreo

### Durante el Entrenamiento
```
Epoch 50/200
Train Loss: 0.023456
Val Loss: 0.028901
Rotation Accuracy: 0.8234
Translation Accuracy: 0.7891
Combined Accuracy: 0.6945
Mean Rotation Error: 8.32°
Epoch Time: 45.2s
--------------------------------------------------
✓ Nuevo mejor modelo guardado (val_loss: 0.028901)
```

### Archivos Generados
- `logs/training_metrics.json`: Métricas completas por época
- `logs/training_curves_epoch_X.png`: Gráficos de progreso
- `checkpoints/checkpoint_epoch_X.pth`: Checkpoints periódicos
- `models/best_model.pth`: Mejor modelo basado en validación
- `results/evaluation/`: Reportes de evaluación completos

### Visualización Manual
```python
# Desde Python
from utils.visualization import PointCloudVisualizer
import numpy as np

visualizer = PointCloudVisualizer()

# Visualizar nube de puntos
points = np.load('mi_nube_puntos.npy')
visualizer.plot_point_cloud_plotly(points, title="Mi Oil Pan")

# Comparar predicciones
true_pose = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
pred_pose = np.array([0.15, 0.18, 0.32, 0.01, -0.01, 0.005])
fig = visualizer.visualize_pose_estimation(points, true_pose, pred_pose)
fig.show()
```

## 🎛️ Entrenamiento Avanzado

### Early Stopping
- **Paciencia**: 20 épocas sin mejora en validación
- **Criterio**: Pérdida de validación (MSE + regularización)
- **Guardado automático**: Mejor modelo según validación

### Learning Rate Scheduling
```python
# StepLR scheduler
initial_lr = 0.001
step_size = 20        # Reduce cada 20 épocas
gamma = 0.7          # Factor de reducción
# LR: 0.001 → 0.0007 → 0.00049 → ...
```

### Regularización
```python
# Pérdida total
total_loss = pose_loss + λ * feature_transform_regularizer
# λ = 0.001 (configurable)
```

### Ejemplo de Entrenamiento Completo
```bash
# Entrenamiento largo con configuración personalizada
python training/train_pose.py \
    --ply_path ./OilPan/CenteredPointClouds/oil_pan_full_pc_10000.ply \
    --epochs 500 \
    --batch_size 8 \
    --lr 0.0005 \
    --device cuda

# Monitorear progreso en tiempo real
tail -f logs/training_log.txt

# Evaluar durante entrenamiento
tensorboard --logdir logs/
```

## 🧪 Testing y Evaluación

### Validación de Dataset
```bash
# Verificar que el .ply se puede cargar
python -c "
from utils.data_prep_pose import validate_dataset
validate_dataset('./OilPan/CenteredPointClouds/oil_pan_full_pc_10000.ply')
"
```

### Test de Modelo
```bash
# Test rápido del modelo
python -c "
from models.pointnet_pose import PointNetPose
import torch

model = PointNetPose()
test_input = torch.randn(1, 3, 1024)
output, transform = model(test_input)
print(f'Output shape: {output.shape}')  # Debe ser [1, 6]
print('✓ Modelo funciona correctamente')
"
```

### Benchmark de Rendimiento
```bash
# Medir tiempo de inferencia
python -c "
import time
from inference.predict_pose import PosePredictor

predictor = PosePredictor('models/best_model.pth')
start_time = time.time()

for i in range(100):
    pose, _ = predictor.predict_pose('test.ply', visualize=False)

avg_time = (time.time() - start_time) / 100
print(f'Tiempo promedio de inferencia: {avg_time*1000:.2f} ms')
"
```

## 🔧 Personalización y Extensión

### Agregar Nuevos Objetos
1. **Preparar archivo .ply**:
   ```bash
   # Colocar en estructura apropiada
   mkdir OilPan/NewObject/
   cp nuevo_objeto.ply OilPan/NewObject/
   ```

2. **Actualizar configuración**:
   ```python
   # En training/config.py
   data_config.oil_pan_ply = "./OilPan/NewObject/nuevo_objeto.ply"
   ```

3. **Ajustar rangos de pose si es necesario**:
   ```python
   # Para objetos con diferentes orientaciones típicas
   pose_config.rotation_x_range = (-30, 30)  # Reducir rango
   pose_config.rotation_y_range = (-60, 60)  # Ampliar rango
   ```

### Modificar Arquitectura del Modelo
```python
# En models/pointnet_pose.py
class CustomPointNetPose(PointNetPose):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Agregar capas adicionales
        self.extra_fc = nn.Linear(1024, 1024)
        self.extra_bn = nn.BatchNorm1d(1024)
        
        # Modificar cabeza final
        self.fc1 = nn.Linear(1024, 1024)  # Aumentar tamaño
        
    def forward(self, x):
        # ... código base ...
        
        # Procesar característica global adicional
        global_feature = F.relu(self.extra_bn(self.extra_fc(global_feature)))
        
        # ... resto del forward ...
```

### Métricas Personalizadas
```python
# En utils/metrics.py o training/evaluate_pose.py
def custom_pose_metric(predictions, ground_truth):
    """
    Métrica personalizada para evaluación específica
    """
    # Implementar lógica personalizada
    custom_score = ...
    return custom_score

# Integrar en evaluación
metrics = evaluator.evaluate_pose_accuracy(pred, gt)
metrics['custom_metric'] = custom_pose_metric(pred, gt)
```

### Funciones de Pérdida Personalizadas
```python
# En training/pose_loss.py
class CustomPoseLoss(nn.Module):
    def __init__(self, rotation_weight=1.0, translation_weight=1.0):
        super().__init__()
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight
        
    def forward(self, pred_pose, true_pose):
        pred_rot, pred_trans = pred_pose[:, :3], pred_pose[:, 3:]
        true_rot, true_trans = true_pose[:, :3], true_pose[:, 3:]
        
        # Pérdida de rotación (puede usar métricas geodésicas)
        rot_loss = F.mse_loss(pred_rot, true_rot)
        
        # Pérdida de traslación
        trans_loss = F.mse_loss(pred_trans, true_trans)
        
        return (self.rotation_weight * rot_loss + 
                self.translation_weight * trans_loss)
```

## 🐛 Solución de Problemas

### Errores Comunes y Soluciones

#### 1. **CUDA Out of Memory**
```bash
# Error: RuntimeError: CUDA out of memory
```
**Soluciones**:
```python
# Reducir batch_size en training/config.py
training_config.batch_size = 8  # En lugar de 32

# Reducir número de puntos si es necesario
model_config.num_points = 512   # En lugar de 1024

# Usar gradient checkpointing
torch.utils.checkpoint.checkpoint_sequential(...)
```

#### 2. **Archivo PLY No Se Carga**
```bash
# Error: OSError: Unable to read file
```
**Debugging**:
```python
# Verificar archivo
python -c "
import open3d as o3d
pcd = o3d.io.read_point_cloud('tu_archivo.ply')
print(f'Puntos cargados: {len(pcd.points)}')
print('✓ Archivo válido' if len(pcd.points) > 0 else '✗ Archivo inválido')
"

# Convertir formato si es necesario
import open3d as o3d
pcd = o3d.io.read_point_cloud('input.obj')  # Otros formatos
o3d.io.write_point_cloud('output.ply', pcd)
```

#### 3. **Convergencia Lenta**
```bash
# Loss se mantiene alto después de muchas épocas
```
**Soluciones**:
```python
# Ajustar learning rate
training_config.learning_rate = 0.0001  # Más conservador

# Cambiar scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

# Aumentar datos de entrenamiento
train_dataset.num_samples = 20000  # Más datos sintéticos

# Verificar normalización
points = pc_normalize(points)  # Asegurar normalización correcta
```

#### 4. **Imports No Funcionan**
```bash
# Error: ModuleNotFoundError: No module named 'models'
```
**Soluciones**:
```bash
# Verificar estructura de archivos
ls models/__init__.py training/__init__.py utils/__init__.py

# Crear archivos faltantes
touch models/__init__.py
touch training/__init__.py
touch utils/__init__.py
touch inference/__init__.py

# Ejecutar desde directorio correcto
pwd  # Debe mostrar la raíz del proyecto
python training/train_pose.py ...
```

#### 5. **Predicciones Incorrectas**
```bash
# El modelo predice poses muy alejadas de la realidad
```
**Debugging**:
```python
# Verificar rango de datos de entrenamiento
python -c "
import numpy as np
# Cargar datos de entrenamiento y verificar rangos
poses = np.load('training_poses.npy')
print(f'Rotación rango: {poses[:, :3].min()} a {poses[:, :3].max()}')
print(f'Traslación rango: {poses[:, 3:].min()} a {poses[:, 3:].max()}')
"

# Verificar normalización de entrada
points = pc_normalize(points)  # Siempre normalizar

# Verificar que el modelo esté en modo eval
model.eval()
with torch.no_grad():
    prediction = model(input_tensor)
```

### Tests de Validación

#### Test Completo del Sistema
```bash
# Script de test completo
python -c "
print('🧪 Testing PointNet Pose System...')

# Test 1: Imports
try:
    from models.pointnet_pose import PointNetPose
    from training.config import get_config
    from utils.data_prep_pose import validate_dataset
    print('✓ Imports working')
except Exception as e:
    print(f'✗ Import error: {e}')
    exit(1)

# Test 2: Configuración
try:
    config = get_config()
    print('✓ Configuration loaded')
except Exception as e:
    print(f'✗ Config error: {e}')
    exit(1)

# Test 3: Modelo
try:
    model = PointNetPose()
    test_input = torch.randn(2, 3, 1024)
    output, transform = model(test_input)
    assert output.shape == (2, 6), f'Wrong output shape: {output.shape}'
    print('✓ Model working')
except Exception as e:
    print(f'✗ Model error: {e}')
    exit(1)

# Test 4: Dataset
try:
    valid = validate_dataset('./OilPan/CenteredPointClouds/oil_pan_full_pc_10000.ply')
    if valid:
        print('✓ Dataset valid')
    else:
        print('✗ Dataset invalid')
except Exception as e:
    print(f'✗ Dataset error: {e}')

print('🎉 All tests passed!')
"
```

#### Test de Entrenamiento Rápido
```bash
# Test con 1 época y batch pequeño
python training/train_pose.py \
    --ply_path ./OilPan/CenteredPointClouds/oil_pan_full_pc_10000.ply \
    --epochs 1 \
    --batch_size 2
```

#### Test de Inferencia
```bash
# Test básico de inferencia
python inference/predict_pose.py \
    --model models/best_model.pth \
    --input OilPan/CenteredPointClouds/oil_pan_full_pc_10000.ply \
    --no-visualize
```

## 📚 Referencias y Recursos

### Papers Fundamentales
- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)
- [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413)

### Implementaciones de Referencia
- [Repositorio Original PointNet](https://github.com/charlesq34/pointnet) - TensorFlow
- [PointNet PyTorch](https://github.com/fxia22/pointnet.pytorch) - PyTorch

### Herramientas y Librerías
- [Open3D Documentation](http://www.open3d.org/docs/) - Procesamiento de nubes de puntos
- [PyTorch Documentation](https://pytorch.org/docs/) - Framework de deep learning
- [Plotly Python](https://plotly.com/python/) - Visualización interactiva

### Datasets Relacionados
- [ModelNet40](http://modelnet.cs.princeton.edu/) - Dataset de objetos 3D
- [ShapeNet](https://www.shapenet.org/) - Dataset de formas 3D
- [ScanNet](http://www.scan-net.org/) - Dataset de escenas 3D

## 📝 Licencia

MIT License - Ver archivo LICENSE para detalles.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. **Fork** el proyecto
2. **Crear rama** para feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** cambios (`git commit -m 'Add: AmazingFeature'`)
4. **Push** a la rama (`git push origin feature/AmazingFeature`)
5. **Abrir Pull Request** con descripción detallada

### Pautas para Contribuciones
- Seguir la estructura de archivos existente
- Agregar tests para nuevas funcionalidades
- Documentar cambios en README.md
- Usar mensajes de commit descriptivos
- Mantener compatibilidad con versiones existentes

## 📞 Contacto y Soporte

### Para Preguntas Técnicas
- Abrir **issue** en el repositorio con:
  - Descripción detallada del problema
  - Código para reproducir el error
  - Información del entorno (OS, Python, CUDA)
  - Logs de error completos

### Para Sugerencias
- Abrir **feature request** con:
  - Descripción de la funcionalidad deseada
  - Casos de uso específicos
  - Beneficios esperados

## 🔄 Changelog

### v1.0.0 (Current)
- ✅ Implementación base de PointNet para estimación de pose
- ✅ Generación automática de datos sintéticos
- ✅ Sistema de entrenamiento con early stopping
- ✅ Métricas de evaluación específicas para pose
- ✅ Visualización interactiva de resultados
- ✅ Scripts de inferencia listos para usar
- ✅ Documentación completa

### Planned Features
- 🔄 Soporte para múltiples objetos en una escena
- 🔄 Integración con ROS para robótica
- 🔄 Exportación a ONNX para deployment
- 🔄 Augmentación de datos más sofisticada
- 🔄 Soporte para datos RGB-D

---

**¡Happy coding!** 🚀🔧

> **Tip**: Para mejores resultados, asegúrate de tener al menos 10,000 samples sintéticos y entrenar por mínimo 100 épocas. ¡La paciencia es clave en deep learning!