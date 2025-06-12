"""
Módulo de entrenamiento y evaluación
"""

from .config import get_config
from .evaluate_pose import PoseEvaluator

__all__ = [
    'get_config',
    'PoseEvaluator'
]