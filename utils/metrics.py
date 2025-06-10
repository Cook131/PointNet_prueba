import numpy as np

def translation_error(pred_pose, gt_pose):
    """Error de translación (distancia euclidiana)"""
    pred_trans = pred_pose[:3]
    gt_trans = gt_pose[:3]
    return np.linalg.norm(pred_trans - gt_trans)

def rotation_error_euler(pred_pose, gt_pose):
    """Error de rotación en ángulos de Euler"""
    pred_rot = pred_pose[3:]
    gt_rot = gt_pose[3:]
    
    # Convertir a grados para mejor interpretación
    error_rad = np.abs(pred_rot - gt_rot)
    error_deg = np.degrees(error_rad)
    return np.mean(error_deg)

def pose_accuracy(pred_poses, gt_poses, trans_threshold=0.1, rot_threshold=15):
    """
    Calcular accuracy de pose basada en umbrales
    
    Args:
        trans_threshold: Umbral de error de translación
        rot_threshold: Umbral de error de rotación en grados
    """
    correct_poses = 0
    
    for pred, gt in zip(pred_poses, gt_poses):
        trans_err = translation_error(pred, gt)
        rot_err = rotation_error_euler(pred, gt)
        
        if trans_err < trans_threshold and rot_err < rot_threshold:
            correct_poses += 1
    
    return correct_poses / len(pred_poses)

def evaluate_pose_estimation(model, test_data, test_poses):
    """Evaluación completa del modelo"""
    pred_poses = model.predict(test_data)
    
    trans_errors = []
    rot_errors = []
    
    for pred, gt in zip(pred_poses, test_poses):
        trans_errors.append(translation_error(pred, gt))
        rot_errors.append(rotation_error_euler(pred, gt))
    
    results = {
        'mean_trans_error': np.mean(trans_errors),
        'mean_rot_error': np.mean(rot_errors),
        'median_trans_error': np.median(trans_errors),
        'median_rot_error': np.median(rot_errors),
        'accuracy_10cm_15deg': pose_accuracy(pred_poses, test_poses, 0.1, 15),
        'accuracy_5cm_10deg': pose_accuracy(pred_poses, test_poses, 0.05, 10)
    }
    
    return results