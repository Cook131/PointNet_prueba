import tensorflow as tf
import numpy as np

def pose_loss(pred_pose, gt_pose, pose_type='6dof'):
    """
    Función de pérdida para estimación de pose
    
    Args:
        pred_pose: Pose predicha [batch, pose_params]
        gt_pose: Pose ground truth [batch, pose_params]
        pose_type: '6dof', 'quaternion', 'euler'
    """
    
    if pose_type == '6dof':
        # Para 6DOF: [tx, ty, tz, rx, ry, rz]
        trans_loss = tf.reduce_mean(tf.square(pred_pose[:,:3] - gt_pose[:,:3]))
        rot_loss = tf.reduce_mean(tf.square(pred_pose[:,3:] - gt_pose[:,3:]))
        
        # Peso diferente para rotación y translación
        total_loss = trans_loss + 0.1 * rot_loss
        
    elif pose_type == 'quaternion':
        # Para quaternion + translación: [qw, qx, qy, qz, tx, ty, tz]
        quat_pred = pred_pose[:,:4]
        quat_gt = gt_pose[:,:4]
        trans_pred = pred_pose[:,4:]
        trans_gt = gt_pose[:,4:]
        
        # Normalizar quaternions
        quat_pred = tf.nn.l2_normalize(quat_pred, axis=1)
        quat_gt = tf.nn.l2_normalize(quat_gt, axis=1)
        
        # Pérdida para quaternion (1 - |q1 · q2|)
        quat_loss = 1.0 - tf.abs(tf.reduce_sum(quat_pred * quat_gt, axis=1))
        quat_loss = tf.reduce_mean(quat_loss)
        
        # Pérdida para translación
        trans_loss = tf.reduce_mean(tf.square(trans_pred - trans_gt))
        
        total_loss = quat_loss + 0.1 * trans_loss
    
    return total_loss, trans_loss, rot_loss

def regularization_loss(transform):
    """Pérdida de regularización para T-Net"""
    # Forzar que la transformación sea cercana a identidad
    I = tf.eye(transform.shape[-1])
    reg_loss = tf.reduce_mean(tf.square(tf.matmul(transform, tf.transpose(transform, [0,2,1])) - I))
    return 0.001 * reg_loss