import tensorflow as tf
import numpy as np
import h5py
from models.pointnet_pose import PointNetPose
from pose_loss import pose_loss, regularization_loss

def train_pose_estimation():
    # Configuración
    BATCH_SIZE = 32
    MAX_EPOCH = 250
    LEARNING_RATE = 0.001
    NUM_POINTS = 1024
    DECAY_STEP = 200000
    DECAY_RATE = 0.7
    
    # Cargar datos
    train_data, train_poses = load_pose_data('train')
    test_data, test_poses = load_pose_data('test')
    
    # Placeholders
    pointclouds_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINTS, 3))
    poses_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 6))
    is_training_pl = tf.placeholder(tf.bool, shape=())
    
    # Modelo
    model = PointNetPose(num_points=NUM_POINTS, num_pose_params=6)
    pred_poses, transform = model.get_model(pointclouds_pl, is_training_pl)
    
    # Pérdidas
    pose_loss_val, trans_loss, rot_loss = pose_loss(pred_poses, poses_pl)
    reg_loss = regularization_loss(transform)
    total_loss = pose_loss_val + reg_loss
    
    # Optimizador
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE, tf.train.get_global_step(),
        DECAY_STEP, DECAY_RATE, staircase=True)
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(total_loss, global_step=tf.train.get_global_step())
    
    # Métricas
    pose_error = tf.reduce_mean(tf.abs(pred_poses - poses_pl))
    
    # Sesión de entrenamiento
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(MAX_EPOCH):
            # Entrenamiento
            for batch in get_batches(train_data, train_poses, BATCH_SIZE):
                feed_dict = {
                    pointclouds_pl: batch[0],
                    poses_pl: batch[1],
                    is_training_pl: True
                }
                
                _, loss_val, trans_l, rot_l = sess.run(
                    [train_op, total_loss, trans_loss, rot_loss], 
                    feed_dict=feed_dict)
            
            # Evaluación
            if epoch % 10 == 0:
                test_loss, test_error = evaluate_model(
                    sess, pointclouds_pl, poses_pl, is_training_pl,
                    total_loss, pose_error, test_data, test_poses)
                
                print(f'Epoch {epoch}: Train Loss: {loss_val:.4f}, '
                      f'Test Loss: {test_loss:.4f}, Test Error: {test_error:.4f}')

def load_pose_data(split):
    """Cargar datos de pose desde HDF5"""
    with h5py.File(f'./data/processed/pose_{split}.h5', 'r') as f:
        data = f['data'][:]
        poses = f['poses'][:]
    return data, poses

def get_batches(data, poses, batch_size):
    """Generador para lotes de datos y poses"""
    num_samples = data.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        excerpt = indices[start_idx:end_idx]
        yield data[excerpt], poses[excerpt]

def evaluate_model(sess, pointclouds_pl, poses_pl, is_training_pl,
                   total_loss, pose_error, test_data, test_poses, batch_size=32):
    """Evalúa el modelo en los datos de prueba"""
    num_samples = test_data.shape[0]
    total_loss_val = 0
    total_error_val = 0
    num_batches = 0
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        feed_dict = {
            pointclouds_pl: test_data[start_idx:end_idx],
            poses_pl: test_poses[start_idx:end_idx],
            is_training_pl: False
        }
        loss_val, error_val = sess.run([total_loss, pose_error], feed_dict=feed_dict)
        total_loss_val += loss_val * (end_idx - start_idx)
        total_error_val += error_val * (end_idx - start_idx)
        num_batches += (end_idx - start_idx)
    return total_loss_val / num_batches, total_error_val / num_batches

if __name__ == '__main__':
    train_pose_estimation()