import tensorflow as tf
import numpy as np
import open3d as o3d
from models.pointnet_pose import PointNetPose

class PosePredictor:
    def __init__(self, model_path, num_points=1024):
        self.model_path = model_path
        self.num_points = num_points
        self.sess = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Cargar modelo entrenado"""
        self.sess = tf.Session()
        
        # Recrear el grafo
        self.pointcloud_pl = tf.placeholder(tf.float32, shape=(1, self.num_points, 3))
        self.is_training_pl = tf.placeholder(tf.bool, shape=())
        
        model = PointNetPose(num_points=self.num_points, num_pose_params=6)
        self.pred_pose, _ = model.get_model(self.pointcloud_pl, self.is_training_pl)
        
        # Restaurar pesos
        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_path)
    
    def predict_pose(self, pointcloud):
        """Predecir pose de una nube de puntos"""
        # Preprocesar
        pc_normalized = self.preprocess_pointcloud(pointcloud)
        pc_batch = np.expand_dims(pc_normalized, 0)
        
        # Predicción
        feed_dict = {
            self.pointcloud_pl: pc_batch,
            self.is_training_pl: False
        }
        
        pose = self.sess.run(self.pred_pose, feed_dict=feed_dict)
        return pose[0]  # Remover dimensión de batch
    
    def preprocess_pointcloud(self, pointcloud):
        """Preprocesar nube de puntos"""
        # Normalizar a esfera unitaria
        centroid = np.mean(pointcloud, axis=0)
        pointcloud_centered = pointcloud - centroid
        max_dist = np.max(np.linalg.norm(pointcloud_centered, axis=1))
        pointcloud_normalized = pointcloud_centered / max_dist
        
        # Submuestrear o pad a num_points
        if len(pointcloud_normalized) > self.num_points:
            indices = np.random.choice(len(pointcloud_normalized), 
                                     self.num_points, replace=False)
            pointcloud_sampled = pointcloud_normalized[indices]
        elif len(pointcloud_normalized) < self.num_points:
            # Pad con repetición de puntos
            padding_size = self.num_points - len(pointcloud_normalized)
            padding_indices = np.random.choice(len(pointcloud_normalized), 
                                             padding_size, replace=True)
            padding_points = pointcloud_normalized[padding_indices]
            pointcloud_sampled = np.vstack([pointcloud_normalized, padding_points])
        else:
            pointcloud_sampled = pointcloud_normalized
            
        return pointcloud_sampled

# Ejemplo de uso
def main():
    # Cargar predictor
    predictor = PosePredictor('./models/pointnet_pose_model.ckpt')
    
    # Cargar nube de puntos desde archivo
    pcd = o3d.io.read_point_cloud('./data/test_object.ply')
    pointcloud = np.asarray(pcd.points)
    
    # Predecir pose
    predicted_pose = predictor.predict_pose(pointcloud)
    
    print(f"Pose predicha: {predicted_pose}")
    print(f"Translación: [{predicted_pose[0]:.3f}, {predicted_pose[1]:.3f}, {predicted_pose[2]:.3f}]")
    print(f"Rotación (Euler): [{predicted_pose[3]:.3f}, {predicted_pose[4]:.3f}, {predicted_pose[5]:.3f}]")
    
    # Visualizar resultado
    from utils.visualization import visualize_pose_prediction
    visualize_pose_prediction(pointcloud, predicted_pose)

if __name__ == '__main__':
    main()