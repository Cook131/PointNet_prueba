import numpy as np

def augment_pointcloud(pointcloud, pose):
    """Aumentar datos con ruido y jitter"""
    # Añadir ruido gaussiano
    noise = np.random.normal(0, 0.02, pointcloud.shape)
    pointcloud_noisy = pointcloud + noise
    
    # Jitter de puntos
    jitter = np.random.uniform(-0.005, 0.005, pointcloud.shape)
    pointcloud_jittered = pointcloud_noisy + jitter
    
    # Dropout aleatorio de puntos
    dropout_ratio = np.random.random() * 0.2
    if dropout_ratio > 0:
        num_points = pointcloud.shape[0]
        keep_indices = np.random.choice(num_points, 
                                      int(num_points * (1 - dropout_ratio)), 
                                      replace=False)
        pointcloud_dropout = pointcloud_jittered[keep_indices]
        
        # Pad con ceros si es necesario
        if len(pointcloud_dropout) < num_points:
            padding = np.zeros((num_points - len(pointcloud_dropout), 3))
            pointcloud_dropout = np.vstack([pointcloud_dropout, padding])
    else:
        pointcloud_dropout = pointcloud_jittered
        
    return pointcloud_dropout, pose