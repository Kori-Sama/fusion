import numpy as np
from pyquaternion import Quaternion

def project_to_image(pts_3d, intrinsic):
    """
    将相机坐标系下的 3D 点投影到图像平面
    pts_3d: (N, 3)
    intrinsic: (3, 3)
    """
    pts_2d = np.dot(intrinsic, pts_3d.T)
    pts_2d[0, :] /= pts_2d[2, :]
    pts_2d[1, :] /= pts_2d[2, :]
    return pts_2d[:2, :].T

def world_to_camera(pts_world, cam_extrinsic, ego_pose):
    """
    将世界坐标系下的点转换到相机坐标系
    1. World -> Ego
    2. Ego -> Camera
    """
    # 示例逻辑，具体需参考 nuScenes 坐标系定义
    # pts_ego = (pts_world - ego_pose['translation']) @ Quaternion(ego_pose['rotation']).rotation_matrix.T
    # pts_cam = (pts_ego - cam_extrinsic['translation']) @ Quaternion(cam_extrinsic['rotation']).rotation_matrix.T
    pass

def decode_rotation(rot_pred):
    """
    解码回归的旋转角 (Bin-based 或 sin/cos)
    CenterNet 3D 常用的 Multi-bin 旋转解码
    """
    # 简化：假设预测的是 (sin, cos)
    yaw = np.arctan2(rot_pred[..., 0], rot_pred[..., 1])
    return yaw
