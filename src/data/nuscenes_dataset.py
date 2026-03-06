import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box, RadarPointCloud
from nuscenes.utils.geometry_utils import box_in_image, view_points

class NuscenesDataset(Dataset):
    def __init__(self, data_root, version='v1.0-mini', split='train', img_size=(448, 800)):
        self.nusc = NuScenes(version=version, dataroot=data_root, verbose=False)
        self.data_root = data_root
        self.img_size = img_size # (H, W)
        self.split = split
        self.samples = [s for s in self.nusc.sample if self.nusc.get('scene', s['scene_token'])['name'].startswith('scene-')] # 简化筛选
        
        self.class_names = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 
                            'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
        self.class_map = {name: i for i, name in enumerate(self.class_names)}
        
        # 预定义下采样率
        self.downsample = 4
        self.out_size = (img_size[0] // self.downsample, img_size[1] // self.downsample)

    def __len__(self):
        return len(self.samples)

    def _get_radar_points(self, sample, cam_token, cam_intrinsic):
        """ 获取投影到相机坐标系的雷达点云 """
        # 获取 5 帧累积的雷达点云以增加密度 (nuScenes 推荐做法)
        all_radar_pcs = RadarPointCloud(np.zeros((18, 0)))
        radar_tokens = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT']
        
        cam_record = self.nusc.get('sample_data', cam_token)
        points, times = RadarPointCloud.from_file_multisweep(
            self.nusc, sample, 'RADAR_FRONT', 'CAM_FRONT', nsweeps=3)
        
        # 投影到图像
        points_img = view_points(points[:3, :], cam_intrinsic, normalize=True)
        
        # 过滤掉图像外的点
        mask = np.ones(points_img.shape[1], dtype=bool)
        mask = np.logical_and(mask, points_img[0, :] > 0)
        mask = np.logical_and(mask, points_img[0, :] < self.img_size[1] - 1)
        mask = np.logical_and(mask, points_img[1, :] > 0)
        mask = np.logical_and(mask, points_img[1, :] < self.img_size[0] - 1)
        mask = np.logical_and(mask, points[2, :] > 1) # 深度 > 1m
        
        valid_points = points[:, mask]
        valid_points_img = points_img[:, mask]
        
        # 返回：[x_img, y_img, depth, vx, vy]
        radar_data = np.concatenate([valid_points_img[:2, :], valid_points[2:3, :], valid_points[8:10, :]], axis=0)
        return radar_data

    def __getitem__(self, idx):
        sample = self.samples[idx]
        cam_token = sample['data']['CAM_FRONT']
        cam_data = self.nusc.get('sample_data', cam_token)
        
        # 1. 图像处理
        img_path = os.path.join(self.data_root, cam_data['filename'])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        
        cs_record = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        intrinsic = np.array(cs_record['camera_intrinsic'])
        
        # 2. 雷达处理
        radar_points = self._get_radar_points(sample, cam_token, intrinsic)
        
        # 3. 标签生成
        _, boxes, _ = self.nusc.get_sample_data(cam_token)
        targets = self._generate_targets(boxes, intrinsic, radar_points)
        
        # 归一化并转 Tensor
        img = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
        img = torch.from_numpy(img).permute(2, 0, 1)
        
        return img, targets

    def _generate_targets(self, boxes, intrinsic, radar_points):
        oh, ow = self.out_size
        hm = np.zeros((len(self.class_names), oh, ow), dtype=np.float32)
        reg = np.zeros((2, oh, ow), dtype=np.float32)
        wh = np.zeros((2, oh, ow), dtype=np.float32)
        dep = np.zeros((1, oh, ow), dtype=np.float32)
        dim = np.zeros((3, oh, ow), dtype=np.float32)
        rot = np.zeros((8, oh, ow), dtype=np.float32)
        vel = np.zeros((2, oh, ow), dtype=np.float32)
        ind = np.zeros((oh, ow), dtype=np.int64) # 用于标识目标中心
        mask = np.zeros((oh, ow), dtype=np.float32)
        
        # 雷达特征图 (深度与速度)
        radar_hm = np.zeros((3, oh, ow), dtype=np.float32) # [depth, vx, vy]

        for box in boxes:
            cls_id = self._get_class_id(box.name)
            if cls_id < 0: continue
            
            center_3d = box.center
            pts_2d = view_points(center_3d.reshape(3, 1), intrinsic, normalize=True)
            x, y = pts_2d[0, 0] / self.downsample, pts_2d[1, 0] / self.downsample
            
            if 0 <= x < ow and 0 <= y < oh:
                ct = np.array([x, y], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                
                # 绘制高斯热力图 (简化版)
                self._draw_gaussian(hm[cls_id], ct_int, radius=3)
                
                # 填充回归值
                reg[:, ct_int[1], ct_int[0]] = ct - ct_int
                wh[:, ct_int[1], ct_int[0]] = [box.wlh[0], box.wlh[1]] # 2D 简化
                dep[0, ct_int[1], ct_int[0]] = box.center[2]
                dim[:, ct_int[1], ct_int[0]] = box.wlh
                # 旋转角和速度 (略，实际需更复杂编码)
                mask[ct_int[1], ct_int[0]] = 1

        # 将雷达点填入特征图
        for i in range(radar_points.shape[1]):
            px, py = radar_points[0, i] / self.downsample, radar_points[1, i] / self.downsample
            if 0 <= px < ow and 0 <= py < oh:
                px, py = int(px), int(py)
                # 如果点云重合，取最近的点 (depth)
                if radar_hm[0, py, px] == 0 or radar_points[2, i] < radar_hm[0, py, px]:
                    radar_hm[0, py, px] = radar_points[2, i]
                    radar_hm[1:, py, px] = radar_points[3:, i]

        return {
            "hm": torch.from_numpy(hm),
            "reg": torch.from_numpy(reg),
            "wh": torch.from_numpy(wh),
            "dep": torch.from_numpy(dep),
            "dim": torch.from_numpy(dim),
            "mask": torch.from_numpy(mask),
            "radar_hm": torch.from_numpy(radar_hm) # [3, oh, ow]
        }

    def _draw_gaussian(self, hm, ct, radius):
        # 简化的 Gaussian 绘制
        x, y = ct
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                if 0 <= x+i < hm.shape[1] and 0 <= y+j < hm.shape[0]:
                    dist = i**2 + j**2
                    hm[y+j, x+i] = max(hm[y+j, x+i], np.exp(-dist / (2 * (radius/3)**2)))

    def _get_class_id(self, name):
        for i, cls in enumerate(self.class_names):
            if cls in name: return i
        return -1
