from __future__ import annotations

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud

from fusion.utils.geometry import project_points


RADAR_FEATURE_NAMES = ["depth", "rcs", "vx_comp", "vy_comp", "count"]


def build_radar_map(
    nusc: NuScenes,
    sample_rec: dict,
    camera_channel: str,
    radar_channels: list[str],
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
    nsweeps: int,
) -> np.ndarray:
    height, width = image_size
    radar_map = np.zeros((len(RADAR_FEATURE_NAMES), height, width), dtype=np.float32)
    depth_acc = np.full((height, width), np.inf, dtype=np.float32)
    feat_acc = np.zeros((4, height, width), dtype=np.float32)
    count_acc = np.zeros((height, width), dtype=np.float32)

    for radar_channel in radar_channels:
        if radar_channel not in sample_rec["data"]:
            continue
        point_cloud, _ = RadarPointCloud.from_file_multisweep(
            nusc,
            sample_rec,
            radar_channel,
            camera_channel,
            nsweeps=nsweeps,
        )
        points = point_cloud.points
        if points.shape[1] == 0:
            continue
        xyz = points[:3, :].T.astype(np.float32)
        projected, mask = project_points(xyz, intrinsics, image_size)
        if not np.any(mask):
            continue
        xyz = xyz[mask]
        projected = projected[mask]
        rcs = points[5, mask].astype(np.float32)
        vx_comp = points[8, mask].astype(np.float32)
        vy_comp = points[9, mask].astype(np.float32)

        pixels = np.round(projected).astype(np.int32)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)

        for idx, (u, v) in enumerate(pixels):
            depth = xyz[idx, 2]
            depth_acc[v, u] = min(depth_acc[v, u], depth)
            feat_acc[0, v, u] += depth
            feat_acc[1, v, u] += rcs[idx]
            feat_acc[2, v, u] += vx_comp[idx]
            feat_acc[3, v, u] += vy_comp[idx]
            count_acc[v, u] += 1.0

    valid = count_acc > 0
    if np.any(valid):
        radar_map[0, valid] = depth_acc[valid]
        radar_map[1, valid] = feat_acc[1, valid] / count_acc[valid]
        radar_map[2, valid] = feat_acc[2, valid] / count_acc[valid]
        radar_map[3, valid] = feat_acc[3, valid] / count_acc[valid]
        radar_map[4, valid] = np.log1p(count_acc[valid])

    if np.any(valid):
        positive_depth = radar_map[0, valid]
        radar_map[0, valid] = np.log(np.clip(positive_depth, 1e-3, None))

    return radar_map
