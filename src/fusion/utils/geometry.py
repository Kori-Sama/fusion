from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resize_intrinsics(
    intrinsics: np.ndarray, src_size: tuple[int, int], dst_size: tuple[int, int]
) -> np.ndarray:
    src_h, src_w = src_size
    dst_h, dst_w = dst_size
    scale_x = dst_w / src_w
    scale_y = dst_h / src_h
    resized = intrinsics.copy().astype(np.float32)
    resized[0, 0] *= scale_x
    resized[0, 2] *= scale_x
    resized[1, 1] *= scale_y
    resized[1, 2] *= scale_y
    return resized


def make_transform(
    translation: Iterable[float], rotation: Iterable[float]
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = Quaternion(rotation).rotation_matrix.astype(np.float32)
    transform[:3, 3] = np.asarray(list(translation), dtype=np.float32)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float32)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inv[:3, :3] = rotation.T
    inv[:3, 3] = -rotation.T @ translation
    return inv


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.reshape(-1, 3)
    homo = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1
    )
    return (transform @ homo.T).T[:, :3]


def project_points(
    points: np.ndarray, intrinsics: np.ndarray, image_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)
    projected = view_points(points.T, intrinsics, normalize=True).T[:, :2]
    h, w = image_size
    mask = (
        (points[:, 2] > 1e-3)
        & (projected[:, 0] >= 0)
        & (projected[:, 0] < w)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < h)
    )
    return projected.astype(np.float32), mask


def yaw_to_quaternion(yaw: float) -> list[float]:
    return Quaternion(axis=[0, 0, 1], radians=float(yaw)).elements.tolist()


def rotation_matrix_z(yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
