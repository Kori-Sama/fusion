from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import create_splits_scenes
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import DataLoader, Dataset

from fusion.config import FusionConfig, resolve_nuscenes_split
from fusion.constants import CATEGORY_TO_DETECTION
from fusion.data.radar import RADAR_FEATURE_NAMES, build_radar_map
from fusion.data.targets import draw_gaussian, gaussian_radius
from fusion.utils.geometry import make_transform, resize_intrinsics


IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


@dataclass(slots=True)
class SampleIndex:
    sample_token: str
    camera_channel: str


class NuScenesCenterFusionDataset(Dataset):
    def __init__(self, config: FusionConfig, split: str, is_train: bool) -> None:
        self.config = config
        self.split = split
        self.is_train = is_train
        self.dataset_cfg = config.dataset
        self.image_size = tuple(self.dataset_cfg.image_size)
        self.output_size = (
            self.image_size[0] // self.dataset_cfg.output_stride,
            self.image_size[1] // self.dataset_cfg.output_stride,
        )
        self.max_objects = self.dataset_cfg.max_objects
        self.classes = self.dataset_cfg.classes
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        dataroot = Path(self.dataset_cfg.dataroot)
        if not dataroot.exists():
            raise FileNotFoundError(
                f"nuScenes dataroot does not exist: {dataroot}. Place the dataset under the configured data directory first."
            )
        self.nusc = NuScenes(
            version=self.dataset_cfg.version,
            dataroot=self.dataset_cfg.dataroot,
            verbose=False,
        )
        self.samples = self._build_index(split)

    def _build_index(self, split: str) -> list[SampleIndex]:
        resolved_split = resolve_nuscenes_split(self.dataset_cfg.version, split)
        split_scenes = set(create_splits_scenes()[resolved_split])
        indices: list[SampleIndex] = []
        for sample in self.nusc.sample:
            scene = self.nusc.get("scene", sample["scene_token"])
            if scene["name"] not in split_scenes:
                continue
            for camera_channel in self.dataset_cfg.camera_channels:
                if camera_channel in sample["data"]:
                    indices.append(
                        SampleIndex(
                            sample_token=sample["token"], camera_channel=camera_channel
                        )
                    )
        return indices

    def __len__(self) -> int:
        return len(self.samples)

    def _map_category(self, category_name: str) -> str | None:
        for prefix, detection_name in CATEGORY_TO_DETECTION.items():
            if category_name.startswith(prefix):
                return detection_name
        return None

    def _load_image(self, file_path: Path) -> tuple[np.ndarray, tuple[int, int]]:
        image = Image.open(file_path).convert("RGB")
        original_size = (image.height, image.width)
        image = image.resize(
            (self.image_size[1], self.image_size[0]), resample=Image.BILINEAR
        )
        image_np = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        image_np = (image_np - IMAGE_MEAN) / IMAGE_STD
        return image_np, original_size

    def _annotation_targets(
        self,
        sample_rec: dict[str, Any],
        camera_rec: dict[str, Any],
        intrinsics: np.ndarray,
        ego_pose: dict[str, Any],
        camera_calib: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        num_classes = len(self.classes)
        out_h, out_w = self.output_size
        heatmap = np.zeros((num_classes, out_h, out_w), dtype=np.float32)
        indices = np.zeros((self.max_objects,), dtype=np.int64)
        masks = np.zeros((self.max_objects,), dtype=np.float32)
        labels = np.zeros((self.max_objects,), dtype=np.int64)
        offsets = np.zeros((self.max_objects, 2), dtype=np.float32)
        depths = np.zeros((self.max_objects, 1), dtype=np.float32)
        size2d = np.zeros((self.max_objects, 2), dtype=np.float32)
        dim3d = np.zeros((self.max_objects, 3), dtype=np.float32)
        rotation = np.zeros((self.max_objects, 2), dtype=np.float32)
        velocity = np.zeros((self.max_objects, 2), dtype=np.float32)

        ego_rotation_inv = Quaternion(ego_pose["rotation"]).inverse
        camera_rotation_inv = Quaternion(camera_calib["rotation"]).inverse
        ego_translation = np.asarray(ego_pose["translation"], dtype=np.float32)
        camera_translation = np.asarray(camera_calib["translation"], dtype=np.float32)

        object_count = 0
        for ann_token in sample_rec["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            detection_name = self._map_category(ann["category_name"])
            if detection_name is None or detection_name not in self.class_to_idx:
                continue

            global_box = Box(
                ann["translation"], ann["size"], Quaternion(ann["rotation"])
            )
            ego_box = Box(
                global_box.center.copy(),
                global_box.wlh.copy(),
                Quaternion(global_box.orientation),
            )
            ego_box.translate(-ego_translation)
            ego_box.rotate(ego_rotation_inv)

            camera_box = Box(
                global_box.center.copy(),
                global_box.wlh.copy(),
                Quaternion(global_box.orientation),
            )
            camera_box.translate(-ego_translation)
            camera_box.rotate(ego_rotation_inv)
            camera_box.translate(-camera_translation)
            camera_box.rotate(camera_rotation_inv)
            if camera_box.center[2] <= 1e-3:
                continue

            center_2d = view_points(
                camera_box.center.reshape(3, 1), intrinsics, normalize=True
            )[:2, 0]
            if not (
                0 <= center_2d[0] < self.image_size[1]
                and 0 <= center_2d[1] < self.image_size[0]
            ):
                continue

            corners_3d = camera_box.corners().T
            visible = corners_3d[:, 2] > 1e-3
            if not np.any(visible):
                continue
            projected = view_points(corners_3d[visible].T, intrinsics, normalize=True)[
                :2
            ].T
            x_min = np.clip(projected[:, 0].min(), 0, self.image_size[1] - 1)
            y_min = np.clip(projected[:, 1].min(), 0, self.image_size[0] - 1)
            x_max = np.clip(projected[:, 0].max(), 0, self.image_size[1] - 1)
            y_max = np.clip(projected[:, 1].max(), 0, self.image_size[0] - 1)
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min
            if bbox_w < 2 or bbox_h < 2:
                continue

            center_down = center_2d / self.dataset_cfg.output_stride
            center_int = center_down.astype(np.int32)
            if not (0 <= center_int[0] < out_w and 0 <= center_int[1] < out_h):
                continue

            cls_idx = self.class_to_idx[detection_name]
            radius = gaussian_radius(
                bbox_h / self.dataset_cfg.output_stride,
                bbox_w / self.dataset_cfg.output_stride,
            )
            draw_gaussian(
                heatmap[cls_idx],
                (int(center_int[0]), int(center_int[1])),
                max(radius, 1),
            )

            if object_count >= self.max_objects:
                continue
            index = center_int[1] * out_w + center_int[0]
            indices[object_count] = index
            masks[object_count] = 1.0
            labels[object_count] = cls_idx
            offsets[object_count] = center_down - center_int.astype(np.float32)
            depths[object_count, 0] = np.log(np.clip(camera_box.center[2], 1e-3, None))
            size2d[object_count] = np.asarray(
                [
                    bbox_w / self.dataset_cfg.output_stride,
                    bbox_h / self.dataset_cfg.output_stride,
                ],
                dtype=np.float32,
            )
            dim3d[object_count] = np.log(
                np.clip(np.asarray(ann["size"], dtype=np.float32), 1e-3, None)
            )
            yaw = float(quaternion_yaw(ego_box.orientation))
            rotation[object_count] = np.asarray(
                [np.sin(yaw), np.cos(yaw)], dtype=np.float32
            )

            ann_velocity = np.asarray(
                self.nusc.box_velocity(ann_token), dtype=np.float32
            )
            if np.isnan(ann_velocity[:2]).any():
                ann_velocity = np.zeros((3,), dtype=np.float32)
            ego_velocity = ego_rotation_inv.rotation_matrix @ ann_velocity.reshape(3, 1)
            velocity[object_count] = ego_velocity[:2, 0]
            object_count += 1

        return {
            "heatmap": heatmap,
            "indices": indices,
            "mask": masks,
            "labels": labels,
            "offset": offsets,
            "depth": depths,
            "size2d": size2d,
            "dim3d": dim3d,
            "rotation": rotation,
            "velocity": velocity,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.samples[index]
        sample_rec = self.nusc.get("sample", item.sample_token)
        camera_token = sample_rec["data"][item.camera_channel]
        camera_rec = self.nusc.get("sample_data", camera_token)
        camera_calib = self.nusc.get(
            "calibrated_sensor", camera_rec["calibrated_sensor_token"]
        )
        ego_pose = self.nusc.get("ego_pose", camera_rec["ego_pose_token"])

        image_path = Path(self.nusc.dataroot) / camera_rec["filename"]
        image_np, original_size = self._load_image(image_path)
        intrinsics = resize_intrinsics(
            np.asarray(camera_calib["camera_intrinsic"], dtype=np.float32),
            original_size,
            self.image_size,
        )
        radar_map = build_radar_map(
            self.nusc,
            sample_rec,
            item.camera_channel,
            self.dataset_cfg.radar_channels,
            intrinsics,
            self.image_size,
            self.dataset_cfg.radar_sweeps,
        )
        targets = self._annotation_targets(
            sample_rec, camera_rec, intrinsics, ego_pose, camera_calib
        )

        meta = {
            "sample_token": item.sample_token,
            "camera_channel": item.camera_channel,
            "camera_token": camera_token,
            "intrinsics": intrinsics.astype(np.float32),
            "camera_to_ego": make_transform(
                camera_calib["translation"], camera_calib["rotation"]
            ),
            "ego_to_global": make_transform(
                ego_pose["translation"], ego_pose["rotation"]
            ),
            "image_size": np.asarray(self.image_size, dtype=np.int64),
        }

        batch = {
            "image": torch.from_numpy(image_np),
            "radar": torch.from_numpy(radar_map),
            "meta": meta,
        }
        for key, value in targets.items():
            batch[key] = torch.from_numpy(value)
        return batch


def centerfusion_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated: dict[str, Any] = {}
    tensor_keys = [key for key in batch[0].keys() if key != "meta"]
    for key in tensor_keys:
        collated[key] = torch.stack([item[key] for item in batch], dim=0)
    collated["meta"] = [item["meta"] for item in batch]
    return collated


def build_dataloaders(config: FusionConfig) -> tuple[DataLoader, DataLoader]:
    pin_memory = torch.cuda.is_available() and config.training.device.startswith("cuda")
    train_dataset = NuScenesCenterFusionDataset(
        config, split=config.dataset.train_split, is_train=True
    )
    val_dataset = NuScenesCenterFusionDataset(
        config, split=config.dataset.val_split, is_train=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
        pin_memory=pin_memory,
        collate_fn=centerfusion_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=pin_memory,
        collate_fn=centerfusion_collate,
    )
    return train_loader, val_loader


__all__ = ["NuScenesCenterFusionDataset", "build_dataloaders", "RADAR_FEATURE_NAMES"]
