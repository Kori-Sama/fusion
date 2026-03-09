from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from pyquaternion import Quaternion

from fusion.config import FusionConfig
from fusion.constants import DEFAULT_ATTRIBUTES


def _safe_quaternion_from_matrix(matrix: np.ndarray) -> Quaternion:
    rotation = np.asarray(matrix, dtype=np.float64)
    u, _, vh = np.linalg.svd(rotation)
    ortho = u @ vh
    if np.linalg.det(ortho) < 0:
        u[:, -1] *= -1.0
        ortho = u @ vh
    return Quaternion(matrix=ortho)


def _nms(heatmap: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    pad = (kernel - 1) // 2
    pooled = F.max_pool2d(heatmap, kernel, stride=1, padding=pad)
    keep = (pooled == heatmap).float()
    return heatmap * keep


def _topk(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, ...]:
    batch, classes, height, width = scores.shape
    topk_scores, topk_indices = torch.topk(scores.view(batch, classes, -1), k)
    topk_indices = topk_indices % (height * width)
    topk_ys = torch.div(topk_indices, width, rounding_mode="floor")
    topk_xs = topk_indices % width

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), k)
    topk_classes = torch.div(topk_ind, k, rounding_mode="floor")
    topk_indices = topk_indices.view(batch, -1).gather(1, topk_ind)
    topk_ys = topk_ys.view(batch, -1).gather(1, topk_ind)
    topk_xs = topk_xs.view(batch, -1).gather(1, topk_ind)
    return topk_score, topk_indices, topk_classes, topk_ys, topk_xs


def _transpose_and_gather(feat: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = feat.shape
    feat = feat.view(batch, channels, -1).permute(0, 2, 1).contiguous()
    return feat.gather(1, indices.unsqueeze(-1).expand(-1, -1, channels))


def _attribute_name(label: str, velocity: np.ndarray) -> str:
    speed = float(np.linalg.norm(velocity[:2]))
    if label == "pedestrian":
        return "pedestrian.moving" if speed > 0.3 else "pedestrian.standing"
    if label in {"bicycle", "motorcycle"}:
        return "cycle.with_rider" if speed > 0.3 else DEFAULT_ATTRIBUTES[label]
    if label in {"bus"}:
        return "vehicle.moving" if speed > 0.5 else "vehicle.stopped"
    if label in {"car", "truck", "construction_vehicle", "trailer"}:
        return "vehicle.moving" if speed > 0.5 else DEFAULT_ATTRIBUTES[label]
    return DEFAULT_ATTRIBUTES[label]


def decode_batch_predictions(
    outputs: dict[str, torch.Tensor],
    meta: list[dict[str, Any]],
    config: FusionConfig,
    include_aux: bool = False,
) -> list[list[dict[str, Any]]]:
    heatmap = _nms(outputs["heatmap"].sigmoid())
    scores, indices, class_ids, ys, xs = _topk(heatmap, config.model.topk)

    offset = _transpose_and_gather(outputs["offset"], indices)
    depth = _transpose_and_gather(outputs["depth"], indices).exp()
    size2d = _transpose_and_gather(outputs["size2d"], indices)
    dim3d = _transpose_and_gather(outputs["dim3d"], indices).exp()
    rotation = _transpose_and_gather(outputs["rotation"], indices)
    velocity = _transpose_and_gather(outputs["velocity"], indices)

    detections: list[list[dict[str, Any]]] = []
    stride = config.dataset.output_stride
    for batch_idx in range(scores.shape[0]):
        intrinsics = np.asarray(meta[batch_idx]["intrinsics"], dtype=np.float32)
        camera_to_ego = np.asarray(meta[batch_idx]["camera_to_ego"], dtype=np.float32)
        ego_to_global = np.asarray(meta[batch_idx]["ego_to_global"], dtype=np.float32)
        sample_token = meta[batch_idx]["sample_token"]
        batch_detections: list[dict[str, Any]] = []
        for det_idx in range(scores.shape[1]):
            score = float(scores[batch_idx, det_idx].item())
            if score < config.model.score_threshold:
                continue
            label_idx = int(class_ids[batch_idx, det_idx].item())
            label = config.dataset.classes[label_idx]
            center_x = float(
                (xs[batch_idx, det_idx] + offset[batch_idx, det_idx, 0]).item() * stride
            )
            center_y = float(
                (ys[batch_idx, det_idx] + offset[batch_idx, det_idx, 1]).item() * stride
            )
            depth_value = float(depth[batch_idx, det_idx, 0].item())
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
            x_cam = (center_x - cx) / fx * depth_value
            y_cam = (center_y - cy) / fy * depth_value
            center_cam = np.asarray([x_cam, y_cam, depth_value, 1.0], dtype=np.float32)
            center_ego = camera_to_ego @ center_cam
            center_global = ego_to_global @ center_ego

            yaw = float(
                torch.atan2(
                    rotation[batch_idx, det_idx, 0], rotation[batch_idx, det_idx, 1]
                ).item()
            )
            yaw_ego = Quaternion(axis=[0, 0, 1], radians=yaw)
            ego_rotation = _safe_quaternion_from_matrix(ego_to_global[:3, :3])
            global_rotation = ego_rotation * yaw_ego

            velocity_ego = np.asarray(
                velocity[batch_idx, det_idx].detach().cpu(), dtype=np.float32
            )
            velocity_global = ego_to_global[:2, :2] @ velocity_ego
            size = np.asarray(
                dim3d[batch_idx, det_idx].detach().cpu(), dtype=np.float32
            )
            batch_detections.append(
                {
                    "sample_token": sample_token,
                    "translation": center_global[:3].tolist(),
                    "size": size.tolist(),
                    "rotation": global_rotation.elements.tolist(),
                    "velocity": velocity_global.tolist(),
                    "detection_name": label,
                    "detection_score": score,
                    "attribute_name": _attribute_name(label, velocity_global),
                }
            )
            if include_aux:
                batch_detections[-1]["center_2d"] = [center_x, center_y]
                batch_detections[-1]["size2d"] = (
                    size2d[batch_idx, det_idx].detach().cpu().tolist()
                )
        detections.append(batch_detections)
    return detections
