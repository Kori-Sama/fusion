from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

from fusion.config import FusionConfig
from fusion.data.dataset import NuScenesCenterFusionDataset, centerfusion_collate
from fusion.engine.trainer import load_checkpoint, move_batch_to_device
from fusion.model import CenterFusionDetector, decode_batch_predictions
from fusion.utils.geometry import ensure_dir

IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

COLORS = {
    "car": (0, 255, 0),
    "truck": (255, 165, 0),
    "construction_vehicle": (255, 140, 0),
    "bus": (255, 255, 0),
    "trailer": (255, 215, 0),
    "barrier": (255, 0, 0),
    "motorcycle": (0, 191, 255),
    "bicycle": (0, 128, 255),
    "pedestrian": (255, 0, 255),
    "traffic_cone": (255, 99, 71),
}


def _to_pil_image(image_tensor: torch.Tensor) -> Image.Image:
    image = image_tensor.detach().cpu().numpy().astype(np.float32)
    image = image * IMAGE_STD + IMAGE_MEAN
    image = np.clip(image, 0.0, 1.0)
    image = (image.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return Image.fromarray(image)


@torch.no_grad()
def visualize_checkpoint(
    config: FusionConfig,
    checkpoint_path: str | Path,
    split: str = "val",
    max_samples: int = 8,
    output_dir: str | Path | None = None,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    vis_dir = ensure_dir(
        output_dir or (Path(config.training.output_dir) / "visualizations")
    )
    dataset = NuScenesCenterFusionDataset(config, split=split, is_train=False)
    subset = Subset(dataset, range(min(max_samples, len(dataset))))
    dataloader = DataLoader(
        subset,
        batch_size=min(config.evaluation.batch_size, max(1, len(subset))),
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=torch.cuda.is_available()
        and config.training.device.startswith("cuda"),
        collate_fn=centerfusion_collate,
    )

    device_name = (
        config.training.device
        if torch.cuda.is_available() and config.training.device.startswith("cuda")
        else "cpu"
    )
    device = torch.device(device_name)
    model = CenterFusionDetector(config).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()

    saved = 0
    for batch in dataloader:
        meta = batch["meta"]
        images = batch["image"]
        batch_on_device = move_batch_to_device(batch, device)
        outputs = model(batch_on_device["image"], batch_on_device["radar"])
        predictions = decode_batch_predictions(outputs, meta, config, include_aux=True)

        for image_tensor, sample_meta, items in zip(images, meta, predictions):
            canvas = _to_pil_image(image_tensor)
            draw = ImageDraw.Draw(canvas)
            for item in items:
                center_x, center_y = item["center_2d"]
                width, height = item["size2d"]
                left = center_x - width / 2.0
                right = center_x + width / 2.0
                top = center_y - height / 2.0
                bottom = center_y + height / 2.0
                color = COLORS.get(item["detection_name"], (0, 255, 255))
                draw.rectangle((left, top, right, bottom), outline=color, width=2)
                draw.ellipse(
                    (center_x - 3, center_y - 3, center_x + 3, center_y + 3), fill=color
                )
                draw.text(
                    (left, max(0.0, top - 14.0)),
                    f"{item['detection_name']} {item['detection_score']:.2f}",
                    fill=color,
                )

            output_path = (
                vis_dir
                / f"{saved:03d}_{sample_meta['sample_token']}_{sample_meta['camera_channel']}.png"
            )
            canvas.save(output_path)
            saved += 1

    return vis_dir
