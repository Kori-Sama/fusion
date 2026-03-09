from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from tqdm import tqdm

from fusion.config import FusionConfig, resolve_nuscenes_split
from fusion.data.dataset import NuScenesCenterFusionDataset, centerfusion_collate
from fusion.model import CenterFusionDetector, decode_batch_predictions
from fusion.utils.geometry import ensure_dir
from fusion.utils.io import save_json
from fusion.engine.trainer import load_checkpoint, move_batch_to_device
from torch.utils.data import DataLoader


def _eval_set_name(version: str, split: str) -> str:
    return resolve_nuscenes_split(version, split)


def evaluate_checkpoint(
    config: FusionConfig, checkpoint_path: str | Path | None = None
) -> Path:
    checkpoint_path = Path(checkpoint_path or config.evaluation.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    split = config.evaluation.split
    dataset = NuScenesCenterFusionDataset(config, split=split, is_train=False)
    dataloader = DataLoader(
        dataset,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
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

    predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="decode"):
            meta = batch["meta"]
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["image"], batch["radar"])
            decoded = decode_batch_predictions(outputs, meta, config)
            for items in decoded:
                for item in items:
                    predictions[item["sample_token"]].append(item)

    serialized = {
        "meta": {
            "use_camera": True,
            "use_radar": True,
            "use_lidar": False,
            "use_map": False,
            "use_external": False,
        },
        "results": predictions,
    }
    output_path = Path(config.evaluation.output_json)
    save_json(serialized, output_path)

    if config.evaluation.official_eval:
        eval_dir = ensure_dir(output_path.parent / "official_eval")
        evaluator = NuScenesEval(
            dataset.nusc,
            config=config_factory("detection_cvpr_2019"),
            result_path=str(output_path),
            eval_set=_eval_set_name(config.dataset.version, split),
            output_dir=str(eval_dir),
            verbose=True,
        )
        evaluator.main(render_curves=False)
    return output_path
