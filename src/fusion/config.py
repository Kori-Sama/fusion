from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from fusion.constants import CAMERA_CHANNELS, DETECTION_CLASSES, RADAR_CHANNELS


@dataclass(slots=True)
class DatasetConfig:
    dataroot: str = "data"
    version: str = "v1.0-trainval"
    train_split: str = "train"
    val_split: str = "val"
    camera_channels: list[str] = field(default_factory=lambda: ["CAM_FRONT"])
    radar_channels: list[str] = field(default_factory=lambda: RADAR_CHANNELS.copy())
    image_size: list[int] = field(default_factory=lambda: [512, 896])
    output_stride: int = 4
    max_objects: int = 64
    radar_sweeps: int = 4
    num_workers: int = 0
    classes: list[str] = field(default_factory=lambda: DETECTION_CLASSES.copy())


@dataclass(slots=True)
class ModelConfig:
    image_channels: int = 3
    radar_channels: int = 5
    width: int = 64
    head_channels: int = 128
    downsample: int = 4
    topk: int = 100
    score_threshold: float = 0.1


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 4
    epochs: int = 12
    learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    grad_clip_norm: float = 5.0
    amp: bool = True
    device: str = "cuda"
    seed: int = 42
    log_every: int = 20
    output_dir: str = "outputs/centerfusion"
    resume_from: str | None = None


@dataclass(slots=True)
class LossConfig:
    heatmap: float = 1.0
    offset: float = 1.0
    depth: float = 1.0
    size2d: float = 0.5
    dim3d: float = 1.0
    rotation: float = 0.5
    velocity: float = 0.2


@dataclass(slots=True)
class EvalConfig:
    checkpoint: str = ""
    output_json: str = "outputs/centerfusion/predictions.json"
    batch_size: int = 2
    official_eval: bool = False
    split: str = "val"


@dataclass(slots=True)
class FusionConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_nuscenes_split(version: str, split: str) -> str:
    if version == "v1.0-mini":
        if split in {"train", "mini_train"}:
            return "mini_train"
        if split in {"val", "mini_val"}:
            return "mini_val"
    return split


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(config_path: str | Path | None = None) -> FusionConfig:
    config = FusionConfig()
    if config_path is None:
        return config
    path = Path(config_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _merge_dataclass(config, data)


def dump_default_config(path: str | Path) -> None:
    config = FusionConfig().to_dict()
    Path(path).write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


__all__ = [
    "CAMERA_CHANNELS",
    "FusionConfig",
    "load_config",
    "dump_default_config",
    "resolve_nuscenes_split",
]
