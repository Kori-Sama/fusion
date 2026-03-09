from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from fusion.config import FusionConfig
from fusion.data import build_dataloaders
from fusion.model import CenterFusionDetector, CenterFusionLoss
from fusion.utils.geometry import ensure_dir
from fusion.utils.seed import seed_everything


TARGET_KEYS = [
    "image",
    "radar",
    "heatmap",
    "indices",
    "mask",
    "labels",
    "offset",
    "depth",
    "size2d",
    "dim3d",
    "rotation",
    "velocity",
]


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {"meta": batch["meta"]}
    for key in TARGET_KEYS:
        if key in batch:
            moved[key] = batch[key].to(device, non_blocking=True)
    return moved


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_val: float,
    config: FusionConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "config": config.to_dict(),
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


@torch.no_grad()
def run_validation(
    model: nn.Module,
    criterion: CenterFusionLoss,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    loss_meter = 0.0
    batches = 0
    for batch in tqdm(dataloader, desc="val", leave=False):
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["image"], batch["radar"])
        loss, _ = criterion(outputs, batch)
        loss_meter += float(loss.item())
        batches += 1
    return loss_meter / max(1, batches)


def train_model(config: FusionConfig) -> Path:
    seed_everything(config.training.seed)
    output_dir = ensure_dir(config.training.output_dir)
    device_name = (
        config.training.device
        if torch.cuda.is_available() and config.training.device.startswith("cuda")
        else "cpu"
    )
    device = torch.device(device_name)

    train_loader, val_loader = build_dataloaders(config)
    model = CenterFusionDetector(config).to(device)
    criterion = CenterFusionLoss(config)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, config.training.epochs))
    scaler = torch.amp.GradScaler(
        device="cuda", enabled=config.training.amp and device.type == "cuda"
    )

    start_epoch = 0
    best_val = float("inf")
    if config.training.resume_from:
        checkpoint = load_checkpoint(
            config.training.resume_from,
            model,
            optimizer,
            scheduler,
            map_location=device,
        )
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = float(checkpoint.get("best_val", float("inf")))

    for epoch in range(start_epoch, config.training.epochs):
        model.train()
        progress = tqdm(
            train_loader, desc=f"train {epoch + 1}/{config.training.epochs}"
        )
        for step, batch in enumerate(progress, start=1):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                enabled=config.training.amp and device.type == "cuda",
            ):
                outputs = model(batch["image"], batch["radar"])
                loss, metrics = criterion(outputs, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.grad_clip_norm
            )
            scaler.step(optimizer)
            scaler.update()
            if step % config.training.log_every == 0 or step == 1:
                progress.set_postfix(loss=f"{metrics['total']:.4f}")

        scheduler.step()
        val_loss = run_validation(model, criterion, val_loader, device)
        latest_path = output_dir / "last.ckpt"
        save_checkpoint(
            latest_path, model, optimizer, scheduler, epoch, best_val, config
        )
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                output_dir / "best.ckpt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_val,
                config,
            )
        print(f"epoch={epoch + 1} val_loss={val_loss:.4f} best_val={best_val:.4f}")

    return output_dir / "best.ckpt"
