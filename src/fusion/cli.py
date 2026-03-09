from __future__ import annotations

import argparse
from pathlib import Path

from fusion.config import dump_default_config, load_config
from fusion.data.dataset import NuScenesCenterFusionDataset
from fusion.engine import evaluate_checkpoint, train_model, visualize_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="nuScenes CenterFusion-style camera + radar baseline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the fusion detector")
    train_parser.add_argument(
        "--config", type=str, default="configs/nuscenes_centerfusion.yaml"
    )

    eval_parser = subparsers.add_parser(
        "evaluate", help="Export or evaluate predictions from a checkpoint"
    )
    eval_parser.add_argument(
        "--config", type=str, default="configs/nuscenes_centerfusion.yaml"
    )
    eval_parser.add_argument("--checkpoint", type=str, default="")

    vis_parser = subparsers.add_parser(
        "visualize", help="Run a checkpoint and save visualization images"
    )
    vis_parser.add_argument(
        "--config", type=str, default="configs/nuscenes_centerfusion.yaml"
    )
    vis_parser.add_argument("--checkpoint", type=str, default="")
    vis_parser.add_argument("--split", type=str, default="val")
    vis_parser.add_argument("--max-samples", type=int, default=8)
    vis_parser.add_argument("--output-dir", type=str, default="")

    dump_parser = subparsers.add_parser(
        "dump-config", help="Write the default configuration file"
    )
    dump_parser.add_argument("--output", type=str, default="configs/default.yaml")

    inspect_parser = subparsers.add_parser(
        "inspect-data", help="Inspect the configured dataset index"
    )
    inspect_parser.add_argument(
        "--config", type=str, default="configs/nuscenes_centerfusion.yaml"
    )
    inspect_parser.add_argument("--split", type=str, default="val")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "dump-config":
        dump_default_config(args.output)
        print(f"default config written to {args.output}")
        return

    config = load_config(args.config)

    if args.command == "train":
        best = train_model(config)
        print(f"best checkpoint saved to {best}")
        return

    if args.command == "evaluate":
        checkpoint = (
            args.checkpoint
            or config.evaluation.checkpoint
            or Path(config.training.output_dir) / "best.ckpt"
        )
        output = evaluate_checkpoint(config, checkpoint)
        print(f"predictions written to {output}")
        return

    if args.command == "visualize":
        checkpoint = (
            args.checkpoint
            or config.evaluation.checkpoint
            or Path(config.training.output_dir) / "best.ckpt"
        )
        output = visualize_checkpoint(
            config,
            checkpoint,
            split=args.split,
            max_samples=args.max_samples,
            output_dir=args.output_dir or None,
        )
        print(f"visualizations written to {output}")
        return

    if args.command == "inspect-data":
        dataset = NuScenesCenterFusionDataset(config, split=args.split, is_train=False)
        print(
            f"split={args.split} samples={len(dataset)} cameras={config.dataset.camera_channels}"
        )
        if len(dataset) > 0:
            sample = dataset[0]
            print(
                f"image={tuple(sample['image'].shape)} radar={tuple(sample['radar'].shape)} "
                f"objects={int(sample['mask'].sum().item())}"
            )
        return


if __name__ == "__main__":
    main()
