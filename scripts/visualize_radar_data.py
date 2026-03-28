from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive nuScenes sensor data viewer for LiDAR + Radar "
            "(overlay, BEV, or both)"
        )
    )
    parser.add_argument("--mode", choices=["overlay", "bev", "both"], default="both")
    parser.add_argument("--config", type=str, default="configs/nuscenes_centerfusion.yaml")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--camera", type=str, default="CAM_FRONT")
    parser.add_argument("--lidar-channel", type=str, default="LIDAR_TOP")
    parser.add_argument("--lidar-sweeps", type=int, default=1)
    parser.add_argument("--radar-sweeps", type=int, default=-1)

    parser.add_argument("--lidar-point-size", type=float, default=2.0)
    parser.add_argument("--radar-point-size", type=float, default=16.0)
    parser.add_argument("--lidar-alpha", type=float, default=0.55)
    parser.add_argument("--radar-alpha", type=float, default=0.9)

    parser.add_argument("--x-range", type=float, default=60.0)
    parser.add_argument("--y-range", type=float, default=60.0)
    parser.add_argument("--show-radar-velocity", action="store_true", default=True)
    parser.add_argument("--no-show-radar-velocity", dest="show_radar_velocity", action="store_false")
    parser.add_argument("--max-velocity-arrows", type=int, default=160)

    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--save-dpi", type=int, default=120)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode in {"overlay", "both"}:
        from visualize_overlay import run_overlay

        run_overlay(args)

    if args.mode in {"bev", "both"}:
        from visualize_bev import run_bev

        run_bev(args)


if __name__ == "__main__":
    main()
