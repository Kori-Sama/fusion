from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.splits import create_splits_scenes
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fusion.config import load_config, resolve_nuscenes_split
from fusion.utils.geometry import ensure_dir, project_points


@dataclass(slots=True)
class FrameRecord:
    sample_token: str
    camera_channel: str


def _build_indices(
    nusc: NuScenes, version: str, split: str, camera_channel: str
) -> list[FrameRecord]:
    resolved_split = resolve_nuscenes_split(version, split)
    splits = create_splits_scenes()
    if resolved_split not in splits:
        raise ValueError(f"unknown split '{split}' resolved to '{resolved_split}'")

    split_scenes = set(splits[resolved_split])
    indices: list[FrameRecord] = []
    for sample in nusc.sample:
        scene = nusc.get("scene", sample["scene_token"])
        if scene["name"] not in split_scenes:
            continue
        if camera_channel not in sample["data"]:
            continue
        indices.append(
            FrameRecord(sample_token=sample["token"], camera_channel=camera_channel)
        )
    return indices


def _collect_lidar_points(
    nusc: NuScenes,
    sample_rec: dict,
    camera_channel: str,
    lidar_channel: str,
    nsweeps: int,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if lidar_channel not in sample_rec["data"]:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    point_cloud, _ = LidarPointCloud.from_file_multisweep(
        nusc,
        sample_rec,
        lidar_channel,
        camera_channel,
        nsweeps=nsweeps,
    )
    points = point_cloud.points
    if points.shape[1] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    xyz = points[:3, :].T.astype(np.float32)
    projected, mask = project_points(xyz, intrinsics, image_size)
    if not np.any(mask):
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    return projected[mask], xyz[mask, 2]


def _collect_radar_points(
    nusc: NuScenes,
    sample_rec: dict,
    camera_channel: str,
    radar_channels: list[str],
    nsweeps: int,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    uv_list: list[np.ndarray] = []
    depth_list: list[np.ndarray] = []

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

        uv_list.append(projected[mask])
        depth_list.append(xyz[mask, 2])

    if not uv_list:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    return np.concatenate(uv_list, axis=0), np.concatenate(depth_list, axis=0)


class OverlayViewer:
    def __init__(self, nusc: NuScenes, frames: list[FrameRecord], args: argparse.Namespace):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        self.plt = plt
        self.Line2D = Line2D
        self.nusc = nusc
        self.frames = frames
        self.args = args
        self.current = 0
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.colorbar = None
        self.save_dir = ensure_dir(args.save_dir) if args.save_dir else None
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._draw()

    def _draw(self) -> None:
        frame = self.frames[self.current]
        sample_rec = self.nusc.get("sample", frame.sample_token)
        camera_token = sample_rec["data"][frame.camera_channel]
        camera_rec = self.nusc.get("sample_data", camera_token)
        camera_calib = self.nusc.get(
            "calibrated_sensor", camera_rec["calibrated_sensor_token"]
        )

        image_path = Path(self.nusc.dataroot) / camera_rec["filename"]
        image = np.asarray(Image.open(image_path).convert("RGB"))
        intrinsics = np.asarray(camera_calib["camera_intrinsic"], dtype=np.float32)
        image_size = (image.shape[0], image.shape[1])

        lidar_uv, lidar_depth = _collect_lidar_points(
            self.nusc,
            sample_rec,
            frame.camera_channel,
            self.args.lidar_channel,
            self.args.lidar_sweeps,
            intrinsics,
            image_size,
        )
        radar_uv, radar_depth = _collect_radar_points(
            self.nusc,
            sample_rec,
            frame.camera_channel,
            self.args.radar_channels,
            self.args.radar_sweeps,
            intrinsics,
            image_size,
        )

        self.ax.clear()
        self.ax.imshow(image)
        self.ax.set_axis_off()

        all_depth: list[np.ndarray] = []
        if lidar_depth.size > 0:
            all_depth.append(lidar_depth)
        if radar_depth.size > 0:
            all_depth.append(radar_depth)

        if all_depth:
            merged_depth = np.concatenate(all_depth, axis=0)
            vmin = float(np.percentile(merged_depth, 2.0))
            vmax = float(np.percentile(merged_depth, 98.0))
            if vmax <= vmin:
                vmax = vmin + 1.0

            color_source = None
            if lidar_uv.shape[0] > 0:
                color_source = self.ax.scatter(
                    lidar_uv[:, 0],
                    lidar_uv[:, 1],
                    c=lidar_depth,
                    s=self.args.lidar_point_size,
                    alpha=self.args.lidar_alpha,
                    cmap="turbo",
                    vmin=vmin,
                    vmax=vmax,
                    marker=".",
                    linewidths=0.0,
                )
            if radar_uv.shape[0] > 0:
                radar_plot = self.ax.scatter(
                    radar_uv[:, 0],
                    radar_uv[:, 1],
                    c=radar_depth,
                    s=self.args.radar_point_size,
                    alpha=self.args.radar_alpha,
                    cmap="turbo",
                    vmin=vmin,
                    vmax=vmax,
                    marker="x",
                    linewidths=0.8,
                )
                if color_source is None:
                    color_source = radar_plot

            if self.colorbar is not None:
                self.colorbar.remove()
                self.colorbar = None
            if color_source is not None:
                self.colorbar = self.fig.colorbar(color_source, ax=self.ax, pad=0.01)
                self.colorbar.set_label("Depth (m)")
        elif self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None

        legend_handles = [
            self.Line2D([0], [0], marker=".", color="w", label="LiDAR", markerfacecolor="black", markersize=8),
            self.Line2D([0], [0], marker="x", color="black", label="Radar", linestyle="None", markersize=8),
        ]
        self.ax.legend(handles=legend_handles, loc="lower right", framealpha=0.9)

        title = (
            f"Overlay [{self.current + 1}/{len(self.frames)}] "
            f"sample={frame.sample_token[:8]} cam={frame.camera_channel} "
            f"lidar={lidar_uv.shape[0]} radar={radar_uv.shape[0]}"
        )
        subtitle = "Keys: n/right next, p/left prev, s save, q quit"
        self.ax.set_title(f"{title}\n{subtitle}")

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def _save_current(self) -> None:
        if self.save_dir is None:
            return
        frame = self.frames[self.current]
        output = (
            self.save_dir
            / f"overlay_{self.current:04d}_{frame.sample_token}_{frame.camera_channel}.png"
        )
        self.fig.savefig(output, dpi=self.args.save_dpi, bbox_inches="tight")
        print(f"saved: {output}")

    def _on_key(self, event) -> None:
        key = event.key
        if key in {"n", "right", " "}:
            if self.current < len(self.frames) - 1:
                self.current += 1
                self._draw()
            return
        if key in {"p", "left"}:
            if self.current > 0:
                self.current -= 1
                self._draw()
            return
        if key == "s":
            self._save_current()
            return
        if key == "q":
            self.plt.close(self.fig)

    def show(self) -> None:
        self.plt.show()


def add_overlay_args(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--save-dpi", type=int, default=120)


def run_overlay(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    dataroot = Path(config.dataset.dataroot)
    if not dataroot.exists():
        raise FileNotFoundError(f"nuScenes dataroot does not exist: {dataroot}")

    radar_sweeps = (
        args.radar_sweeps if args.radar_sweeps > 0 else int(config.dataset.radar_sweeps)
    )
    args.radar_sweeps = radar_sweeps
    args.radar_channels = list(config.dataset.radar_channels)

    nusc = NuScenes(
        version=config.dataset.version,
        dataroot=config.dataset.dataroot,
        verbose=False,
    )
    all_frames = _build_indices(
        nusc,
        version=config.dataset.version,
        split=args.split,
        camera_channel=args.camera,
    )

    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    selected = all_frames[args.start_index : args.start_index + max(1, args.max_samples)]
    if not selected:
        raise RuntimeError("no frames selected, check split/camera/start-index/max-samples")

    print(
        f"overlay viewer: split={args.split} camera={args.camera} "
        f"frames={len(selected)} lidar_sweeps={args.lidar_sweeps} radar_sweeps={args.radar_sweeps}"
    )
    viewer = OverlayViewer(nusc, selected, args)
    viewer.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive LiDAR + Radar overlay viewer on camera images"
    )
    add_overlay_args(parser)
    args = parser.parse_args()

    try:
        import matplotlib  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required; install dependencies first (e.g. uv sync)."
        ) from exc

    run_overlay(args)


if __name__ == "__main__":
    main()
