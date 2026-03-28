from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.splits import create_splits_scenes

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fusion.config import load_config, resolve_nuscenes_split
from fusion.utils.geometry import make_transform, transform_points


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


def _collect_lidar_ego_points(
    nusc: NuScenes,
    sample_rec: dict,
    lidar_channel: str,
    lidar_sweeps: int,
) -> np.ndarray:
    if lidar_channel not in sample_rec["data"]:
        return np.zeros((0, 3), dtype=np.float32)

    lidar_token = sample_rec["data"][lidar_channel]
    lidar_rec = nusc.get("sample_data", lidar_token)
    lidar_calib = nusc.get("calibrated_sensor", lidar_rec["calibrated_sensor_token"])
    lidar_to_ego = make_transform(lidar_calib["translation"], lidar_calib["rotation"])

    point_cloud, _ = LidarPointCloud.from_file_multisweep(
        nusc,
        sample_rec,
        lidar_channel,
        lidar_channel,
        nsweeps=lidar_sweeps,
    )
    points = point_cloud.points
    if points.shape[1] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    lidar_xyz = points[:3, :].T.astype(np.float32)
    return transform_points(lidar_xyz, lidar_to_ego)


def _collect_radar_ego_points(
    nusc: NuScenes,
    sample_rec: dict,
    radar_channels: list[str],
    radar_sweeps: int,
    lidar_channel: str,
) -> tuple[np.ndarray, np.ndarray]:
    if lidar_channel not in sample_rec["data"]:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    lidar_token = sample_rec["data"][lidar_channel]
    lidar_rec = nusc.get("sample_data", lidar_token)
    lidar_calib = nusc.get("calibrated_sensor", lidar_rec["calibrated_sensor_token"])
    lidar_to_ego = make_transform(lidar_calib["translation"], lidar_calib["rotation"])
    lidar_to_ego_rotation = lidar_to_ego[:3, :3]

    xyz_list: list[np.ndarray] = []
    velocity_list: list[np.ndarray] = []

    for radar_channel in radar_channels:
        if radar_channel not in sample_rec["data"]:
            continue

        point_cloud, _ = RadarPointCloud.from_file_multisweep(
            nusc,
            sample_rec,
            radar_channel,
            lidar_channel,
            nsweeps=radar_sweeps,
        )
        points = point_cloud.points
        if points.shape[1] == 0:
            continue

        radar_xyz_lidar = points[:3, :].T.astype(np.float32)
        radar_xyz_ego = transform_points(radar_xyz_lidar, lidar_to_ego)
        xyz_list.append(radar_xyz_ego)

        if points.shape[0] > 9:
            velocity_lidar = np.zeros((points.shape[1], 3), dtype=np.float32)
            velocity_lidar[:, 0] = points[8, :].astype(np.float32)
            velocity_lidar[:, 1] = points[9, :].astype(np.float32)
            velocity_ego = (lidar_to_ego_rotation @ velocity_lidar.T).T[:, :2]
            velocity_list.append(velocity_ego)
        else:
            velocity_list.append(np.zeros((points.shape[1], 2), dtype=np.float32))

    if not xyz_list:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    return np.concatenate(xyz_list, axis=0), np.concatenate(velocity_list, axis=0)


class BevViewer:
    def __init__(self, nusc: NuScenes, frames: list[FrameRecord], args: argparse.Namespace):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        self.plt = plt
        self.Line2D = Line2D
        self.nusc = nusc
        self.frames = frames
        self.args = args
        self.current = 0
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.colorbar = None
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._draw()

    def _draw(self) -> None:
        frame = self.frames[self.current]
        sample_rec = self.nusc.get("sample", frame.sample_token)

        lidar_xyz = _collect_lidar_ego_points(
            self.nusc,
            sample_rec,
            self.args.lidar_channel,
            self.args.lidar_sweeps,
        )
        radar_xyz, radar_velocity = _collect_radar_ego_points(
            self.nusc,
            sample_rec,
            self.args.radar_channels,
            self.args.radar_sweeps,
            self.args.lidar_channel,
        )

        self.ax.clear()

        all_depth: list[np.ndarray] = []
        if lidar_xyz.shape[0] > 0:
            all_depth.append(np.linalg.norm(lidar_xyz[:, :2], axis=1))
        if radar_xyz.shape[0] > 0:
            all_depth.append(np.linalg.norm(radar_xyz[:, :2], axis=1))

        if all_depth:
            merged_depth = np.concatenate(all_depth, axis=0)
            vmin = float(np.percentile(merged_depth, 2.0))
            vmax = float(np.percentile(merged_depth, 98.0))
            if vmax <= vmin:
                vmax = vmin + 1.0

            color_source = None
            if lidar_xyz.shape[0] > 0:
                lidar_depth = np.linalg.norm(lidar_xyz[:, :2], axis=1)
                color_source = self.ax.scatter(
                    lidar_xyz[:, 0],
                    lidar_xyz[:, 1],
                    c=lidar_depth,
                    s=self.args.lidar_point_size,
                    alpha=self.args.lidar_alpha,
                    cmap="turbo",
                    vmin=vmin,
                    vmax=vmax,
                    marker=".",
                    linewidths=0.0,
                )
            if radar_xyz.shape[0] > 0:
                radar_depth = np.linalg.norm(radar_xyz[:, :2], axis=1)
                radar_plot = self.ax.scatter(
                    radar_xyz[:, 0],
                    radar_xyz[:, 1],
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
                self.colorbar.set_label("Range (m)")
        elif self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None

        if self.args.show_radar_velocity and radar_xyz.shape[0] > 0:
            count = radar_xyz.shape[0]
            stride = max(1, count // max(1, self.args.max_velocity_arrows))
            indices = np.arange(0, count, stride)
            self.ax.quiver(
                radar_xyz[indices, 0],
                radar_xyz[indices, 1],
                radar_velocity[indices, 0],
                radar_velocity[indices, 1],
                color="red",
                alpha=0.7,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.002,
            )

        self.ax.scatter([0.0], [0.0], marker="^", color="black", s=50)
        self.ax.arrow(0.0, 0.0, 3.0, 0.0, color="black", width=0.05, head_width=0.8)
        self.ax.set_xlim(-self.args.x_range, self.args.x_range)
        self.ax.set_ylim(-self.args.y_range, self.args.y_range)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        self.ax.set_xlabel("Ego X (m)")
        self.ax.set_ylabel("Ego Y (m)")

        legend_handles = [
            self.Line2D([0], [0], marker=".", color="w", label="LiDAR", markerfacecolor="black", markersize=8),
            self.Line2D([0], [0], marker="x", color="black", label="Radar", linestyle="None", markersize=8),
        ]
        self.ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9)

        title = (
            f"BEV [{self.current + 1}/{len(self.frames)}] "
            f"sample={frame.sample_token[:8]} lidar={lidar_xyz.shape[0]} radar={radar_xyz.shape[0]}"
        )
        subtitle = "Keys: n/right next, p/left prev, q quit"
        self.ax.set_title(f"{title}\n{subtitle}")

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

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
        if key == "q":
            self.plt.close(self.fig)

    def show(self) -> None:
        self.plt.show()


def add_bev_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default="configs/nuscenes_centerfusion.yaml")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--camera", type=str, default="CAM_FRONT")
    parser.add_argument("--lidar-channel", type=str, default="LIDAR_TOP")
    parser.add_argument("--lidar-sweeps", type=int, default=1)
    parser.add_argument("--radar-sweeps", type=int, default=-1)
    parser.add_argument("--x-range", type=float, default=60.0)
    parser.add_argument("--y-range", type=float, default=60.0)
    parser.add_argument("--lidar-point-size", type=float, default=2.0)
    parser.add_argument("--radar-point-size", type=float, default=16.0)
    parser.add_argument("--lidar-alpha", type=float, default=0.55)
    parser.add_argument("--radar-alpha", type=float, default=0.9)
    parser.add_argument("--show-radar-velocity", action="store_true", default=True)
    parser.add_argument("--no-show-radar-velocity", dest="show_radar_velocity", action="store_false")
    parser.add_argument("--max-velocity-arrows", type=int, default=160)


def run_bev(args: argparse.Namespace) -> None:
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
        f"bev viewer: split={args.split} camera={args.camera} frames={len(selected)} "
        f"lidar_sweeps={args.lidar_sweeps} radar_sweeps={args.radar_sweeps}"
    )
    viewer = BevViewer(nusc, selected, args)
    viewer.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive BEV viewer for LiDAR + Radar points"
    )
    add_bev_args(parser)
    args = parser.parse_args()

    try:
        import matplotlib  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required; install dependencies first (e.g. uv sync)."
        ) from exc

    run_bev(args)


if __name__ == "__main__":
    main()
