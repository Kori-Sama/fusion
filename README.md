# nuScenes CenterFusion 风格视觉+雷达融合基线

这是一个基于 `uv` 管理环境、基于 nuScenes 数据集、采用 **center-based** 检测范式的视觉+雷达融合基线实现。

## 特性

- 前视相机为默认输入，可扩展到多相机
- 聚合多雷达 sweep，并投影到图像平面形成雷达特征图
- 使用 center-based 检测头回归：
  - 类别热力图
  - 2D center offset
  - 深度
  - 2D bbox size
  - 3D size
  - ego 坐标系航向角
  - ego 坐标系速度
- 支持训练、导出预测、可选官方 nuScenes detection eval
- 默认通过 `uv run fusion ...` 调用

## 目录结构

- [configs/nuscenes_centerfusion.yaml](configs/nuscenes_centerfusion.yaml)
- [src/fusion/data/dataset.py](src/fusion/data/dataset.py)
- [src/fusion/data/radar.py](src/fusion/data/radar.py)
- [src/fusion/model/detector.py](src/fusion/model/detector.py)
- [src/fusion/model/losses.py](src/fusion/model/losses.py)
- [src/fusion/model/decode.py](src/fusion/model/decode.py)
- [src/fusion/engine/trainer.py](src/fusion/engine/trainer.py)
- [src/fusion/engine/evaluator.py](src/fusion/engine/evaluator.py)
- [src/fusion/cli.py](src/fusion/cli.py)

## 数据准备

当前工作区里还没有检测到 `data/` 目录，因此运行前请先将 nuScenes 数据集放到项目根目录下的 `data/` 中。

推荐结构：

```text
fusion/
  data/
    samples/
    sweeps/
    maps/
    v1.0-trainval/
```

如果你使用 mini 版本，可将配置中的 `dataset.version` 改为 `v1.0-mini`，并把 split 对应改成 `mini_train` / `mini_val` 的语义配置。

## 使用 uv

首次同步依赖：

```bash
uv sync
```

查看帮助：

```bash
uv run fusion --help
```

导出默认配置：

```bash
uv run fusion dump-config --output configs/default.yaml
```

检查数据索引：

```bash
uv run fusion inspect-data --config configs/nuscenes_centerfusion.yaml --split val
```

训练：

```bash
uv run fusion train --config configs/nuscenes_centerfusion.yaml
```

导出预测：

```bash
uv run fusion evaluate --config configs/nuscenes_centerfusion.yaml --checkpoint outputs/centerfusion/best.ckpt
```

## 算法说明

该实现是一个 **CenterFusion 风格** 的实用基线：

1. 图像经过轻量 CNN 编码；
2. 多雷达点云通过 nuScenes 标定信息对齐到参考相机；
3. 雷达点被栅格化为稀疏图像平面特征图；
4. 图像特征与雷达特征通过 gated fusion 融合；
5. 检测头在低分辨率特征图上预测对象中心及多任务回归量；
6. 解码阶段将图像中心 + 深度恢复为 3D 中心，再变换到 ego / global 坐标；
7. 可导出 nuScenes detection JSON。

## 说明

- 默认配置是单相机 `CAM_FRONT`，这样更稳妥，也更容易先跑通。
- 如果希望改成多相机，只需修改 [configs/nuscenes_centerfusion.yaml](configs/nuscenes_centerfusion.yaml) 中的 `camera_channels`。
- 当前实现更偏向 **完整训练/推理基线**，不是 leaderboard 级别复现。
- 如果你希望，我下一步可以继续补：
  - 多相机结果合并与 NMS
  - 更强的 backbone/FPN
  - 可视化脚本
  - Windows 下的数据缓存与加速
