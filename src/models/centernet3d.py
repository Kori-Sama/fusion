import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- DLA-34 核心组件 ---

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)
        return x

class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels, stride, root_dim=0, dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels, root_dim=root_dim + out_channels, dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        dest = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        if self.levels == 1:
            x1 = self.tree1(x, dest)
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            x1 = self.tree1(x, dest)
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x

class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000, block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2, level_root=False, root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2, level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2, level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2, level_root=True, root_residual=residual_root)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.append(nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride if i == 0 else 1, padding=dilation, bias=False, dilation=dilation))
            modules.append(nn.BatchNorm2d(planes))
            modules.append(nn.ReLU(inplace=True))
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

# --- DLAUp: 实现下采样后的特征图上采样融合 (1/4 分辨率输出) ---

class IDAUp(nn.Module):
    def __init__(self, node_kernel, out_dim, channels):
        super(IDAUp, self).__init__()
        for i, c in enumerate(channels):
            if c == out_dim:
                proj = nn.Identity()
            else:
                proj = nn.Sequential(nn.Conv2d(c, out_dim, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True))
            f = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=node_kernel, stride=1, padding=(node_kernel - 1) // 2, bias=False),
                              nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True))
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'node_' + str(i), f)
            if i > 0:
                up = nn.ConvTranspose2d(out_dim, out_dim, 4, stride=2, padding=1, groups=out_dim, bias=False)
                setattr(self, 'up_' + str(i), up)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(project(layers[i]) + upsample(layers[i - 1]))

class DLAUp(nn.Module):
    def __init__(self, channels, scales=(1, 2, 4, 8, 16)):
        super(DLAUp, self).__init__()
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i), IDAUp(3, channels[j], channels[j:]))
            scales[j + 1:] = scales[j + 1:] // 2
            channels[j + 1:] = [channels[j] for _ in channels[j+1:]]

    def forward(self, layers):
        out = [layers[-1]] # 最后一层作为起点
        for i in range(len(layers) - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out[0]

# --- 完整的 CenterFusion 模型 (带 DLA-34) ---

import numpy as np

class CenterFusion(nn.Module):
    def __init__(self, num_classes=10, head_conv=256):
        super(CenterFusion, self).__init__()
        
        # 1. 加载 DLA-34 主干网络 (1/4 下采样输出)
        # channels = [16, 32, 64, 128, 256, 512]
        self.base = DLA([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512], block=BasicBlock)
        
        # 2. DLA-Up 用于聚合多尺度特征到 1/4 分辨率 (最终输出通道 64)
        self.dla_up = DLAUp(channels=[64, 128, 256, 512], scales=(1, 2, 4, 8))
        self.out_channels = 64
        
        # 3. 雷达特征处理 (针对 3 通道: depth, vx, vy)
        self.radar_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1)
        )
        
        # 4. 融合后的总通道数 (64 视觉 + 64 雷达)
        self.combined_channels = 64 + 64
        
        # 5. 检测头
        self.heads = {
            'hm': num_classes, 'reg': 2, 'wh': 2, 'dep': 1, 
            'dim': 3, 'rot': 8, 'vel': 2, 'radar_dep': 1
        }
        
        for head in self.heads:
            classes = self.heads[head]
            out_conv = nn.Sequential(
                nn.Conv2d(self.combined_channels, head_conv, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, classes, kernel_size=1)
            )
            if 'hm' in head:
                out_conv[-1].bias.data.fill_(-2.19)
            self.add_module(head, out_conv)

    def forward(self, img, radar_hm):
        # 1. DLA 主干网络前向传播 (得到不同阶段的特征图)
        # y[0]=1, y[1]=1/2, y[2]=1/4, y[3]=1/8, y[4]=1/16, y[5]=1/32
        y = self.base(img)
        
        # 2. 上采样融合至 1/4 分辨率 (即 y[2] 所在的尺度)
        # 我们只取 level2 到 level5 的特征进行聚合
        img_feat = self.dla_up(y[2:]) # 此时输出通道为 64, 分辨率为 H/4, W/4
        
        # 3. 提取雷达特征
        radar_feat = self.radar_conv(radar_hm)
            
        # 4. 融合
        combined_feat = torch.cat([img_feat, radar_feat], dim=1)
        
        # 5. 各检测头输出
        z = {}
        for head in self.heads:
            z[head] = getattr(self, head)(combined_feat)
            
        return z

def get_model(num_classes=10):
    return CenterFusion(num_classes=num_classes)
