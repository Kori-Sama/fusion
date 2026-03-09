from __future__ import annotations

import math

import numpy as np


def gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> int:
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(0.0, b1**2 - 4 * a1 * c1))
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(max(0.0, b2**2 - 4 * a2 * c2))
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(max(0.0, b3**2 - 4 * a3 * c3))
    r3 = (b3 + sq3) / (2 * a3)

    return max(0, int(min(r1, r2, r3)))


def gaussian2d(shape: tuple[int, int], sigma: float = 1.0) -> np.ndarray:
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    gaussian = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    gaussian[gaussian < np.finfo(gaussian.dtype).eps * gaussian.max()] = 0
    return gaussian


def draw_gaussian(heatmap: np.ndarray, center: tuple[int, int], radius: int) -> None:
    diameter = 2 * radius + 1
    gaussian = gaussian2d((diameter, diameter), sigma=max(diameter / 6.0, 1e-6))
    x, y = center
    height, width = heatmap.shape[-2:]

    left = min(x, radius)
    right = min(width - x, radius + 1)
    top = min(y, radius)
    bottom = min(height - y, radius + 1)
    if left < 0 or top < 0 or right <= 0 or bottom <= 0:
        return

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]
    np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
