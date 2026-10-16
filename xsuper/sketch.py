"""Turn a rough drawing (any colours, any background) into a clean ControlNet scribble map."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def prepare_sketch(path: str | Path, width: int, height: int) -> Image.Image:
    """White lines on black, fitted (not cropped) into width x height."""
    img = Image.open(path)
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img.convert("RGBA"))
    gray = np.asarray(img.convert("L"), np.uint8)
    if gray.mean() > 127:          # dark lines on light paper -> invert
        gray = 255 - gray
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, lines = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    scale = min(width / lines.shape[1], height / lines.shape[0])
    nw, nh = max(1, int(lines.shape[1] * scale)), max(1, int(lines.shape[0] * scale))
    lines = cv2.resize(lines, (nw, nh), interpolation=cv2.INTER_AREA)
    thick = max(2, min(width, height) // 300)
    lines = cv2.dilate((lines > 64).astype(np.uint8) * 255, np.ones((thick, thick), np.uint8))

    canvas = np.zeros((height, width), np.uint8)
    x, y = (width - nw) // 2, (height - nh) // 2
    canvas[y:y + nh, x:x + nw] = lines
    return Image.fromarray(canvas).convert("RGB")
