"""Anime face detection and face-region masks (for mouth flaps and blinks).

Uses nagadomi's lbpcascade_animeface (MIT) - a tiny OpenCV cascade trained on anime faces.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageFilter

from .backends.base import Box

CASCADE_URL = "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml"

_cascade = None
_cascade_error: str | None = None


def cache_dir() -> Path:
    path = Path(os.environ.get("XSUPER_CACHE") or Path.home() / ".cache" / "xsuper")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_cascade():
    global _cascade, _cascade_error
    if _cascade is not None or _cascade_error:
        return _cascade
    path = cache_dir() / "lbpcascade_animeface.xml"
    try:
        if not path.exists():
            r = requests.get(CASCADE_URL, timeout=30)
            r.raise_for_status()
            path.write_bytes(r.content)
        cascade = cv2.CascadeClassifier(str(path))
        if cascade.empty():
            raise RuntimeError("cascade file is empty or corrupt")
        _cascade = cascade
    except Exception as exc:  # noqa: BLE001 - detection is optional
        _cascade_error = f"anime face detector unavailable ({exc}); lip flaps/blinks need 'face:' set manually"
    return _cascade


def detector_problem() -> str | None:
    _get_cascade()
    return _cascade_error


def detect_faces(img: Image.Image) -> list[Box]:
    cascade = _get_cascade()
    if cascade is None:
        return []
    gray = cv2.equalizeHist(cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY))
    min_side = max(24, min(img.size) // 16)
    found = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(min_side, min_side))
    boxes = [tuple(int(v) for v in f) for f in found]
    return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)


def eyes_box(face: Box) -> Box:
    x, y, w, h = face
    return (int(x + 0.1 * w), int(y + 0.36 * h), int(0.8 * w), int(0.28 * h))


def mouth_box(face: Box) -> Box:
    x, y, w, h = face
    return (int(x + 0.3 * w), int(y + 0.66 * h), int(0.4 * w), int(0.26 * h))


def crop_around(face: Box, size: tuple[int, int], factor: float = 1.8) -> tuple[int, int, int, int]:
    """Square-ish crop around a face, clipped to the image (x0, y0, x1, y1)."""
    x, y, w, h = face
    cx, cy = x + w / 2, y + h / 2
    half = max(w, h) * factor / 2
    W, H = size
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(W, cx + half)), int(min(H, cy + half))
    return x0, y0, x1, y1


def box_mask(size: tuple[int, int], box: Box, feather: float = 0.0, grow: float = 0.0) -> Image.Image:
    x, y, w, h = box
    gx, gy = w * grow, h * grow
    mask = Image.new("L", size, 0)
    from PIL import ImageDraw
    ImageDraw.Draw(mask).ellipse([x - gx, y - gy, x + w + gx, y + h + gy], fill=255)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    return mask
