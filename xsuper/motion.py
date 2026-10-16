"""Limited animation: camera moves over held drawings, lip flaps and blinks.

TV anime gets most of its runtime from a few tricks: pans/zooms over a single drawing,
3-position mouth flaps during dialogue, blinks, and characters animated "on twos".
Smooth digital camera moves on top of on-twos drawings is exactly the modern TV look.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

OVERSCAN = 1.12

# zoom start/end, horizontal position start/end, vertical position start/end (0..1 of travel)
CAMERA_PATHS = {
    "hold":       (1.0, 1.015, 0.5, 0.5, 0.5, 0.5),
    "drift":      (1.0, 1.05, 0.42, 0.58, 0.5, 0.5),
    "pan_left":   (1.0, 1.0, 1.0, 0.0, 0.5, 0.5),
    "pan_right":  (1.0, 1.0, 0.0, 1.0, 0.5, 0.5),
    "tilt_up":    (1.0, 1.0, 0.5, 0.5, 1.0, 0.0),
    "tilt_down":  (1.0, 1.0, 0.5, 0.5, 0.0, 1.0),
    "push_in":    (1.0, 1.25, 0.5, None, 0.5, None),
    "pull_out":   (1.28, 1.0, None, 0.5, None, 0.5),
    "zoom_punch": (1.0, 1.32, 0.5, None, 0.5, None),
    "shake":      (1.04, 1.04, 0.5, 0.5, 0.5, 0.5),
}


def pick_camera(camera: str, kind: str, framing: str, fx: list[str], index: int) -> str:
    """Choose a camera move when the script says 'auto' - mimics common anime shot grammar."""
    if camera != "auto":
        return camera
    if "impact" in fx or "shake" in fx:
        return "zoom_punch" if index % 2 else "shake"
    if kind == "ai":
        return "hold"
    if framing == "wide":
        return ("pan_right", "pull_out", "pan_left")[index % 3]
    if framing in ("closeup", "extreme_closeup"):
        return "hold" if kind == "dialogue" else "push_in"
    return ("drift", "hold", "push_in")[index % 3]


def generation_size(camera: str) -> tuple[int, int]:
    """SDXL-friendly canvas: wide for pans, tall for tilts, 16:9-ish otherwise."""
    if camera in ("pan_left", "pan_right"):
        return 1536, 640
    if camera in ("tilt_up", "tilt_down"):
        return 896, 1152
    return 1344, 768


def prescale(img: np.ndarray, out_w: int, out_h: int, overscan: float = OVERSCAN) -> np.ndarray:
    """Resize once so the image covers the output frame with some room for camera moves."""
    h, w = img.shape[:2]
    s = max(out_w * overscan / w, out_h * overscan / h)
    size = (max(out_w, round(w * s)), max(out_h, round(h * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LANCZOS4
    return cv2.resize(img, size, interpolation=interp)


def _ease(t: float) -> float:
    return t * t * (3 - 2 * t)


def _ease_out(t: float) -> float:
    return 1 - (1 - t) ** 3


def camera_matrices(camera: str, n: int, src_w: int, src_h: int, out_w: int, out_h: int,
                    focus: tuple[float, float] | None = None) -> list[np.ndarray]:
    """2x3 affine matrices mapping the (prescaled) source onto each output frame."""
    z0, z1, px0, px1, py0, py1 = CAMERA_PATHS.get(camera, CAMERA_PATHS["hold"])
    base = max(out_w / src_w, out_h / src_h)

    def travel_pos(p, z, axis):
        """Pixel centre for travel fraction p (0..1) at zoom z along axis 0=x 1=y."""
        size_src, size_out = (src_w, out_w) if axis == 0 else (src_h, out_h)
        half = size_out / (base * z) / 2
        lo, hi = half, size_src - half
        return lo + p * (hi - lo) if hi > lo else size_src / 2

    def focus_frac(z, axis):
        if focus is None:
            return 0.5
        size_src, size_out = (src_w, out_w) if axis == 0 else (src_h, out_h)
        half = size_out / (base * z) / 2
        lo, hi = half, size_src - half
        return float(np.clip((focus[axis] - lo) / (hi - lo), 0, 1)) if hi > lo else 0.5

    if px1 is None:
        px1, py1 = focus_frac(z1, 0), focus_frac(z1, 1)
    if px0 is None:
        px0, py0 = focus_frac(z0, 0), focus_frac(z0, 1)

    mats = []
    punch = max(2, min(6, n // 3))
    for i in range(n):
        t = i / max(1, n - 1)
        if camera == "zoom_punch":
            k = _ease_out(min(1.0, i / punch))
            z = z0 + (z1 - z0) * k + 0.03 * t
        else:
            k = _ease(t)
            z = z0 + (z1 - z0) * k
        cx = travel_pos(px0 + (px1 - px0) * k, z, 0)
        cy = travel_pos(py0 + (py1 - py0) * k, z, 1)
        s = base * z
        mats.append(np.array([[s, 0, out_w / 2 - s * cx], [0, s, out_h / 2 - s * cy]], np.float32))
    return mats


def warp(src: np.ndarray, m: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    return cv2.warpAffine(src, m, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def mouth_track(envelope: np.ndarray, rng: np.random.Generator, step: int = 2,
                threshold: float = 0.22) -> np.ndarray:
    """Open/closed mouth per frame. Updates on twos and alternates like hand-timed flaps."""
    n = len(envelope)
    out = np.zeros(n, bool)
    state, held = False, 0
    for i in range(0, n, step):
        loud = float(envelope[i:i + step].max()) > threshold
        if not loud:
            state, held = False, 0
        elif state:
            held += 1
            if held >= rng.integers(1, 3):  # open for 1-2 beats then snap shut
                state, held = False, 0
        else:
            state, held = True, 0
        out[i:i + step] = state
    return out


def blink_track(n: int, fps: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros(n, bool)
    t = int(fps * rng.uniform(0.4, 1.8))
    while t < n:
        out[t:t + 3] = True
        t += int(fps * rng.uniform(2.2, 4.8))
    return out


def on_twos(n_out: int, n_src: int, step: int = 2) -> list[int]:
    """Map output frames to source frames, holding each drawing for `step` frames."""
    return [min(n_src - 1, (i // step) * step) for i in range(n_out)]


def ease_alpha(i: int, length: int) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * min(1.0, i / max(1, length)))
