"""Compositing effects: colour grade, speed lines, impact frames, shake, flashes, rain, grain.

These are the 'finishing' (撮影 / satsuei) tricks that make TV anime feel like TV anime.
All of them are deterministic for a given seed so re-renders are stable.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .styles import Grade


# ------------------------------------------------------------------ colour
def grade_image(rgb: np.ndarray, g: Grade, bloom_extra: float = 0.0) -> np.ndarray:
    x = rgb.astype(np.float32) / 255.0
    luma = (x @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
    if g.saturation != 1.0:
        x = luma + (x - luma) * g.saturation
    if g.contrast != 1.0:
        x = (x - 0.5) * g.contrast + 0.5
    x = x * np.array(g.gain, np.float32) + np.array(g.lift, np.float32) * (1.0 - x)
    x = x + np.array(g.shadow_tint, np.float32) * (1 - luma) ** 2 + np.array(g.highlight_tint, np.float32) * luma ** 2
    x = np.clip(x, 0, 1)
    if g.gamma != 1.0:
        x = x ** (1.0 / g.gamma)
    bloom = g.bloom + bloom_extra
    if bloom > 0:
        bright = np.clip((luma - 0.62) / 0.38, 0, 1) * x
        sigma = max(rgb.shape[1] / 70.0, 2.0)
        glow = cv2.GaussianBlur(bright, (0, 0), sigma)
        x = 1 - (1 - x) * (1 - np.clip(glow * bloom * 1.6, 0, 1))
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def lab_stats(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    small = cv2.resize(rgb, (256, max(1, int(256 * rgb.shape[0] / rgb.shape[1]))), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    return lab.mean(0), lab.std(0) + 1e-3


def match_colors(rgb: np.ndarray, ref: tuple[np.ndarray, np.ndarray], strength: float) -> np.ndarray:
    """Pull an image's palette toward a reference (keeps every shot in a scene on one palette)."""
    if strength <= 0:
        return rgb
    mean, std = lab_stats(rgb)
    rmean, rstd = ref
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ratio = np.clip(rstd / std, 0.7, 1.4)
    matched = (lab - mean) * ratio + rmean
    out = lab * (1 - strength) + matched * strength
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


# ------------------------------------------------------------------ per-frame finishing
def vignette_map(w: int, h: int, strength: float) -> np.ndarray | None:
    if strength <= 0:
        return None
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = (xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)
    r = np.sqrt(nx ** 2 * 0.8 + ny ** 2 * 1.0)
    v = 1.0 - strength * np.clip((r - 0.55) / 0.9, 0, 1) ** 1.6
    return np.repeat((v * 255).astype(np.uint8)[..., None], 3, axis=2)


def apply_vignette(frame: np.ndarray, vmap: np.ndarray | None) -> np.ndarray:
    return frame if vmap is None else cv2.multiply(frame, vmap, scale=1 / 255.0)


def grain_textures(w: int, h: int, sigma: float, n: int = 4, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    if sigma <= 0:
        return []
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        noise = rng.normal(0, sigma, (h, w, 1)).astype(np.float32)
        noise = np.repeat(noise, 3, axis=2)
        out.append((np.clip(noise, 0, 255).astype(np.uint8), np.clip(-noise, 0, 255).astype(np.uint8)))
    return out


def apply_grain(frame: np.ndarray, tex: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    if tex is None:
        return frame
    return cv2.subtract(cv2.add(frame, tex[0]), tex[1])


def overlay_mask(frame: np.ndarray, mask: np.ndarray, color, opacity: float = 1.0) -> np.ndarray:
    """Paint `color` over frame through an 8-bit mask (fast uint8 path)."""
    a = mask if opacity >= 1 else cv2.convertScaleAbs(mask, alpha=opacity)
    a3 = cv2.merge([a, a, a])
    solid = np.empty_like(frame)
    solid[:] = color
    return cv2.add(cv2.multiply(frame, cv2.bitwise_not(a3), scale=1 / 255.0),
                   cv2.multiply(solid, a3, scale=1 / 255.0))


def line_color(frame: np.ndarray) -> tuple[int, int, int]:
    return (255, 255, 255) if frame[::8, ::8].mean() < 110 else (10, 10, 14)


def speed_lines_mask(w: int, h: int, rng: np.random.Generator, center: tuple[float, float],
                     density: int = 90) -> np.ndarray:
    """Radial focus lines (集中線) converging on the subject."""
    cx, cy = center
    diag = math.hypot(w, h)
    mask = np.zeros((h, w), np.uint8)
    for _ in range(density):
        a = rng.uniform(0, 2 * math.pi)
        spread = rng.uniform(0.002, 0.011)
        r_in = diag * rng.uniform(0.2, 0.38)
        tip = (cx + math.cos(a) * r_in, cy + math.sin(a) * r_in)
        p1 = (cx + math.cos(a - spread) * diag, cy + math.sin(a - spread) * diag)
        p2 = (cx + math.cos(a + spread) * diag, cy + math.sin(a + spread) * diag)
        cv2.fillPoly(mask, [np.array([tip, p1, p2], np.int32)], 255, lineType=cv2.LINE_AA)
    return cv2.convertScaleAbs(mask, alpha=0.85)


def streaks_mask(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """Horizontal motion streaks for fast movement."""
    mask = np.zeros((h, w), np.uint8)
    for _ in range(int(h / 14)):
        y = int(rng.uniform(0, h))
        x0 = int(rng.uniform(-w * 0.2, w * 0.6))
        length = int(rng.uniform(w * 0.2, w * 0.7))
        cv2.line(mask, (x0, y), (x0 + length, y), 255, max(1, int(rng.uniform(1, h / 300 + 2))), cv2.LINE_AA)
    return cv2.convertScaleAbs(cv2.GaussianBlur(mask, (0, 0), 1.2), alpha=0.6)


def impact_frame(frame: np.ndarray, variant: int) -> np.ndarray:
    """High-contrast inverted 'impact frame' flash used on big hits."""
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    thr = float(np.percentile(gray[::4, ::4], 50))
    bright = gray > thr
    edges = cv2.dilate(cv2.Canny(gray, 60, 160), np.ones((3, 3), np.uint8)) > 0
    out = np.empty_like(frame)
    if variant % 3 == 0:     # inverted mono
        out[:] = 255
        out[bright | edges] = 0
    elif variant % 3 == 1:   # red / black
        out[:] = (205, 18, 32)
        out[bright | edges] = 0
    else:                    # mono
        out[:] = 0
        out[bright & ~edges] = 255
    return out


def flash(frame: np.ndarray, alpha: float, color=(255, 255, 255)) -> np.ndarray:
    if alpha <= 0:
        return frame
    return cv2.addWeighted(frame, 1 - alpha, np.full_like(frame, color), alpha, 0)


def fade(frame: np.ndarray, alpha: float) -> np.ndarray:
    return frame if alpha >= 1 else cv2.convertScaleAbs(frame, alpha=max(alpha, 0.0))


def rain_layers(w: int, h: int, n: int = 3, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    layers = []
    for _ in range(n):
        mask = np.zeros((h, w), np.uint8)
        for _ in range(int(w * h / 9000)):
            x, y = int(rng.uniform(-w * 0.1, w)), int(rng.uniform(0, h))
            length = rng.uniform(h * 0.03, h * 0.08)
            cv2.line(mask, (x, y), (int(x + length * 0.25), int(y + length)), int(rng.uniform(90, 200)), 1, cv2.LINE_AA)
        layers.append(mask)
    return layers


def letterbox(frame: np.ndarray, ratio: float = 2.39) -> np.ndarray:
    h, w = frame.shape[:2]
    bar = max(0, int((h - w / ratio) / 2))
    if bar:
        frame = frame.copy()
        frame[:bar] = 0
        frame[h - bar:] = 0
    return frame


def shake_offsets(n: int, fps: int, strength: float, rng: np.random.Generator,
                  start: int = 0, length_s: float = 0.7) -> np.ndarray:
    """Camera shake on 2s (new offset every other frame), decaying."""
    offs = np.zeros((n, 2), np.float32)
    length = max(1, int(length_s * fps))
    current = np.zeros(2, np.float32)
    for i in range(start, min(n, start + length)):
        if (i - start) % 2 == 0:
            decay = 1 - (i - start) / length
            current = rng.uniform(-1, 1, 2).astype(np.float32) * strength * decay
        offs[i] = current
    return offs
