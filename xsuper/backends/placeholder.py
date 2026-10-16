"""Animatic backend: draws simple storyboard stand-ins instantly, no GPU needed.

Real studios cut an animatic (moving storyboard) before drawing anything final. This lets
you check timing, dialogue, camera moves and pacing of a full episode in minutes, then
swap in the real art by rendering again in 'final' mode.
"""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageDraw

from ..textutil import font
from .base import Backend, ImageRequest, Keyframe

SKIES = {
    "day": [(110, 180, 235), (205, 235, 255)],
    "morning": [(150, 190, 230), (250, 225, 190)],
    "sunset": [(45, 30, 80), (240, 130, 70), (255, 210, 140)],
    "dusk": [(30, 25, 70), (120, 70, 130)],
    "night": [(5, 8, 20), (30, 40, 75)],
    "rain": [(70, 80, 92), (125, 135, 148)],
    "snow": [(170, 180, 195), (225, 230, 240)],
    "indoor": [(70, 55, 50), (120, 100, 85)],
    "indoor_night": [(25, 20, 25), (60, 45, 40)],
}

HEAD_RADIUS = {  # fraction of image height
    "extreme_closeup": 0.44, "closeup": 0.28, "over_shoulder": 0.2, "pov": 0.16,
    "medium": 0.13, "cowboy": 0.10, "full": 0.07, "wide": 0.035,
}
HEAD_Y = {
    "extreme_closeup": 0.5, "closeup": 0.48, "over_shoulder": 0.42, "pov": 0.42,
    "medium": 0.36, "cowboy": 0.3, "full": 0.25, "wide": 0.62,
}
SKIN = (250, 222, 200)


def _gradient(w: int, h: int, stops: list[tuple[int, int, int]]) -> Image.Image:
    t = np.linspace(0, 1, h)[:, None]
    seg = np.clip(t * (len(stops) - 1), 0, len(stops) - 1 - 1e-6)
    i = seg.astype(int)
    f = seg - i
    a = np.array(stops, dtype=np.float32)
    col = a[i[:, 0]] * (1 - f) + a[i[:, 0] + 1] * f
    arr = np.repeat(col[:, None, :], w, axis=1)
    return Image.fromarray(arr.astype(np.uint8))


class AnimaticBackend(Backend):
    name = "animatic-v1"

    def __init__(self, video_frames: int = 49, video_size: tuple[int, int] = (832, 480)):
        self.video_frames = video_frames
        self.video_size = video_size

    def txt2img(self, req: ImageRequest) -> Keyframe:
        w, h = req.width, req.height
        hint = req.hint
        rng = random.Random(req.seed)
        time = hint.get("time", "day")
        img = _gradient(w, h, SKIES.get(time, SKIES["day"]))
        d = ImageDraw.Draw(img, "RGBA")

        # scenery silhouettes, deterministic per location so a place always looks the same
        loc_rng = random.Random(str(hint.get("location")))
        horizon = int(h * 0.62)
        dark = tuple(int(c * 0.45) for c in SKIES.get(time, SKIES["day"])[0])
        if time.startswith("indoor"):
            d.rectangle([0, horizon, w, h], fill=(*dark, 255))
            for _ in range(3):
                x = loc_rng.randint(0, w)
                d.rectangle([x, int(h * 0.15), x + w // 8, int(h * 0.45)], fill=(255, 240, 200, 60))
        else:
            x = 0
            while x < w:
                bw = loc_rng.randint(w // 20, w // 7)
                bh = loc_rng.randint(h // 10, h // 2)
                d.rectangle([x, horizon - bh, x + bw, h], fill=(*dark, 255))
                if time in ("night", "dusk"):
                    for _ in range(6):
                        wx, wy = loc_rng.randint(x, x + bw), loc_rng.randint(horizon - bh, horizon)
                        d.rectangle([wx, wy, wx + 6, wy + 9], fill=(255, 220, 140, 200))
                x += bw + loc_rng.randint(0, w // 40)
            d.rectangle([0, horizon, w, h], fill=(*tuple(int(c * 0.3) for c in dark), 255))

        # characters
        faces = []
        chars = hint.get("characters", [])
        framing = hint.get("framing", "medium")
        r = int(h * HEAD_RADIUS.get(framing, 0.13))
        cy = int(h * HEAD_Y.get(framing, 0.36))
        for i, ch in enumerate(chars):
            cx = int(w * (i + 1) / (len(chars) + 1)) + rng.randint(-w // 40, w // 40)
            color = tuple(ch.get("color", (200, 80, 80)))
            outfit = tuple(ch.get("outfit_color", (40, 40, 50)))
            # body
            d.rounded_rectangle([cx - 1.3 * r, cy + 0.85 * r, cx + 1.3 * r, cy + 0.85 * r + 6 * r],
                                radius=int(r * 0.6), fill=(*outfit, 255), outline=(0, 0, 0, 255), width=max(2, r // 25))
            # head + hair
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*SKIN, 255), outline=(0, 0, 0, 255), width=max(2, r // 25))
            d.chord([cx - 1.08 * r, cy - 1.12 * r, cx + 1.08 * r, cy + 0.2 * r], 180, 360, fill=(*color, 255))
            d.polygon([(cx - 1.05 * r, cy - 0.4 * r), (cx - 0.7 * r, cy - 0.27 * r), (cx - 0.35 * r, cy - 0.45 * r),
                       (cx, cy - 0.27 * r), (cx + 0.35 * r, cy - 0.45 * r), (cx + 0.7 * r, cy - 0.27 * r),
                       (cx + 1.05 * r, cy - 0.4 * r)], fill=(*color, 255))
            # eyes + mouth (positions match faces.py regions so flaps/blinks line up)
            for sx in (-1, 1):
                ex = cx + sx * 0.38 * r
                d.ellipse([ex - 0.13 * r, cy - 0.12 * r, ex + 0.13 * r, cy + 0.22 * r], fill=(25, 20, 30, 255))
                d.ellipse([ex - 0.06 * r, cy - 0.07 * r, ex + 0.02 * r, cy + 0.03 * r], fill=(255, 255, 255, 255))
            d.line([cx - 0.17 * r, cy + 0.55 * r, cx + 0.17 * r, cy + 0.55 * r], fill=(90, 30, 30, 255), width=max(2, r // 18))
            if r > 12:
                d.text((cx, cy - 1.25 * r), ch.get("name", "?"), font=font(max(14, int(r * 0.28))),
                       fill=(255, 255, 255, 255), anchor="ms", stroke_width=3, stroke_fill=(0, 0, 0, 255))
            faces.append((int(cx - r), int(cy - r), int(2 * r), int(2 * r)))

        # the user's sketch, so the animatic shows their drawing
        if req.control_image is not None:
            sk = req.control_image.convert("L").resize((w, h))
            ink = Image.new("RGBA", (w, h), (20, 20, 30, 255))
            img.paste(ink, (0, 0), sk.point(lambda v: 210 if v > 100 else 0))

        return Keyframe(img, faces)

    def inpaint(self, image: Image.Image, mask: Image.Image, req: ImageRequest, denoise: float) -> Image.Image:
        out = image.copy()
        box = mask.convert("L").point(lambda v: 255 if v > 128 else 0).getbbox()
        if not box:
            return out
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        d = ImageDraw.Draw(out)
        prompt = req.positive.lower()
        if prompt.startswith("open mouth"):
            d.ellipse([cx - 0.22 * bw, cy - 0.3 * bh, cx + 0.22 * bw, cy + 0.32 * bh], fill=(120, 25, 35), outline=(40, 0, 0), width=2)
            d.chord([cx - 0.16 * bw, cy + 0.02 * bh, cx + 0.16 * bw, cy + 0.32 * bh], 0, 180, fill=(230, 110, 120))
        elif prompt.startswith("closed eyes"):
            skin = image.getpixel((int(cx), int(cy)))
            r = bw / 1.6 / (1 + 2 * 0.2)  # head radius (mask = eyes box grown by 20%)
            for sx in (-1, 1):
                ex = cx + sx * 0.38 * r
                d.ellipse([ex - 0.16 * r, cy - 0.2 * r, ex + 0.16 * r, cy + 0.3 * r], fill=skin)
                d.arc([ex - 0.15 * r, cy - 0.05 * r, ex + 0.15 * r, cy + 0.2 * r], 20, 160,
                      fill=(25, 20, 30), width=max(2, int(r * 0.05)))
        return out

    def img2video(self, image: Image.Image, prompt: str, negative: str, seed: int) -> list[Image.Image]:
        w, h = self.video_size
        base = image.convert("RGB").resize((int(w * 1.12), int(h * 1.12)), Image.LANCZOS)
        frames = []
        for i in range(self.video_frames):
            t = i / max(1, self.video_frames - 1)
            x = int((base.width - w) * t)
            frame = base.crop((x, (base.height - h) // 2, x + w, (base.height - h) // 2 + h))
            d = ImageDraw.Draw(frame)
            d.text((w * 0.97, h * 0.04), "AI MOTION", font=font(18), fill=(255, 240, 120), anchor="ra",
                   stroke_width=2, stroke_fill=(0, 0, 0))
            frames.append(frame)
        return frames
