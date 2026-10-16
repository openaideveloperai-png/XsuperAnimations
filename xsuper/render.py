"""The production pipeline: script -> voices -> keyframes -> lip-flap/blink frames ->
sakuga clips -> finished shots -> episode.

Everything is cached per shot. Change one line of dialogue and only that shot is redone;
'reroll' a shot to get a new drawing of it while the rest of the episode stays identical.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import ENGINE_VERSION, audio as A, fx, motion
from .backends import ImageRequest, make_backend
from .faces import box_mask, crop_around, detect_faces, detector_problem, eyes_box, mouth_box
from .project import Episode, Shot
from .prompts import WAN_NEGATIVE, region_prompt, shot_negative, shot_prompt, video_prompt
from .settings import StudioSettings
from .sketch import prepare_sketch
from .styles import get_style
from .textutil import font, wrap
from .video import VideoWriter, conform_clip, count_frames, mux, require_ffmpeg
from .video import configure as configure_encoder

LogFn = Callable[[str], None]
ProgressFn = Callable[[float, str], None]

DIALOGUE_LEAD = 0.15   # seconds of silence before a line starts
DIALOGUE_TAIL = 0.4


def h(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


@lru_cache(maxsize=256)
def _file_hash(path: str, mtime: float, size: int) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:16]


def file_hash(path: Path | None) -> str:
    if not path:
        return "-"
    st = os.stat(path)
    return _file_hash(str(path), st.st_mtime, st.st_size)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "episode"


class Cache:
    def __init__(self, path: Path):
        self.path = path
        try:
            self.data = json.loads(path.read_text())
        except (OSError, ValueError):
            self.data = {}

    def ok(self, key: str, value: str, *paths: Path) -> bool:
        return self.data.get(key) == value and all(Path(p).exists() for p in paths)

    def set(self, key: str, value: str) -> None:
        self.data[key] = value
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        os.replace(tmp, self.path)


@dataclass
class KeyframeInfo:
    path: Path
    key: str
    faces: list
    size: tuple[int, int]


@dataclass
class ShotResult:
    shot: Shot
    path: Path
    frames: int
    sounds: list[tuple[float, np.ndarray, float, str]] = field(default_factory=list)  # offset, audio, gain, track
    subtitles: list[tuple[float, float, str]] = field(default_factory=list)          # offset, duration, text


class Studio:
    def __init__(self, ep: Episode, settings: StudioSettings, mode: str = "animatic",
                 log: LogFn = print, progress: ProgressFn | None = None):
        require_ffmpeg()
        configure_encoder(settings.encode_preset, settings.encode_crf)
        self.ep, self.s, self.mode = ep, settings, mode
        self.style = get_style(ep.style)
        self.backend = make_backend(mode, settings)
        self.draft = mode == "draft"
        self.log = log
        self.progress = progress or (lambda frac, msg: None)
        self.dir = ep.root / "build" / mode
        for sub in ("voices", "keyframes", "clips", "shots"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.cache = Cache(self.dir / "cache.json")
        self.rerolls_path = self.dir / "rerolls.json"
        try:
            self.rerolls: dict[str, int] = json.loads(self.rerolls_path.read_text())
        except (OSError, ValueError):
            self.rerolls = {}
        self._voices = self._assign_voices()
        self._finish = None

    # ------------------------------------------------------------------ helpers
    def _assign_voices(self) -> dict[str, str]:
        counters: dict[str, int] = {}
        out = {}
        for key, ch in self.ep.characters.items():
            if ch.voice:
                out[key] = ch.voice
                continue
            group = "girl" if ch.kind in ("girl", "woman", "female") else "boy"
            out[key] = A.default_voice(self.ep.language, ch.kind, counters.get(group, 0))
            counters[group] = counters.get(group, 0) + 1
        return out

    def _seed(self, shot: Shot) -> int:
        return (shot.seed + 7919 * self.rerolls.get(shot.id, 0)) % (2 ** 31)

    def _loras(self, shot: Shot) -> list[tuple[str, float]]:
        chars = [self.ep.characters[k] for k in shot.characters]
        loras = [(c.lora, c.lora_strength) for c in chars if c.lora]
        if len(loras) > 1:  # stacked character LoRAs bleed into each other; soften them
            loras = [(n, s * 0.75) for n, s in loras]
        return loras

    def _hint(self, shot: Shot) -> dict:
        chars = []
        for key in shot.characters:
            ch = self.ep.characters[key]
            outfit = shot.outfits.get(key, "default")
            oc = int(h(key, outfit), 16)
            chars.append({"name": ch.name, "color": ch.color,
                          "outfit_color": (40 + oc % 60, 40 + (oc >> 8) % 60, 50 + (oc >> 16) % 70)})
        return {"characters": chars, "time": shot.time, "location": shot.location, "framing": shot.framing,
                "action": shot.action or shot.expression or "", "shot_id": shot.id, "line": shot.line}

    def _frames_for(self, shot: Shot, voice_len: float) -> int:
        fps = self.ep.fps
        if voice_len:
            secs = max(shot.seconds or 0, voice_len + DIALOGUE_LEAD + DIALOGUE_TAIL)
        else:
            secs = shot.estimate_seconds(self.ep.language)
        return max(2, int(round(secs * fps)))

    # ------------------------------------------------------------------ voices
    def voice(self, shot: Shot) -> tuple[np.ndarray, str]:
        ch = self.ep.characters[shot.speaker]
        voice_id = self._voices[ch.key]
        speed = self.s.tts.speed * ch.voice_speed
        key = h("voice", shot.line, voice_id, speed, self.s.tts.engine, self.ep.language)
        path = self.dir / "voices" / f"{shot.id}.wav"
        if self.cache.ok(f"voice:{shot.id}", key, path):
            return A.read_wav(path), key
        self.log(f"[voice] {shot.id}: {ch.name}: {shot.line}")
        clip = A.synthesize(shot.line, voice_id, speed, self.s.tts.engine, self.ep.language, self.log)
        A.write_wav(path, clip)
        self.cache.set(f"voice:{shot.id}", key)
        return clip, key

    # ------------------------------------------------------------------ keyframes
    def keyframe(self, shot: Shot) -> KeyframeInfo:
        camera = motion.pick_camera(shot.camera, shot.kind, shot.framing, shot.fx, shot.index)
        w, h_ = motion.generation_size(camera)
        positive = shot_prompt(self.ep, shot, self.style)
        negative = shot_negative(self.style)
        seed = self._seed(shot)
        loras = self._loras(shot)
        key = h("kf", self.backend.identity(), positive, negative, seed, w, h_, loras, file_hash(shot.sketch),
                shot.sketch_strength, self.draft, shot.face)
        path = self.dir / "keyframes" / f"{shot.id}.png"
        meta_path = path.with_suffix(".json")
        if self.cache.ok(f"kf:{shot.id}", key, path, meta_path):
            meta = json.loads(meta_path.read_text())
            return KeyframeInfo(path, key, [tuple(f) for f in meta["faces"]], tuple(meta["size"]))

        self.log(f"[draw] {shot.id}  ({w}x{h_}, seed {seed})")
        control = prepare_sketch(shot.sketch, w, h_) if shot.sketch else None
        req = ImageRequest(positive=positive, negative=negative, width=w, height=h_, seed=seed, loras=loras,
                           control_image=control, control_strength=shot.sketch_strength,
                           hires=False if self.draft else None, hint=self._hint(shot))
        kf = self.backend.txt2img(req)
        img = kf.image.convert("RGB")
        if shot.face:
            x, y, fw, fh = shot.face
            faces = [(int(x * img.width), int(y * img.height), int(fw * img.width), int(fh * img.height))]
        elif kf.faces is not None:
            sx, sy = img.width / w, img.height / h_
            faces = [(int(a * sx), int(b * sy), int(c * sx), int(d * sy)) for a, b, c, d in kf.faces]
        else:
            faces = detect_faces(img) if shot.characters else []
        img.save(path)
        meta_path.write_text(json.dumps({"faces": faces, "size": img.size, "seed": seed, "camera": camera,
                                         "positive": positive, "negative": negative}, indent=1))
        self.cache.set(f"kf:{shot.id}", key)
        return KeyframeInfo(path, key, faces, img.size)

    def _paint_region(self, shot: Shot, kf: KeyframeInfo, face, which: str) -> Image.Image:
        """Repaint just the mouth (open) or eyes (closed) of a keyframe, then blend it back."""
        base = Image.open(kf.path).convert("RGB")
        region = mouth_box(face) if which == "mouth" else eyes_box(face)
        x0, y0, x1, y1 = crop_around(face, base.size, 1.8)
        crop = base.crop((x0, y0, x1, y1))
        scale = 1024 / max(crop.size)
        cw = max(512, int(round(crop.width * scale / 64)) * 64)
        ch = max(512, int(round(crop.height * scale / 64)) * 64)
        sx, sy = cw / crop.width, ch / crop.height
        work = crop.resize((cw, ch), Image.LANCZOS)
        rx, ry, rw, rh = region
        local = (int((rx - x0) * sx), int((ry - y0) * sy), int(rw * sx), int(rh * sy))
        mask = box_mask((cw, ch), local, grow=0.2)
        change = "open mouth" if which == "mouth" else "closed eyes"
        avoid = "closed mouth" if which == "mouth" else "open eyes"
        speaker = shot.speaker if which == "mouth" else (shot.speaker or shot.characters[0])
        if speaker not in shot.characters:
            speaker = shot.characters[0]
        req = ImageRequest(positive=region_prompt(self.ep, shot, self.style, speaker, change),
                           negative=shot_negative(self.style, avoid), width=cw, height=ch,
                           seed=self._seed(shot) + (101 if which == "mouth" else 202),
                           loras=self._loras(shot), hint=self._hint(shot))
        painted = self.backend.inpaint(work, mask, req, self.s.image.inpaint_denoise).resize(crop.size, Image.LANCZOS)
        feather = max(2.0, rw * 0.08)
        blend = box_mask(crop.size, (rx - x0, ry - y0, rw, rh), feather=feather, grow=0.22)
        out = base.copy()
        out.paste(painted, (x0, y0), blend)
        return out

    def variants(self, shot: Shot, kf: KeyframeInfo) -> dict:
        if not kf.faces or not shot.characters:
            return {}
        face = kf.faces[0]
        frac = face[2] / kf.size[0]
        wants = []
        if shot.kind == "dialogue" and shot.speaker in shot.characters and frac >= 0.06:
            wants.append("mouth")
        if shot.kind in ("dialogue", "still") and frac >= 0.09:
            wants.append("blink")
        out: dict = {"face": face}
        for which in wants:
            key = h("var", kf.key, which, face, self.s.image.inpaint_denoise, self.backend.identity())
            path = self.dir / "keyframes" / f"{shot.id}_{which}.png"
            if not self.cache.ok(f"{which}:{shot.id}", key, path):
                self.log(f"[draw] {shot.id} {which} frame")
                self._paint_region(shot, kf, face, which).save(path)
                self.cache.set(f"{which}:{shot.id}", key)
            out[which] = (path, key)
        return out

    # ------------------------------------------------------------------ sakuga clips
    def clip(self, shot: Shot, kf: KeyframeInfo) -> tuple[list[Path], str] | None:
        if shot.kind != "ai" or self.draft or (self.mode != "animatic" and not self.s.video.enabled):
            return None
        prompt = video_prompt(self.ep, shot, self.style)
        seed = self._seed(shot)
        ident = getattr(self.backend, "video_identity", lambda: self.backend.identity())()
        key = h("clip", ident, kf.key, prompt, seed)
        folder = self.dir / "clips" / shot.id
        listing = folder / "frames.json"
        if self.cache.ok(f"clip:{shot.id}", key, listing):
            names = json.loads(listing.read_text())
            frames = [folder / n for n in names]
            if all(p.exists() for p in frames):
                return frames, key
        self.log(f"[animate] {shot.id}: AI in-betweening (this is the slow part)")
        images = self.backend.img2video(Image.open(kf.path).convert("RGB"), prompt, WAN_NEGATIVE, seed)
        folder.mkdir(parents=True, exist_ok=True)
        for old in [*folder.glob("*.png"), *folder.glob("*.jpg")]:
            old.unlink()
        names = []
        for i, im in enumerate(images):  # high-quality JPEG: a 23-min episode's clips would be GBs as PNG
            names.append(f"{i:04d}.jpg")
            im.convert("RGB").save(folder / names[-1], quality=95, subsampling=0)
        listing.write_text(json.dumps(names))
        self.cache.set(f"clip:{shot.id}", key)
        return [folder / n for n in names], key

    # ------------------------------------------------------------------ shot rendering
    def _finishing(self):
        if self._finish is None:
            W, H = self.ep.width, self.ep.height
            g = self.style.grade
            self._finish = {
                "vignette": fx.vignette_map(W, H, g.vignette),
                "grain": fx.grain_textures(W, H, g.grain, seed=self.ep.seed),
                "rain": None,
            }
        return self._finish

    def _rain(self):
        fin = self._finishing()
        if fin["rain"] is None:
            fin["rain"] = fx.rain_layers(self.ep.width, self.ep.height, seed=self.ep.seed)
        return fin["rain"]

    def _post(self, frame: np.ndarray, i: int, n: int, shot: Shot, fade_in: str, fade_out: str) -> np.ndarray:
        fin = self._finishing()
        fps = self.ep.fps
        if "flash" in shot.fx or fade_in == "flash":
            start = 6 if "impact" in shot.fx else 0   # flash *after* the impact frames
            k = int(fps * 0.25)
            if start <= i < start + k:
                frame = fx.flash(frame, 1 - (i - start) / k)
        if "letterbox" in shot.fx:
            frame = fx.letterbox(frame)
        frame = fx.apply_vignette(frame, fin["vignette"])
        if fin["grain"]:
            frame = fx.apply_grain(frame, fin["grain"][(i // 2) % len(fin["grain"])])
        k = int(fps * 0.4)
        if fade_in == "fade" and i < k:
            frame = fx.fade(frame, motion.ease_alpha(i, k))
        if fade_out == "fade" and i >= n - k:
            frame = fx.fade(frame, motion.ease_alpha(n - 1 - i, k))
        return frame

    def render_shot(self, shot: Shot, fade_in: str, fade_out: str, scene_ref: tuple | None,
                    scene_ref_key: str) -> ShotResult:
        if shot.kind == "card":
            return self._render_card(shot, fade_in, fade_out)
        if shot.kind == "video":
            return self._render_insert(shot)

        W, H, fps = self.ep.width, self.ep.height, self.ep.fps
        voice_clip, voice_key = (self.voice(shot) if shot.line else (np.zeros(0, np.float32), "-"))
        voice_len = len(voice_clip) / A.SR
        n = self._frames_for(shot, voice_len)
        kf = self.keyframe(shot)
        var = self.variants(shot, kf)
        clip = self.clip(shot, kf)
        camera = motion.pick_camera(shot.camera, shot.kind, shot.framing, shot.fx, shot.index)

        sounds, subs = [], []
        if shot.line:
            sounds.append((DIALOGUE_LEAD, voice_clip, 1.0, "voice"))
            subs.append((DIALOGUE_LEAD, max(voice_len, 0.8), shot.line))
        if shot.sfx:
            sounds.append((0.0, A.load_audio(shot.sfx), shot.sfx_volume, "sfx"))

        key = h("shot", ENGINE_VERSION, kf.key, {k: v[1] for k, v in var.items() if k != "face"},
                clip[1] if clip else "-", camera, shot.fx, n, self.ep.style, voice_key, fade_in, fade_out,
                W, H, fps, scene_ref_key, shot.time, self.mode, self.s.encode_preset, self.s.encode_crf)
        path = self.dir / "shots" / f"{shot.id}.mp4"
        if self.cache.ok(f"shot:{shot.id}", key, path):
            return ShotResult(shot, path, n, sounds, subs)

        self.log(f"[film] {shot.id}  {n / fps:.1f}s  camera={camera}" + (f" fx={','.join(shot.fx)}" if shot.fx else ""))
        glow = 0.35 if "glow" in shot.fx else 0.0
        base_img = np.asarray(Image.open(kf.path).convert("RGB"))
        kf_stats = fx.lab_stats(base_img)

        def prep(arr: np.ndarray, match=None, strength=0.0) -> np.ndarray:
            if match is not None and strength > 0:
                arr = fx.match_colors(arr, match, strength)
            return motion.prescale(fx.grade_image(arr, self.style.grade, glow), W, H)

        ref_strength = 0.3 if scene_ref is not None else 0.0
        states = {"": prep(base_img, scene_ref, ref_strength)}
        if "mouth" in var:
            states["m"] = prep(np.asarray(Image.open(var["mouth"][0]).convert("RGB")), scene_ref, ref_strength)
        if "blink" in var:
            states["b"] = prep(np.asarray(Image.open(var["blink"][0]).convert("RGB")), scene_ref, ref_strength)
        if "m" in states and "b" in states:
            fx_, fy_, fw_, fh_ = mouth_box(var["face"])
            scale = states[""].shape[1] / base_img.shape[1]
            mask = np.asarray(box_mask((states[""].shape[1], states[""].shape[0]),
                                       (fx_ * scale, fy_ * scale, fw_ * scale, fh_ * scale), feather=4, grow=0.25),
                              np.float32)[..., None] / 255
            states["mb"] = (states["b"] * (1 - mask) + states["m"] * mask).astype(np.uint8)

        src_h, src_w = states[""].shape[:2]
        scale = src_w / base_img.shape[1]
        focus = None
        if var.get("face") or kf.faces:
            x, y, fw, fh = var.get("face") or kf.faces[0]
            focus = ((x + fw / 2) * scale, (y + fh / 2) * scale)
        mats = motion.camera_matrices(camera, n, src_w, src_h, W, H, focus)

        rng = np.random.default_rng(self._seed(shot))
        shake = np.zeros((n, 2), np.float32)
        if camera == "shake" or "shake" in shot.fx or "impact" in shot.fx:
            start = 6 if "impact" in shot.fx else 0
            shake = fx.shake_offsets(n, fps, H * 0.014, rng, start=start, length_s=0.8 if camera != "shake" else n / fps)

        env = A.envelope(voice_clip, A.SR, fps, n, DIALOGUE_LEAD) if shot.line else np.zeros(n)
        mouth = motion.mouth_track(env, rng) if "m" in states else np.zeros(n, bool)
        blink = motion.blink_track(n, fps, rng) if "b" in states else np.zeros(n, bool)

        clip_paths = clip[0] if clip else None
        clip_index = motion.on_twos(n, len(clip_paths)) if clip_paths else None
        clip_cache: dict[int, np.ndarray] = {}

        def clip_frame(idx: int) -> np.ndarray:
            if idx not in clip_cache:
                if len(clip_cache) > 4:
                    clip_cache.clear()
                arr = np.asarray(Image.open(clip_paths[idx]).convert("RGB"))
                arr = fx.match_colors(arr, kf_stats, 0.6)  # stop AI video drifting off the keyframe palette
                clip_cache[idx] = prep(arr, scene_ref, ref_strength)
            return clip_cache[idx]

        def to_out(pt):
            m = mats[0]
            return (pt[0] * m[0, 0] + m[0, 2], pt[1] * m[1, 1] + m[1, 2])

        center = to_out(focus) if focus else (W / 2, H / 2)
        rain = self._rain() if ("rain" in shot.fx or shot.time == "rain") else None
        first = motion.warp(clip_frame(0) if clip_paths else states[""], mats[0], W, H)
        ink = fx.line_color(first)  # pick once per shot so lines don't flicker between black and white
        caption = self._caption_overlay(shot) if self.mode == "animatic" else None
        masks: dict[str, tuple[int, np.ndarray]] = {}

        def beat_mask(name: str, beat: int, make) -> np.ndarray:
            """Effect drawings change on twos, like hand-drawn effects animation."""
            if name not in masks or masks[name][0] != beat:
                masks[name] = (beat, make(np.random.default_rng(shot.seed + sum(map(ord, name)) + beat)))
            return masks[name][1]

        writer = VideoWriter(path, W, H, fps)
        try:
            for i in range(n):
                if clip_paths:
                    src = clip_frame(clip_index[i])
                else:
                    src = states.get(("m" if mouth[i] else "") + ("b" if blink[i] else ""), states[""])
                m = mats[i].copy()
                m[:, 2] += shake[i]
                frame = motion.warp(src, m, W, H)
                if "speed_lines" in shot.fx:
                    density = self.style.speed_line_density
                    frame = fx.overlay_mask(frame, beat_mask("speed", i // 2, lambda r: fx.speed_lines_mask(
                        W, H, r, center, density)), ink)
                if "streaks" in shot.fx:
                    frame = fx.overlay_mask(frame, beat_mask("streaks", i // 2, lambda r: fx.streaks_mask(W, H, r)), ink)
                if rain is not None:
                    frame = fx.overlay_mask(frame, rain[(i // 2) % len(rain)], (215, 225, 240), 0.55)
                if "impact" in shot.fx and i < 6:
                    frame = fx.impact_frame(frame, i // 2)
                frame = self._post(frame, i, n, shot, fade_in, fade_out)
                if caption is not None:
                    frame = cv2.add(cv2.multiply(frame, caption[1], scale=1 / 255.0), caption[0])
                writer.write(frame)
            writer.close()
        except BaseException:
            writer.abort()
            raise
        self.cache.set(f"shot:{shot.id}", key)
        return ShotResult(shot, path, n, sounds, subs)

    def _caption_overlay(self, shot: Shot) -> tuple[np.ndarray, np.ndarray]:
        """Animatic caption bar (shot id, action, dialogue) as premultiplied colour + inverse alpha."""
        W, H = self.ep.width, self.ep.height
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        text = f"{shot.id}  {shot.action or shot.expression or ''}".strip()
        if shot.line:
            text += f'   {self.ep.characters[shot.speaker].name}: "{shot.line}"'
        fnt = font(max(16, H // 34))
        lines = wrap(d, text, fnt, int(W * 0.92))[:3]
        bar = (len(lines) + 1) * H // 30
        d.rectangle([0, H - bar, W, H], fill=(0, 0, 0, 160))
        for li, t in enumerate(lines):
            d.text((W * 0.04, H - bar + H / 60 + li * H / 30), t, font=fnt, fill=(255, 255, 255, 240))
        d.text((W * 0.02, H * 0.03), "ANIMATIC", font=font(max(14, H // 30)), fill=(255, 255, 255, 110))
        rgba = np.asarray(layer)
        alpha = cv2.merge([rgba[..., 3]] * 3)
        premult = cv2.multiply(np.ascontiguousarray(rgba[..., :3]), alpha, scale=1 / 255.0)
        return premult, cv2.bitwise_not(alpha)

    def _render_card(self, shot: Shot, fade_in: str, fade_out: str) -> ShotResult:
        W, H, fps = self.ep.width, self.ep.height, self.ep.fps
        n = max(2, int(round((shot.seconds or 3.0) * fps)))
        key = h("card", ENGINE_VERSION, shot.card, n, W, H, fade_in, fade_out, self.ep.style)
        path = self.dir / "shots" / f"{shot.id}.mp4"
        sounds = [(0.0, A.load_audio(shot.sfx), shot.sfx_volume, "sfx")] if shot.sfx else []
        if self.cache.ok(f"shot:{shot.id}", key, path):
            return ShotResult(shot, path, n, sounds)
        self.log(f"[card] {shot.id}: {shot.card!r}")
        img = Image.new("RGB", (W, H), (6, 6, 8))
        d = ImageDraw.Draw(img)
        lines = (shot.card or "").split("\n")
        title_font, sub_font = font(H // 11), font(H // 26)
        heights = [H // 9] + [H // 20] * (len(lines) - 1)
        y = H / 2 - sum(heights) / 2
        for i, line in enumerate(lines):
            fnt = title_font if i == 0 else sub_font
            for wrapped in wrap(d, line, fnt, int(W * 0.85)):
                d.text((W / 2, y), wrapped, font=fnt, fill=(240, 236, 228), anchor="mt")
                y += heights[min(i, 1)] * (1.0 if i == 0 else 1.2)
        frame0 = np.asarray(img)
        writer = VideoWriter(path, W, H, fps)
        try:
            k = int(fps * 0.4)
            for i in range(n):
                frame = frame0
                if fade_in != "cut" and i < k:
                    frame = fx.fade(frame, motion.ease_alpha(i, k))
                if fade_out == "fade" and i >= n - k:
                    frame = fx.fade(frame, motion.ease_alpha(n - 1 - i, k))
                writer.write(frame)
            writer.close()
        except BaseException:
            writer.abort()
            raise
        self.cache.set(f"shot:{shot.id}", key)
        return ShotResult(shot, path, n, sounds)

    def _render_insert(self, shot: Shot) -> ShotResult:
        W, H, fps = self.ep.width, self.ep.height, self.ep.fps
        key = h("insert", file_hash(shot.video), W, H, fps)
        path = self.dir / "shots" / f"{shot.id}.mp4"
        wav = self.dir / "voices" / f"{shot.id}_insert.wav"
        if not self.cache.ok(f"shot:{shot.id}", key, path, wav):
            self.log(f"[insert] {shot.id}: {shot.video.name}")
            conform_clip(shot.video, path, W, H, fps)
            A.write_wav(wav, A.load_audio(shot.video))
            self.cache.set(f"shot:{shot.id}", key)
        sounds = [(0.0, A.read_wav(wav), 1.0, "sfx")]
        return ShotResult(shot, path, count_frames(path), sounds)

    # ------------------------------------------------------------------ whole episode
    def _neighbours(self) -> dict[str, tuple[str, str]]:
        shots = self.ep.shots
        out = {}
        for i, s in enumerate(shots):
            nxt = shots[i + 1].transition if i + 1 < len(shots) else "fade"
            out[s.id] = (s.transition if i > 0 else "fade", nxt)
        return out

    def _scene_refs(self, keyframes: dict[str, KeyframeInfo]) -> dict[str, tuple]:
        refs = {}
        for scene in self.ep.scenes:
            for s in scene.shots:
                if s.id in keyframes:
                    kf = keyframes[s.id]
                    refs[scene.id] = (fx.lab_stats(np.asarray(Image.open(kf.path).convert("RGB"))), kf.key, s.id)
                    break
        return refs

    def render(self, only: list[str] | None = None, reroll: list[str] | None = None,
               keyframes_only: bool = False) -> Path:
        for sid in reroll or []:
            self.ep.find_shot(sid)
            self.rerolls[sid] = self.rerolls.get(sid, 0) + 1
        if reroll:
            self.rerolls_path.write_text(json.dumps(self.rerolls, indent=1))
        shots = [self.ep.find_shot(s) for s in only] if only else self.ep.shots
        self.backend.check()
        problem = detector_problem() if self.mode != "animatic" else None
        if problem:
            self.log(f"[warn] {problem}")

        drawable = [s for s in shots if s.kind in ("still", "dialogue", "ai")]
        total = len(drawable) + (0 if keyframes_only else len(shots))
        done = 0

        # Pass 1 - all the GPU work (keyframes, flap/blink frames, sakuga clips).
        keyframes: dict[str, KeyframeInfo] = {}
        for s in drawable:
            self.progress(done / max(1, total), f"drawing {s.id}")
            keyframes[s.id] = self.keyframe(s)
            self.variants(s, keyframes[s.id])
            self.clip(s, keyframes[s.id])
            done += 1
        if only:  # reuse scene reference frames that already exist from earlier full renders
            for scene in self.ep.scenes:
                for s in scene.shots:
                    p = self.dir / "keyframes" / f"{s.id}.png"
                    if s.id not in keyframes and s.kind in ("still", "dialogue", "ai") and p.exists():
                        meta = json.loads(p.with_suffix(".json").read_text())
                        keyframes[s.id] = KeyframeInfo(p, self.cache.data.get(f"kf:{s.id}", "-"),
                                                       meta["faces"], tuple(meta["size"]))
        self.write_review(keyframes)
        if keyframes_only:
            self.progress(1.0, "keyframes done")
            return self.dir / "review.html"

        # Pass 2 - film every shot (CPU), then mix and mux.
        refs = self._scene_refs(keyframes)
        neighbours = self._neighbours()
        results = []
        for s in shots:
            self.progress(done / max(1, total), f"filming {s.id}")
            ref = refs.get(s.scene_id)
            use_ref = ref if ref and ref[2] != s.id else None
            fade_in, fade_out = neighbours[s.id]
            results.append(self.render_shot(s, fade_in, fade_out, use_ref[0] if use_ref else None,
                                            use_ref[1] if use_ref else "-"))
            done += 1
        self.progress(0.99, "mixing sound")
        name = f"{slug(self.ep.title)}_ep{self.ep.number:02d}" if not only else "partial"
        out = self.assemble(results, self.ep.root / "build" / f"{name}_{self.mode}.mp4")
        self.progress(1.0, "done")
        self.log(f"[done] {out}")
        return out

    def assemble(self, results: list[ShotResult], out: Path) -> Path:
        fps = self.ep.fps
        total_frames = sum(r.frames for r in results)
        n = int(total_frames / fps * A.SR) + 1
        voice, sfx, music = (np.zeros(n, np.float32) for _ in range(3))
        srt, t, frame_pos = [], 0.0, 0
        scene_spans: dict[str, list[float]] = {}
        for r in results:
            t = frame_pos / fps
            for off, clip, gain, track in r.sounds:
                A.place(voice if track == "voice" else sfx, clip, t + off, gain)
            for off, dur, text in r.subtitles:
                srt.append((t + off, t + off + dur, text))
            span = scene_spans.setdefault(r.shot.scene_id, [t, t])
            frame_pos += r.frames
            span[1] = frame_pos / fps
        for scene in self.ep.scenes:
            if scene.music and scene.id in scene_spans:
                a, b = scene_spans[scene.id]
                track = A.load_audio(scene.music)
                A.place(music, A.loop_to(track, int((b - a) * A.SR)), a, scene.music_volume)
        mix = voice + sfx * 0.9 + A.duck(music, voice)
        peak = float(np.abs(mix).max()) if mix.size else 0
        if peak > 0.98:
            mix = np.tanh(mix / peak * 1.4) / np.tanh(1.4) * 0.97
        wav = out.with_suffix(".wav")
        A.write_wav(wav, mix)
        srt_path = out.with_suffix(".srt")
        srt_path.write_text("".join(
            f"{i + 1}\n{_ts(a)} --> {_ts(b)}\n{text}\n\n" for i, (a, b, text) in enumerate(srt)), encoding="utf-8")
        try:
            mux([r.path for r in results], wav, srt_path, out)
        finally:
            wav.unlink(missing_ok=True)
        return out

    def write_review(self, keyframes: dict[str, KeyframeInfo]) -> Path:
        """Contact sheet of every keyframe so you can spot bad drawings and reroll them."""
        cards = []
        for s in self.ep.shots:
            kf = keyframes.get(s.id)
            if not kf:
                continue
            rel = os.path.relpath(kf.path, self.dir)
            text = html.escape(f'{self.ep.characters[s.speaker].name}: "{s.line}"' if s.line else s.action or "")
            extras = "".join(f'<img src="{os.path.relpath(self.dir / "keyframes" / f"{s.id}_{w}.png", self.dir)}" '
                             f'title="{w}">' for w in ("mouth", "blink")
                             if (self.dir / "keyframes" / f"{s.id}_{w}.png").exists())
            cards.append(f'<figure><img src="{html.escape(rel)}"><figcaption><b>{s.id}</b> '
                         f'<span>{s.kind}</span><br>{text}</figcaption><div class="v">{extras}</div></figure>')
        page = f"""<!doctype html><meta charset="utf-8"><title>{html.escape(self.ep.title)} - keyframes</title>
<style>body{{font:14px system-ui;background:#111;color:#ddd;margin:16px}}main{{display:grid;
grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}}figure{{margin:0;background:#1b1b1f;
border-radius:8px;overflow:hidden}}figure>img{{width:100%;display:block}}figcaption{{padding:8px}}
span{{color:#999;font-size:12px}}.v img{{width:30%;margin:0 0 6px 8px;border-radius:4px}}</style>
<h1>{html.escape(self.ep.title)} &mdash; keyframes ({self.mode})</h1>
<p>Bad drawing? Reroll it: <code>xsuper render &lt;project&gt; --mode {self.mode} --reroll SHOT_ID</code></p>
<main>{''.join(cards)}</main>"""
        path = self.dir / "review.html"
        path.write_text(page, encoding="utf-8")
        return path


def _ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
