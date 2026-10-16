"""Load and validate episode.yaml - the single source of truth for an episode.

The format is written for humans first: most fields are optional and get sensible
anime-grammar defaults (dialogue -> close-up, empty shot -> wide establishing shot...).
"""

from __future__ import annotations

import difflib
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .prompts import ANGLE_TAGS, CAMERAS, FRAMING_TAGS, FX_NAMES, TIME_TAGS, TRANSITIONS
from .styles import STYLES


class ProjectError(ValueError):
    """Raised with a friendly, multi-line explanation of everything wrong in episode.yaml."""


def stable_int(*parts) -> int:
    digest = hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:8], 16)


def _suggest(value: str, options) -> str:
    match = difflib.get_close_matches(str(value), list(options), n=1)
    return f" (did you mean '{match[0]}'?)" if match else ""


@dataclass
class Character:
    key: str
    name: str
    tags: str
    kind: str = "other"
    outfits: dict[str, str] = field(default_factory=dict)
    voice: str | None = None
    voice_speed: float = 1.0
    lora: str | None = None
    lora_strength: float = 0.8
    trigger: str | None = None
    color: tuple[int, int, int] = (200, 80, 80)   # used for animatic placeholders


@dataclass
class Shot:
    id: str
    scene_id: str
    index: int                       # position in the whole episode
    location: str | None
    time: str
    characters: list[str]
    outfits: dict[str, str]
    speaker: str | None = None
    line: str | None = None
    expression: str | None = None
    action: str = ""
    framing: str = "medium"
    angle: str | None = None
    camera: str = "auto"
    motion: str = "still"            # still | ai
    fx: list[str] = field(default_factory=list)
    seconds: float | None = None
    sketch: Path | None = None
    sketch_strength: float = 0.65
    seed: int = 0
    transition: str = "cut"
    sfx: Path | None = None
    sfx_volume: float = 1.0
    video: Path | None = None
    card: str | None = None
    face: list[float] | None = None  # manual face box [x, y, w, h] in 0-1 image coordinates

    @property
    def kind(self) -> str:
        if self.card is not None:
            return "card"
        if self.video is not None:
            return "video"
        if self.motion == "ai":
            return "ai"
        if self.line:
            return "dialogue"
        return "still"

    def estimate_seconds(self, language: str = "en") -> float:
        if self.seconds:
            return float(self.seconds)
        if self.kind == "card":
            return 3.0
        if self.line:
            if language in ("ja", "zh"):
                return round(len(self.line) * 0.13 + 0.9, 2)
            return round(len(self.line.split()) * 0.38 + 0.9, 2)
        return 3.0 if self.kind != "ai" else 2.5


@dataclass
class Scene:
    id: str
    location: str | None
    time: str
    music: Path | None = None
    music_volume: float = 0.35
    shots: list[Shot] = field(default_factory=list)


@dataclass
class Episode:
    root: Path
    title: str
    number: int
    style: str
    language: str
    width: int
    height: int
    fps: int
    seed: int
    characters: dict[str, Character]
    locations: dict[str, str]
    scenes: list[Scene]
    warnings: list[str] = field(default_factory=list)

    @property
    def shots(self) -> list[Shot]:
        return [s for sc in self.scenes for s in sc.shots]

    def find_shot(self, shot_id: str) -> Shot:
        for s in self.shots:
            if s.id == shot_id:
                return s
        raise KeyError(f"No shot called '{shot_id}'" + _suggest(shot_id, [s.id for s in self.shots]))

    def estimated_seconds(self) -> float:
        return sum(s.estimate_seconds(self.language) for s in self.shots)


PALETTE = [(230, 70, 70), (70, 140, 230), (90, 200, 120), (230, 180, 60), (170, 90, 220),
           (240, 120, 180), (60, 200, 210), (200, 200, 200)]

SHOT_FIELDS = {
    "id", "say", "line", "characters", "expression", "action", "prompt", "framing", "angle",
    "camera", "motion", "fx", "seconds", "sketch", "sketch_strength", "seed", "transition", "sfx",
    "sfx_volume", "video", "card", "outfit", "location", "time", "face",
}


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v).strip() for v in value]


def load_episode(path: str | Path) -> Episode:
    path = Path(path)
    if path.is_dir():
        path = path / "episode.yaml"
    if not path.exists():
        raise ProjectError(f"Can't find {path}. Create one with:  xsuper new <folder>")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ProjectError(f"episode.yaml is not valid YAML:\n{exc}") from exc
    return parse_episode(data, path.parent)


def parse_episode(data: dict, root: Path) -> Episode:
    errors: list[str] = []
    warnings: list[str] = []
    root = Path(root)

    style = str(data.get("style") or "bright_fantasy")
    if style not in STYLES:
        errors.append(f"style '{style}' doesn't exist{_suggest(style, STYLES)}. Options: {', '.join(STYLES)}")

    res = data.get("resolution") or [1920, 1080]
    try:
        width, height = int(res[0]), int(res[1])
    except (TypeError, ValueError, IndexError):
        errors.append("resolution must look like [1920, 1080]")
        width, height = 1920, 1080
    width, height = width // 2 * 2, height // 2 * 2

    # --- characters -------------------------------------------------------------------
    characters: dict[str, Character] = {}
    for i, (key, c) in enumerate((data.get("characters") or {}).items()):
        key = str(key)
        if isinstance(c, str):
            c = {"tags": c}
        c = c or {}
        outfits = c.get("outfits") or {}
        if isinstance(outfits, str):
            outfits = {"default": outfits}
        tags = str(c.get("tags") or c.get("look") or "")
        if not tags:
            errors.append(f"character '{key}' needs 'tags' describing their design (hair, eyes, build...)")
        kind = str(c.get("kind") or "").lower()
        if not kind:
            kind = "girl" if "1girl" in tags else "boy" if "1boy" in tags else "other"
        characters[key] = Character(
            key=key,
            name=str(c.get("name") or key.title()),
            tags=tags.replace("1girl", "").replace("1boy", "").strip(" ,"),
            kind=kind,
            outfits={str(k): str(v) for k, v in outfits.items()},
            voice=c.get("voice"),
            voice_speed=float(c.get("voice_speed") or 1.0),
            lora=c.get("lora"),
            lora_strength=float(c.get("lora_strength") or 0.8),
            trigger=c.get("trigger"),
            color=PALETTE[i % len(PALETTE)],
        )

    locations = {str(k): str(v) for k, v in (data.get("locations") or {}).items()}
    master_seed = int(data.get("seed") or 1)

    def check_char(k: str, where: str) -> bool:
        if k in characters:
            return True
        errors.append(f"{where}: unknown character '{k}'{_suggest(k, characters)}")
        return False

    def resolve_file(p, where: str) -> Path | None:
        if not p:
            return None
        fp = (root / str(p)).resolve()
        if not fp.exists():
            errors.append(f"{where}: file not found: {p}")
            return None
        return fp

    # --- scenes & shots -----------------------------------------------------------------
    scenes: list[Scene] = []
    seen_ids: set[str] = set()
    index = 0
    for si, sc in enumerate(data.get("scenes") or []):
        sc = sc or {}
        scene_id = str(sc.get("id") or f"sc{si + 1:02d}")
        scene_loc = sc.get("location")
        if scene_loc and scene_loc not in locations:
            warnings.append(f"scene {scene_id}: location '{scene_loc}' isn't in 'locations' - using it as a description")
        scene_time = str(sc.get("time") or "day")
        if scene_time not in TIME_TAGS:
            warnings.append(f"scene {scene_id}: time '{scene_time}' isn't a preset - used as raw tags")
        scene_outfits = {str(k): str(v) for k, v in (sc.get("outfits") or {}).items()}
        for k in scene_outfits:
            check_char(k, f"scene {scene_id} outfits")
        scene = Scene(
            id=scene_id, location=scene_loc, time=scene_time,
            music=resolve_file(sc.get("music"), f"scene {scene_id} music"),
            music_volume=float(sc.get("music_volume") or 0.35),
        )
        for shi, d in enumerate(sc.get("shots") or []):
            if isinstance(d, str):
                d = {"action": d}
            d = d or {}
            shot_id = str(d.get("id") or f"{scene_id}_{shi + 1:03d}")
            where = f"shot {shot_id}"
            if shot_id in seen_ids:
                errors.append(f"{where}: duplicate shot id")
            seen_ids.add(shot_id)
            for k in set(d) - SHOT_FIELDS:
                warnings.append(f"{where}: unknown field '{k}'{_suggest(k, SHOT_FIELDS)} (ignored)")

            speaker = d.get("say")
            speaker = str(speaker) if speaker else None
            if speaker and not check_char(speaker, where):
                speaker = None
            line = d.get("line")
            if speaker and not line:
                errors.append(f"{where}: 'say: {speaker}' needs a 'line'")
            chars = _as_list(d.get("characters")) if "characters" in d else ([speaker] if speaker else [])
            chars = [c for c in chars if check_char(c, where)]

            outfits = {k: scene_outfits[k] for k in chars if k in scene_outfits}
            shot_outfit = d.get("outfit")
            if isinstance(shot_outfit, dict):
                outfits.update({str(k): str(v) for k, v in shot_outfit.items()})
            elif shot_outfit:
                for k in chars:
                    outfits[k] = str(shot_outfit)

            motion = str(d.get("motion") or "still").lower()
            if motion not in ("still", "ai"):
                errors.append(f"{where}: motion must be 'still' or 'ai'")
                motion = "still"
            card = d.get("card")
            video = resolve_file(d.get("video"), where) if d.get("video") else None

            default_framing = ("closeup" if line else "medium" if chars else "wide")
            framing = str(d.get("framing") or default_framing)
            if framing not in FRAMING_TAGS:
                warnings.append(f"{where}: framing '{framing}' isn't a preset{_suggest(framing, FRAMING_TAGS)} - used as raw tags")
            angle = d.get("angle")
            if angle and angle not in ANGLE_TAGS:
                warnings.append(f"{where}: angle '{angle}' isn't a preset{_suggest(angle, ANGLE_TAGS)} - used as raw tags")
            camera = str(d.get("camera") or "auto")
            if camera not in CAMERAS:
                errors.append(f"{where}: camera '{camera}' unknown{_suggest(camera, CAMERAS)}. Options: {', '.join(CAMERAS)}")
                camera = "auto"
            fx = _as_list(d.get("fx"))
            for f in fx:
                if f not in FX_NAMES:
                    errors.append(f"{where}: fx '{f}' unknown{_suggest(f, FX_NAMES)}. Options: {', '.join(FX_NAMES)}")
            fx = [f for f in fx if f in FX_NAMES]
            transition = str(d.get("transition") or "cut")
            if transition not in TRANSITIONS:
                errors.append(f"{where}: transition must be one of {', '.join(TRANSITIONS)}")
                transition = "cut"
            face = d.get("face")
            if face is not None and (not isinstance(face, (list, tuple)) or len(face) != 4):
                errors.append(f"{where}: face must be [x, y, w, h] with values between 0 and 1")
                face = None

            shot = Shot(
                id=shot_id, scene_id=scene_id, index=index,
                location=d.get("location") or scene_loc,
                time=str(d.get("time") or scene_time),
                characters=chars, outfits=outfits,
                speaker=speaker, line=str(line).strip() if line else None,
                expression=d.get("expression"),
                action=str(d.get("action") or d.get("prompt") or ""),
                framing=framing, angle=angle, camera=camera, motion=motion, fx=fx,
                seconds=float(d["seconds"]) if d.get("seconds") else None,
                sketch=resolve_file(d.get("sketch"), where),
                sketch_strength=float(d.get("sketch_strength") or 0.65),
                seed=int(d["seed"]) if d.get("seed") is not None else stable_int(master_seed, shot_id),
                transition=transition,
                sfx=resolve_file(d.get("sfx"), where),
                sfx_volume=float(d.get("sfx_volume") or 1.0),
                video=video,
                card=str(card) if card is not None else None,
                face=[float(v) for v in face] if face else None,
            )
            scene.shots.append(shot)
            index += 1
        if not scene.shots:
            warnings.append(f"scene {scene_id} has no shots")
        scenes.append(scene)

    if not scenes:
        errors.append("the episode has no scenes yet")
    if errors:
        raise ProjectError("episode.yaml has problems:\n  - " + "\n  - ".join(errors))

    return Episode(
        root=root,
        title=str(data.get("title") or "Untitled"),
        number=int(data.get("episode") or 1),
        style=style,
        language=str(data.get("language") or "en"),
        width=width, height=height,
        fps=int(data.get("fps") or 24),
        seed=master_seed,
        characters=characters,
        locations=locations,
        scenes=scenes,
        warnings=warnings,
    )


def summarize(ep: Episode) -> str:
    secs = ep.estimated_seconds()
    kinds: dict[str, int] = {}
    for s in ep.shots:
        kinds[s.kind] = kinds.get(s.kind, 0) + 1
    lines = [
        f"{ep.title} (episode {ep.number}) - style: {ep.style}",
        f"{len(ep.scenes)} scenes, {len(ep.shots)} shots, about {int(secs // 60)}m {int(secs % 60)}s",
        "shot types: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())),
        "characters: " + (", ".join(f"{c.name} ({k})" for k, c in ep.characters.items()) or "none"),
    ]
    if ep.warnings:
        lines.append("warnings:\n  - " + "\n  - ".join(ep.warnings))
    return "\n".join(lines)
