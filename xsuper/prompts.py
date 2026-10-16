"""Shot vocabulary and prompt building.

Every frame of a character is built from the *same* character block (trigger word + design
tags + outfit), so the model draws the same person in every shot. Only the expression,
framing and action change.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .styles import BASE_NEGATIVE, QUALITY_TAGS, Style

if TYPE_CHECKING:  # pragma: no cover
    from .project import Character, Episode, Shot

FRAMING_TAGS = {
    "extreme_closeup": "extreme close-up, face focus, eye focus",
    "closeup": "close-up, portrait, face focus",
    "medium": "upper body",
    "cowboy": "cowboy shot",
    "full": "full body",
    "wide": "wide shot, very wide shot, scenery",
    "over_shoulder": "over shoulder, from behind, looking at another",
    "pov": "pov",
}

ANGLE_TAGS = {
    "eye": "",
    "low": "from below",
    "high": "from above",
    "dutch": "dutch angle",
    "side": "from side",
    "behind": "from behind",
    "aerial": "from above, aerial view",
}

TIME_TAGS = {
    "day": "day, blue sky, sunlight",
    "morning": "morning, soft sunlight",
    "sunset": "sunset, orange sky, golden hour, backlighting",
    "dusk": "evening, twilight, purple sky",
    "night": "night, night sky, moonlight, dark",
    "rain": "rain, overcast, wet, puddle",
    "snow": "snow, snowing, winter",
    "indoor": "indoors",
    "indoor_night": "indoors, night, dim lighting, lamp",
}

EXPRESSION_TAGS = {
    "neutral": "expressionless, closed mouth",
    "serious": "serious, closed mouth, furrowed brow",
    "happy": "smile",
    "laughing": "laughing, smile, open mouth",
    "angry": "angry, v-shaped eyebrows, clenched teeth",
    "furious": "furious, angry, shouting, open mouth, v-shaped eyebrows",
    "sad": "sad, frown",
    "crying": "crying, tears, sad",
    "shocked": "surprised, wide-eyed, constricted pupils",
    "scared": "scared, sweat, wide-eyed",
    "smirk": "smirk, smug",
    "embarrassed": "blush, embarrassed, looking away",
    "determined": "determined, serious, glaring",
    "tired": "tired, half-closed eyes",
    "crazy": "crazy smile, wide-eyed, constricted pupils, shaded face",
}

CAMERAS = ["auto", "hold", "drift", "pan_left", "pan_right", "tilt_up", "tilt_down",
           "push_in", "pull_out", "zoom_punch", "shake"]

FX_NAMES = ["speed_lines", "streaks", "impact", "shake", "flash", "rain", "glow", "letterbox"]

TRANSITIONS = ["cut", "fade", "flash"]

KIND_GROUP = {"boy": "boy", "man": "boy", "male": "boy", "girl": "girl", "woman": "girl", "female": "girl"}

WAN_NEGATIVE = (
    "static, still image, frozen, blurry, low quality, worst quality, jpeg artifacts, overexposed, "
    "deformed, extra fingers, malformed limbs, fused fingers, poorly drawn hands, poorly drawn face, "
    "morphing, flickering, 3d, realistic, photorealistic, subtitles, text, watermark, walking backwards"
)


def join_tags(*parts: str | None) -> str:
    """Join comma-separated tag strings, dropping blanks and case-insensitive duplicates."""
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        for tag in part.split(","):
            tag = tag.strip()
            key = tag.lower()
            if tag and key not in seen:
                seen.add(key)
                out.append(tag)
    return ", ".join(out)


def count_tags(characters: list["Character"]) -> str:
    groups = {"boy": 0, "girl": 0}
    for ch in characters:
        group = KIND_GROUP.get(ch.kind)
        if group:
            groups[group] += 1
    tags = []
    for group, n in groups.items():
        if n == 1:
            tags.append(f"1{group}")
        elif 1 < n < 6:
            tags.append(f"{n}{group}s")
        elif n >= 6:
            tags.append(f"6+{group}s")
    return ", ".join(tags)


def expression_tags(expression: str | None) -> str:
    if not expression:
        return ""
    return EXPRESSION_TAGS.get(expression.strip().lower(), expression)


def outfit_tags(ch: "Character", outfit: str | None) -> str:
    key = outfit or "default"
    if key in ch.outfits:
        return ch.outfits[key]
    if outfit:  # not a named outfit -> treat as literal clothing tags
        return outfit
    return ""


def character_block(ch: "Character", outfit: str | None, expression: str | None) -> str:
    return join_tags(ch.trigger, ch.tags, outfit_tags(ch, outfit), expression_tags(expression))


def _mouth_is_open(tags: str) -> bool:
    return bool(re.search(r"open mouth|shouting|laughing|screaming|teeth", tags))


def shot_prompt(ep: "Episode", shot: "Shot", style: Style) -> str:
    chars = [ep.characters[k] for k in shot.characters]
    blocks = []
    for ch in chars:
        expr = shot.expression if (shot.speaker == ch.key or len(chars) == 1) else None
        block = character_block(ch, shot.outfits.get(ch.key), expr)
        # Dialogue keyframes are drawn with a closed mouth; the open-mouth frame is painted
        # separately so the two can be swapped for anime-style lip flaps.
        if shot.kind == "dialogue" and ch.key == shot.speaker and not _mouth_is_open(block):
            block = join_tags(block, "closed mouth")
        blocks.append(block)
    location = ep.locations.get(shot.location or "", shot.location or "")
    return join_tags(
        count_tags(chars) if chars else "no humans, scenery",
        *blocks,
        shot.action,
        FRAMING_TAGS.get(shot.framing, shot.framing),
        ANGLE_TAGS.get(shot.angle or "eye", shot.angle or ""),
        location,
        TIME_TAGS.get(shot.time, shot.time),
        style.tags,
        QUALITY_TAGS,
    )


def shot_negative(style: Style, extra: str = "") -> str:
    return join_tags(BASE_NEGATIVE, style.negative, extra)


def region_prompt(ep: "Episode", shot: "Shot", style: Style, character_key: str, change: str) -> str:
    """Prompt for repainting part of a face (mouth flap / blink frame)."""
    ch = ep.characters[character_key]
    block = character_block(ch, shot.outfits.get(character_key), shot.expression)
    block = re.sub(r"\bclosed mouth\b,?\s*", "", block)
    return join_tags(change, block, "portrait, face focus", style.tags, QUALITY_TAGS)


def video_prompt(ep: "Episode", shot: "Shot", style: Style) -> str:
    """Natural-language prompt for the image-to-video model (it prefers sentences over tags)."""
    who = []
    for key in shot.characters:
        ch = ep.characters[key]
        who.append(f"{ch.name} ({outfit_tags(ch, shot.outfits.get(key)) or ch.tags})")
    action = shot.action or (f"{ep.characters[shot.speaker].name} talks" if shot.speaker else "the scene breathes")
    camera = {
        "push_in": "The camera pushes in.", "pull_out": "The camera pulls back.",
        "pan_left": "The camera pans left.", "pan_right": "The camera pans right.",
        "tilt_up": "The camera tilts up.", "tilt_down": "The camera tilts down.",
        "shake": "Handheld camera shake.", "zoom_punch": "Fast dramatic zoom.",
    }.get(shot.camera, "")
    parts = [
        style.video_hint,
        f"{action}.",
        ("Characters: " + "; ".join(who) + ".") if who else "",
        camera,
        "Fluid hand-drawn 2D animation, sakuga, cel shading, clean lineart, consistent character design, "
        "the art style stays exactly the same as the first frame.",
    ]
    return " ".join(p for p in parts if p)
