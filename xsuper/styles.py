"""Art-direction presets.

A style controls three things that together make frames read as *TV anime* instead of
"AI illustration": the prompt tags (cel shading, anime screencap, no painterly/3D look),
the negative prompt, and a colour grade that is applied identically to every shot so the
whole episode shares one palette.
"""

from __future__ import annotations

from dataclasses import dataclass, field

QUALITY_TAGS = "masterpiece, best quality, high score, great score, absurdres, safe"

BASE_NEGATIVE = (
    "lowres, bad anatomy, bad hands, missing fingers, extra digits, fewer digits, extra arms, "
    "deformed, disfigured, cropped, worst quality, low quality, low score, bad score, average score, "
    "jpeg artifacts, text, error, signature, watermark, username, logo, blurry, "
    # keep it looking like a cel-animated TV frame, not generic AI art
    "3d, realistic, photorealistic, cgi, render, semi-realistic, painting, oil painting, watercolor, "
    "sketch, traditional media, concept art, airbrushed, glossy skin, multiple views, comic, manga panel"
)


@dataclass(frozen=True)
class Grade:
    saturation: float = 1.0
    contrast: float = 1.0
    gamma: float = 1.0
    lift: tuple = (0.0, 0.0, 0.0)          # added to shadows (0-1 scale)
    gain: tuple = (1.0, 1.0, 1.0)          # multiplied into highlights
    shadow_tint: tuple = (0.0, 0.0, 0.0)
    highlight_tint: tuple = (0.0, 0.0, 0.0)
    bloom: float = 0.0
    vignette: float = 0.25
    grain: float = 2.5                     # std-dev in 0-255 units


@dataclass(frozen=True)
class Style:
    key: str
    label: str
    tags: str
    negative: str = ""
    video_hint: str = ""
    grade: Grade = field(default_factory=Grade)
    speed_line_density: int = 90


STYLES: dict[str, Style] = {
    "dark_action": Style(
        key="dark_action",
        label="Dark action - heavy shadows, gritty, cinematic (modern seinen TV look)",
        tags=(
            "anime screencap, anime coloring, cel shading, sharp lineart, thick outlines, "
            "dramatic lighting, high contrast, deep shadows, cinematic lighting, dynamic angle, "
            "detailed background, newest"
        ),
        negative="pastel colors, chibi, flat background",
        video_hint="Dark, gritty modern TV anime fight scene with heavy shadows and dramatic lighting.",
        grade=Grade(saturation=0.88, contrast=1.14, gamma=0.96, lift=(0.0, 0.006, 0.02),
                    shadow_tint=(-0.01, 0.0, 0.03), highlight_tint=(0.02, 0.01, -0.01),
                    vignette=0.38, grain=3.0),
        speed_line_density=120,
    ),
    "bright_fantasy": Style(
        key="bright_fantasy",
        label="Bright fantasy - clean modern TV anime: big shiny eyes, soft cel shading, pastel colours, sparkly magic",
        tags=(
            "anime screencap, anime coloring, cel shading, clean lineart, thin lineart, shiny eyes, detailed eyes, "
            "sparkling eyes, blush, soft lighting, bright colors, pastel colors, sparkle, light particles, "
            "detailed background, newest"
        ),
        negative="dark, gritty, muted colors, desaturated, thick lineart, heavy shadows",
        video_hint="Bright, clean modern fantasy TV anime with soft cel shading and sparkling magic effects.",
        grade=Grade(saturation=1.06, contrast=0.98, gamma=1.03, lift=(0.015, 0.012, 0.02),
                    highlight_tint=(0.01, 0.005, 0.015), bloom=0.16, vignette=0.12, grain=1.5),
        speed_line_density=80,
    ),
    "bright_shonen": Style(
        key="bright_shonen",
        label="Bright shonen - vivid colours, clean lines, high energy",
        tags=(
            "anime screencap, anime coloring, cel shading, clean lineart, vibrant colors, "
            "bright lighting, dynamic pose, detailed background, newest"
        ),
        negative="muted colors, desaturated",
        video_hint="Energetic, colourful shonen TV anime.",
        grade=Grade(saturation=1.1, contrast=1.06, vignette=0.2, grain=2.5),
    ),
    "soft_drama": Style(
        key="soft_drama",
        label="Soft drama - gentle light, pastel, emotional slice-of-life",
        tags=(
            "anime screencap, anime coloring, soft lighting, light particles, bloom, pastel colors, "
            "depth of field, detailed background, beautiful scenery, newest"
        ),
        negative="dark, gritty, gore",
        video_hint="Gentle, emotional slice-of-life TV anime with soft light.",
        grade=Grade(saturation=0.95, contrast=0.96, gamma=1.04, lift=(0.03, 0.025, 0.03),
                    highlight_tint=(0.02, 0.01, 0.0), bloom=0.25, vignette=0.18, grain=2.0),
        speed_line_density=60,
    ),
    "retro_90s": Style(
        key="retro_90s",
        label="Retro 90s - cel-painted, warm, film grain",
        tags=(
            "anime screencap, 1990s (style), retro artstyle, cel animation, anime coloring, "
            "film grain, detailed background"
        ),
        negative="newest, digital art",
        video_hint="1990s hand-painted cel anime, film look.",
        grade=Grade(saturation=0.92, contrast=1.02, gamma=1.02, lift=(0.03, 0.02, 0.0),
                    gain=(1.03, 1.0, 0.94), highlight_tint=(0.03, 0.015, -0.02),
                    bloom=0.12, vignette=0.32, grain=5.0),
    ),
    "noir_horror": Style(
        key="noir_horror",
        label="Noir horror - muted, sickly palette, oppressive shadows",
        tags=(
            "anime screencap, cel shading, dark, horror (theme), muted colors, limited palette, "
            "heavy shadows, shaded face, detailed background, newest"
        ),
        negative="vibrant colors, cheerful",
        video_hint="Unsettling dark horror TV anime.",
        grade=Grade(saturation=0.6, contrast=1.18, gamma=0.92, shadow_tint=(0.0, 0.02, 0.01),
                    highlight_tint=(0.0, 0.015, -0.01), vignette=0.5, grain=4.5),
        speed_line_density=110,
    ),
}

DEFAULT_STYLE = "bright_fantasy"


def get_style(key: str | None) -> Style:
    return STYLES.get(key or DEFAULT_STYLE, STYLES[DEFAULT_STYLE])
