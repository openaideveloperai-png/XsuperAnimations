"""Per-project machine settings (studio.yaml): which models to use and where ComfyUI lives.

The story itself lives in episode.yaml; this file only describes the "studio equipment".
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml


@dataclass
class ImageSettings:
    checkpoint: str = "animagine-xl-4.0.safetensors"
    steps: int = 28
    cfg: float = 5.0
    sampler: str = "euler_ancestral"
    scheduler: str = "normal"
    clip_skip: int = 1               # 2 is common for Illustrious / NoobAI based checkpoints
    hires: bool = True               # second detail pass at hires_scale
    hires_scale: float = 1.5
    hires_denoise: float = 0.35
    controlnet: str = "controlnet-scribble-sdxl-1.0.safetensors"
    controlnet_union_type: str = ""  # e.g. "canny/lineart/anime_lineart/mlsd" for union ControlNets
    inpaint_denoise: float = 0.62    # mouth/blink variants
    lora_dir_hint: str = "ComfyUI/models/loras"


@dataclass
class VideoSettings:
    enabled: bool = True
    model: str = "wan2.2_ti2v_5B_fp16.safetensors"
    text_encoder: str = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    vae: str = "wan2.2_vae.safetensors"
    width: int = 832
    height: int = 480
    frames: int = 49                 # must be 4*k + 1
    fps: int = 24
    steps: int = 20
    cfg: float = 5.0
    shift: float = 8.0
    sampler: str = "uni_pc"
    scheduler: str = "simple"
    upscale_model: str = "RealESRGAN_x4plus_anime_6B.pth"   # "" to skip
    workflow: str = ""               # optional custom ComfyUI API workflow with %PLACEHOLDERS%


@dataclass
class TTSSettings:
    engine: str = "kokoro"           # kokoro | espeak | none
    speed: float = 1.0


@dataclass
class LLMSettings:
    # Any OpenAI-compatible endpoint works: Ollama (free, local), LM Studio, Groq / Gemini free tiers...
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen2.5:14b"
    api_key: str = ""


@dataclass
class StudioSettings:
    comfy_url: str = "http://127.0.0.1:8188"
    timeout_image: int = 900
    timeout_video: int = 3600
    encode_preset: str = "faster"    # x264 preset: veryfast (quick) ... slow (smaller files)
    encode_crf: int = 18             # lower = better quality, bigger files
    image: ImageSettings = field(default_factory=ImageSettings)
    video: VideoSettings = field(default_factory=VideoSettings)
    tts: TTSSettings = field(default_factory=TTSSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)


def _merge(obj, data: dict, path: str, warnings: list[str]):
    known = {f.name: f for f in fields(obj)}
    for key, value in (data or {}).items():
        if key not in known:
            warnings.append(f"studio.yaml: unknown setting '{path}{key}' (ignored)")
            continue
        current = getattr(obj, key)
        if is_dataclass(current):
            if isinstance(value, dict):
                _merge(current, value, f"{path}{key}.", warnings)
            else:
                warnings.append(f"studio.yaml: '{path}{key}' should be a section")
        else:
            setattr(obj, key, type(current)(value) if value is not None and current is not None else value)


def load_settings(project_dir: str | Path, warnings: list[str] | None = None) -> StudioSettings:
    warnings = warnings if warnings is not None else []
    settings = StudioSettings()
    path = Path(project_dir) / "studio.yaml"
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        _merge(settings, data, "", warnings)
    if os.environ.get("XSUPER_COMFY_URL"):
        settings.comfy_url = os.environ["XSUPER_COMFY_URL"]
    if os.environ.get("XSUPER_LLM_KEY"):
        settings.llm.api_key = os.environ["XSUPER_LLM_KEY"]
    return settings


def write_default_settings(path: str | Path) -> None:
    header = (
        "# XsuperAnimations studio settings - which AI models to use and where ComfyUI runs.\n"
        "# File names must match what is inside your ComfyUI/models/... folders.\n"
    )
    Path(path).write_text(header + yaml.safe_dump(asdict(StudioSettings()), sort_keys=False), encoding="utf-8")
