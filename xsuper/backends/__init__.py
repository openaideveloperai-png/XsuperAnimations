from __future__ import annotations

from ..settings import StudioSettings
from .base import Backend, ImageRequest, Keyframe


def make_backend(mode: str, settings: StudioSettings) -> Backend:
    """mode: 'animatic' (no GPU, instant storyboard) or 'final' (ComfyUI + open models)."""
    if mode == "animatic":
        from .placeholder import AnimaticBackend
        return AnimaticBackend()
    if mode in ("final", "draft", "comfy"):
        from .comfy import ComfyBackend
        return ComfyBackend(settings)
    raise ValueError(f"unknown render mode '{mode}' (use 'animatic', 'draft' or 'final')")


__all__ = ["Backend", "ImageRequest", "Keyframe", "make_backend"]
