from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

Box = tuple[int, int, int, int]  # x, y, w, h in pixels


@dataclass
class ImageRequest:
    positive: str
    negative: str
    width: int
    height: int
    seed: int
    loras: list[tuple[str, float]] = field(default_factory=list)
    control_image: Image.Image | None = None   # preprocessed sketch: white lines on black
    control_strength: float = 0.65
    hires: bool | None = None                  # None -> use settings
    # Only used by the animatic backend, which draws simple stand-ins instead of art.
    hint: dict = field(default_factory=dict)


@dataclass
class Keyframe:
    image: Image.Image
    faces: list[Box] | None = None             # None -> run the anime face detector


class Backend:
    """Interface every image/video generator implements."""

    name = "base"

    def identity(self) -> str:
        """String that changes whenever results would change (used for caching)."""
        return self.name

    def check(self) -> None:
        """Raise a friendly error if the backend isn't reachable / configured."""

    def txt2img(self, req: ImageRequest) -> Keyframe:
        raise NotImplementedError

    def inpaint(self, image: Image.Image, mask: Image.Image, req: ImageRequest, denoise: float) -> Image.Image:
        """Repaint the white area of mask (same size as image)."""
        raise NotImplementedError

    def img2video(self, image: Image.Image, prompt: str, negative: str, seed: int) -> list[Image.Image]:
        raise NotImplementedError
