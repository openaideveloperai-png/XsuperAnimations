"""Character model sheets and LoRA training sets.

How studios keep characters on-model: a settei (model sheet) every animator follows.
Our equivalent: draw a character many times with fixed design tags, keep the drawings that
match, and train a small LoRA on them. With the LoRA loaded, the model draws *that*
character in every shot of every episode.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw

from .backends import ImageRequest, make_backend
from .project import Episode, stable_int
from .prompts import QUALITY_TAGS, count_tags, expression_tags, join_tags, outfit_tags
from .settings import StudioSettings
from .styles import get_style
from .textutil import font

# (framing/pose tags, expression, portrait canvas?)
VARIATIONS = [
    ("full body, standing, front view, arms at sides", "neutral", True),
    ("full body, standing, from side", "neutral", True),
    ("full body, from behind, looking back", "neutral", True),
    ("upper body, front view", "happy", False),
    ("upper body, three-quarter view", "serious", False),
    ("portrait, close-up, front view", "neutral", False),
    ("portrait, close-up, looking to the side", "angry", False),
    ("upper body, arms crossed", "smirk", False),
    ("cowboy shot, hand on hip", "happy", True),
    ("portrait, close-up", "shocked", False),
    ("full body, running, dynamic pose", "determined", True),
    ("upper body, from above", "sad", False),
    ("portrait, close-up, three-quarter view", "embarrassed", False),
    ("full body, fighting stance", "determined", True),
    ("upper body, looking at viewer, waving", "happy", False),
    ("portrait, close-up, from below", "furious", False),
]


def make_character_sheet(ep: Episode, settings: StudioSettings, key: str, count: int = 24, mode: str = "final",
                         outfit: str | None = None, log: Callable[[str], None] = print) -> Path:
    if key not in ep.characters:
        raise KeyError(f"No character '{key}'. Characters: {', '.join(ep.characters)}")
    ch = ep.characters[key]
    backend = make_backend(mode, settings)
    backend.check()
    style = get_style(ep.style)
    # model sheets want plain backgrounds, so drop the style's background tags
    style_tags = ", ".join(t for t in style.tags.split(", ") if "background" not in t)
    trigger = ch.trigger or f"{key}_xs"
    out_dir = ep.root / "characters" / key
    data_dir = out_dir / "dataset"
    data_dir.mkdir(parents=True, exist_ok=True)
    clothes = outfit_tags(ch, outfit)

    paths = []
    for i in range(count):
        pose, expr, tall = VARIATIONS[i % len(VARIATIONS)]
        w, h = (896, 1152) if tall else (1024, 1024)
        positive = join_tags(count_tags([ch]), ch.trigger, ch.tags, clothes, expression_tags(expr), pose,
                             "simple background, white background", style_tags, QUALITY_TAGS)
        seed = stable_int(ep.seed, "sheet", key, outfit, i)
        log(f"[sheet] {ch.name} {i + 1}/{count}: {pose}, {expr}")
        hint = {"characters": [{"name": ch.name, "color": ch.color}], "time": "indoor", "framing":
                "full" if tall else "closeup", "location": "sheet", "shot_id": f"{key} #{i + 1}", "action": pose}
        img = backend.txt2img(ImageRequest(positive=positive, negative=join_tags(
            "multiple views, reference sheet, text", style.negative), width=w, height=h, seed=seed,
            loras=[(ch.lora, ch.lora_strength)] if ch.lora else [], hint=hint)).image
        path = data_dir / f"{i:03d}.png"
        img.save(path)
        # caption: trigger first; leave out the fixed design tags so the trigger word learns them
        path.with_suffix(".txt").write_text(join_tags(
            trigger, count_tags([ch]), clothes, expression_tags(expr), pose, "simple background, white background"))
        paths.append(path)

    sheet = _contact_sheet(paths, ch.name, out_dir / "sheet.png")
    _training_files(ep, settings, key, trigger, out_dir, data_dir)
    log(f"[sheet] Delete any drawing in {data_dir} that is off-model (and its .txt), then train the LoRA with "
        f"{out_dir / 'train_lora.sh'} - or skip training and just use the tags.")
    return sheet


def _contact_sheet(paths: list[Path], title: str, out: Path) -> Path:
    thumbs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        im.thumbnail((300, 380))
        thumbs.append((p.stem, im))
    cols = 6
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 310 + 10, rows * 410 + 70), (24, 24, 28))
    d = ImageDraw.Draw(sheet)
    d.text((16, 16), f"{title} - model sheet", font=font(32), fill=(240, 240, 240))
    for i, (name, im) in enumerate(thumbs):
        x, y = 10 + (i % cols) * 310, 70 + (i // cols) * 410
        sheet.paste(im, (x + (300 - im.width) // 2, y))
        d.text((x + 4, y + 384), name, font=font(18), fill=(200, 200, 200))
    sheet.save(out)
    return out


def _training_files(ep: Episode, settings: StudioSettings, key: str, trigger: str, out_dir: Path, data_dir: Path):
    (out_dir / "dataset.toml").write_text(textwrap.dedent(f"""\
        [general]
        shuffle_caption = true
        keep_tokens = 1
        caption_extension = ".txt"

        [[datasets]]
        resolution = 1024
        batch_size = 1
        enable_bucket = true

          [[datasets.subsets]]
          image_dir = "{data_dir.resolve().as_posix()}"
          num_repeats = 8
        """))
    script = out_dir / "train_lora.sh"
    script.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        # Train a LoRA for {ep.characters[key].name} with kohya-ss/sd-scripts (free, open source).
        # Needs an NVIDIA GPU with ~12 GB VRAM (a free Colab T4 works; ~45-90 min).
        #   git clone https://github.com/kohya-ss/sd-scripts && pip install -r sd-scripts/requirements.txt
        set -e
        SD_SCRIPTS="${{SD_SCRIPTS:-./sd-scripts}}"
        CHECKPOINT="${{CHECKPOINT:-ComfyUI/models/checkpoints/{settings.image.checkpoint}}}"
        OUT="${{OUT:-ComfyUI/models/loras}}"
        accelerate launch --num_cpu_threads_per_process 1 "$SD_SCRIPTS/sdxl_train_network.py" \\
          --pretrained_model_name_or_path="$CHECKPOINT" \\
          --dataset_config="{(out_dir / 'dataset.toml').resolve().as_posix()}" \\
          --output_dir="$OUT" --output_name="{key}" --save_model_as=safetensors \\
          --network_module=networks.lora --network_dim=16 --network_alpha=8 --network_train_unet_only \\
          --learning_rate=1e-4 --lr_scheduler=cosine --lr_warmup_steps=50 --optimizer_type=AdamW8bit \\
          --max_train_epochs=10 --save_every_n_epochs=5 --train_batch_size=1 --seed=42 \\
          --mixed_precision=fp16 --save_precision=fp16 --gradient_checkpointing --sdpa --no_half_vae \\
          --cache_latents --cache_text_encoder_outputs
        echo
        echo "Done. In episode.yaml add to {key}:   lora: {key}.safetensors   and   trigger: {trigger}"
        """))
    script.chmod(0o755)
