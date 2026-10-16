"""ComfyUI backend - drives free, open-weight models running in ComfyUI (local GPU or Colab).

* keyframes:    SDXL anime checkpoint (Animagine XL 4 / Illustrious / NoobAI) + character LoRAs,
                optional hires detail pass, optional scribble ControlNet for your doodles
* lip flaps:    masked repaint of just the mouth / eyes of an existing keyframe
* sakuga shots: Wan 2.2 image-to-video (or any workflow you export from ComfyUI)

Only ComfyUI *core* nodes are used, so a stock ComfyUI install is enough.
"""

from __future__ import annotations

import copy
import io
import json
import time
import uuid
from pathlib import Path

import requests
from PIL import Image

from ..settings import StudioSettings
from .base import Backend, ImageRequest, Keyframe


class ComfyError(RuntimeError):
    pass


class Graph:
    """Tiny helper to build ComfyUI API-format prompts."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}

    def add(self, class_type: str, **inputs) -> str:
        node_id = str(len(self.nodes) + 1)
        self.nodes[node_id] = {"class_type": class_type, "inputs": inputs}
        return node_id


def _cover(img: Image.Image, w: int, h: int) -> Image.Image:
    img = img.convert("RGB")
    scale = max(w / img.width, h / img.height)
    nw, nh = max(w, round(img.width * scale)), max(h, round(img.height * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    x, y = (nw - w) // 2, (nh - h) // 2
    return img.crop((x, y, x + w, y + h))


class ComfyBackend(Backend):
    name = "comfy"

    def __init__(self, settings: StudioSettings):
        self.s = settings
        self.url = settings.comfy_url.rstrip("/")
        self.client_id = uuid.uuid4().hex
        self.http = requests.Session()
        self._choice_cache: dict[tuple[str, str], list[str] | None] = {}
        self._video_checked = False

    # ---------------------------------------------------------------- identity / checks
    def identity(self) -> str:
        i = self.s.image
        return (f"comfy|{i.checkpoint}|{i.steps}|{i.cfg}|{i.sampler}|{i.scheduler}|{i.clip_skip}|"
                f"{i.hires}|{i.hires_scale}|{i.hires_denoise}|{i.controlnet}")

    def video_identity(self) -> str:
        v = self.s.video
        return (f"{v.model}|{v.width}x{v.height}|{v.frames}|{v.steps}|{v.cfg}|{v.shift}|{v.sampler}|"
                f"{v.upscale_model}|{v.workflow}")

    def check(self) -> None:
        try:
            self.http.get(f"{self.url}/system_stats", timeout=10).raise_for_status()
        except requests.RequestException as exc:
            raise ComfyError(
                f"Can't reach ComfyUI at {self.url}.\n"
                "Start ComfyUI first (see README / the Colab notebook), set comfy_url in studio.yaml,\n"
                "or render with --mode animatic to preview without a GPU."
            ) from exc
        self._require("CheckpointLoaderSimple", "ckpt_name", self.s.image.checkpoint, "checkpoint",
                      "ComfyUI/models/checkpoints")

    def _choices(self, node: str, input_name: str) -> list[str] | None:
        key = (node, input_name)
        if key not in self._choice_cache:
            try:
                info = self.http.get(f"{self.url}/object_info/{node}", timeout=20).json().get(node, {})
                spec = (info.get("input", {}).get("required", {}).get(input_name)
                        or info.get("input", {}).get("optional", {}).get(input_name))
                options = None
                if spec:
                    if isinstance(spec[0], list):
                        options = spec[0]
                    elif spec[0] == "COMBO" and len(spec) > 1:
                        options = spec[1].get("options")
                self._choice_cache[key] = options
            except (requests.RequestException, ValueError, AttributeError, IndexError):
                self._choice_cache[key] = None
        return self._choice_cache[key]

    def _require(self, node: str, input_name: str, value: str, what: str, folder: str) -> None:
        options = self._choices(node, input_name)
        if options is None or value in options:
            return
        found = ", ".join(options[:12]) or "nothing"
        raise ComfyError(
            f"The {what} '{value}' isn't installed in ComfyUI.\n"
            f"Put it in {folder}/ (then restart ComfyUI) or change the name in studio.yaml.\n"
            f"Installed right now: {found}"
        )

    # ---------------------------------------------------------------- transport
    def _upload(self, img: Image.Image, prefix: str) -> str:
        buf = io.BytesIO()
        img.save(buf, "PNG")
        name = f"xsuper_{prefix}_{uuid.uuid4().hex[:10]}.png"
        r = self.http.post(f"{self.url}/upload/image",
                           files={"image": (name, buf.getvalue(), "image/png")},
                           data={"overwrite": "true", "type": "input"}, timeout=120)
        r.raise_for_status()
        j = r.json()
        return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]

    @staticmethod
    def _explain(payload: dict) -> str:
        parts = []
        err = payload.get("error") or {}
        if err:
            parts.append(f"{err.get('message', '')} {err.get('details', '')}".strip())
        for node_id, info in (payload.get("node_errors") or {}).items():
            for e in info.get("errors", []):
                parts.append(f"node {node_id} ({info.get('class_type')}): {e.get('message')} {e.get('details', '')}".strip())
        return "\n".join(parts) or json.dumps(payload)[:500]

    def _fetch(self, ref: dict) -> Image.Image:
        r = self.http.get(f"{self.url}/view", params={
            "filename": ref["filename"], "subfolder": ref.get("subfolder", ""), "type": ref.get("type", "output")},
            timeout=120)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")

    def run(self, nodes: dict, timeout: int) -> list[Image.Image]:
        r = self.http.post(f"{self.url}/prompt", json={"prompt": nodes, "client_id": self.client_id}, timeout=60)
        if r.status_code != 200:
            try:
                raise ComfyError("ComfyUI rejected the workflow:\n" + self._explain(r.json()))
            except ValueError:
                raise ComfyError(f"ComfyUI error {r.status_code}: {r.text[:300]}") from None
        prompt_id = r.json()["prompt_id"]
        start = time.time()
        while True:
            hist = self.http.get(f"{self.url}/history/{prompt_id}", timeout=60).json()
            entry = hist.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    detail = "unknown error"
                    for kind, data in status.get("messages", []):
                        if kind == "execution_error":
                            detail = f"{data.get('node_type')}: {data.get('exception_message', '').strip()}"
                    raise ComfyError(f"ComfyUI failed while generating - {detail}")
                images = []
                for out in entry.get("outputs", {}).values():
                    images.extend(self._fetch(ref) for ref in out.get("images", []))
                if images:
                    return images
                if status.get("completed"):
                    raise ComfyError("ComfyUI finished but returned no images")
            if time.time() - start > timeout:
                raise ComfyError(f"ComfyUI took longer than {timeout}s - raise timeout_image/timeout_video in studio.yaml")
            time.sleep(1.0)

    # ---------------------------------------------------------------- graphs
    def _base(self, g: Graph, loras: list[tuple[str, float]]):
        i = self.s.image
        ck = g.add("CheckpointLoaderSimple", ckpt_name=i.checkpoint)
        model, clip, vae = [ck, 0], [ck, 1], [ck, 2]
        if i.clip_skip and i.clip_skip > 1:
            cs = g.add("CLIPSetLastLayer", clip=clip, stop_at_clip_layer=-int(i.clip_skip))
            clip = [cs, 0]
        for name, strength in loras:
            self._require("LoraLoader", "lora_name", name, "LoRA", "ComfyUI/models/loras")
            lo = g.add("LoraLoader", model=model, clip=clip, lora_name=name,
                       strength_model=float(strength), strength_clip=float(strength))
            model, clip = [lo, 0], [lo, 1]
        return model, clip, vae

    def _sampler(self, g: Graph, model, pos, neg, latent, seed: int, denoise: float) -> str:
        i = self.s.image
        return g.add("KSampler", model=model, seed=int(seed), steps=int(i.steps), cfg=float(i.cfg),
                     sampler_name=i.sampler, scheduler=i.scheduler, positive=pos, negative=neg,
                     latent_image=latent, denoise=float(denoise))

    def txt2img(self, req: ImageRequest) -> Keyframe:
        i = self.s.image
        g = Graph()
        model, clip, vae = self._base(g, req.loras)
        pos = [g.add("CLIPTextEncode", text=req.positive, clip=clip), 0]
        neg = [g.add("CLIPTextEncode", text=req.negative, clip=clip), 0]
        if req.control_image is not None:
            self._require("ControlNetLoader", "control_net_name", i.controlnet, "ControlNet",
                          "ComfyUI/models/controlnet")
            cn = [g.add("ControlNetLoader", control_net_name=i.controlnet), 0]
            if i.controlnet_union_type:
                cn = [g.add("SetUnionControlNetType", control_net=cn, type=i.controlnet_union_type), 0]
            hint = [g.add("LoadImage", image=self._upload(req.control_image.convert("RGB"), "sketch")), 0]
            ap = g.add("ControlNetApplyAdvanced", positive=pos, negative=neg, control_net=cn, image=hint,
                       strength=float(req.control_strength), start_percent=0.0, end_percent=0.85)
            pos, neg = [ap, 0], [ap, 1]
        latent = [g.add("EmptyLatentImage", width=req.width, height=req.height, batch_size=1), 0]
        ks = self._sampler(g, model, pos, neg, latent, req.seed, 1.0)
        image = [g.add("VAEDecode", samples=[ks, 0], vae=vae), 0]
        if i.hires if req.hires is None else req.hires:
            up = g.add("ImageScaleBy", image=image, upscale_method="lanczos", scale_by=float(i.hires_scale))
            enc = g.add("VAEEncode", pixels=[up, 0], vae=vae)
            ks2 = self._sampler(g, model, pos, neg, [enc, 0], req.seed + 1, i.hires_denoise)
            image = [g.add("VAEDecode", samples=[ks2, 0], vae=vae), 0]
        g.add("PreviewImage", images=image)
        return Keyframe(self.run(g.nodes, self.s.timeout_image)[0], None)

    def inpaint(self, image: Image.Image, mask: Image.Image, req: ImageRequest, denoise: float) -> Image.Image:
        g = Graph()
        model, clip, vae = self._base(g, req.loras)
        pos = [g.add("CLIPTextEncode", text=req.positive, clip=clip), 0]
        neg = [g.add("CLIPTextEncode", text=req.negative, clip=clip), 0]
        img = [g.add("LoadImage", image=self._upload(image.convert("RGB"), "src")), 0]
        msk = [g.add("LoadImageMask", image=self._upload(mask.convert("RGB"), "mask"), channel="red"), 0]
        enc = g.add("VAEEncode", pixels=img, vae=vae)
        noised = g.add("SetLatentNoiseMask", samples=[enc, 0], mask=msk)
        ks = self._sampler(g, model, pos, neg, [noised, 0], req.seed, denoise)
        out = g.add("VAEDecode", samples=[ks, 0], vae=vae)
        g.add("PreviewImage", images=[out, 0])
        result = self.run(g.nodes, self.s.timeout_image)[0]
        return result.resize(image.size, Image.LANCZOS) if result.size != image.size else result

    def _check_video(self) -> None:
        if self._video_checked:
            return
        v = self.s.video
        if not v.workflow:
            self._require("UNETLoader", "unet_name", v.model, "video model", "ComfyUI/models/diffusion_models")
            self._require("CLIPLoader", "clip_name", v.text_encoder, "video text encoder", "ComfyUI/models/text_encoders")
            self._require("VAELoader", "vae_name", v.vae, "video VAE", "ComfyUI/models/vae")
        if v.upscale_model:
            self._require("UpscaleModelLoader", "model_name", v.upscale_model, "upscale model",
                          "ComfyUI/models/upscale_models")
        self._video_checked = True

    def img2video(self, image: Image.Image, prompt: str, negative: str, seed: int) -> list[Image.Image]:
        v = self.s.video
        self._check_video()
        start = _cover(image, v.width, v.height)
        name = self._upload(start, "start")
        if v.workflow:
            nodes = self._fill_template(Path(v.workflow), {
                "%PROMPT%": prompt, "%NEGATIVE%": negative, "%IMAGE%": name, "%SEED%": int(seed),
                "%WIDTH%": v.width, "%HEIGHT%": v.height, "%FRAMES%": v.frames,
            })
            return self.run(nodes, self.s.timeout_video)

        g = Graph()
        unet = g.add("UNETLoader", unet_name=v.model, weight_dtype="default")
        model = g.add("ModelSamplingSD3", model=[unet, 0], shift=float(v.shift))
        clip = g.add("CLIPLoader", clip_name=v.text_encoder, type="wan")
        vae = g.add("VAELoader", vae_name=v.vae)
        pos = g.add("CLIPTextEncode", text=prompt, clip=[clip, 0])
        neg = g.add("CLIPTextEncode", text=negative, clip=[clip, 0])
        img = g.add("LoadImage", image=name)
        latent = g.add("Wan22ImageToVideoLatent", vae=[vae, 0], width=v.width, height=v.height,
                       length=v.frames, batch_size=1, start_image=[img, 0])
        ks = g.add("KSampler", model=[model, 0], seed=int(seed), steps=int(v.steps), cfg=float(v.cfg),
                   sampler_name=v.sampler, scheduler=v.scheduler, positive=[pos, 0], negative=[neg, 0],
                   latent_image=[latent, 0], denoise=1.0)
        frames = [g.add("VAEDecode", samples=[ks, 0], vae=[vae, 0]), 0]
        if v.upscale_model:
            um = g.add("UpscaleModelLoader", model_name=v.upscale_model)
            up = g.add("ImageUpscaleWithModel", upscale_model=[um, 0], image=frames)
            out_w = 1920
            out_h = round(1920 * v.height / v.width / 8) * 8
            frames = [g.add("ImageScale", image=[up, 0], upscale_method="lanczos", width=out_w, height=out_h,
                            crop="disabled"), 0]
        g.add("PreviewImage", images=frames)
        return self.run(g.nodes, self.s.timeout_video)

    @staticmethod
    def _fill_template(path: Path, values: dict) -> dict:
        nodes = json.loads(path.read_text(encoding="utf-8"))
        if "nodes" in nodes and "links" in nodes:
            raise ComfyError(f"{path} is a UI workflow. In ComfyUI use 'Export (API)' and point video.workflow at that file.")

        def fill(obj):
            if isinstance(obj, dict):
                return {k: fill(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [fill(v) for v in obj]
            if isinstance(obj, str):
                if obj in values:
                    return values[obj]
                for key, val in values.items():
                    if key in obj:
                        obj = obj.replace(key, str(val))
            return obj

        return fill(copy.deepcopy(nodes))
