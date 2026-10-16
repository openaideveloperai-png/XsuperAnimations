"""Protocol tests against small fake servers (no GPU, no network)."""

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import pytest
import yaml
from PIL import Image

from xsuper.backends.base import ImageRequest
from xsuper.backends.comfy import ComfyBackend, ComfyError
from xsuper.settings import LLMSettings, StudioSettings


def serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


class FakeComfy(BaseHTTPRequestHandler):
    graphs: list = []
    fail_with: dict | None = None

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/system_stats":
            return self._json({"system": {}})
        if url.path.startswith("/object_info/"):
            node = url.path.rsplit("/", 1)[1]
            options = {"CheckpointLoaderSimple": ("ckpt_name", ["animagine-xl-4.0.safetensors"]),
                       "LoraLoader": ("lora_name", ["ren.safetensors"])}.get(node)
            if not options:
                return self._json({})
            return self._json({node: {"input": {"required": {options[0]: [options[1]]}}}})
        if url.path.startswith("/history/"):
            graph = FakeComfy.graphs[-1]
            frames = 1
            for node in graph.values():
                if node["class_type"] == "Wan22ImageToVideoLatent":
                    frames = node["inputs"]["length"]
            return self._json({"p1": {"status": {"status_str": "success", "completed": True},
                                      "outputs": {"9": {"images": [{"filename": f"{i}.png", "subfolder": "",
                                                                     "type": "temp"} for i in range(frames)]}}}})
        if url.path == "/view":
            buf = io.BytesIO()
            Image.new("RGB", (64, 32), (10, 200, 30)).save(buf, "PNG")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(buf.getvalue())
            return
        self._json({}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        if self.path == "/upload/image":
            return self._json({"name": "up.png", "subfolder": "", "type": "input"})
        if self.path == "/prompt":
            if FakeComfy.fail_with:
                return self._json(FakeComfy.fail_with, 400)
            graph = json.loads(body)["prompt"]
            # every link must point at an existing node
            for node in graph.values():
                for value in node["inputs"].values():
                    if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
                        assert value[0] in graph, f"dangling link {value}"
            FakeComfy.graphs.append(graph)
            return self._json({"prompt_id": "p1", "number": 1, "node_errors": {}})
        self._json({}, 404)


@pytest.fixture
def comfy():
    FakeComfy.graphs = []
    FakeComfy.fail_with = None
    srv, url = serve(FakeComfy)
    settings = StudioSettings(comfy_url=url)
    yield ComfyBackend(settings), settings
    srv.shutdown()


def classes(graph):
    return [n["class_type"] for n in graph.values()]


def test_txt2img_graph_with_lora_sketch_and_hires(comfy):
    backend, settings = comfy
    backend.check()
    req = ImageRequest("1boy", "lowres", 1344, 768, 5, loras=[("ren.safetensors", 0.8)],
                       control_image=Image.new("RGB", (1344, 768)))
    kf = backend.txt2img(req)
    assert kf.image.size == (64, 32) and kf.faces is None
    graph = FakeComfy.graphs[-1]
    cls = classes(graph)
    assert cls.count("KSampler") == 2 and "LoraLoader" in cls and "ControlNetApplyAdvanced" in cls
    assert "ImageScaleBy" in cls and cls[-1] == "PreviewImage"


def test_missing_model_is_explained(comfy):
    backend, settings = comfy
    settings.image.checkpoint = "nope.safetensors"
    with pytest.raises(ComfyError, match="isn't installed in ComfyUI"):
        backend.check()
    with pytest.raises(ComfyError, match="LoRA 'other.safetensors'"):
        backend.txt2img(ImageRequest("a", "b", 64, 64, 1, loras=[("other.safetensors", 1.0)]))


def test_rejected_workflow_message(comfy):
    backend, _ = comfy
    FakeComfy.fail_with = {"error": {"message": "Prompt outputs failed validation"},
                           "node_errors": {"3": {"class_type": "KSampler",
                                                 "errors": [{"message": "Value not in list"}]}}}
    with pytest.raises(ComfyError, match="node 3 \\(KSampler\\): Value not in list"):
        backend.txt2img(ImageRequest("a", "b", 64, 64, 1))


def test_inpaint_and_video_graphs(comfy):
    backend, settings = comfy
    out = backend.inpaint(Image.new("RGB", (512, 512)), Image.new("L", (512, 512)), ImageRequest("a", "b", 512, 512, 1), 0.6)
    assert out.size == (512, 512)
    assert "SetLatentNoiseMask" in classes(FakeComfy.graphs[-1])
    frames = backend.img2video(Image.new("RGB", (1344, 768)), "anime", "bad", 3)
    assert len(frames) == settings.video.frames
    cls = classes(FakeComfy.graphs[-1])
    assert "Wan22ImageToVideoLatent" in cls and "ImageUpscaleWithModel" in cls


def test_custom_video_workflow_placeholders(comfy, tmp_path):
    backend, settings = comfy
    wf = {"1": {"class_type": "LoadImage", "inputs": {"image": "%IMAGE%"}},
          "2": {"class_type": "Wan22ImageToVideoLatent", "inputs": {"length": "%FRAMES%", "start_image": ["1", 0]}},
          "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "%PROMPT%, masterpiece"}}}
    path = tmp_path / "wf.json"
    path.write_text(json.dumps(wf))
    settings.video.workflow = str(path)
    settings.video.upscale_model = ""
    frames = backend.img2video(Image.new("RGB", (100, 100)), "sword slash", "bad", 3)
    sent = FakeComfy.graphs[-1]
    assert sent["1"]["inputs"]["image"] == "up.png"
    assert sent["2"]["inputs"]["length"] == settings.video.frames == len(frames)
    assert sent["3"]["inputs"]["text"] == "sword slash, masterpiece"


def test_writer_with_fake_llm(tmp_path):
    from xsuper.project import parse_episode
    from xsuper.writer import chatbot_prompt, write_episode

    outline = {"title": "Neon Ronin", "episode_title": "Rain",
               "characters": {"Kaito": {"name": "Kaito", "kind": "boy", "tags": "black hair",
                                        "outfits": {"default": "haori"}}},
               "locations": {"Market": "night market"},
               "scenes": [{"id": "open", "location": "Market", "time": "night", "minutes": 0.2, "summary": "walk"}]}
    shots = {"shots": [{"action": "market", "camera": "pan_right"},
                       {"say": "Kaito", "line": "Stay close.", "fx": ["bogus"], "camera": "orbit"},
                       {"characters": ["kaito", "ghost"], "action": "draws sword", "motion": "ai"}, "junk"]}

    class LLM(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            content = outline if "series bible" in req["messages"][1]["content"] else shots
            body = json.dumps({"choices": [{"message": {"content": "```json\n" + json.dumps(content) + "```"}}]})
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body.encode())

    srv, url = serve(LLM)
    try:
        text = write_episode("ronin", LLMSettings(base_url=url + "/v1", model="m"), minutes=0.4, log=lambda m: None)
    finally:
        srv.shutdown()
    ep = parse_episode(yaml.safe_load(text), tmp_path)
    say = [s for s in ep.shots if s.line][0]
    assert say.speaker == "kaito" and say.camera == "auto" and say.fx == []
    ai = [s for s in ep.shots if s.kind == "ai"][0]
    assert ai.characters == ["kaito"]
    assert "Write an original anime episode" in chatbot_prompt("x")
