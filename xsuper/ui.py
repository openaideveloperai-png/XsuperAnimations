"""Web app (Gradio): write, edit, draw and render an episode from the browser.

    xsuper ui my_show            # local
    xsuper ui my_show --share    # public link (use this on Colab)
"""

from __future__ import annotations

import io
import queue
import random
import threading
from contextlib import redirect_stdout
from pathlib import Path

import yaml

from .project import ProjectError, Shot, load_episode, parse_episode, summarize
from .prompts import FRAMING_TAGS, TIME_TAGS
from .settings import load_settings
from .styles import STYLES

INTRO = """# XsuperAnimations
**Prompt / script / doodle → anime episode.** Workflow: **1** write the story → **2** check the script →
**3** draw character sheets → **4** preview as an *animatic* (free, instant) → **5** render *final* with the AI models.
"""


def run_job(fn):
    """Run fn(log) in a thread and stream its log lines. Yields (log_text, result_or_None)."""
    q: queue.Queue = queue.Queue()
    result: dict = {}

    def target():
        try:
            result["value"] = fn(q.put)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            result["error"] = exc
            q.put(f"\nERROR: {exc}")
        finally:
            q.put(None)

    threading.Thread(target=target, daemon=True).start()
    lines: list[str] = []
    while True:
        msg = q.get()
        if msg is None:
            break
        lines.append(str(msg))
        yield "\n".join(lines[-300:]), None
    yield "\n".join(lines[-300:]), result


def launch(project: str, share: bool = False, port: int = 7860) -> None:
    import gradio as gr

    from .cli import create_project

    root = Path(project).resolve()
    if create_project(root, example=True):
        print(f"Created a new project with the demo episode in {root}")
    ep_file = root / "episode.yaml"
    studio_file = root / "studio.yaml"

    def episode():
        return load_episode(root)

    def char_keys():
        try:
            return list(episode().characters)
        except ProjectError:
            return []

    def location_keys():
        try:
            return ["(none)"] + list(episode().locations)
        except ProjectError:
            return ["(none)"]

    def keyframe_gallery(mode):
        folder = root / "build" / mode / "keyframes"
        if not folder.exists():
            return []
        order = {}
        try:
            order = {s.id: i for i, s in enumerate(episode().shots)}
        except ProjectError:
            pass
        files = [p for p in folder.glob("*.png") if not p.stem.endswith(("_mouth", "_blink"))]
        files.sort(key=lambda p: order.get(p.stem, 10 ** 6))
        return [(str(p), p.stem) for p in files]

    # ------------------------------------------------------------------ actions
    def save_script(text):
        try:
            data = yaml.safe_load(text) or {}
            ep = parse_episode(data, root)
        except (yaml.YAMLError, ProjectError) as exc:
            return f"**Not saved - fix this first:**\n```\n{exc}\n```"
        ep_file.write_text(text, encoding="utf-8")
        return "**Saved.**\n```\n" + summarize(ep) + "\n```"

    def write_story(idea, minutes, style, language, base_url, model, api_key):
        from .writer import write_episode
        settings = load_settings(root)
        settings.llm.base_url = base_url or settings.llm.base_url
        settings.llm.model = model or settings.llm.model
        settings.llm.api_key = api_key or settings.llm.api_key
        text_out = ep_file.read_text(encoding="utf-8")
        for log, res in run_job(lambda log: write_episode(idea, settings.llm, float(minutes), style, language, log)):
            if res and "value" in res:
                if ep_file.exists():
                    (root / "episode.backup.yaml").write_text(ep_file.read_text(encoding="utf-8"), encoding="utf-8")
                text_out = res["value"]
                ep_file.write_text(text_out, encoding="utf-8")
                log += "\n\nSaved to episode.yaml (old version kept as episode.backup.yaml). Check tab 2."
            yield log, text_out

    def chatbot(idea, minutes, style):
        from .writer import chatbot_prompt
        return chatbot_prompt(idea or "(describe your idea here)", float(minutes), style)

    def make_sheet(key, count, mode, outfit, comfy_url):
        from .characters import make_character_sheet
        if not key:
            yield "Pick a character first.", None
            return
        settings = load_settings(root)
        settings.comfy_url = comfy_url or settings.comfy_url
        for log, res in run_job(lambda log: make_character_sheet(episode(), settings, key, int(count), mode,
                                                                 outfit or None, log)):
            yield log, (str(res["value"]) if res and "value" in res else None)

    def sketch_frame(sketch, chars, action, framing, location, time, strength, mode, comfy_url):
        from .render import Studio
        ep = episode()
        settings = load_settings(root)
        settings.comfy_url = comfy_url or settings.comfy_url
        shot = Shot(id=f"sketchlab_{random.randint(0, 99999):05d}", scene_id="sketchlab", index=0,
                    location=None if location == "(none)" else location, time=time,
                    characters=list(chars or []), outfits={}, action=action or "", framing=framing,
                    sketch=Path(sketch) if sketch else None, sketch_strength=float(strength),
                    seed=random.randint(0, 2 ** 31 - 1))
        studio = Studio(ep, settings, mode, log=print)
        studio.backend.check()
        return str(studio.keyframe(shot).path)

    def render(mode, comfy_url, only, reroll, keyframes_only):
        from .render import Studio
        try:
            ep = episode()
        except ProjectError as exc:
            yield str(exc), None, []
            return
        settings = load_settings(root)
        settings.comfy_url = comfy_url or settings.comfy_url
        split = lambda v: [x.strip() for x in (v or "").split(",") if x.strip()] or None  # noqa: E731

        def job(log):
            studio = Studio(ep, settings, mode, log=log,
                            progress=lambda f, m: log(f"[{int(f * 100):3d}%] {m}") if m.startswith("drawing") else None)
            return studio.render(only=split(only), reroll=split(reroll), keyframes_only=keyframes_only)

        for log, res in run_job(job):
            if res is None:
                yield log, None, []
            else:
                out = res.get("value")
                video = str(out) if out and str(out).endswith(".mp4") else None
                yield log, video, keyframe_gallery(mode)

    def doctor():
        from .cli import cmd_doctor
        buf = io.StringIO()
        with redirect_stdout(buf):
            cmd_doctor(type("A", (), {"project": str(root)})())
        return buf.getvalue()

    def save_studio(text):
        try:
            yaml.safe_load(text)
        except yaml.YAMLError as exc:
            return f"Not saved: {exc}"
        studio_file.write_text(text, encoding="utf-8")
        warnings: list[str] = []
        load_settings(root, warnings)
        return "Saved." + ("\n" + "\n".join(warnings) if warnings else "")

    styles = list(STYLES)
    settings = load_settings(root)

    with gr.Blocks(title="XsuperAnimations") as app:
        gr.Markdown(INTRO + f"\nProject folder: `{root}`")
        with gr.Tab("1 · Story"):
            gr.Markdown("Describe your anime. A free LLM writes the whole episode as a script you can edit. "
                        "No LLM? Use **Copy-paste prompt** with any free chatbot and paste its answer in tab 2.")
            idea = gr.Textbox(label="Idea", lines=4, placeholder="A girl who can rewind time by 10 seconds joins...")
            with gr.Row():
                minutes = gr.Slider(1, 30, value=23, step=0.5, label="Length (minutes)")
                style = gr.Dropdown(styles, value="bright_fantasy", label="Style")
                language = gr.Dropdown(["en", "ja"], value="en", label="Voice language")
            with gr.Accordion("LLM settings (free: Ollama locally, or Groq / Google AI Studio keys)", open=False):
                base_url = gr.Textbox(value=settings.llm.base_url, label="OpenAI-compatible base URL")
                model = gr.Textbox(value=settings.llm.model, label="Model")
                api_key = gr.Textbox(value=settings.llm.api_key, label="API key (if needed)", type="password")
            with gr.Row():
                write_btn = gr.Button("Write episode with AI", variant="primary")
                prompt_btn = gr.Button("Copy-paste prompt instead")
            write_log = gr.Textbox(label="Progress", lines=10)
        with gr.Tab("2 · Script"):
            gr.Markdown("The whole episode: cast, places and every shot. Edit freely, then **Save & check**.")
            script = gr.Code(value=ep_file.read_text(encoding="utf-8"), language="yaml", label="episode.yaml")
            save_btn = gr.Button("Save & check", variant="primary")
            check_out = gr.Markdown()
        with gr.Tab("3 · Characters"):
            gr.Markdown("Draw a **model sheet** to check a character's design. The drawings double as a "
                        "LoRA training set that locks the design in for every episode (see README).")
            with gr.Row():
                char = gr.Dropdown(char_keys(), label="Character")
                count = gr.Slider(4, 40, value=24, step=1, label="Drawings")
                sheet_mode = gr.Radio(["animatic", "final"], value="final", label="Mode")
                outfit = gr.Textbox(label="Outfit (optional)")
            sheet_btn = gr.Button("Draw model sheet", variant="primary")
            sheet_img = gr.Image(label="Model sheet", type="filepath")
            sheet_log = gr.Textbox(label="Progress", lines=6)
        with gr.Tab("4 · Doodle → Anime"):
            gr.Markdown("Upload any drawing - stick figures are fine. Lower strength = the AI fixes more of your "
                        "drawing; higher = it follows your lines more closely. Use the file in a shot with `sketch:`.")
            with gr.Row():
                sketch = gr.Image(label="Your drawing", type="filepath")
                sketch_out = gr.Image(label="Anime frame", type="filepath")
            with gr.Row():
                sk_chars = gr.Dropdown(char_keys(), multiselect=True, label="Characters in it")
                sk_action = gr.Textbox(label="What's happening (tags)", value="dynamic pose")
            with gr.Row():
                sk_framing = gr.Dropdown(list(FRAMING_TAGS), value="full", label="Framing")
                sk_loc = gr.Dropdown(location_keys(), value="(none)", label="Location")
                sk_time = gr.Dropdown(list(TIME_TAGS), value="day", label="Time")
                sk_strength = gr.Slider(0.2, 1.0, value=0.6, step=0.05, label="Follow my lines")
                sk_mode = gr.Radio(["animatic", "draft", "final"], value="final", label="Mode")
            sk_btn = gr.Button("Make it anime", variant="primary")
        with gr.Tab("5 · Render"):
            gr.Markdown("**animatic** = instant storyboard preview, no GPU · **draft** = AI art, no AI motion "
                        "(fast) · **final** = everything. Everything is cached: re-running only redoes what changed.")
            with gr.Row():
                mode = gr.Radio(["animatic", "draft", "final"], value="animatic", label="Mode")
                comfy_url = gr.Textbox(value=settings.comfy_url, label="ComfyUI URL")
            with gr.Row():
                only = gr.Textbox(label="Only these shots (ids, comma-separated)")
                reroll = gr.Textbox(label="Redraw these shots with a new seed")
                kf_only = gr.Checkbox(label="Keyframes only (do the GPU part first)")
            render_btn = gr.Button("Render", variant="primary")
            with gr.Row():
                video = gr.Video(label="Episode")
                render_log = gr.Textbox(label="Log", lines=18)
            gallery = gr.Gallery(value=keyframe_gallery("animatic"), label="Keyframes (bad one? put its id in "
                                 "'Redraw these shots')", columns=5)
        with gr.Tab("Setup"):
            doctor_btn = gr.Button("Check installation")
            doctor_out = gr.Textbox(label="Doctor", lines=8)
            studio_text = gr.Code(value=studio_file.read_text(encoding="utf-8") if studio_file.exists() else "",
                                  language="yaml", label="studio.yaml (models & ComfyUI)")
            studio_btn = gr.Button("Save studio.yaml")
            studio_out = gr.Markdown()

        write_btn.click(write_story, [idea, minutes, style, language, base_url, model, api_key], [write_log, script])
        prompt_btn.click(chatbot, [idea, minutes, style], write_log)
        save_btn.click(save_script, script, check_out)
        sheet_btn.click(make_sheet, [char, count, sheet_mode, outfit, comfy_url], [sheet_log, sheet_img])
        sk_btn.click(sketch_frame, [sketch, sk_chars, sk_action, sk_framing, sk_loc, sk_time, sk_strength, sk_mode,
                                    comfy_url], sketch_out)
        render_btn.click(render, [mode, comfy_url, only, reroll, kf_only], [render_log, video, gallery])
        doctor_btn.click(doctor, None, doctor_out)
        studio_btn.click(save_studio, studio_text, studio_out)
        for refresh in (save_btn, write_btn):
            refresh.click(lambda: (gr.update(choices=char_keys()), gr.update(choices=char_keys()),
                                   gr.update(choices=location_keys())), None, [char, sk_chars, sk_loc])

    app.queue().launch(share=share, server_port=port, allowed_paths=[str(root)], server_name="0.0.0.0")
