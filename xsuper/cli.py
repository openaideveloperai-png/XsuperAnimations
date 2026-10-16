"""Command line:  xsuper <command> ...   (run `xsuper -h` for help)"""

from __future__ import annotations

import argparse
import shutil
import sys
from importlib import resources
from pathlib import Path

from .project import ProjectError, load_episode, summarize
from .settings import load_settings, write_default_settings


def _template_dir() -> Path:
    return Path(str(resources.files("xsuper") / "templates"))


def create_project(dest: Path, example: bool = False) -> bool:
    """Create a project folder. Returns False if an episode.yaml already exists there."""
    dest = Path(dest)
    if (dest / "episode.yaml").exists():
        return False
    dest.mkdir(parents=True, exist_ok=True)
    templates = _template_dir()
    if example:
        shutil.copytree(templates / "demo", dest, dirs_exist_ok=True)
    else:
        shutil.copy(templates / "blank.yaml", dest / "episode.yaml")
    for sub in ("sketches", "music", "sfx", "characters"):
        (dest / sub).mkdir(exist_ok=True)
    if not (dest / "studio.yaml").exists():
        write_default_settings(dest / "studio.yaml")
    return True


def cmd_new(args) -> int:
    dest = Path(args.folder)
    if not create_project(dest, args.example):
        print(f"{dest}/episode.yaml already exists - not overwriting.")
        return 1
    print(f"Created project in {dest}/\n"
          f"  1. edit {dest}/episode.yaml (or: xsuper write {dest} \"your idea\")\n"
          f"  2. preview it free:   xsuper render {dest}\n"
          f"  3. real anime render: xsuper render {dest} --mode final   (needs ComfyUI, see README)")
    return 0


def cmd_check(args) -> int:
    ep = load_episode(args.project)
    print(summarize(ep))
    return 0


def _split(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def cmd_render(args) -> int:
    from .render import Studio

    ep = load_episode(args.project)
    for w in ep.warnings:
        print(f"[warn] {w}")
    warnings: list[str] = []
    settings = load_settings(ep.root, warnings)
    for w in warnings:
        print(f"[warn] {w}")
    if args.comfy:
        settings.comfy_url = args.comfy
    studio = Studio(ep, settings, args.mode)
    out = studio.render(only=_split(args.only), reroll=_split(args.reroll), keyframes_only=args.keyframes_only)
    print(f"\nSaved: {out}")
    if not args.keyframes_only:
        print(f"Keyframe review sheet: {studio.dir / 'review.html'}")
    return 0


def cmd_write(args) -> int:
    from .writer import chatbot_prompt, write_episode

    project = Path(args.project)
    if args.prompt_only:
        print(chatbot_prompt(args.idea, args.minutes, args.style))
        return 0
    settings = load_settings(project)
    if args.llm_url:
        settings.llm.base_url = args.llm_url
    if args.model:
        settings.llm.model = args.model
    target = project / "episode.yaml"
    if target.exists() and not args.force:
        print(f"{target} exists. Use --force to replace it (the old one is kept as episode.backup.yaml).")
        return 1
    text = write_episode(args.idea, settings.llm, minutes=args.minutes, style=args.style, log=print)
    project.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.copy(target, project / "episode.backup.yaml")
    target.write_text(text, encoding="utf-8")
    if not (project / "studio.yaml").exists():
        write_default_settings(project / "studio.yaml")
    print(f"\nWrote {target}\n")
    print(summarize(load_episode(target)))
    return 0


def cmd_sheet(args) -> int:
    from .characters import make_character_sheet

    ep = load_episode(args.project)
    settings = load_settings(ep.root)
    out = make_character_sheet(ep, settings, args.character, count=args.count, mode=args.mode,
                               outfit=args.outfit, log=print)
    print(f"\nCharacter sheet: {out}")
    return 0


def cmd_ui(args) -> int:
    from .ui import launch

    launch(args.project, share=args.share, port=args.port)
    return 0


def cmd_doctor(args) -> int:
    import importlib.util

    from .faces import detector_problem

    ok = True

    def line(good: bool, text: str):
        nonlocal ok
        ok &= good
        print(("  OK   " if good else "  !!   ") + text)

    print("XsuperAnimations doctor")
    line(bool(shutil.which("ffmpeg")), "ffmpeg installed")
    line(importlib.util.find_spec("cv2") is not None, "opencv installed")
    problem = detector_problem()
    line(problem is None, "anime face detector" + (f" - {problem}" if problem else ""))
    kokoro = importlib.util.find_spec("kokoro") is not None
    print(("  OK   " if kokoro else "  --   ") + "kokoro voices" + ("" if kokoro else " (optional: pip install kokoro soundfile)"))
    if args.project:
        settings = load_settings(args.project)
        from .backends.comfy import ComfyBackend, ComfyError
        try:
            ComfyBackend(settings).check()
            line(True, f"ComfyUI reachable at {settings.comfy_url} with checkpoint {settings.image.checkpoint}")
        except ComfyError as exc:
            print("  --   ComfyUI: " + str(exc).replace("\n", "\n         "))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="xsuper", description="XsuperAnimations - make anime episodes from prompts and doodles")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("new", help="create a project folder")
    s.add_argument("folder")
    s.add_argument("--example", action="store_true", help="start from the 'Ashfall' demo episode")
    s.set_defaults(fn=cmd_new)

    s = sub.add_parser("check", help="validate episode.yaml and show a summary")
    s.add_argument("project", nargs="?", default=".")
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser("render", help="render the episode")
    s.add_argument("project", nargs="?", default=".")
    s.add_argument("--mode", default="animatic", choices=["animatic", "draft", "final"],
                   help="animatic = instant storyboard preview (no GPU); draft = fast AI art, no AI motion; "
                        "final = full quality")
    s.add_argument("--only", help="comma-separated shot ids to render (makes build/partial_*.mp4)")
    s.add_argument("--reroll", help="comma-separated shot ids to redraw with a new seed")
    s.add_argument("--keyframes-only", action="store_true", help="only draw keyframes + review sheet (GPU part)")
    s.add_argument("--comfy", help="ComfyUI url (overrides studio.yaml)")
    s.set_defaults(fn=cmd_render)

    s = sub.add_parser("write", help="write episode.yaml from an idea with a (free) LLM")
    s.add_argument("project")
    s.add_argument("idea")
    s.add_argument("--minutes", type=float, default=23)
    s.add_argument("--style", default="bright_fantasy")
    s.add_argument("--llm-url", help="OpenAI-compatible base url (default: local Ollama)")
    s.add_argument("--model")
    s.add_argument("--force", action="store_true")
    s.add_argument("--prompt-only", action="store_true",
                   help="just print a prompt to paste into any free chatbot")
    s.set_defaults(fn=cmd_write)

    s = sub.add_parser("sheet", help="draw a character model sheet + LoRA training set")
    s.add_argument("project")
    s.add_argument("character")
    s.add_argument("--count", type=int, default=24)
    s.add_argument("--outfit")
    s.add_argument("--mode", default="final", choices=["animatic", "draft", "final"])
    s.set_defaults(fn=cmd_sheet)

    s = sub.add_parser("ui", help="open the web app")
    s.add_argument("project", nargs="?", default="my_anime")
    s.add_argument("--share", action="store_true", help="public link (needed on Colab)")
    s.add_argument("--port", type=int, default=7860)
    s.set_defaults(fn=cmd_ui)

    s = sub.add_parser("doctor", help="check that everything is installed")
    s.add_argument("project", nargs="?")
    s.set_defaults(fn=cmd_doctor)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except ProjectError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nStopped. Everything finished so far is cached - run the same command to resume.")
        return 130
    except Exception as exc:  # friendly message for the common failures
        name = type(exc).__name__
        if name in ("ComfyError", "FFmpegError", "LLMError", "KeyError"):
            print(f"\n{exc}", file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
