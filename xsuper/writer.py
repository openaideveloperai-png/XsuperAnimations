"""Prompt -> episode.yaml, written by any OpenAI-compatible LLM (free options: Ollama locally,
or the free tiers of Groq / Google AI Studio / OpenRouter).

The episode is written in two passes so a full 23-minute episode fits in small local models:
  1. series bible + scene outline (cast with fixed design tags, locations, beats)
  2. each scene expanded into shots, with the story so far as context
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import requests
import yaml

from .audio import default_voice
from .project import ProjectError, parse_episode
from .prompts import ANGLE_TAGS, CAMERAS, EXPRESSION_TAGS, FRAMING_TAGS, FX_NAMES, TIME_TAGS
from .settings import LLMSettings
from .styles import STYLES


class LLMError(RuntimeError):
    pass


SHOT_GUIDE = f"""Each shot is a JSON object. Fields (all optional except what the shot needs):
- "say": character key of the speaker, "line": what they say (short, natural, anime-style dialogue)
- "characters": list of character keys visible in the shot (a reaction shot can show someone who is NOT speaking)
- "expression": one of {", ".join(EXPRESSION_TAGS)}
- "action": what is visible, as short comma-separated danbooru-style tags (e.g. "running, looking back, scarf flowing")
- "framing": one of {", ".join(FRAMING_TAGS)}
- "angle": one of {", ".join(ANGLE_TAGS)}
- "camera": one of {", ".join(CAMERAS)}
- "motion": "ai" ONLY for important moving moments (fights, running, big gestures) - at most 1 in 8 shots
- "fx": list from {", ".join(FX_NAMES)}
- "outfit": outfit name if different from default
- "seconds": duration for shots without dialogue (1.5-5)
- "transition": "fade" at the start of a new time/place, "flash" for shocking cuts
- "card": text for a black title card instead of a picture (title, eyecatch, "To be continued")
Anime shot grammar: open a scene with a wide establishing shot (no characters, camera pan_left/pan_right/tilt_down),
then medium shots, then close-ups for dialogue, cut to reaction shots, use impact/speed_lines/shake for hits."""


def chat(llm: LLMSettings, system: str, user: str, json_mode: bool = True, temperature: float = 0.8) -> str:
    url = llm.base_url.rstrip("/") + "/chat/completions"
    payload = {"model": llm.model, "temperature": temperature,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {llm.api_key}"} if llm.api_key else {}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=900)
        if r.status_code == 400 and json_mode:  # some servers don't support response_format
            payload.pop("response_format")
            r = requests.post(url, json=payload, headers=headers, timeout=900)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except requests.ConnectionError as exc:
        raise LLMError(
            f"Can't reach the LLM at {llm.base_url}.\n"
            "Free options: install Ollama (https://ollama.com) and run 'ollama pull qwen2.5:14b',\n"
            "or use a free API key (Groq, Google AI Studio, OpenRouter) in studio.yaml -> llm,\n"
            "or run 'xsuper write <project> \"idea\" --prompt-only' and paste the prompt into any chatbot."
        ) from exc
    except (requests.HTTPError, KeyError, ValueError) as exc:
        raise LLMError(f"LLM request failed: {exc}") from exc


def extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except ValueError:
                pass
    raise LLMError("The LLM didn't return valid JSON. Try again or use a bigger model.")


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(name).lower()).strip("_") or "x"


def _structure(minutes: float) -> str:
    if minutes >= 15:
        return ("TV episode structure: cold open (~2 min) -> title card -> Part A (~9 min) -> eyecatch card -> "
                "Part B (~9 min) -> cliffhanger -> 'To be continued' card.")
    if minutes >= 5:
        return "Structure: hook -> title card -> rising conflict -> climax -> ending beat."
    return "Structure: one strong beginning, middle and end."


def outline(idea: str, llm: LLMSettings, minutes: float, style: str) -> dict:
    n_scenes = max(2, min(18, round(minutes / 1.5)))
    system = "You are the series composer and character designer at a top anime studio. Reply with JSON only."
    user = f"""Create the series bible and scene outline for an original anime episode.
Idea: {idea}
Runtime: {minutes} minutes. About {n_scenes} scenes. {_structure(minutes)}
Visual style: {STYLES[style].label}

Return JSON:
{{"title": "...", "episode_title": "...",
 "characters": {{"<short_key>": {{"name": "...", "kind": "boy|girl|man|woman|other",
    "tags": "fixed visual design as danbooru tags: hair colour+style, eye colour, build, distinguishing marks (NO clothes)",
    "outfits": {{"default": "clothing tags", "<other>": "clothing tags"}}, "personality": "..."}}}},
 "locations": {{"<short_key>": "danbooru-style scenery tags"}},
 "scenes": [{{"id": "<short_key>", "location": "<location key>", "time": "{"|".join(TIME_TAGS)}",
    "minutes": 1.5, "summary": "what happens, who is there, emotional beat", "outfits": {{"<char>": "<outfit name>"}}}}]}}
Rules: 2-6 characters with very distinctive designs. Scene minutes must add up to about {minutes}."""
    data = extract_json(chat(llm, system, user))
    if not data.get("characters") or not data.get("scenes"):
        raise LLMError("The outline came back without characters or scenes - try again.")
    return data


def scene_shots(data: dict, scene: dict, story_so_far: list[str], llm: LLMSettings, minutes: float) -> list[dict]:
    seconds = float(scene.get("minutes") or 1.5) * 60
    want = max(4, round(seconds / 3.6))
    cast = "\n".join(f'- {k}: {c.get("name")} ({c.get("kind")}), {c.get("personality", "")}'
                     for k, c in data["characters"].items())
    system = "You are an anime storyboard artist and scriptwriter. Reply with JSON only."
    shots: list[dict] = []
    while len(shots) < want:
        batch = min(40, want - len(shots))
        recent = json.dumps(shots[-4:], ensure_ascii=False) if shots else "none (this is the start of the scene)"
        user = f"""Episode: {data.get('title')} - {data.get('episode_title', '')}
Cast:
{cast}
Story so far: {' '.join(story_so_far[-6:]) or 'This is the first scene.'}
THIS scene ({scene.get('id')}, at {scene.get('location')}, {scene.get('time')}): {scene.get('summary')}
Previous shots in this scene: {recent}

Write the next {batch} shots of this scene{' (continue exactly where it left off)' if shots else ''}.
{SHOT_GUIDE}
Return JSON: {{"shots": [ ... ]}}"""
        got = extract_json(chat(llm, system, user)).get("shots") or []
        if not got:
            break
        shots.extend(got)
    return shots[:want]


def clean_shot(raw: dict, cast: set[str]) -> dict | None:
    if not isinstance(raw, dict):
        return None
    out: dict = {}
    if raw.get("card"):
        return {"card": str(raw["card"]), "seconds": 3, "transition": "fade"}
    say = _key(raw["say"]) if raw.get("say") else None
    if say in cast and raw.get("line"):
        out["say"] = say
        out["line"] = str(raw["line"]).strip()
    chars = [_key(c) for c in (raw.get("characters") or []) if _key(c) in cast]
    if chars or "characters" in raw:
        out["characters"] = list(dict.fromkeys(chars))
    for name, allowed in (("expression", None), ("framing", FRAMING_TAGS), ("angle", ANGLE_TAGS),
                          ("camera", CAMERAS)):
        value = raw.get(name)
        if value and (allowed is None or value in allowed):
            out[name] = str(value)
    if raw.get("action"):
        out["action"] = str(raw["action"])
    if raw.get("motion") == "ai":
        out["motion"] = "ai"
    fx = [f for f in (raw.get("fx") or []) if f in FX_NAMES]
    if fx:
        out["fx"] = fx
    if raw.get("outfit"):
        out["outfit"] = str(raw["outfit"])
    if raw.get("transition") in ("fade", "flash"):
        out["transition"] = raw["transition"]
    if raw.get("seconds") and "line" not in out:
        try:
            out["seconds"] = round(min(8.0, max(1.0, float(raw["seconds"]))), 1)
        except (TypeError, ValueError):
            pass
    if not out.get("line") and not out.get("action") and not out.get("characters"):
        return None
    return out


def build_episode(data: dict, scenes: list[tuple[dict, list[dict]]], minutes: float, style: str,
                  language: str = "en") -> dict:
    characters, counters = {}, {}
    for key, c in data["characters"].items():
        k = _key(key)
        kind = str(c.get("kind") or "other").lower()
        group = "girl" if kind in ("girl", "woman") else "boy"
        outfits = c.get("outfits") or {}
        if isinstance(outfits, str):
            outfits = {"default": outfits}
        characters[k] = {
            "name": c.get("name") or k.title(),
            "kind": kind,
            "tags": c.get("tags") or "",
            "outfits": {_key(o): str(v) for o, v in outfits.items()} or {"default": ""},
            "voice": default_voice(language, kind, counters.get(group, 0)),
        }
        counters[group] = counters.get(group, 0) + 1
    locations = {_key(k): str(v) for k, v in (data.get("locations") or {}).items()}
    cast = set(characters)

    out_scenes = []
    for i, (scene, shots) in enumerate(scenes):
        cleaned = [s for s in (clean_shot(r, cast) for r in shots) if s]
        if not cleaned:
            continue
        sc = {"id": _key(scene.get("id") or f"sc{i + 1:02d}"),
              "location": _key(scene.get("location", "")) if scene.get("location") else None,
              "time": scene.get("time") if scene.get("time") in TIME_TAGS else "day"}
        outfits = {_key(k): _key(v) for k, v in (scene.get("outfits") or {}).items() if _key(k) in cast}
        if outfits:
            sc["outfits"] = outfits
        sc["shots"] = cleaned
        out_scenes.append({k: v for k, v in sc.items() if v is not None})
        if i == 0 and minutes >= 5:
            title = f"{data.get('title', 'Untitled')}\n{data.get('episode_title', '')}".strip()
            out_scenes.append({"id": "title_card", "shots": [{"card": title, "seconds": 4, "transition": "fade"}]})
    out_scenes.append({"id": "next_time", "shots": [{"card": "To be continued", "seconds": 3, "transition": "fade"}]})
    # scene ids must be unique
    seen: dict[str, int] = {}
    for sc in out_scenes:
        n = seen.get(sc["id"], 0)
        seen[sc["id"]] = n + 1
        if n:
            sc["id"] = f"{sc['id']}_{n + 1}"
    return {"title": data.get("title") or "Untitled", "episode": 1, "style": style, "language": language,
            "resolution": [1920, 1080], "fps": 24,
            "seed": sum(map(ord, str(data.get("title") or "x"))) * 7 % 100000,
            "characters": characters, "locations": locations, "scenes": out_scenes}


def write_episode(idea: str, llm: LLMSettings, minutes: float = 23, style: str = "bright_fantasy",
                  language: str = "en", log: Callable[[str], None] = print) -> str:
    if style not in STYLES:
        raise LLMError(f"unknown style {style}")
    log(f"[write] outlining a {minutes:g}-minute episode with {llm.model} ...")
    data = outline(idea, llm, minutes, style)
    log(f"[write] '{data.get('title')}' - {len(data['characters'])} characters, {len(data['scenes'])} scenes")
    story, scenes = [], []
    for i, scene in enumerate(data["scenes"]):
        log(f"[write] scene {i + 1}/{len(data['scenes'])}: {scene.get('summary', '')[:80]}")
        scenes.append((scene, scene_shots(data, scene, story, llm, minutes)))
        story.append(str(scene.get("summary", "")))
    episode = build_episode(data, scenes, minutes, style, language)
    text = "# Written by XsuperAnimations from the idea:\n#   " + idea.replace("\n", " ") + "\n" + \
        yaml.safe_dump(episode, sort_keys=False, allow_unicode=True, width=120)
    try:
        parse_episode(yaml.safe_load(text), Path("."))
    except ProjectError as exc:
        log(f"[write] note: the script needs small fixes before rendering:\n{exc}")
    return text


def chatbot_prompt(idea: str, minutes: float = 23, style: str = "bright_fantasy") -> str:
    """A prompt to paste into any free chatbot when no LLM endpoint is available."""
    example = """title: Ashfall
episode: 1
style: dark_action
language: en
seed: 4242
characters:
  ren:
    name: Ren Kagari
    kind: boy
    tags: messy white hair, red eyes, small scar on left cheek, slim build
    outfits:
      default: black gakuran, long red scarf
      battle: torn black gakuran, long red scarf, bandaged arms
locations:
  rooftop: school rooftop, chain-link fence, city skyline in background
scenes:
  - id: rooftop
    location: rooftop
    time: sunset
    shots:
      - action: empty rooftop, wind, clouds
        camera: pan_left
        transition: fade
      - say: ren
        line: Go home. It's not safe here.
        expression: serious
      - characters: [ren]
        action: jumping over the fence, midair, scarf flowing
        motion: ai
        fx: [speed_lines]"""
    return f"""You are an anime series composer and storyboard artist. Write an original anime episode as YAML
for the XsuperAnimations tool. Output ONLY YAML in a code block.

IDEA: {idea}
RUNTIME: {minutes:g} minutes ({_structure(minutes)}). Plan roughly {round(minutes * 60 / 3.6)} shots.
STYLE: {style}

Format (follow exactly):
{example}

Character "tags" = fixed visual design as danbooru tags (hair, eyes, build, marks) and never change.
Clothes go in "outfits". Scene "time" is one of: {", ".join(TIME_TAGS)}.
{SHOT_GUIDE}

The episode is long, so write it in parts: first the header (title, characters, locations) and the first
scenes, then stop. When I say "continue", write the next scenes (only the scene list items) until the
episode is complete, then write "# END"."""
