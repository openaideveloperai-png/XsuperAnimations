import subprocess

import yaml

from xsuper.project import load_episode
from xsuper.render import Studio
from xsuper.settings import StudioSettings
from xsuper.video import count_frames


def small_project(demo, shots_from=("cold_open",), keep=3):
    data = yaml.safe_load((demo / "episode.yaml").read_text())
    data["resolution"] = [640, 360]
    scenes = [s for s in data["scenes"] if s["id"] in shots_from]
    for s in scenes:
        s["shots"] = s["shots"][:keep]
    data["scenes"] = scenes + [{"id": "end", "shots": [{"card": "Fin", "seconds": 1}]}]
    (demo / "episode.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return load_episode(demo)


def probe_streams(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.split()
    return sorted(set(out))


def test_animatic_end_to_end(demo, fake_voice):
    ep = small_project(demo)
    logs = []
    studio = Studio(ep, StudioSettings(), "animatic", log=logs.append)
    out = studio.render()
    assert out.exists()
    assert probe_streams(out) == ["audio", "subtitle", "video"]
    shot_frames = sum(count_frames(studio.dir / "shots" / f"{s.id}.mp4") for s in ep.shots)
    assert count_frames(out) == shot_frames
    srt = out.with_suffix(".srt").read_text()
    assert "Who... who's there?" in srt
    assert (studio.dir / "review.html").exists()
    # dialogue close-up got lip-flap + blink frames
    assert (studio.dir / "keyframes" / "cold_open_003_mouth.png").exists()

    # second run: everything cached, nothing re-filmed
    logs.clear()
    Studio(ep, StudioSettings(), "animatic", log=logs.append).render()
    assert not [line for line in logs if line.startswith(("[film]", "[draw]", "[card]"))]


def test_reroll_redraws_only_that_shot(demo, fake_voice):
    ep = small_project(demo)
    Studio(ep, StudioSettings(), "animatic", log=lambda m: None).render(keyframes_only=True)
    logs = []
    Studio(ep, StudioSettings(), "animatic", log=logs.append).render(reroll=["cold_open_002"], keyframes_only=True)
    drawn = [line for line in logs if line.startswith("[draw]")]
    assert drawn and all("cold_open_002" in line for line in drawn)


def test_only_renders_partial(demo, fake_voice):
    ep = small_project(demo)
    out = Studio(ep, StudioSettings(), "animatic", log=lambda m: None).render(only=["cold_open_001"])
    assert out.name == "partial_animatic.mp4"
    assert count_frames(out) == 4 * 24
