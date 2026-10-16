import pytest

from xsuper.project import ProjectError, load_episode, parse_episode
from xsuper.prompts import join_tags, shot_prompt
from xsuper.styles import get_style


def minimal(**shot):
    return {
        "characters": {"ren": {"kind": "boy", "tags": "white hair, red eyes",
                               "outfits": {"default": "black gakuran", "battle": "torn gakuran"}}},
        "locations": {"roof": "school rooftop"},
        "scenes": [{"id": "a", "location": "roof", "shots": [shot]}],
    }


def test_demo_episode_loads(demo):
    ep = load_episode(demo)
    assert ep.title == "Ashfall"
    assert len(ep.shots) == 23
    kinds = {s.kind for s in ep.shots}
    assert kinds == {"still", "dialogue", "ai", "card"}
    assert ep.find_shot("rooftop_008").sketch.exists()
    assert not ep.warnings


def test_defaults_follow_anime_grammar(tmp_path):
    ep = parse_episode(minimal(say="ren", line="Hello."), tmp_path)
    shot = ep.shots[0]
    assert shot.kind == "dialogue" and shot.framing == "closeup" and shot.characters == ["ren"]
    ep = parse_episode(minimal(action="empty rooftop"), tmp_path)
    assert ep.shots[0].framing == "wide" and ep.shots[0].characters == []


def test_friendly_errors(tmp_path):
    with pytest.raises(ProjectError) as err:
        parse_episode(minimal(say="rne", line="hi", fx=["speedlines"], camera="zoom"), tmp_path)
    msg = str(err.value)
    assert "unknown character 'rne' (did you mean 'ren'?)" in msg
    assert "fx 'speedlines' unknown (did you mean 'speed_lines'?)" in msg
    assert "camera 'zoom' unknown" in msg


def test_seeds_are_stable_and_unique(tmp_path):
    data = minimal(action="x")
    data["scenes"][0]["shots"] = ["one", "two"]
    a = parse_episode(data, tmp_path)
    b = parse_episode(data, tmp_path)
    assert [s.seed for s in a.shots] == [s.seed for s in b.shots]
    assert a.shots[0].seed != a.shots[1].seed


def test_prompt_keeps_character_on_model(tmp_path):
    data = minimal(say="ren", line="Let's go.", outfit="battle", expression="angry")
    ep = parse_episode(data, tmp_path)
    prompt = shot_prompt(ep, ep.shots[0], get_style("dark_action"))
    assert prompt.startswith("1boy, white hair, red eyes, torn gakuran")
    assert "school rooftop" in prompt and "anime screencap" in prompt
    # angry has clenched teeth -> mouth already open-ish, so no forced 'closed mouth'
    assert "clenched teeth" in prompt


def test_dialogue_base_frame_has_closed_mouth(tmp_path):
    ep = parse_episode(minimal(say="ren", line="Hi.", expression="neutral"), tmp_path)
    assert "closed mouth" in shot_prompt(ep, ep.shots[0], get_style(None))


def test_join_tags_dedupes():
    assert join_tags("a, b", "B, c", None, "") == "a, b, c"
