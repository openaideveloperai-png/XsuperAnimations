import shutil
from pathlib import Path

import numpy as np
import pytest

from xsuper import audio as A

TEMPLATES = Path(__file__).resolve().parents[1] / "xsuper" / "templates"


@pytest.fixture
def demo(tmp_path) -> Path:
    shutil.copytree(TEMPLATES / "demo", tmp_path / "demo")
    return tmp_path / "demo"


@pytest.fixture
def fake_voice(monkeypatch):
    """Speech-like noise instead of a real TTS engine (deterministic, no downloads)."""
    def synth(text, voice, speed, engine, language, log=print):
        n = int(A.SR * (0.3 * len(text.split()) + 0.3))
        t = np.arange(n) / A.SR
        env = (np.sin(2 * np.pi * 4.5 * t) > -0.2).astype(np.float32)
        return (np.random.default_rng(len(text)).normal(0, 0.3, n) * env).astype(np.float32)

    import xsuper.render as R
    monkeypatch.setattr(R.A, "synthesize", synth)
    return synth
