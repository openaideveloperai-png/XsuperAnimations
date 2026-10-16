"""Voices (free, local TTS), lip-sync envelopes and the episode sound mix."""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

SR = 24000

# Kokoro-82M voices (Apache-2.0). The first letter of the id picks the language.
VOICES = {
    ("en", "boy"): ["am_michael", "am_adam", "am_eric", "am_liam", "am_onyx", "am_puck", "bm_george", "bm_lewis"],
    ("en", "girl"): ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "af_nova", "bf_emma", "bf_isabella"],
    ("ja", "boy"): ["jm_kumo"],
    ("ja", "girl"): ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro"],
}

_pipelines: dict[str, object] = {}
_warned: set[str] = set()


def default_voice(language: str, kind: str, index: int) -> str:
    group = "girl" if kind in ("girl", "woman", "female") else "boy"
    options = VOICES.get((language, group)) or VOICES[("en", group)]
    return options[index % len(options)]


def _kokoro(text: str, voice: str, speed: float) -> np.ndarray:
    from kokoro import KPipeline  # imported lazily: optional dependency

    lang_code = voice[0]
    if lang_code not in _pipelines:
        _pipelines[lang_code] = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M")
    chunks = []
    for result in _pipelines[lang_code](text, voice=voice, speed=speed):
        audio = getattr(result, "audio", None)
        if audio is None and isinstance(result, tuple):
            audio = result[2]
        if audio is None:
            continue
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        chunks.append(np.asarray(audio, np.float32).reshape(-1))
    return np.concatenate(chunks) if chunks else np.zeros(SR // 2, np.float32)


def _espeak(text: str, voice: str, speed: float) -> np.ndarray:
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if not exe:
        raise RuntimeError("espeak not installed")
    lang = "ja" if voice.startswith("j") else "en"
    wav = subprocess.run([exe, "-v", lang, "-s", str(int(165 * speed)), "--stdout", text],
                         capture_output=True, check=True).stdout
    return decode_audio_bytes(wav)


def silence_for(text: str, language: str) -> np.ndarray:
    secs = len(text) * 0.12 if language in ("ja", "zh") else len(text.split()) * 0.36
    return np.zeros(int(SR * (secs + 0.3)), np.float32)


def synthesize(text: str, voice: str, speed: float, engine: str, language: str, log=print) -> np.ndarray:
    engines = {"kokoro": [_kokoro, _espeak], "espeak": [_espeak]}.get(engine, [])
    for fn in engines:
        try:
            audio = fn(text, voice, speed)
            peak = float(np.abs(audio).max()) if audio.size else 0
            return audio / peak * 0.89 if peak > 0 else audio
        except Exception as exc:  # noqa: BLE001 - fall through to the next engine
            if fn.__name__ not in _warned:
                _warned.add(fn.__name__)
                log(f"[voice] {fn.__name__.strip('_')} unavailable ({exc}); trying fallback")
    if "silence" not in _warned:
        _warned.add("silence")
        log("[voice] no TTS engine available - dialogue will be silent (subtitles still work). "
            "Install with: pip install kokoro soundfile  (and espeak-ng)")
    return silence_for(text, language)


def envelope(audio: np.ndarray, sr: int, fps: int, n_frames: int, offset_s: float = 0.0) -> np.ndarray:
    """Per-video-frame loudness 0..1 used to drive mouth flaps."""
    env = np.zeros(n_frames, np.float32)
    if audio.size == 0:
        return env
    hop = sr / fps
    start = int(round(offset_s * fps))
    for i in range(n_frames - start):
        a = int(i * hop)
        seg = audio[a:int(a + hop)]
        if seg.size == 0:
            break
        env[start + i] = float(np.sqrt(np.mean(seg ** 2)))
    peak = np.percentile(env[env > 0], 95) if np.any(env > 0) else 0
    return np.clip(env / peak, 0, 1) if peak > 0 else env


# ---------------------------------------------------------------------------- file io
def write_wav(path: str | Path, audio: np.ndarray, sr: int = SR) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def read_wav(path: str | Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32767
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels()).mean(1)
        if w.getframerate() != SR:
            data = resample(data, w.getframerate(), SR)
    return data


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or audio.size == 0:
        return audio
    n = int(len(audio) * sr_out / sr_in)
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


def decode_audio_bytes(data: bytes) -> np.ndarray:
    return _ffmpeg_decode(["-i", "pipe:0"], data)


def load_audio(path: str | Path) -> np.ndarray:
    """Decode any audio/video file (mp3, ogg, mp4...) to mono float32 at SR. Empty if no audio."""
    return _ffmpeg_decode(["-i", str(path)], None)


def _ffmpeg_decode(input_args: list[str], data: bytes | None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", *input_args, "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"]
    proc = subprocess.run(cmd, input=data, capture_output=True)
    if proc.returncode != 0:
        return np.zeros(0, np.float32)
    return np.frombuffer(proc.stdout, np.float32).copy()


# ---------------------------------------------------------------------------- mixing
def place(track: np.ndarray, clip: np.ndarray, start_s: float, gain: float = 1.0) -> None:
    a = int(start_s * SR)
    if a >= len(track) or clip.size == 0:
        return
    b = min(len(track), a + len(clip))
    track[a:b] += clip[:b - a] * gain


def duck(music: np.ndarray, voice: np.ndarray, amount: float = 0.35) -> np.ndarray:
    """Lower music under dialogue, with smooth attack/release."""
    win = SR // 20
    n = len(voice) // win + 1
    level = np.array([np.abs(voice[i * win:(i + 1) * win]).max() if voice[i * win:(i + 1) * win].size else 0
                      for i in range(n)])
    gate = (level > 0.02).astype(np.float32)
    gate = np.convolve(gate, np.ones(8) / 8, mode="same")  # ~0.4s smoothing
    gain = 1 - (1 - amount) * np.clip(gate * 1.5, 0, 1)
    return music * np.repeat(gain, win)[:len(music)]


def loop_to(clip: np.ndarray, n: int) -> np.ndarray:
    if clip.size == 0:
        return np.zeros(n, np.float32)
    reps = int(np.ceil(n / len(clip)))
    out = np.tile(clip, reps)[:n].copy()
    fade_n = min(n // 4, SR)
    if fade_n:
        out[-fade_n:] *= np.linspace(1, 0, fade_n)
        out[:SR // 4] *= np.linspace(0, 1, len(out[:SR // 4]))
    return out
