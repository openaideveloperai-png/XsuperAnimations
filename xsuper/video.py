"""ffmpeg helpers: frame writer, inserting existing clips (OP/ED), final mux."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np


class FFmpegError(RuntimeError):
    pass


def require_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise FFmpegError("ffmpeg is not installed. Install it (e.g. 'apt install ffmpeg', 'brew install ffmpeg', "
                          "or 'winget install ffmpeg') and try again.")


ENCODE = {"preset": "faster", "crf": 18}


def configure(preset: str = "faster", crf: int = 18) -> None:
    """x264 settings for every shot segment (the final file is a lossless concat of them)."""
    ENCODE.update(preset=preset, crf=crf)


def _x264_args(fps: int) -> list[str]:
    return ["-c:v", "libx264", "-preset", ENCODE["preset"], "-tune", "animation", "-crf", str(ENCODE["crf"]),
            "-pix_fmt", "yuv420p", "-r", str(fps), "-video_track_timescale", str(fps * 1000)]


class VideoWriter:
    """Pipe raw RGB frames into ffmpeg. Writes to a temp file and renames on success."""

    def __init__(self, path: Path, width: int, height: int, fps: int):
        self.path = Path(path)
        self.tmp = self.path.with_suffix(".tmp.mp4")
        self.size = (width, height)
        self.frames = 0
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-r", str(fps), "-i", "pipe:0", "-an", *_x264_args(fps), str(self.tmp)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1::-1] != self.size:
            raise ValueError(f"frame size {frame.shape[1::-1]} != {self.size}")
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame, np.uint8).tobytes())
        except BrokenPipeError:
            raise FFmpegError("ffmpeg stopped unexpectedly: " + self.proc.stderr.read().decode(errors="ignore")) from None
        self.frames += 1

    def close(self) -> None:
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="ignore")
        if self.proc.wait() != 0:
            raise FFmpegError(f"ffmpeg failed writing {self.path.name}: {err}")
        os.replace(self.tmp, self.path)

    def abort(self) -> None:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        self.proc.kill()
        self.tmp.unlink(missing_ok=True)


def count_frames(path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True)
    try:
        return int(out.stdout.strip().split(",")[0])
    except ValueError:
        raise FFmpegError(f"can't read video {path}: {out.stderr.strip()}") from None


def conform_clip(src: Path, dst: Path, width: int, height: int, fps: int) -> int:
    """Re-encode an existing video (e.g. your OP/ED) to the episode format. Returns frame count."""
    vf = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
          f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={fps},format=yuv420p")
    tmp = dst.with_suffix(".tmp.mp4")
    proc = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", vf, "-an", *_x264_args(fps), str(tmp)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"couldn't convert {src.name}: {proc.stderr.strip()[-400:]}")
    os.replace(tmp, dst)
    return count_frames(dst)


def mux(segments: list[Path], audio_wav: Path, srt: Path | None, out: Path) -> None:
    list_file = out.with_suffix(".concat.txt")
    def quote(p: Path) -> str:
        return "'" + str(p.resolve()).replace("'", "'\\''") + "'"

    list_file.write_text("".join(f"file {quote(p)}\n" for p in segments), encoding="utf-8")
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(list_file), "-i", str(audio_wav)]
    maps = ["-map", "0:v", "-map", "1:a"]
    if srt is not None and srt.exists() and srt.stat().st_size > 0:
        cmd += ["-i", str(srt)]
        maps += ["-map", "2:s", "-c:s", "mov_text", "-metadata:s:s:0", "language=eng"]
    tmp = out.with_suffix(".tmp.mp4")
    cmd += [*maps, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart", str(tmp)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    list_file.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise FFmpegError(f"final mux failed: {proc.stderr.strip()[-600:]}")
    os.replace(tmp, out)
