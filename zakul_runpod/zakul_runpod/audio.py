"""Bounded FFmpeg conversion, exact clip trimming, and basic audio validation."""

import json
import math
import subprocess
from pathlib import Path


def run_audio_tool(command: list[str]) -> str:
    """Run a fixed audio tool without a shell or user-supplied command fragments."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg/FFprobe is missing from the container") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Audio conversion exceeded 180 seconds") from exc
    if result.returncode:
        raise RuntimeError(f"Audio tool failed: {result.stderr[-800:]}")
    return result.stdout


def duration_seconds(path: Path) -> float:
    """Measure an audio file and reject empty, non-audio, or non-finite duration."""
    data = json.loads(run_audio_tool([
        "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
        "stream=codec_type:format=duration", "-of", "json", str(path),
    ]))
    if not data.get("streams"):
        raise RuntimeError("Generated file contains no audio stream")
    duration = float(data.get("format", {}).get("duration", 0))
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("Generated file has no measurable audio duration")
    return duration


def encode_take(source: Path, folder: Path, target: float | None, lossless: bool) -> dict[str, Path]:
    """Trim to the requested length and create MP3, plus FLAC for durable storage."""
    available = duration_seconds(source)
    automatic = target is None
    if automatic:
        if not 10 <= available <= 240.5:
            raise RuntimeError("Automatic audio is outside the worker duration limit; not truncating")
        target = available
    if available + 0.05 < target:
        raise RuntimeError(f"ACE-Step generated only {available:.2f}s; requested {target:.2f}s")
    common = [
        "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
        "-map", "0:a:0", "-vn", *([] if automatic else ["-t", str(target)]), "-ac", "2", "-ar", "48000",
    ]
    mp3 = folder / "stream.mp3"
    run_audio_tool(common + ["-c:a", "libmp3lame", "-b:a", "320k", str(mp3)])
    assets = {"mp3": mp3}
    if lossless:
        flac = folder / "master.flac"
        run_audio_tool(common + ["-c:a", "flac", "-compression_level", "8", str(flac)])
        if abs(duration_seconds(flac) - target) > 0.025:
            raise RuntimeError("Lossless output duration differs from the requested clip length")
        assets["flac"] = flac
    if any(not path.is_file() or path.stat().st_size <= 0 for path in assets.values()):
        raise RuntimeError("An audio encoder returned an empty file")
    return assets
