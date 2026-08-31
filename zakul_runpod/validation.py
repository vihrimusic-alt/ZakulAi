"""Strict request validation before any paid model work or downloads."""

import math
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

FIELDS = {
    "operation", "prompt", "lyrics", "instrumental", "duration_seconds", "requested_outputs",
    "max_duration_seconds", "seed", "thinking", "bpm", "keyscale", "vocal_language", "output_mode",
    "reference_audio_url", "reference_audio_token",
}
OPERATIONS = {"health", "warmup", "generate"}


def number(value: Any, name: str, low: float, high: float) -> float:
    """Reject booleans, strings, NaNs and values outside explicit bounds."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a JSON number")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return float(value)


def boolean(value: Any, name: str) -> bool:
    """Require a JSON boolean, not truthy text such as 'false'."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a JSON boolean")
    return value


def text(value: Any, name: str, limit: int) -> str:
    """Validate text without silently truncating a musical instruction."""
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"{name} must be text of at most {limit} characters")
    return value.strip()


@dataclass(frozen=True)
class GenerateRequest:
    """Normalized request; short clips are generated at 10s then trimmed."""

    prompt: str
    lyrics: str
    instrumental: bool
    duration: float | None
    outputs: int
    seed: int
    thinking: bool
    bpm: int | None
    keyscale: str
    language: str
    output_mode: str
    max_duration: float = 240
    reference_audio_url: str = ""
    reference_audio_token: str = ""


def validate_job(job: Any) -> tuple[str, dict]:
    """Validate the RunPod envelope and operation without any GPU imports."""
    if not isinstance(job, dict) or not isinstance(job.get("input"), dict):
        raise ValueError("Expected a RunPod job with an input object")
    data = job["input"]
    if data.get("operation") == "assist":
        if set(data) - {"operation","action","query","vocal_language","instrumental"}: raise ValueError("Unknown assistant fields")
        if data.get("action") not in {"lyrics","style"}: raise ValueError("Invalid assistant action")
        if not text(data.get("query"),"query",12000): raise ValueError("query is required")
        language=text(data.get("vocal_language","unknown"),"vocal_language",12)
        if language!="unknown" and not (len(language) in {2,3} and language.isascii() and language.isalpha() and language.islower()): raise ValueError("Invalid language")
        boolean(data.get("instrumental",False),"instrumental")
        return "assist",data
    unknown = set(data) - FIELDS
    if unknown:
        raise ValueError("Unknown input fields: " + ", ".join(sorted(unknown)))
    operation = data.get("operation", "generate")
    if not isinstance(operation, str) or operation not in OPERATIONS:
        raise ValueError("operation must be health, warmup or generate")
    if operation != "generate" and set(data) - {"operation"}:
        raise ValueError("health/warmup accept only operation; use generate for audio")
    return operation, data


def parse_generate(data: dict) -> GenerateRequest:
    """Require an explicit duration; reject unsupported requests before generation."""
    prompt = text(data.get("prompt", ""), "prompt", 2400)
    if not prompt:
        raise ValueError("prompt is required")
    lyrics = text(data.get("lyrics", ""), "lyrics", 4096)
    instrumental = boolean(data.get("instrumental", False), "instrumental")
    if instrumental and lyrics and lyrics != "[Instrumental]":
        raise ValueError("Remove lyrics for an instrumental request")
    if not instrumental and (not lyrics or lyrics == "[Instrumental]"):
        raise ValueError("Provide lyrics, or set instrumental=true")
    raw_duration = data.get("duration_seconds")
    duration = None if raw_duration == "auto" else number(raw_duration, "duration_seconds", 0.1, 240)
    outputs = number(data.get("requested_outputs", 1), "requested_outputs", 1, 2)
    seed = number(data.get("seed", -1), "seed", -1, 2**31 - 2)
    if outputs != int(outputs) or seed != int(seed):
        raise ValueError("requested_outputs and seed must be integers")
    mode = data.get("output_mode", "inline")
    if not isinstance(mode, str) or mode not in {"inline", "s3"}:
        raise ValueError("output_mode must be inline or s3")
    if duration is None and (mode != "s3" or not boolean(data.get("thinking", True), "thinking")):
        raise ValueError("Auto duration requires S3 output and thinking=true")
    if mode == "inline" and duration is not None and duration > 60:
        raise ValueError("Inline tests are limited to 60s per take; configure S3 for longer songs")
    bpm = data.get("bpm")
    if bpm is not None:
        bpm_value = number(bpm, "bpm", 30, 300)
        if bpm_value != int(bpm_value):
            raise ValueError("bpm must be an integer")
        bpm = int(bpm_value)
    language = text(data.get("vocal_language", "unknown"), "vocal_language", 12)
    if language != "unknown" and not (len(language) in {2, 3} and language.isascii()
                                      and language.isalpha() and language.islower()):
        raise ValueError("vocal_language must be a lowercase language code or unknown")
    reference_url = text(data.get("reference_audio_url", ""), "reference_audio_url", 500)
    reference_token = text(data.get("reference_audio_token", ""), "reference_audio_token", 64)
    if bool(reference_url) != bool(reference_token):
        raise ValueError("reference audio URL and token must be provided together")
    if reference_url:
        parsed = urlsplit(reference_url)
        if (parsed.scheme != "https" or parsed.hostname != "zakul-ai.com"
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or not parsed.path.startswith("/api/voice-reference/")):
            raise ValueError("reference_audio_url must be a ZaKul HTTPS reference endpoint")
        if len(reference_token) != 64 or any(char not in "0123456789abcdef" for char in reference_token):
            raise ValueError("reference_audio_token must be 64 lowercase hexadecimal characters")
    return GenerateRequest(
        prompt, "[Instrumental]" if instrumental else lyrics, instrumental, duration,
        int(outputs), int(seed), boolean(data.get("thinking", True), "thinking"), bpm,
        text(data.get("keyscale", ""), "keyscale", 30), language, mode,
        number(data.get("max_duration_seconds", 240), "max_duration_seconds", 10, 240),
        reference_url, reference_token,
    )
