"""Bounded download of one private ZaKul vocal reference."""

from pathlib import Path
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .validation import GenerateRequest

MAX_REFERENCE_BYTES = 25 * 1024 * 1024
AUDIO_SUFFIXES = {
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/webm": ".webm",
    "audio/mp4": ".m4a",
    "audio/ogg": ".ogg",
}


class _NoRedirect(HTTPRedirectHandler):
    """Reject redirects so an approved ZaKul URL cannot become an SSRF hop."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Return no redirect request; urllib then raises HTTPError."""
        return None


def download_reference(
    request: GenerateRequest,
    folder: Path,
    opener=None,
) -> Path | None:
    """Download a bounded authenticated audio reference before GPU model loading."""
    if not request.reference_audio_url:
        return None
    client = opener or build_opener(_NoRedirect())
    http_request = Request(
        request.reference_audio_url,
        headers={
            "Authorization": f"Bearer {request.reference_audio_token}",
            "Accept": ", ".join(AUDIO_SUFFIXES),
            "User-Agent": "ZaKul-RunPod-Voice-Reference/1",
        },
        method="GET",
    )
    try:
        response = client.open(http_request, timeout=30)
    except HTTPError as exc:
        raise ValueError(f"Voice reference download returned HTTP {exc.code}") from exc
    with response:
        if getattr(response, "status", 200) != 200:
            raise ValueError("Voice reference download was not successful")
        content_type = response.headers.get_content_type()
        suffix = AUDIO_SUFFIXES.get(content_type)
        if not suffix:
            raise ValueError("Voice reference uses an unsupported audio format")
        declared = int(response.headers.get("Content-Length", "0") or 0)
        if declared > MAX_REFERENCE_BYTES:
            raise ValueError("Voice reference exceeds the 25 MB limit")
        target = folder / f"voice-reference{suffix}"
        total = 0
        with target.open("wb") as output:
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_REFERENCE_BYTES:
                    target.unlink(missing_ok=True)
                    raise ValueError("Voice reference exceeds the 25 MB limit")
                output.write(chunk)
        if total < 1024:
            target.unlink(missing_ok=True)
            raise ValueError("Voice reference is empty or too small")
        return target
