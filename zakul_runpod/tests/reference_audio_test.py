"""Tests for bounded authenticated private voice-reference downloads."""

from email.message import Message
from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from zakul_runpod.reference_audio import MAX_REFERENCE_BYTES, download_reference
from zakul_runpod.validation import parse_generate


class FakeResponse(BytesIO):
    """Provide the small urllib response surface used by the downloader."""

    def __init__(self, body: bytes, content_type: str = "audio/mpeg", declared: int | None = None):
        super().__init__(body)
        self.status = 200
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        self.headers["Content-Length"] = str(len(body) if declared is None else declared)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class FakeOpener:
    """Return one response and retain the authenticated request for assertions."""

    def __init__(self, response):
        self.response = response
        self.request = None
        self.timeout = None

    def open(self, request, timeout):
        self.request = request
        self.timeout = timeout
        return self.response


def voice_request():
    """Build a valid request containing a private ZaKul reference endpoint."""
    return parse_generate({
        "prompt": "Warm intimate vocal", "lyrics": "Sing these original lyrics",
        "instrumental": False, "duration_seconds": 20,
        "reference_audio_url": "https://zakul-ai.com/api/voice-reference/job-1",
        "reference_audio_token": "a" * 64,
    })


class ReferenceAudioTests(unittest.TestCase):
    """Only bounded MP3 data from the authenticated endpoint may reach inference."""

    def test_success_downloads_mp3_with_bearer_token(self):
        opener = FakeOpener(FakeResponse(b"x" * 2048))
        with tempfile.TemporaryDirectory() as temporary:
            result = download_reference(voice_request(), Path(temporary), opener)
            self.assertEqual(result.read_bytes(), b"x" * 2048)
        self.assertEqual(opener.request.get_header("Authorization"), "Bearer " + "a" * 64)
        self.assertEqual(opener.timeout, 30)

    def test_request_without_reference_does_not_open_network(self):
        request = parse_generate({
            "prompt": "Instrumental", "instrumental": True, "duration_seconds": 20,
        })
        opener = FakeOpener(FakeResponse(b"x" * 2048))
        with tempfile.TemporaryDirectory() as temporary:
            self.assertIsNone(download_reference(request, Path(temporary), opener))
        self.assertIsNone(opener.request)

    def test_wrong_content_type_is_rejected(self):
        opener = FakeOpener(FakeResponse(b"x" * 2048, "text/html"))
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "MP3"):
            download_reference(voice_request(), Path(temporary), opener)

    def test_declared_oversize_is_rejected_before_writing(self):
        opener = FakeOpener(FakeResponse(b"x" * 2048, declared=MAX_REFERENCE_BYTES + 1))
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "25 MB"):
            download_reference(voice_request(), Path(temporary), opener)

    def test_too_small_body_is_rejected_and_removed(self):
        opener = FakeOpener(FakeResponse(b"x" * 100))
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "too small"):
            download_reference(voice_request(), Path(temporary), opener)
            self.assertFalse((Path(temporary) / "voice-reference.mp3").exists())


if __name__ == "__main__":
    unittest.main()
