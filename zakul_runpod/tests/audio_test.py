"""Real CPU FFmpeg tests for sample-length trimming and valid encoders."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from zakul_runpod.audio import duration_seconds, encode_take, run_audio_tool


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class AudioTests(unittest.TestCase):
    """Exercise actual conversion without model downloads or a GPU."""

    def setUp(self):
        """Create an owned synthetic sine wave, not a generated or licensed song."""
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.source = self.folder / "source.wav"
        run_audio_tool([
            "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
            "-i", "sine=frequency=100:duration=1.5", "-ar", "48000", str(self.source),
        ])

    def tearDown(self):
        """Delete only the test-owned temporary directory."""
        self.temporary.cleanup()

    def test_exact_flac_and_playable_mp3(self):
        assets = encode_take(self.source, self.folder, 0.75, True)
        self.assertAlmostEqual(duration_seconds(assets["flac"]), 0.75, places=3)
        self.assertLess(abs(duration_seconds(assets["mp3"]) - 0.75), 0.06)
        self.assertGreater(assets["mp3"].stat().st_size, 100)

    def test_inline_does_not_create_flac(self):
        self.assertEqual(set(encode_take(self.source, self.folder, 1.0, False)), {"mp3"})

    def test_short_generated_source_is_error_not_silence_padding(self):
        with self.assertRaisesRegex(RuntimeError, "only"):
            encode_take(self.source, self.folder, 2.0, True)

    def test_invalid_audio_is_rejected(self):
        invalid = self.folder / "invalid.wav"
        invalid.write_bytes(b"not audio")
        with self.assertRaises(RuntimeError):
            duration_seconds(invalid)

    def test_missing_tool_has_actionable_error(self):
        with patch("zakul_runpod.audio.subprocess.run", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(RuntimeError, "missing"):
                duration_seconds(self.source)


if __name__ == "__main__":
    unittest.main()
