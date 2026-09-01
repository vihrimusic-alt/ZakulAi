"""Tests for Muse archive dataset preparation."""

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from pipeline.muse.prepare_archive import lyrics_text, prepare_archive


class PrepareArchiveTest(unittest.TestCase):
    """Verify safe, bucketed ACE-Step dataset preparation."""

    def _write_archive(self, path: Path, name: str = "song.mp3") -> None:
        """Write a tiny TAR containing one fake audio file."""
        payload = b"audio"
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        with tarfile.open(path, "w") as archive:
            archive.addfile(info, io.BytesIO(payload))

    def _write_catalog(self, path: Path) -> None:
        """Write one representative assigned catalog record."""
        record = {
            "audio_path": "songs/song.mp3",
            "primary_bucket": "pop",
            "quality_tier": "a",
            "split": "train",
            "style": "Pop, Female Vocal",
            "language": "cn",
            "sections": [
                {"section": "Intro", "text": ""},
                {"section": "Verse 1", "text": "第一行"},
            ],
        }
        path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    def test_prepares_audio_and_sidecars(self) -> None:
        """Extract a selected track with lyrics and annotation files."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "part.tar"
            catalog = root / "catalog.jsonl"
            output = root / "dataset"
            self._write_archive(archive)
            self._write_catalog(catalog)

            summary = prepare_archive(
                archive,
                catalog,
                output,
                {"train"},
                {"a", "b"},
            )

            self.assertEqual(summary["included"], 1)
            self.assertEqual((output / "pop/song.mp3").read_bytes(), b"audio")
            self.assertEqual(
                (output / "pop/song.lyrics.txt").read_text(encoding="utf-8"),
                "[Verse 1]\n第一行\n",
            )
            annotation = json.loads(
                (output / "pop/song.json").read_text(encoding="utf-8")
            )
            self.assertEqual(annotation["caption"], "Pop, Female Vocal")

    def test_rejects_unsafe_archive_member(self) -> None:
        """Reject path traversal before writing archive contents."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "part.tar"
            catalog = root / "catalog.jsonl"
            self._write_archive(archive, "../song.mp3")
            self._write_catalog(catalog)

            with self.assertRaisesRegex(ValueError, "Unsafe archive member"):
                prepare_archive(
                    archive,
                    catalog,
                    root / "dataset",
                    {"train"},
                    {"a"},
                )

    def test_formats_only_nonempty_sections(self) -> None:
        """Omit instrumental sections from lyric sidecars."""
        sections = [
            {"section": "Intro", "text": ""},
            {"section": "Chorus", "text": "Hello"},
        ]
        self.assertEqual(lyrics_text(sections), "[Chorus]\nHello\n")


if __name__ == "__main__":
    unittest.main()
