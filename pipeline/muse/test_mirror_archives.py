"""Tests for resumable Muse archive mirroring."""

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.muse.mirror_archives import archive_names, write_manifest


class MirrorArchivesTest(unittest.TestCase):
    """Verify archive planning and durable progress state."""

    def test_lists_all_archives(self) -> None:
        """Generate the exact 25 CN and 35 EN filenames."""
        names = archive_names()
        self.assertEqual(len(names), 60)
        self.assertEqual(names[0], "cn_part01_of_25.tar")
        self.assertEqual(names[24], "cn_part25_of_25.tar")
        self.assertEqual(names[25], "en_part01_of_35.tar")
        self.assertEqual(names[-1], "en_part35_of_35.tar")
        self.assertEqual(len(set(names)), 60)

    def test_writes_manifest_atomically(self) -> None:
        """Persist progress as valid JSON without leaving a temp file."""
        entries = {
            "cn_part01_of_25.tar": {
                "sha256": "abc",
                "size": 123,
                "status": "verified_remote",
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive_manifest.json"
            write_manifest(entries, path)
            restored = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(restored, entries)
            self.assertFalse(path.with_suffix(".tmp").exists())


if __name__ == "__main__":
    unittest.main()
