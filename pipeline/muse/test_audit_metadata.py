"""Tests for the Muse metadata audit."""

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.muse.audit_metadata import audit_files, normalize_tag


class AuditMetadataTest(unittest.TestCase):
    """Verify Muse metadata auditing behavior."""

    def test_audits_valid_records(self) -> None:
        """Summarize valid rows, tags, duplicates, and duration."""
        record = {
            "audio_path": "songs/example.mp3",
            "sections": [{"endS": 12.5}, {"endS": 30.0}],
            "song_id": "example",
            "style": " Pop, Female   Vocal ",
            "style_sim": 0.75,
            "track_index": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(
                json.dumps(record) + "\n" + json.dumps(record) + "\n",
                encoding="utf-8",
            )

            summary = audit_files([path])

        self.assertEqual(summary["totals"]["rows"], 2)
        self.assertEqual(summary["unique_audio_paths"], 1)
        self.assertEqual(summary["duplicate_audio_rows"], 1)
        self.assertEqual(summary["top_style_tags"]["pop"], 2)
        self.assertEqual(summary["duration"]["max_seconds"], 30.0)
        self.assertEqual(summary["style_sim_mean"], 0.75)

    def test_reports_invalid_and_incomplete_records(self) -> None:
        """Retain counts for invalid JSON and missing required fields."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text('{invalid}\n{"song_id": "partial"}\n', encoding="utf-8")

            summary = audit_files([path])

        self.assertEqual(summary["totals"]["rows"], 2)
        self.assertEqual(summary["totals"]["invalid_json"], 1)
        self.assertEqual(summary["totals"]["missing_required"], 1)
        self.assertEqual(summary["missing_fields"]["audio_path"], 1)
        self.assertEqual(len(summary["error_samples"]), 1)

    def test_normalizes_style_tags(self) -> None:
        """Normalize tag case and repeated whitespace."""
        self.assertEqual(normalize_tag("  Female   Vocal "), "female vocal")


if __name__ == "__main__":
    unittest.main()
