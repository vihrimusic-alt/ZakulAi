"""Tests for the clean Muse catalog builder."""

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.muse.build_catalog import (
    build_catalog,
    candidate_buckets,
    quality_tier,
    split_for_song,
)


class BuildCatalogTest(unittest.TestCase):
    """Verify catalog filtering, classification, and splitting."""

    def test_classifies_and_writes_clean_record(self) -> None:
        """Include a valid row with deterministic metadata."""
        record = {
            "audio_path": "songs/example.mp3",
            "sections": [{"endS": 180.0}],
            "song_id": "example",
            "style": "Pop, Electronic, Female Vocal",
            "style_sim": 0.7,
            "track_index": 0,
        }
        taxonomy = {"pop": {"pop"}, "electronic": {"electronic"}}

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train_en.jsonl"
            output = Path(directory) / "catalog.jsonl"
            source.write_text(json.dumps(record) + "\n", encoding="utf-8")

            summary = build_catalog([source], taxonomy, output)
            catalog = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(summary["counts"]["included_rows"], 1)
        self.assertEqual(catalog["candidate_buckets"], ["pop", "electronic"])
        self.assertEqual(catalog["quality_tier"], "a")
        self.assertEqual(catalog["language"], "en")

    def test_excludes_source_error(self) -> None:
        """Exclude a row carrying a source error."""
        record = {
            "audio_path": "songs/broken.mp3",
            "error": "generation failed",
            "sections": [],
            "song_id": "broken",
            "style": "Pop",
            "style_sim": None,
            "track_index": 0,
        }

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train_cn.jsonl"
            output = Path(directory) / "catalog.jsonl"
            source.write_text(json.dumps(record) + "\n", encoding="utf-8")

            summary = build_catalog([source], {"pop": {"pop"}}, output)

        self.assertEqual(summary["counts"]["excluded_source_error"], 1)
        self.assertEqual(output.read_text(encoding="utf-8"), "")

    def test_song_split_is_stable(self) -> None:
        """Keep all track variants for the same song in one split."""
        self.assertEqual(split_for_song("same-song"), split_for_song("same-song"))

    def test_quality_tiers(self) -> None:
        """Assign quality tiers at configured boundaries."""
        self.assertEqual(quality_tier(0.65), "a")
        self.assertEqual(quality_tier(0.50), "b")
        self.assertEqual(quality_tier(0.49), "c")

    def test_candidate_bucket_fallback(self) -> None:
        """Return no candidates when taxonomy has no matching tags."""
        self.assertEqual(candidate_buckets("Unknown", {"pop": {"pop"}}), [])


if __name__ == "__main__":
    unittest.main()
