"""Tests for Muse tensor routing index generation."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from pipeline.muse.build_tensor_index import build_tensor_index


class BuildTensorIndexTest(unittest.TestCase):
    """Verify catalog records are routed only for completed tensors."""

    def test_indexes_existing_tensor_with_lora_metadata(self) -> None:
        """Include a final tensor and preserve all routing fields."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl.gz"
            tensors = root / "tensors"
            output = root / "index.jsonl"
            tensors.mkdir()
            (tensors / "song_0.pt").touch()
            record = {
                "audio_path": "songs/song_0.mp3",
                "primary_bucket": "rock",
                "candidate_buckets": ["rock", "pop"],
                "quality_tier": "a",
                "training_weight": 1.0,
                "vocal_profile": "female",
                "language": "en",
                "style": "Rock, Pop",
                "split": "train",
            }
            with gzip.open(catalog, "wt", encoding="utf-8") as target:
                target.write(json.dumps(record) + "\n")

            counts = build_tensor_index(catalog, tensors, output)
            row = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(counts, {"rock": 1})
            self.assertEqual(row["tensor"], "song_0.pt")
            self.assertEqual(row["candidate_buckets"], ["rock", "pop"])
            self.assertEqual(row["training_weight"], 1.0)

    def test_rejects_tensor_absent_from_catalog(self) -> None:
        """Raise when a completed tensor cannot be routed to a LoRA."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            tensors = root / "tensors"
            tensors.mkdir()
            catalog.write_text("", encoding="utf-8")
            (tensors / "unknown.pt").touch()

            with self.assertRaisesRegex(ValueError, "absent from catalog"):
                build_tensor_index(catalog, tensors, root / "index.jsonl")


if __name__ == "__main__":
    unittest.main()
