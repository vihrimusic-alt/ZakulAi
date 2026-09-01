"""Tests for Muse primary bucket assignment."""

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from pipeline.muse.assign_catalog import (
    assign_catalog,
    choose_primary,
    vocal_profile,
)


class AssignCatalogTest(unittest.TestCase):
    """Verify deterministic rarity and vocal assignments."""

    def test_chooses_rarest_candidate(self) -> None:
        """Prefer underrepresented candidate buckets."""
        counts = Counter({"pop": 100, "electronic": 20})
        self.assertEqual(choose_primary(["pop", "electronic"], counts), "electronic")

    def test_detects_vocal_profiles(self) -> None:
        """Recognize vocal types from exact normalized style tags."""
        self.assertEqual(vocal_profile("Pop, Female Vocal"), "female")
        self.assertEqual(vocal_profile("Rock, Male Vocal, Female Vocal"), "duet_choir")
        self.assertEqual(vocal_profile("Ambient, Instrumental"), "instrumental")
        self.assertEqual(vocal_profile("Classical"), "unspecified")

    def test_assigns_catalog_and_weights(self) -> None:
        """Write assignments and summarize their distributions."""
        records = [
            {
                "candidate_buckets": ["pop", "rock"],
                "quality_tier": "a",
                "style": "Pop, Male Vocal",
            },
            {
                "candidate_buckets": ["pop"],
                "quality_tier": "c",
                "style": "Pop, Instrumental",
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "catalog.jsonl"
            output = Path(directory) / "assigned.jsonl"
            source.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            summary = assign_catalog(source, output)
            assigned = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(summary["rows"], 2)
        self.assertEqual(assigned[0]["primary_bucket"], "rock")
        self.assertEqual(assigned[0]["training_weight"], 1.0)
        self.assertEqual(assigned[1]["training_weight"], 0.35)


if __name__ == "__main__":
    unittest.main()
