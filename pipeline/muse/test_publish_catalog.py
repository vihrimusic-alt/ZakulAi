"""Tests for Muse catalog publication helpers."""

import gzip
import tempfile
import unittest
from pathlib import Path

from pipeline.muse.publish_catalog import (
    compress_file,
    required_environment,
    sha256,
    write_checksums,
)


class PublishCatalogTest(unittest.TestCase):
    """Verify safe publication preparation."""

    def test_sanitizes_environment(self) -> None:
        """Strip accidental whitespace from RunPod secrets."""
        config = required_environment(
            {
                "R2_ACCESS_KEY_ID": " access\n",
                "R2_SECRET_ACCESS_KEY": " secret\r\n",
                "R2_ENDPOINT_URL": " https://example.invalid ",
                "R2_BUCKET": " bucket ",
            }
        )
        self.assertEqual(config["R2_ACCESS_KEY_ID"], "access")
        self.assertEqual(config["R2_SECRET_ACCESS_KEY"], "secret")

    def test_rejects_missing_environment(self) -> None:
        """Fail before upload when required configuration is absent."""
        with self.assertRaises(ValueError):
            required_environment({})

    def test_compresses_and_checksums_artifact(self) -> None:
        """Create readable deterministic gzip and checksum files."""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "catalog.jsonl"
            compressed = Path(directory) / "catalog.jsonl.gz"
            checksums = Path(directory) / "SHA256SUMS"
            source.write_bytes(b'{"song_id":"example"}\n')

            compress_file(source, compressed)
            write_checksums([compressed], checksums)

            with gzip.open(compressed, "rb") as archive:
                restored = archive.read()

            self.assertEqual(restored, source.read_bytes())
            self.assertIn(sha256(compressed), checksums.read_text(encoding="utf-8"))
            self.assertIn(compressed.name, checksums.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
