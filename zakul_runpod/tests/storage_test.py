"""Test bounded inline output and S3 publication with mocked cloud storage."""

import base64
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from zakul_runpod.storage import ResultStore

S3_ENV = {
    "ZAKUL_S3_BUCKET": "test-bucket", "ZAKUL_S3_ACCESS_KEY_ID": "test-key",
    "ZAKUL_S3_SECRET_ACCESS_KEY": "test-secret", "ZAKUL_S3_ENDPOINT_URL": "https://s3.example.test",
}


class StorageTests(unittest.TestCase):
    """No live bucket is contacted by this test suite."""

    def setUp(self):
        """Create one private test file."""
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "stream.mp3"
        self.path.write_bytes(b"test-mp3-bytes")

    def tearDown(self):
        """Release the owned temporary folder."""
        self.temporary.cleanup()

    def test_inline_is_self_contained_after_worker_files_disappear(self):
        store = ResultStore("inline", "job-id")
        output = store.publish({"mp3": self.path}, 1, 20, 42)
        self.assertEqual(base64.b64decode(output["mp3"]["base64"]), b"test-mp3-bytes")
        self.assertNotIn("url", output["mp3"])

    def test_total_inline_budget_covers_all_takes(self):
        store = ResultStore("inline", "job-id")
        with patch("zakul_runpod.storage.MAX_INLINE_BYTES", self.path.stat().st_size):
            store.publish({"mp3": self.path}, 1, 20, 42)
            with self.assertRaisesRegex(ValueError, "payload limit"):
                store.publish({"mp3": self.path}, 2, 20, 43)

    def test_flac_is_not_returned_as_giant_inline_payload(self):
        with self.assertRaises(ValueError):
            ResultStore("inline", "job-id").publish({"flac": self.path}, 1, 20, 42)

    def test_missing_s3_secrets_fail_during_preflight(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            ResultStore("s3", "job-id")

    def test_non_https_destination_is_rejected(self):
        with patch.dict(os.environ, {**S3_ENV, "ZAKUL_S3_ENDPOINT_URL": "http://example.test"}, clear=True):
            with self.assertRaises(ValueError):
                ResultStore("s3", "job-id")

    def test_s3_uploads_audio_before_publishing_url_and_keeps_key(self):
        client = MagicMock()
        client.generate_presigned_url.return_value = "https://s3.example.test/signed"
        with patch.dict(os.environ, S3_ENV, clear=True), patch("boto3.client", return_value=client):
            store = ResultStore("s3", "../../untrusted-job-id")
            output = store.publish({"mp3": self.path, "flac": self.path}, 1, 20, 42)
        self.assertEqual(client.upload_file.call_count, 2)
        self.assertEqual(client.method_calls[0][0], "upload_file")
        self.assertNotIn("..", output["mp3"]["key"])
        self.assertIn("key", output["flac"])
        self.assertNotIn("base64", output["mp3"])

    def test_failed_upload_never_returns_completed_audio(self):
        client = MagicMock()
        client.upload_file.side_effect = RuntimeError("storage unavailable")
        with patch.dict(os.environ, S3_ENV, clear=True), patch("boto3.client", return_value=client):
            store = ResultStore("s3", "job-id")
            with self.assertRaises(RuntimeError):
                store.publish({"mp3": self.path}, 1, 20, 42)
        client.generate_presigned_url.assert_not_called()

    def test_object_prefix_cannot_escape_namespace(self):
        with patch.dict(os.environ, {**S3_ENV, "ZAKUL_S3_PREFIX": "zakul/../other"}, clear=True):
            with self.assertRaises(ValueError):
                ResultStore("s3", "job-id")


if __name__ == "__main__":
    unittest.main()
