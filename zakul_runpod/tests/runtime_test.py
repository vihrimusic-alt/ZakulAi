"""Queue lifecycle and validation-order tests with mocked model inference."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from zakul_runpod.config import Settings
from zakul_runpod.runtime import QueueWorker


class RuntimeTests(unittest.TestCase):
    """Keep paid GPU activity and local cleanup boundaries observable."""

    def setUp(self):
        """Create test-owned settings and fake inference while using real publication code."""
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.settings = Settings(root, root / "checkpoints", root / "tmp",
                                 "acestep-v15-turbo", "acestep-5Hz-lm-0.6B", False, False)
        self.settings.temporary.mkdir()
        self.worker = QueueWorker(self.settings)
        self.worker.models = MagicMock()
        self.worker.models.health.return_value = {"cuda_available": True, "models_initialized": False}
        self.worker.models.generate.side_effect = self._fake_generate
        self.progress_patch = patch("zakul_runpod.runtime.report_progress")
        self.progress_patch.start()

    def tearDown(self):
        """Close patches and clean only owned test data."""
        self.progress_patch.stop()
        self.temporary.cleanup()

    @staticmethod
    def _fake_generate(request, seed, folder):
        path = folder / "source.wav"
        path.write_bytes(b"placeholder-wav")
        return path

    @staticmethod
    def _fake_encode(source, folder, target, lossless):
        path = folder / "stream.mp3"
        path.write_bytes(b"mp3-bytes")
        return {"mp3": path}

    def _job(self, **changes):
        return {"id": "queue-job-id", "input": {
            "prompt": "Sopilka and bass", "instrumental": True, "duration_seconds": 20,
            "seed": 42, **changes,
        }}

    def test_health_does_not_load_models(self):
        output = self.worker.handle({"input": {"operation": "health"}})
        self.assertFalse(output["models_initialized"])
        self.worker.models.ensure_loaded.assert_not_called()

    def test_warmup_explicitly_loads_lm(self):
        self.worker.handle({"input": {"operation": "warmup"}})
        self.assertTrue(self.worker.models.ensure_loaded.call_args.args[0])

    def test_bad_duration_fails_before_model_download(self):
        with self.assertRaises(ValueError):
            self.worker.handle(self._job(duration_seconds=-1))
        self.worker.models.ensure_loaded.assert_not_called()

    def test_missing_storage_config_fails_before_gpu_work(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            self.worker.handle(self._job(output_mode="s3", duration_seconds=180))
        self.worker.models.ensure_loaded.assert_not_called()

    def test_two_takes_are_sequential_and_have_distinct_explicit_seeds(self):
        with patch("zakul_runpod.runtime.encode_take", side_effect=self._fake_encode):
            output = self.worker.handle(self._job(requested_outputs=2))
        calls = self.worker.models.generate.call_args_list
        self.assertEqual([call.args[1] for call in calls], [42, 43])
        self.assertEqual(len(output["tracks"]), 2)
        self.assertEqual(output["inference_steps"], 8)
        self.assertIn("base64", output["tracks"][0]["mp3"])
        self.assertEqual(list(self.settings.temporary.iterdir()), [])

    def test_clip_target_is_not_changed_to_engine_minimum(self):
        with patch("zakul_runpod.runtime.encode_take", side_effect=self._fake_encode) as encode:
            output = self.worker.handle(self._job(duration_seconds=2.12, thinking=False))
        self.assertEqual(encode.call_args.args[2], 2.12)
        self.assertEqual(output["tracks"][0]["duration_seconds"], 2.12)
        self.assertIsNone(output["lm_model"])
        self.assertFalse(self.worker.models.ensure_loaded.call_args.args[0])

    def test_failed_conversion_never_claims_success_and_cleans_temp(self):
        with patch("zakul_runpod.runtime.encode_take", side_effect=RuntimeError("FFmpeg failed")):
            with self.assertRaises(RuntimeError):
                self.worker.handle(self._job())
        self.assertEqual(list(self.settings.temporary.iterdir()), [])

    def test_model_failure_is_not_reported_as_completed(self):
        self.worker.models.ensure_loaded.side_effect = RuntimeError("CUDA unavailable")
        with self.assertRaises(RuntimeError):
            self.worker.handle(self._job())
        self.worker.models.generate.assert_not_called()

    def test_unknown_studio_operation_is_rejected_not_faked(self):
        with self.assertRaises(ValueError):
            self.worker.handle({"input": {"operation": "remix"}})
        self.worker.models.ensure_loaded.assert_not_called()

    def test_checkpoint_files_survive_job_cleanup(self):
        self.settings.checkpoints.mkdir()
        checkpoint = self.settings.checkpoints / "model.safetensors"
        checkpoint.write_bytes(b"keep")
        with patch("zakul_runpod.runtime.encode_take", side_effect=self._fake_encode):
            self.worker.handle(self._job())
        self.assertEqual(checkpoint.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
