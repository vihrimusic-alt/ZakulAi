"""Check model loading and inference contracts without importing PyTorch."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from zakul_runpod.config import Settings
from zakul_runpod.models import Models
from zakul_runpod.validation import parse_generate


class ModelTests(unittest.TestCase):
    """Mock only upstream boundary objects; exercise the adapter's real logic."""

    def setUp(self):
        """Set up a fake CUDA device and fake handlers with the actual public signatures."""
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.settings = Settings(root, root / "checkpoints", root / "tmp",
                                 "acestep-v15-turbo", "acestep-5Hz-lm-0.6B", False, False)
        self.models = Models(self.settings)
        self.torch = MagicMock()
        self.torch.cuda.is_available.return_value = True
        self.torch.cuda.get_device_properties.return_value.total_memory = 24 * 2**30
        self.dit = MagicMock()
        self.dit.initialize_service.return_value = ("loaded", True)
        self.lm = MagicMock()
        self.lm.initialize.return_value = ("loaded", True)
        self.loader = SimpleNamespace(ensure_lm_model=MagicMock(return_value=(True, "ok")),
                                      ensure_main_model=MagicMock(return_value=(True, "ok")))
        self.modules = {
            "torch": self.torch,
            "acestep.gpu_config": SimpleNamespace(get_gpu_config=MagicMock(), set_global_gpu_config=MagicMock()),
            "acestep.handler": SimpleNamespace(AceStepHandler=MagicMock(return_value=self.dit)),
            "acestep.llm_inference": SimpleNamespace(LLMHandler=MagicMock(return_value=self.lm)),
            "acestep.model_downloader": self.loader,
        }
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()

    def tearDown(self):
        """Restore environment and temporary folders."""
        self.env_patch.stop()
        self.temporary.cleanup()

    def test_no_cuda_never_silently_generates_on_cpu(self):
        self.torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, self.modules), self.assertRaisesRegex(RuntimeError, "CUDA"):
            self.models.ensure_loaded(True, MagicMock())
        self.dit.initialize_service.assert_not_called()

    def test_loaded_models_are_reused_across_requests(self):
        with patch.dict(sys.modules, self.modules):
            self.models.ensure_loaded(True, MagicMock())
            self.models.ensure_loaded(True, MagicMock())
        self.dit.initialize_service.assert_called_once()
        self.lm.initialize.assert_called_once()
        self.assertEqual(self.lm.initialize.call_args.kwargs["backend"], "pt")

    def test_thinking_false_avoids_lm_load(self):
        with patch.dict(sys.modules, self.modules):
            self.models.ensure_loaded(False, MagicMock())
        self.lm.initialize.assert_not_called()

    def test_failed_dit_load_not_cached_as_ready(self):
        self.dit.initialize_service.return_value = ("OOM", False)
        with patch.dict(sys.modules, self.modules), self.assertRaises(RuntimeError):
            self.models.ensure_loaded(True, MagicMock())
        self.assertIsNone(self.models.dit)

    def test_failed_lm_load_is_error_not_silent_downgrade(self):
        self.lm.initialize.return_value = ("OOM", False)
        with patch.dict(sys.modules, self.modules), self.assertRaises(RuntimeError):
            self.models.ensure_loaded(True, MagicMock())
        self.assertIsNone(self.models.lm)

    def test_short_duration_lyrics_and_reference_map_to_upstream_fields(self):
        root = self.settings.root
        output = root / "generated.wav"
        output.write_bytes(b"source")
        reference = root / "voice-reference.mp3"
        reference.write_bytes(b"reference")
        generator = MagicMock(return_value=SimpleNamespace(success=True, audios=[{"path": str(output)}]))
        inference = SimpleNamespace(GenerationParams=SimpleNamespace, GenerationConfig=SimpleNamespace,
                                    generate_music=generator)
        request = parse_generate({"prompt": "Bass", "lyrics": "Ти поруч", "instrumental": False,
                                  "duration_seconds": 2.12, "thinking": False})
        with patch.dict(sys.modules, {"acestep.inference": inference}):
            self.models.generate(request, 123, root, reference)
        params, config = generator.call_args.args[2:4]
        self.assertEqual(params.duration, 10)
        self.assertEqual(params.lyrics, "Ти поруч")
        self.assertEqual(params.reference_audio, str(reference))
        self.assertFalse(params.use_cot_caption)
        self.assertFalse(params.use_cot_metas)
        self.assertEqual(config.batch_size, 1)
        self.assertEqual(config.seeds, [123])
        self.assertEqual(config.audio_format, "wav")

    def test_output_path_cannot_point_to_unrelated_file(self):
        inference = SimpleNamespace(
            GenerationParams=SimpleNamespace, GenerationConfig=SimpleNamespace,
            generate_music=MagicMock(return_value=SimpleNamespace(success=True,
                                      audios=[{"path": "/etc/passwd"}])),
        )
        request = parse_generate({"prompt": "Bass", "instrumental": True, "duration_seconds": 20})
        with patch.dict(sys.modules, {"acestep.inference": inference}), self.assertRaises(RuntimeError):
            self.models.generate(request, 123, self.settings.root)


if __name__ == "__main__":
    unittest.main()
