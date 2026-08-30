"""Build-time compatibility and non-target source preservation checks."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from zakul_runpod.compatibility import check_compatibility
from zakul_runpod.config import Settings, env_bool

ROOT = Path(__file__).resolve().parents[2]


class CompatibilityTests(unittest.TestCase):
    """Catch incorrect repository/API choices before cloud GPU billing starts."""

    def test_uploaded_snapshot_matches_adapter(self):
        check_compatibility(ROOT)

    def test_wrong_repository_fails_with_useful_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "Unsupported ACE-Step snapshot"):
                check_compatibility(Path(temporary))

    def test_original_api_dockerfile_not_replaced(self):
        original = (ROOT / "Dockerfile").read_text()
        self.assertIn("ACESTEP_MODE=gradio", original)
        self.assertNotIn("zakul_runpod", original)
        self.assertIn("handler.py", (ROOT / "Dockerfile.runpod").read_text())

    def test_configuration_rejects_arbitrary_checkpoint_names(self):
        with patch.dict(os.environ, {"ACESTEP_CONFIG_PATH": "../../arbitrary-model"}, clear=True):
            with self.assertRaises(ValueError):
                Settings.from_env()

    def test_boolean_env_values_are_not_guessed(self):
        with patch.dict(os.environ, {"ACESTEP_OFFLOAD_TO_CPU": "flase"}, clear=True):
            with self.assertRaises(ValueError):
                env_bool("ACESTEP_OFFLOAD_TO_CPU")

    def test_default_model_is_previous_turbo_not_unrequested_xl(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.model, "acestep-v15-turbo")
        self.assertEqual(settings.lm_model, "acestep-5Hz-lm-0.6B")


if __name__ == "__main__":
    unittest.main()
