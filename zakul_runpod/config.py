"""Validated worker configuration and dedicated runtime directories."""

import os
from dataclasses import dataclass
from pathlib import Path

MODELS = {
    "acestep-v15-turbo": (8, 1.0),
    "acestep-v15-sft": (50, 7.0),
    "acestep-v15-base": (50, 7.0),
    "acestep-v15-xl-turbo": (8, 1.0),
}
LM_MODELS = {"acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-4B"}


def env_bool(name: str, default: bool = False) -> bool:
    """Parse explicit boolean environment values; reject misspellings."""
    value = os.environ.get(name, str(default)).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


@dataclass(frozen=True)
class Settings:
    """Fixed model selection for one endpoint, shared across its jobs."""

    root: Path
    checkpoints: Path
    temporary: Path
    model: str
    lm_model: str
    offload: bool
    lm_offload: bool

    @classmethod
    def from_env(cls) -> "Settings":
        """Resolve paths without loading models, downloading, or opening sockets."""
        root = Path(__file__).resolve().parents[1]
        volume = Path("/runpod-volume")
        default_checkpoints = volume / "checkpoints" if volume.is_dir() else root / "checkpoints"
        model = os.environ.get("ACESTEP_CONFIG_PATH", "acestep-v15-turbo").strip()
        lm_model = os.environ.get("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-0.6B").strip()
        if model not in MODELS:
            raise ValueError("Unsupported ACESTEP_CONFIG_PATH; see RUNPOD_SETUP_UA.md")
        if lm_model not in LM_MODELS:
            raise ValueError("Unsupported ACESTEP_LM_MODEL_PATH")
        return cls(
            root=root,
            checkpoints=Path(os.environ.get("ACESTEP_CHECKPOINTS_DIR", default_checkpoints)),
            temporary=Path(os.environ.get("ZAKUL_RUNPOD_TMPDIR", "/tmp/zakul-runpod")),
            model=model,
            lm_model=lm_model,
            offload=env_bool("ACESTEP_OFFLOAD_TO_CPU"),
            lm_offload=env_bool("ACESTEP_LM_OFFLOAD_TO_CPU"),
        )

    def prepare(self) -> None:
        """Create only configured working folders, never remove existing models."""
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.temporary.mkdir(parents=True, exist_ok=True)
        os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(self.checkpoints.resolve())
        os.environ.setdefault("ACESTEP_PROJECT_ROOT", str(self.root))
        os.environ.setdefault("ACESTEP_TMPDIR", str(self.temporary / "acestep"))
        os.environ.setdefault("HF_HOME", str(self.checkpoints / ".hf-cache"))
