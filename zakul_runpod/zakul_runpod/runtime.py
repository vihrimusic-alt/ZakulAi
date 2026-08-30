"""Sequential queue orchestration with strict validation and bounded per-job files."""

from dataclasses import replace
import secrets
import threading
from datetime import datetime, timezone
from tempfile import TemporaryDirectory
from typing import Callable
from pathlib import Path

from loguru import logger

from . import VERSION
from .audio import encode_take, duration_seconds
from .config import MODELS, Settings
from .models import Models
from .storage import ResultStore
from .validation import parse_generate, validate_job


def report_progress(job: dict, message: str) -> None:
    """Report coarse phases without leaking lyrics, keys or signed output URLs."""
    logger.info("{}", message)
    if not job.get("id"):
        return
    try:
        import runpod

        runpod.serverless.progress_update(job, message)
    except Exception:
        # A best-effort UI update must not discard successfully generated audio.
        logger.warning("RunPod progress update could not be sent")


class QueueWorker:
    """Reuse initialized models across jobs and serialize all GPU work."""

    def __init__(self, settings: Settings | None = None):
        """Initialize configuration only; no downloads occur at module import."""
        self.settings = settings or Settings.from_env()
        self.models = Models(self.settings)
        self.lock = threading.Lock()

    def handle(self, job: dict) -> dict:
        """Process health, warmup or text-to-music; unsupported tasks fail explicitly."""
        operation, data = validate_job(job)
        progress = lambda message: report_progress(job, message)
        with self.lock:
            if operation == "health":
                return {"worker_version": VERSION, "operation": operation, **self.models.health()}
            if operation == "warmup":
                self.models.ensure_loaded(True, progress)
                return {"worker_version": VERSION, "operation": operation, **self.models.health()}
            request = parse_generate(data)
            job_id = str(job.get("id") or secrets.token_hex(16))
            store = ResultStore(request.output_mode, job_id)
            self.models.ensure_loaded(request.thinking, progress)
            return self._generate(request, store, progress)

    def _generate(self, request, store: ResultStore, progress: Callable[[str], None]) -> dict:
        automatic = request.duration is None
        if automatic:
            progress("AI is choosing song duration from lyrics and style")
            planned = self.models.plan_duration(request)
            request = replace(request, duration=planned)
        tracks = []
        # Only this newly created directory is ever removed. Model weights stay cached.
        with TemporaryDirectory(prefix="job-", dir=self.settings.temporary) as temporary:
            for index in range(1, request.outputs + 1):
                folder = Path(temporary) / f"take-{index}"
                folder.mkdir()
                seed = secrets.randbelow(2**31 - 1) if request.seed == -1 else request.seed + index - 1
                progress(f"Generating take {index}/{request.outputs}")
                source = self.models.generate(request, seed, folder)
                progress(f"Encoding take {index}/{request.outputs}")
                assets = encode_take(source, folder, None if automatic else request.duration, request.output_mode == "s3")
                progress(f"Saving take {index}/{request.outputs}")
                actual_duration = duration_seconds(assets["flac"]) if automatic else request.duration
                tracks.append(store.publish(assets, index, actual_duration, seed))
        steps, guidance = MODELS[self.settings.model]
        return {
            "worker_version": VERSION, "operation": "generate", "tracks": tracks,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": self.settings.model,
            "lm_model": self.settings.lm_model if request.thinking else None,
            "inference_steps": steps, "guidance_scale": guidance,
            "output_mode": request.output_mode,
            "duration_mode": "auto" if automatic else "fixed",
            "planned_duration_seconds": request.duration,
            "note": "MP3 may include encoder delay. Auto mode preserves the generated ending without fixed-length trimming.",
        }
