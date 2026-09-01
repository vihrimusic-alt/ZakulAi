"""RunPod Queue entrypoint; the original ACE-Step API/UI entrypoints stay unchanged."""

import sys
import os

import runpod
from loguru import logger

from zakul_runpod.runtime import QueueWorker

worker = QueueWorker()


def handler(job: dict):
    """Stream each completed take while one validated job keeps the GPU worker warm."""
    yield from worker.stream(job)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO", diagnose=False)
    if os.getenv("ZAKUL_PRELOAD_MODELS", "false").lower() == "true":
        worker.models.ensure_loaded(True, lambda message: logger.info("{}",message))
    # A synchronous handler intentionally runs one GPU job at a time.
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
