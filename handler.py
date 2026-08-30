"""RunPod Queue entrypoint; the original ACE-Step API/UI entrypoints stay unchanged."""

import sys

import runpod
from loguru import logger

from zakul_runpod.runtime import QueueWorker

worker = QueueWorker()


def handler(job: dict) -> dict:
    """Run one validated job and publish audio before reporting success."""
    return worker.handle(job)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO", diagnose=False)
    # A synchronous handler intentionally runs one GPU job at a time.
    runpod.serverless.start({"handler": handler})
