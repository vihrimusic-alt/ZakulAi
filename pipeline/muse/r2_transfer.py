"""R2 transfer helpers for the Muse preprocessing worker."""

import json
import subprocess
from pathlib import Path
from typing import Any

from pipeline.muse.mirror_archives import aws_environment


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    """Run one external command and fail on a non-zero exit status."""
    subprocess.run(command, check=True, cwd=cwd, env=env)


def download_file(
    key: str,
    destination: Path,
    config: dict[str, str],
    aws_path: str,
) -> None:
    """Download one R2 object to a local path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            aws_path,
            "s3",
            "cp",
            f"s3://{config['R2_BUCKET']}/{key}",
            str(destination),
            "--endpoint-url",
            config["R2_ENDPOINT_URL"],
            "--no-progress",
        ],
        env=aws_environment(config),
    )


def upload_file(
    source: Path,
    key: str,
    config: dict[str, str],
    aws_path: str,
) -> None:
    """Upload one local file to R2."""
    run(
        [
            aws_path,
            "s3",
            "cp",
            str(source),
            f"s3://{config['R2_BUCKET']}/{key}",
            "--endpoint-url",
            config["R2_ENDPOINT_URL"],
            "--only-show-errors",
        ],
        env=aws_environment(config),
    )


def upload_directory(
    source: Path,
    prefix: str,
    config: dict[str, str],
    aws_path: str,
) -> None:
    """Synchronize a local directory to R2."""
    run(
        [
            aws_path,
            "s3",
            "sync",
            str(source),
            f"s3://{config['R2_BUCKET']}/{prefix}/tensors/",
            "--endpoint-url",
            config["R2_ENDPOINT_URL"],
            "--only-show-errors",
        ],
        env=aws_environment(config),
    )


def load_manifest(
    work_dir: Path,
    raw_prefix: str,
    config: dict[str, str],
    aws_path: str,
) -> dict[str, Any]:
    """Download and parse the verified raw-archive manifest."""
    path = work_dir / "archive_manifest.json"
    download_file(f"{raw_prefix}/archive_manifest.json", path, config, aws_path)
    return json.loads(path.read_text(encoding="utf-8"))
