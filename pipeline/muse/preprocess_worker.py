"""Run resumable archive-by-archive ACE-Step preprocessing from R2."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from pipeline.muse.mirror_archives import archive_names, aws_environment, head_object
from pipeline.muse.publish_catalog import required_environment


RAW_PREFIX = "muse/raw_archives"
OUTPUT_PREFIX = "muse/preprocessed/turbo/v1"


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    """Run one external command and fail on a non-zero exit status."""
    subprocess.run(command, check=True, cwd=cwd, env=env)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest for a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def upload_directory(
    source: Path,
    prefix: str,
    config: dict[str, str],
    aws_path: str,
) -> None:
    """Synchronize a tensor directory to R2."""
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


def load_manifest(
    work_dir: Path,
    config: dict[str, str],
    aws_path: str,
) -> dict[str, Any]:
    """Download and parse the verified raw-archive manifest."""
    path = work_dir / "archive_manifest.json"
    download_file(f"{RAW_PREFIX}/archive_manifest.json", path, config, aws_path)
    return json.loads(path.read_text(encoding="utf-8"))


def preprocess(
    dataset_dir: Path,
    tensor_dir: Path,
    ace_dir: Path,
    max_duration: int,
) -> None:
    """Run ACE-Step Turbo preprocessing for one prepared archive."""
    run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "acestep.training_v2.cli.train_fixed",
            "--preprocess",
            "--audio-dir",
            str(dataset_dir),
            "--tensor-output",
            str(tensor_dir),
            "--checkpoint-dir",
            str(ace_dir / "checkpoints"),
            "--model-variant",
            "turbo",
            "--max-duration",
            str(max_duration),
            "--device",
            "cuda:0",
            "--precision",
            "bf16",
            "--yes",
        ],
        cwd=ace_dir,
    )


def process_archive(
    filename: str,
    args: argparse.Namespace,
    config: dict[str, str],
) -> bool:
    """Process one available archive; return False while waiting for R2."""
    output_prefix = f"{OUTPUT_PREFIX}/{Path(filename).stem}"
    if head_object(f"{output_prefix}/_SUCCESS.json", config, args.aws_path):
        print(f"[SKIP] {filename} already completed", flush=True)
        return True

    manifest = load_manifest(args.work_dir, config, args.aws_path)
    entry = manifest.get(filename)
    remote = head_object(f"{RAW_PREFIX}/{filename}", config, args.aws_path)
    if not entry or entry.get("status") != "verified_remote" or remote is None:
        print(f"[WAIT] {filename} is not verified in R2", flush=True)
        return False

    archive = args.work_dir / "archives" / filename
    dataset = args.work_dir / "datasets" / Path(filename).stem
    tensors = args.work_dir / "tensors" / Path(filename).stem
    summary = args.work_dir / "reports" / f"{Path(filename).stem}.json"
    stage = tensors / ".preprocess_complete"

    if not archive.exists():
        download_file(f"{RAW_PREFIX}/{filename}", archive, config, args.aws_path)
    if archive.stat().st_size != int(entry["size"]) or sha256(archive) != entry["sha256"]:
        archive.unlink(missing_ok=True)
        raise RuntimeError(f"Archive verification failed: {filename}")

    if not summary.exists():
        run(
            [
                "uv",
                "run",
                "--no-project",
                "--python",
                "3.11",
                "python",
                "-m",
                "pipeline.muse.prepare_archive",
                "--archive",
                str(archive),
                "--catalog",
                str(args.catalog),
                "--output-dir",
                str(dataset),
                "--summary",
                str(summary),
            ],
            cwd=args.zakul_dir,
        )

    if not stage.exists():
        tensors.mkdir(parents=True, exist_ok=True)
        preprocess(dataset, tensors, args.ace_dir, args.max_duration)
        stage.touch()

    upload_directory(tensors, output_prefix, config, args.aws_path)
    upload_file(summary, f"{output_prefix}/summary.json", config, args.aws_path)
    success = args.work_dir / "reports" / f"{Path(filename).stem}.success.json"
    success.write_text(
        json.dumps({"archive": filename, "status": "complete"}, indent=2) + "\n",
        encoding="utf-8",
    )
    upload_file(success, f"{output_prefix}/_SUCCESS.json", config, args.aws_path)
    if not head_object(f"{output_prefix}/_SUCCESS.json", config, args.aws_path):
        raise RuntimeError(f"Success marker verification failed: {filename}")

    archive.unlink(missing_ok=True)
    shutil.rmtree(dataset, ignore_errors=True)
    shutil.rmtree(tensors, ignore_errors=True)
    print(f"[DONE] {filename}", flush=True)
    return True


def parse_args() -> argparse.Namespace:
    """Parse worker paths and retry settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--zakul-dir", type=Path, required=True)
    parser.add_argument("--ace-dir", type=Path, required=True)
    parser.add_argument("--aws-path", default="/usr/local/bin/aws")
    parser.add_argument("--max-duration", type=int, default=240)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--retry-seconds", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    """Process every Muse archive sequentially with retry and resume."""
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    config = required_environment(os.environ)
    for filename in archive_names():
        while True:
            try:
                if process_archive(filename, args, config):
                    break
                time.sleep(args.poll_seconds)
            except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
                print(f"[RETRY] {filename}: {error}", flush=True)
                time.sleep(args.retry_seconds)
    print("[COMPLETE] All Muse archives preprocessed", flush=True)


if __name__ == "__main__":
    main()
