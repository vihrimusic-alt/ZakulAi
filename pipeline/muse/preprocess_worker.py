"""Run resumable archive-by-archive ACE-Step preprocessing from R2."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from pipeline.muse.build_tensor_index import build_tensor_index
from pipeline.muse.mirror_archives import archive_names, head_object
from pipeline.muse.mirror_shard import selected_names
from pipeline.muse.publish_catalog import required_environment
from pipeline.muse.r2_transfer import (
    download_file,
    load_manifest,
    run,
    upload_directory,
    upload_file,
)


RAW_PREFIX = "muse/raw_archives"
OUTPUT_PREFIX = "muse/preprocessed/turbo/v1"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest for a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(
    dataset_dir: Path,
    tensor_dir: Path,
    ace_dir: Path,
    max_duration: int,
) -> None:
    """Run ACE-Step Turbo preprocessing for one prepared archive."""
    run(
        [
            "uv", "run", "python", "-m",
            "acestep.training_v2.cli.train_fixed",
            "--preprocess",
            "--audio-dir", str(dataset_dir),
            "--tensor-output", str(tensor_dir),
            "--checkpoint-dir", str(ace_dir / "checkpoints"),
            "--model-variant", "turbo",
            "--max-duration", str(max_duration),
            "--device", "cuda:0",
            "--precision", "bf16",
            "--yes",
        ],
        cwd=ace_dir,
    )


def prepare(
    archive: Path,
    dataset: Path,
    summary: Path,
    catalog: Path,
    zakul_dir: Path,
) -> None:
    """Extract selected training records and create ACE-Step sidecars."""
    run(
        [
            "uv", "run", "--no-project", "--python", "3.11",
            "python", "-m", "pipeline.muse.prepare_archive",
            "--archive", str(archive),
            "--catalog", str(catalog),
            "--output-dir", str(dataset),
            "--summary", str(summary),
        ],
        cwd=zakul_dir,
    )


def publish_result(
    filename: str,
    tensors: Path,
    summary: Path,
    index: Path,
    args: argparse.Namespace,
    config: dict[str, str],
) -> None:
    """Upload tensors and publish the completion marker last."""
    prefix = f"{OUTPUT_PREFIX}/{Path(filename).stem}"
    upload_directory(tensors, prefix, config, args.aws_path)
    upload_file(summary, f"{prefix}/summary.json", config, args.aws_path)
    upload_file(index, f"{prefix}/training_index.jsonl", config, args.aws_path)
    success = args.work_dir / "reports" / f"{Path(filename).stem}.success.json"
    success.write_text(
        json.dumps({"archive": filename, "status": "complete"}, indent=2) + "\n",
        encoding="utf-8",
    )
    upload_file(success, f"{prefix}/_SUCCESS.json", config, args.aws_path)
    if not head_object(f"{prefix}/_SUCCESS.json", config, args.aws_path):
        raise RuntimeError(f"Success marker verification failed: {filename}")


def process_archive(
    filename: str,
    args: argparse.Namespace,
    config: dict[str, str],
) -> bool:
    """Process one available archive; return False while waiting for R2."""
    prefix = f"{OUTPUT_PREFIX}/{Path(filename).stem}"
    if head_object(f"{prefix}/_SUCCESS.json", config, args.aws_path):
        print(f"[SKIP] {filename} already completed", flush=True)
        return True

    manifest = load_manifest(args.work_dir, RAW_PREFIX, config, args.aws_path)
    entry = manifest.get(filename)
    remote = head_object(f"{RAW_PREFIX}/{filename}", config, args.aws_path)
    if not entry or entry.get("status") != "verified_remote" or remote is None:
        print(f"[WAIT] {filename} is not verified in R2", flush=True)
        return False

    stem = Path(filename).stem
    archive = args.work_dir / "archive" / filename
    dataset = args.work_dir / "dataset" / stem
    tensors = args.work_dir / "tensors" / stem
    summary = args.work_dir / "reports" / f"{stem}.json"
    index = args.work_dir / "reports" / f"{stem}.training_index.jsonl"
    stage = tensors / ".preprocess_complete"

    if not archive.exists():
        download_file(f"{RAW_PREFIX}/{filename}", archive, config, args.aws_path)
    if archive.stat().st_size != int(entry["size"]) or sha256(archive) != entry["sha256"]:
        archive.unlink(missing_ok=True)
        raise RuntimeError(f"Archive verification failed: {filename}")

    if not summary.exists():
        prepare(archive, dataset, summary, args.catalog, args.zakul_dir)
    if not stage.exists():
        tensors.mkdir(parents=True, exist_ok=True)
        preprocess(dataset, tensors, args.ace_dir, args.max_duration)
        stage.touch()
    build_tensor_index(args.catalog, tensors, index)

    publish_result(filename, tensors, summary, index, args, config)
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
    parser.add_argument("--language", choices=("cn", "en"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    """Process every Muse archive sequentially with retry and resume."""
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    config = required_environment(os.environ)
    names = archive_names()
    if args.language:
        names = selected_names(args.language, args.shard_index, args.shard_count)
    for filename in names:
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
