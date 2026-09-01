"""Download all Muse TAR archives to R2 with resume and verification."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from pipeline.muse.publish_catalog import required_environment


MIN_ARCHIVE_BYTES = 1_000_000_000
R2_PREFIX = "muse/raw_archives"


def archive_names() -> list[str]:
    """Return all CN and EN archive filenames in repository order."""
    chinese = [f"cn_part{part:02d}_of_25.tar" for part in range(1, 26)]
    english = [f"en_part{part:02d}_of_35.tar" for part in range(1, 36)]
    return chinese + english


def sha256(path: Path) -> str:
    """Calculate an archive checksum incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aws_environment(config: dict[str, str]) -> dict[str, str]:
    """Build a sanitized environment for AWS CLI calls."""
    environment = os.environ.copy()
    environment["AWS_ACCESS_KEY_ID"] = config["R2_ACCESS_KEY_ID"]
    environment["AWS_SECRET_ACCESS_KEY"] = config["R2_SECRET_ACCESS_KEY"]
    environment["AWS_DEFAULT_REGION"] = "auto"
    environment["AWS_EC2_METADATA_DISABLED"] = "true"
    return environment


def head_object(
    key: str,
    config: dict[str, str],
    aws_path: str,
) -> dict[str, Any] | None:
    """Return R2 object metadata, or None when the key is absent."""
    command = [
        aws_path,
        "s3api",
        "head-object",
        "--bucket",
        config["R2_BUCKET"],
        "--key",
        key,
        "--endpoint-url",
        config["R2_ENDPOINT_URL"],
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        check=False,
        env=aws_environment(config),
        text=True,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


def download_archive(filename: str, destination: Path) -> None:
    """Download an archive with retries and local resume support."""
    url = (
        "https://huggingface.co/datasets/bolshyC/Muse/resolve/main/"
        f"{filename}?download=true"
    )
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "8",
            "--retry-delay",
            "5",
            "--continue-at",
            "-",
            "--output",
            str(destination),
            url,
        ],
        check=True,
    )


def upload_archive(
    source: Path,
    key: str,
    config: dict[str, str],
    aws_path: str,
) -> None:
    """Upload one archive to R2."""
    subprocess.run(
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
        check=True,
        env=aws_environment(config),
    )


def write_manifest(entries: dict[str, Any], path: Path) -> None:
    """Write archive progress atomically."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(entries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def publish_manifest(
    path: Path,
    config: dict[str, str],
    aws_path: str,
) -> None:
    """Upload the current archive manifest to R2."""
    subprocess.run(
        [
            aws_path,
            "s3",
            "cp",
            str(path),
            f"s3://{config['R2_BUCKET']}/{R2_PREFIX}/archive_manifest.json",
            "--endpoint-url",
            config["R2_ENDPOINT_URL"],
            "--only-show-errors",
        ],
        check=True,
        env=aws_environment(config),
    )


def mirror_archives(work_dir: Path, aws_path: str) -> None:
    """Mirror every Muse archive to R2, resuming safely after interruption."""
    config = required_environment(os.environ)
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / "archive_manifest.json"
    entries: dict[str, Any] = {}
    if manifest_path.exists():
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))

    for filename in archive_names():
        key = f"{R2_PREFIX}/{filename}"
        remote = head_object(key, config, aws_path)
        if remote and int(remote.get("ContentLength", 0)) >= MIN_ARCHIVE_BYTES:
            entries[filename] = {
                **entries.get(filename, {}),
                "size": int(remote["ContentLength"]),
                "status": "verified_remote",
            }
            write_manifest(entries, manifest_path)
            continue

        local_path = work_dir / filename
        download_archive(filename, local_path)
        size = local_path.stat().st_size
        if size < MIN_ARCHIVE_BYTES:
            raise RuntimeError(f"Archive is unexpectedly small: {filename} ({size} bytes)")

        digest = sha256(local_path)
        upload_archive(local_path, key, config, aws_path)
        verified = head_object(key, config, aws_path)
        if not verified or int(verified.get("ContentLength", 0)) != size:
            raise RuntimeError(f"R2 size verification failed for {filename}")

        entries[filename] = {
            "sha256": digest,
            "size": size,
            "status": "verified_remote",
        }
        write_manifest(entries, manifest_path)
        publish_manifest(manifest_path, config, aws_path)
        local_path.unlink()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--aws-path", default="/usr/local/bin/aws")
    return parser.parse_args()


def main() -> None:
    """Run the resumable archive mirror."""
    args = parse_args()
    mirror_archives(args.work_dir, args.aws_path)


if __name__ == "__main__":
    main()
