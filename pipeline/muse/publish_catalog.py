"""Publish Muse catalog artifacts to Cloudflare R2."""

import argparse
import gzip
import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Mapping


DEFAULT_PREFIX = "muse/catalog/v1"


def required_environment(source: Mapping[str, str]) -> dict[str, str]:
    """Return sanitized R2 configuration or raise for missing values."""
    names = (
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_ENDPOINT_URL",
        "R2_BUCKET",
    )
    values = {name: source.get(name, "").strip() for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")
    return values


def compress_file(source: Path, destination: Path) -> None:
    """Create a deterministic gzip copy of a catalog file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source.open("rb") as input_file:
        with temporary.open("wb") as raw_output:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_output,
                mtime=0,
            ) as output_file:
                shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
    temporary.replace(destination)


def sha256(path: Path) -> str:
    """Calculate a SHA-256 digest without loading the file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(paths: list[Path], destination: Path) -> None:
    """Write standard SHA-256 checksum lines for artifacts."""
    lines = [f"{sha256(path)}  {path.name}" for path in paths]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aws_environment(config: dict[str, str]) -> dict[str, str]:
    """Build an AWS CLI environment without exposing R2 secrets."""
    environment = os.environ.copy()
    environment["AWS_ACCESS_KEY_ID"] = config["R2_ACCESS_KEY_ID"]
    environment["AWS_SECRET_ACCESS_KEY"] = config["R2_SECRET_ACCESS_KEY"]
    environment["AWS_DEFAULT_REGION"] = "auto"
    environment["AWS_EC2_METADATA_DISABLED"] = "true"
    return environment


def run_aws(arguments: list[str], config: dict[str, str], aws_path: str) -> None:
    """Run an AWS CLI command and fail on a non-zero exit status."""
    subprocess.run(
        [aws_path, *arguments],
        check=True,
        env=aws_environment(config),
    )


def publish(
    catalog: Path,
    summary: Path,
    prefix: str,
    aws_path: str,
) -> None:
    """Compress, checksum, upload, and verify catalog artifacts."""
    config = required_environment(os.environ)
    compressed = catalog.with_suffix(catalog.suffix + ".gz")
    checksums = catalog.parent / "ASSIGNMENT_SHA256SUMS"
    compress_file(catalog, compressed)
    write_checksums([compressed, summary], checksums)

    artifacts = [compressed, summary, checksums]
    for artifact in artifacts:
        key = f"{prefix.rstrip('/')}/{artifact.name}"
        run_aws(
            [
                "s3",
                "cp",
                str(artifact),
                f"s3://{config['R2_BUCKET']}/{key}",
                "--endpoint-url",
                config["R2_ENDPOINT_URL"],
                "--only-show-errors",
            ],
            config,
            aws_path,
        )
        run_aws(
            [
                "s3api",
                "head-object",
                "--bucket",
                config["R2_BUCKET"],
                "--key",
                key,
                "--endpoint-url",
                config["R2_ENDPOINT_URL"],
            ],
            config,
            aws_path,
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--aws-path", default="/usr/local/bin/aws")
    return parser.parse_args()


def main() -> None:
    """Run catalog publication."""
    args = parse_args()
    publish(args.catalog, args.summary, args.prefix, args.aws_path)


if __name__ == "__main__":
    main()
