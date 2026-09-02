"""Mirror a disjoint Muse archive shard to R2 with its own manifest."""

import argparse
import json
import os
from pathlib import Path

from pipeline.muse.mirror_archives import (
    MIN_ARCHIVE_BYTES,
    R2_PREFIX,
    archive_names,
    download_archive,
    head_object,
    sha256,
    upload_archive,
    write_manifest,
)
from pipeline.muse.publish_catalog import required_environment
from pipeline.muse.r2_transfer import download_file, upload_file


def selected_names(language: str, shard_index: int, shard_count: int) -> list[str]:
    """Return one deterministic language shard without overlap."""
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard-index must be between zero and shard-count minus one")
    prefix = f"{language}_"
    names = [name for name in archive_names() if name.startswith(prefix)]
    return names[shard_index::shard_count]


def load_entries(
    work_dir: Path,
    manifest_key: str,
    config: dict[str, str],
    aws_path: str,
) -> dict[str, dict[str, object]]:
    """Restore the worker manifest from local disk or R2."""
    path = work_dir / Path(manifest_key).name
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if head_object(manifest_key, config, aws_path):
        download_file(manifest_key, path, config, aws_path)
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def mirror_shard(args: argparse.Namespace) -> None:
    """Mirror assigned archives and publish progress to an isolated manifest."""
    config = required_environment(os.environ)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    manifest_key = f"{R2_PREFIX}/{args.manifest_name}"
    manifest_path = args.work_dir / args.manifest_name
    entries = load_entries(args.work_dir, manifest_key, config, args.aws_path)

    for filename in selected_names(args.language, args.shard_index, args.shard_count):
        key = f"{R2_PREFIX}/{filename}"
        remote = head_object(key, config, args.aws_path)
        entry = entries.get(filename)
        if (
            remote
            and entry
            and entry.get("status") == "verified_remote"
            and int(remote.get("ContentLength", 0)) == int(entry.get("size", -1))
        ):
            print(f"[SKIP] {filename}", flush=True)
            continue

        local_path = args.work_dir / filename
        if remote and int(remote.get("ContentLength", 0)) >= MIN_ARCHIVE_BYTES:
            local_path.unlink(missing_ok=True)
            download_file(key, local_path, config, args.aws_path)
        else:
            download_archive(filename, local_path)

        size = local_path.stat().st_size
        if size < MIN_ARCHIVE_BYTES:
            raise RuntimeError(f"Archive is unexpectedly small: {filename} ({size} bytes)")
        digest = sha256(local_path)

        if not remote or int(remote.get("ContentLength", 0)) != size:
            upload_archive(local_path, key, config, args.aws_path)
        verified = head_object(key, config, args.aws_path)
        if not verified or int(verified.get("ContentLength", 0)) != size:
            raise RuntimeError(f"R2 size verification failed for {filename}")

        entries[filename] = {
            "sha256": digest,
            "size": size,
            "status": "verified_remote",
        }
        write_manifest(entries, manifest_path)
        upload_file(manifest_path, manifest_key, config, args.aws_path)
        local_path.unlink()
        print(f"[DONE] {filename}", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse shard assignment and storage paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--language", choices=("cn", "en"), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--manifest-name", required=True)
    parser.add_argument("--aws-path", default="/usr/local/bin/aws")
    return parser.parse_args()


def main() -> None:
    """Run one resumable archive shard worker."""
    mirror_shard(parse_args())


if __name__ == "__main__":
    main()
