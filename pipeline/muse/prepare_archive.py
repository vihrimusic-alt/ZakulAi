"""Prepare one Muse TAR archive as bucketed ACE-Step training datasets."""

import argparse
import gzip
import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


AUDIO_SUFFIXES = {".flac", ".mp3", ".ogg", ".opus", ".wav"}


def iter_catalog(path: Path) -> Iterable[dict[str, Any]]:
    """Yield assigned catalog records from plain JSONL or gzip JSONL."""
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def lyrics_text(sections: list[dict[str, Any]]) -> str:
    """Convert structured Muse sections into ACE-Step lyric text."""
    blocks: list[str] = []
    for section in sections:
        name = str(section.get("section", "")).strip()
        text = str(section.get("text", "")).strip()
        if not text:
            continue
        heading = f"[{name}]" if name else ""
        blocks.append("\n".join(part for part in (heading, text) if part))
    return "\n\n".join(blocks).strip() + "\n"


def annotation(record: dict[str, Any]) -> dict[str, Any]:
    """Build the supported ACE-Step annotation fields for one track."""
    return {
        "caption": record["style"],
        "language": record["language"],
    }


def selected_records(
    catalog_path: Path,
    splits: set[str],
    quality_tiers: set[str],
) -> dict[str, dict[str, Any]]:
    """Index records selected for training by their archive basename."""
    records: dict[str, dict[str, Any]] = {}
    for record in iter_catalog(catalog_path):
        if record["split"] not in splits:
            continue
        if record["quality_tier"] not in quality_tiers:
            continue
        basename = Path(record["audio_path"]).name
        if basename in records:
            raise ValueError(f"Duplicate audio basename in catalog: {basename}")
        records[basename] = record
    return records


def validate_member(member: tarfile.TarInfo) -> str:
    """Return a safe flat audio filename or raise for an unsafe member."""
    path = Path(member.name)
    if path.name != member.name or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive member: {member.name}")
    if path.suffix.lower() not in AUDIO_SUFFIXES:
        raise ValueError(f"Unexpected archive member: {member.name}")
    if not member.isfile():
        raise ValueError(f"Archive member is not a regular file: {member.name}")
    return path.name


def prepare_archive(
    archive_path: Path,
    catalog_path: Path,
    output_dir: Path,
    splits: set[str],
    quality_tiers: set[str],
) -> dict[str, Any]:
    """Extract selected tracks and write ACE-Step sidecars by primary bucket."""
    records = selected_records(catalog_path, splits, quality_tiers)
    bucket_counts: Counter[str] = Counter()
    skipped = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r") as archive:
        for member in archive:
            basename = validate_member(member)
            record = records.get(basename)
            if record is None:
                skipped += 1
                continue
            bucket = record["primary_bucket"]
            destination_dir = output_dir / bucket
            destination_dir.mkdir(parents=True, exist_ok=True)
            audio_path = destination_dir / basename
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read archive member: {basename}")
            with source, audio_path.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)

            stem = audio_path.with_suffix("")
            stem.with_suffix(".lyrics.txt").write_text(
                lyrics_text(record["sections"]),
                encoding="utf-8",
            )
            stem.with_suffix(".json").write_text(
                json.dumps(annotation(record), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            bucket_counts[bucket] += 1

    return {
        "archive": archive_path.name,
        "included": sum(bucket_counts.values()),
        "skipped": skipped,
        "bucket_counts": dict(bucket_counts.most_common()),
        "splits": sorted(splits),
        "quality_tiers": sorted(quality_tiers),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--quality-tiers", nargs="+", default=["a", "b", "c"])
    return parser.parse_args()


def main() -> None:
    """Prepare a Muse archive for ACE-Step training."""
    args = parse_args()
    summary = prepare_archive(
        args.archive,
        args.catalog,
        args.output_dir,
        set(args.splits),
        set(args.quality_tiers),
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
