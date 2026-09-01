"""Build a clean Muse catalog with style candidates and leakage-safe splits."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from pipeline.muse.audit_metadata import REQUIRED_FIELDS, iter_records, normalize_tag


QUALITY_THRESHOLDS = (("a", 0.65), ("b", 0.50))


def load_taxonomy(path: Path) -> dict[str, set[str]]:
    """Load and normalize style tags for each catalog bucket."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        bucket: {normalize_tag(tag) for tag in tags}
        for bucket, tags in raw.items()
    }


def split_for_song(song_id: str) -> str:
    """Assign every track for a song to a deterministic dataset split."""
    value = int(hashlib.sha256(song_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if value < 90:
        return "train"
    if value < 95:
        return "validation"
    return "test"


def quality_tier(style_sim: float) -> str:
    """Convert style similarity to a coarse quality tier."""
    for tier, threshold in QUALITY_THRESHOLDS:
        if style_sim >= threshold:
            return tier
    return "c"


def candidate_buckets(
    style: str,
    taxonomy: dict[str, set[str]],
) -> list[str]:
    """Return taxonomy buckets whose tags occur in a style description."""
    tags = {normalize_tag(tag) for tag in style.split(",")}
    return [
        bucket
        for bucket, bucket_tags in taxonomy.items()
        if tags & bucket_tags
    ]


def catalog_record(
    record: dict[str, Any],
    language: str,
    taxonomy: dict[str, set[str]],
) -> tuple[dict[str, Any] | None, str | None]:
    """Create a normalized catalog row or return its exclusion reason."""
    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        return None, "missing_required"
    if record.get("error"):
        return None, "source_error"

    style_sim = record.get("style_sim")
    if not isinstance(style_sim, (int, float)):
        return None, "missing_style_sim"
    song_id = record.get("song_id")
    style = record.get("style")
    if not isinstance(song_id, str) or not isinstance(style, str):
        return None, "invalid_text_fields"

    buckets = candidate_buckets(style, taxonomy)
    if not buckets:
        buckets = ["other"]

    return {
        "audio_path": record["audio_path"],
        "candidate_buckets": buckets,
        "language": language,
        "quality_tier": quality_tier(float(style_sim)),
        "sections": record["sections"],
        "song_id": song_id,
        "split": split_for_song(song_id),
        "style": style,
        "style_sim": float(style_sim),
        "track_index": record["track_index"],
    }, None


def iter_sources(paths: list[Path]) -> Iterable[tuple[Path, str]]:
    """Yield source paths with their language inferred from the filename."""
    for path in paths:
        language = "cn" if "_cn" in path.stem else "en"
        yield path, language


def build_catalog(
    paths: list[Path],
    taxonomy: dict[str, set[str]],
    output_path: Path,
) -> dict[str, Any]:
    """Write clean catalog rows and return summary counters."""
    counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output:
        for path, language in iter_sources(paths):
            for _, record, parse_error in iter_records(path):
                counts["input_rows"] += 1
                if parse_error or record is None:
                    counts["excluded_invalid_json"] += 1
                    continue

                catalog, exclusion = catalog_record(record, language, taxonomy)
                if exclusion:
                    counts[f"excluded_{exclusion}"] += 1
                    continue

                assert catalog is not None
                output.write(json.dumps(catalog, ensure_ascii=False) + "\n")
                counts["included_rows"] += 1
                counts[f"language_{language}"] += 1
                counts[f"split_{catalog['split']}"] += 1
                counts[f"quality_{catalog['quality_tier']}"] += 1
                bucket_counts.update(catalog["candidate_buckets"])

    return {
        "counts": dict(sorted(counts.items())),
        "candidate_bucket_rows": dict(bucket_counts.most_common()),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the catalog builder CLI."""
    args = parse_args()
    summary = build_catalog(
        args.input,
        load_taxonomy(args.taxonomy),
        args.output,
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
