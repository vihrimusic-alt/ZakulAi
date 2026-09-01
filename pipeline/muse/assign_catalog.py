"""Assign each clean Muse track to one primary training bucket."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


QUALITY_WEIGHTS = {"a": 1.0, "b": 0.7, "c": 0.35}
VOCAL_TAGS = {
    "duet_choir": {"duet", "choir", "male vocal and female vocal"},
    "female": {"female vocal", "female vocals"},
    "male": {"male vocal", "male vocals"},
    "instrumental": {"instrumental"},
}


def iter_catalog(path: Path) -> Iterable[dict[str, Any]]:
    """Yield JSON objects from a clean catalog."""
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def count_candidates(path: Path) -> Counter[str]:
    """Count candidate membership for rarity-aware assignment."""
    counts: Counter[str] = Counter()
    for record in iter_catalog(path):
        counts.update(record["candidate_buckets"])
    return counts


def choose_primary(
    candidates: list[str],
    candidate_counts: Counter[str],
) -> str:
    """Choose the rarest candidate bucket with deterministic tie-breaking."""
    return min(candidates, key=lambda bucket: (candidate_counts[bucket], bucket))


def vocal_profile(style: str) -> str:
    """Infer a coarse vocal profile from normalized comma-separated tags."""
    tags = {" ".join(tag.strip().lower().split()) for tag in style.split(",")}
    if tags & VOCAL_TAGS["duet_choir"]:
        return "duet_choir"
    female = bool(tags & VOCAL_TAGS["female"])
    male = bool(tags & VOCAL_TAGS["male"])
    if female and male:
        return "duet_choir"
    if female:
        return "female"
    if male:
        return "male"
    if tags & VOCAL_TAGS["instrumental"]:
        return "instrumental"
    return "unspecified"


def assign_record(
    record: dict[str, Any],
    candidate_counts: Counter[str],
) -> dict[str, Any]:
    """Add primary bucket, training weight, and vocal profile."""
    result = dict(record)
    result["primary_bucket"] = choose_primary(
        record["candidate_buckets"],
        candidate_counts,
    )
    result["training_weight"] = QUALITY_WEIGHTS[record["quality_tier"]]
    result["vocal_profile"] = vocal_profile(record["style"])
    return result


def assign_catalog(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Write assigned catalog and return its distribution summary."""
    candidate_counts = count_candidates(input_path)
    primary_counts: Counter[str] = Counter()
    vocal_counts: Counter[str] = Counter()
    weighted_totals: Counter[str] = Counter()
    rows = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output:
        for record in iter_catalog(input_path):
            assigned = assign_record(record, candidate_counts)
            output.write(json.dumps(assigned, ensure_ascii=False) + "\n")
            rows += 1
            primary = assigned["primary_bucket"]
            primary_counts[primary] += 1
            vocal_counts[assigned["vocal_profile"]] += 1
            weighted_totals[primary] += assigned["training_weight"]

    return {
        "rows": rows,
        "candidate_bucket_rows": dict(candidate_counts.most_common()),
        "primary_bucket_rows": dict(primary_counts.most_common()),
        "weighted_primary_examples": {
            bucket: round(value, 2)
            for bucket, value in weighted_totals.most_common()
        },
        "vocal_profile_rows": dict(vocal_counts.most_common()),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run primary bucket assignment."""
    args = parse_args()
    summary = assign_catalog(args.input, args.output)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
