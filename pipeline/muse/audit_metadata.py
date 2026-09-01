"""Audit Muse JSONL metadata before downloading audio."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REQUIRED_FIELDS = {
    "audio_path",
    "sections",
    "song_id",
    "style",
    "style_sim",
    "track_index",
}


def normalize_tag(tag: str) -> str:
    """Return a stable lowercase representation of a style tag."""
    return " ".join(tag.strip().lower().split())


def iter_records(path: Path) -> Iterable[tuple[int, dict[str, Any] | None, str | None]]:
    """Yield line number, parsed record, and an optional parse error."""
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                yield line_number, None, str(error)
                continue
            if not isinstance(record, dict):
                yield line_number, None, "record is not a JSON object"
                continue
            yield line_number, record, None


def record_duration(record: dict[str, Any]) -> float | None:
    """Estimate song duration from the largest section end timestamp."""
    ends = [
        section.get("endS")
        for section in record.get("sections", [])
        if isinstance(section, dict) and isinstance(section.get("endS"), (int, float))
    ]
    return float(max(ends)) if ends else None


def audit_files(paths: list[Path]) -> dict[str, Any]:
    """Audit Muse metadata files and return a serializable summary."""
    tags: Counter[str] = Counter()
    missing_fields: Counter[str] = Counter()
    audio_paths: Counter[str] = Counter()
    song_ids: Counter[str] = Counter()
    totals = Counter()
    style_sim_sum = 0.0
    durations: list[float] = []
    error_samples: list[dict[str, Any]] = []

    for path in paths:
        file_total = 0
        for line_number, record, parse_error in iter_records(path):
            file_total += 1
            totals["rows"] += 1
            if parse_error:
                totals["invalid_json"] += 1
                if len(error_samples) < 25:
                    error_samples.append(
                        {"file": path.name, "line": line_number, "error": parse_error}
                    )
                continue

            assert record is not None
            missing = REQUIRED_FIELDS - record.keys()
            if missing:
                totals["missing_required"] += 1
                missing_fields.update(missing)
            if record.get("error"):
                totals["source_error_rows"] += 1

            audio_path = record.get("audio_path")
            song_id = record.get("song_id")
            if isinstance(audio_path, str):
                audio_paths[audio_path] += 1
            if isinstance(song_id, str):
                song_ids[song_id] += 1

            style = record.get("style")
            if isinstance(style, str):
                tags.update(
                    normalized
                    for raw_tag in style.split(",")
                    if (normalized := normalize_tag(raw_tag))
                )

            style_sim = record.get("style_sim")
            if isinstance(style_sim, (int, float)):
                totals["style_sim_count"] += 1
                style_sim_sum += float(style_sim)

            duration = record_duration(record)
            if duration is not None:
                durations.append(duration)

        totals[f"rows_{path.stem}"] = file_total

    duplicate_audio = sum(count - 1 for count in audio_paths.values() if count > 1)
    duplicate_songs = sum(count - 1 for count in song_ids.values() if count > 1)
    duration_summary = {}
    if durations:
        duration_summary = {
            "count": len(durations),
            "min_seconds": min(durations),
            "max_seconds": max(durations),
            "mean_seconds": sum(durations) / len(durations),
        }

    return {
        "totals": dict(sorted(totals.items())),
        "unique_audio_paths": len(audio_paths),
        "unique_song_ids": len(song_ids),
        "duplicate_audio_rows": duplicate_audio,
        "duplicate_song_rows": duplicate_songs,
        "missing_fields": dict(missing_fields.most_common()),
        "style_sim_mean": (
            style_sim_sum / totals["style_sim_count"] if totals["style_sim_count"] else None
        ),
        "duration": duration_summary,
        "top_style_tags": dict(tags.most_common(200)),
        "error_samples": error_samples,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the metadata audit CLI."""
    args = parse_args()
    summary = audit_files(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
