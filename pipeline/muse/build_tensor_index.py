"""Build a LoRA routing index for successfully preprocessed Muse tensors."""

import gzip
import json
from pathlib import Path
from typing import Any, Iterable


INDEX_FIELDS = (
    "primary_bucket",
    "candidate_buckets",
    "quality_tier",
    "training_weight",
    "vocal_profile",
    "language",
    "style",
    "split",
)


def iter_catalog(path: Path) -> Iterable[dict[str, Any]]:
    """Yield catalog records from JSONL or gzip-compressed JSONL."""
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def build_tensor_index(
    catalog_path: Path,
    tensor_dir: Path,
    output_path: Path,
) -> dict[str, int]:
    """Write routing metadata for final tensors and return bucket counts."""
    tensor_names = {
        path.name
        for path in tensor_dir.glob("*.pt")
        if not path.name.endswith(".tmp.pt")
    }
    counts: dict[str, int] = {}
    indexed: set[str] = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as target:
        for record in iter_catalog(catalog_path):
            tensor_name = f"{Path(record['audio_path']).stem}.pt"
            if tensor_name not in tensor_names:
                continue
            row = {"tensor": tensor_name}
            row.update({field: record[field] for field in INDEX_FIELDS})
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            indexed.add(tensor_name)
            bucket = str(record["primary_bucket"])
            counts[bucket] = counts.get(bucket, 0) + 1

    missing = tensor_names - indexed
    if missing:
        examples = ", ".join(sorted(missing)[:5])
        raise ValueError(f"{len(missing)} tensors are absent from catalog: {examples}")
    return counts
