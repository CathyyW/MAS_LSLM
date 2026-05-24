#!/usr/bin/env python3
"""
Split the MATH dataset into:
  1. a balanced public set sampled from every subject
  2. three non-IID private client sets, each drawn from its expert subjects

Supported input layouts:
  - Hugging Face datasets:
      load_dataset("qwedsacf/competition_math")
  - Hendrycks MATH style:
      input_dir/train/algebra/*.json
      input_dir/train/geometry/*.json
      ...
  - Flat JSONL/JSON files containing records with a subject/type/category field.

Output layout:
  output_dir/public.jsonl
  output_dir/clients/client_0/train.jsonl
  output_dir/clients/client_0/infer.jsonl
  output_dir/clients/client_1/train.jsonl
  output_dir/clients/client_1/infer.jsonl
  output_dir/clients/client_2/train.jsonl
  output_dir/clients/client_2/infer.jsonl
  output_dir/metadata.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


SUBJECTS = [
    "Algebra",
    "Geometry",
    "Number Theory",
    "Counting & Probability",
    "Prealgebra",
    "Precalculus",
    "Intermediate Algebra",
]

# A deliberately skewed non-IID split. Adjust with --client-map if needed.
DEFAULT_CLIENT_MAP = {
    "client_0": ["Algebra", "Intermediate Algebra", "Prealgebra"],
    "client_1": ["Geometry", "Precalculus"],
    "client_2": ["Number Theory", "Counting & Probability"],
}


def canonical_subject(value: str) -> str:
    """Normalize folder names / dataset labels to the subject names above."""
    normalized = re.sub(r"[_\-\s&]+", " ", value.strip().lower())
    aliases = {
        "algebra": "Algebra",
        "geometry": "Geometry",
        "number theory": "Number Theory",
        "counting probability": "Counting & Probability",
        "counting and probability": "Counting & Probability",
        "prealgebra": "Prealgebra",
        "pre algebra": "Prealgebra",
        "precalculus": "Precalculus",
        "pre calculus": "Precalculus",
        "intermediate algebra": "Intermediate Algebra",
    }
    if normalized not in aliases:
        raise ValueError(f"Unknown MATH subject: {value!r}")
    return aliases[normalized]


def read_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported JSON payload in {path}")


def infer_subject(path: Path, record: dict[str, Any]) -> str:
    for key in ("subject", "category", "type"):
        if key in record and record[key]:
            return canonical_subject(str(record[key]))

    for part in reversed(path.parts):
        try:
            return canonical_subject(part)
        except ValueError:
            continue
    raise ValueError(f"Cannot infer subject for record from {path}")


def load_math_records(input_dir: Path, input_split: str) -> dict[str, list[dict[str, Any]]]:
    split_root = input_dir / input_split
    if split_root.exists():
        roots = [split_root]
    else:
        roots = [input_dir]

    files: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix.lower() in {".json", ".jsonl"} and path not in seen:
                files.append(path)
                seen.add(path)

    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in files:
        for record in read_json_records(path):
            subject = infer_subject(path, record)
            enriched = dict(record)
            enriched.setdefault("subject", subject)
            enriched.setdefault("source_file", str(path.relative_to(input_dir)))
            by_subject[subject].append(enriched)

    missing = [subject for subject in SUBJECTS if subject not in by_subject]
    if missing:
        raise ValueError(
            "Missing subjects in input data: "
            + ", ".join(missing)
            + ". Check --input-dir/--input-split or record subject labels."
        )

    return dict(by_subject)


def load_hf_math_records(dataset_name: str, input_split: str) -> dict[str, list[dict[str, Any]]]:
    try:
        from datasets import Dataset, DatasetDict, load_dataset
    except ImportError as exc:
        raise ImportError(
            "Hugging Face input requires a working `datasets` installation. "
            "If you see an xxhash error, reinstall it with: "
            "pip install --force-reinstall xxhash datasets"
        ) from exc

    ds = load_dataset(dataset_name)
    if isinstance(ds, DatasetDict):
        if input_split not in ds:
            raise ValueError(
                f"Split {input_split!r} not found in {dataset_name}. "
                f"Available splits: {list(ds.keys())}"
            )
        split_ds = ds[input_split]
    elif isinstance(ds, Dataset):
        split_ds = ds
    else:
        raise TypeError(f"Unsupported Hugging Face dataset object: {type(ds)}")

    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(split_ds):
        plain_record = dict(record)
        subject = infer_subject(Path(dataset_name), plain_record)
        enriched = dict(plain_record)
        enriched["subject"] = subject
        enriched.setdefault("source_dataset", dataset_name)
        enriched.setdefault("source_split", input_split)
        enriched.setdefault("source_index", index)
        by_subject[subject].append(enriched)

    missing = [subject for subject in SUBJECTS if subject not in by_subject]
    if missing:
        raise ValueError(
            "Missing subjects in Hugging Face data: "
            + ", ".join(missing)
            + ". Check --input-split or dataset subject labels."
        )

    return dict(by_subject)


def parse_client_map(raw: str | None) -> dict[str, list[str]]:
    if raw is None:
        return DEFAULT_CLIENT_MAP

    loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise ValueError("--client-map must be a JSON object")
    parsed = {
        str(client): [canonical_subject(str(subject)) for subject in subjects]
        for client, subjects in loaded.items()
    }
    if len(parsed) != 3:
        raise ValueError("--client-map must define exactly three clients")
    return parsed


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_records(
    by_subject: dict[str, list[dict[str, Any]]],
    client_map: dict[str, list[str]],
    public_per_subject: int,
    train_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    rng = random.Random(seed)

    public: list[dict[str, Any]] = []
    remaining: dict[str, list[dict[str, Any]]] = {}

    for subject in SUBJECTS:
        records = list(by_subject[subject])
        rng.shuffle(records)
        take = min(public_per_subject, len(records))
        public.extend(records[:take])
        remaining[subject] = records[take:]

    rng.shuffle(public)

    client_splits: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for client, subjects in client_map.items():
        private_records: list[dict[str, Any]] = []
        for subject in subjects:
            private_records.extend(remaining[subject])
        rng.shuffle(private_records)

        train_size = int(len(private_records) * train_ratio)
        client_splits[client] = {
            "train": private_records[:train_size],
            "infer": private_records[train_size:],
        }

    assigned_subjects = [subject for subjects in client_map.values() for subject in subjects]
    unassigned_subjects = sorted(set(SUBJECTS) - set(assigned_subjects))
    duplicate_subjects = sorted(
        subject for subject in set(assigned_subjects) if assigned_subjects.count(subject) > 1
    )
    if unassigned_subjects or duplicate_subjects:
        raise ValueError(
            f"Invalid client subject map. Unassigned={unassigned_subjects}, "
            f"duplicate={duplicate_subjects}"
        )

    metadata = {
        "seed": seed,
        "subjects": SUBJECTS,
        "client_map": client_map,
        "public_per_subject_requested": public_per_subject,
        "train_ratio": train_ratio,
        "counts": {
            "input_by_subject": {
                subject: len(by_subject[subject])
                for subject in SUBJECTS
            },
            "public_by_subject": {
                subject: sum(1 for item in public if item["subject"] == subject)
                for subject in SUBJECTS
            },
            "clients": {
                client: {
                    "train": len(splits["train"]),
                    "infer": len(splits["infer"]),
                    "total": len(splits["train"]) + len(splits["infer"]),
                    "by_subject": {
                        subject: sum(
                            1
                            for split_name in ("train", "infer")
                            for item in splits[split_name]
                            if item["subject"] == subject
                        )
                        for subject in subjects
                    },
                }
                for client, (subjects, splits) in zip(
                    client_map.keys(),
                    [(client_map[c], client_splits[c]) for c in client_map],
                )
            },
        },
    }
    return public, client_splits, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-dir", type=Path)
    input_group.add_argument(
        "--hf-dataset",
        default=None,
        help='Hugging Face dataset name, e.g. "qwedsacf/competition_math"',
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-split", default="train")
    parser.add_argument("--public-per-subject", type=int, default=50)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--client-map",
        default=None,
        help=(
            "Optional JSON object, e.g. "
            '\'{"client_0":["Algebra"],"client_1":["Geometry"],'
            '"client_2":["Number Theory","Counting & Probability",'
            '"Prealgebra","Precalculus","Intermediate Algebra"]}\''
        ),
    )
    args = parser.parse_args()

    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1")
    if args.public_per_subject < 0:
        raise ValueError("--public-per-subject must be non-negative")

    if args.hf_dataset:
        by_subject = load_hf_math_records(args.hf_dataset, args.input_split)
    else:
        by_subject = load_math_records(args.input_dir, args.input_split)
    client_map = parse_client_map(args.client_map)
    public, client_splits, metadata = split_records(
        by_subject=by_subject,
        client_map=client_map,
        public_per_subject=args.public_per_subject,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    write_jsonl(args.output_dir / "public.jsonl", public)
    for client, splits in client_splits.items():
        write_jsonl(args.output_dir / "clients" / client / "train.jsonl", splits["train"])
        write_jsonl(args.output_dir / "clients" / client / "infer.jsonl", splits["infer"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(json.dumps(metadata["counts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
