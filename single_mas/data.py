from __future__ import annotations

import json
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

from .schemas import Task


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f):
            if limit is not None and len(records) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_line_number"] = line_number
            records.append(record)
    return records


def load_tasks(path: str | Path, limit: int | None = None) -> list[Task]:
    tasks: list[Task] = []
    for record in load_jsonl(path, limit=limit):
        problem = record.get("problem") or record.get("question") or record.get("prompt")
        if not problem:
            raise ValueError(f"Record has no problem/question/prompt field: {record}")
        task_id = str(record.get("source_index", record.get("id", record["_line_number"])))
        solution = record.get("solution") or record.get("answer")
        metadata = {k: v for k, v in record.items() if k not in {"problem", "question", "prompt", "solution", "answer"}}
        tasks.append(Task(problem=problem, solution=solution, task_id=task_id, metadata=metadata))
    return tasks


def write_jsonl(path: str | Path, records: Iterable[Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            if is_dataclass(record):
                payload = asdict(record)
            else:
                payload = record
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
