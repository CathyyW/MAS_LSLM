from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


class LeaderTrainer(Protocol):
    """Adapter boundary for real SFT and MLPO/GRPO implementations."""

    def train_sft(self, sft_data_path: str | Path, output_dir: str | Path) -> None:
        ...

    def train_mlpo(self, mlpo_data_path: str | Path, output_dir: str | Path) -> None:
        ...


@dataclass(slots=True)
class NoOpLeaderTrainer:
    """Records training intent without touching model weights.

    Replace this with a TRL, verl, or custom trainer. The MLPO data contains the
    prompt, reference answer, and grouped task ids needed for GRPO-style sampling.
    """

    name: str = "noop"

    def train_sft(self, sft_data_path: str | Path, output_dir: str | Path) -> None:
        self._write_manifest("sft", sft_data_path, output_dir)

    def train_mlpo(self, mlpo_data_path: str | Path, output_dir: str | Path) -> None:
        self._write_manifest("mlpo", mlpo_data_path, output_dir)

    def _write_manifest(self, stage: str, data_path: str | Path, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        manifest = {
            "trainer": asdict(self),
            "stage": stage,
            "data_path": str(data_path),
            "note": "NoOpLeaderTrainer does not train weights. Plug in a real trainer adapter here.",
        }
        (out / f"{stage}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
