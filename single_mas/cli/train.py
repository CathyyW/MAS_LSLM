from __future__ import annotations

import argparse

from ..training.trainer import NoOpLeaderTrainer
from .common import load_object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the leader through pluggable SFT/MLPO adapters.")
    parser.add_argument("--sft-data", help="SFT JSONL path.")
    parser.add_argument("--mlpo-data", help="MLPO prompt JSONL path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trainer-factory", help="Optional 'module:function' returning a LeaderTrainer.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = load_object(args.trainer_factory)() if args.trainer_factory else NoOpLeaderTrainer()
    if args.sft_data:
        trainer.train_sft(args.sft_data, args.output_dir)
    if args.mlpo_data:
        trainer.train_mlpo(args.mlpo_data, args.output_dir)
    if not args.sft_data and not args.mlpo_data:
        raise SystemExit("Provide --sft-data and/or --mlpo-data.")


if __name__ == "__main__":
    main()
