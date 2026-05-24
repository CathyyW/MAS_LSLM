from __future__ import annotations

import argparse

from ..data import load_tasks, write_jsonl
from ..schemas import GenerationConfig
from ..training.datasets import build_sft_examples, build_sft_prompt_records
from .common import build_team


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate backtracking-style SFT examples for the leader.")
    parser.add_argument("--input", required=True, help="Task JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path for SFT examples.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--team-size", type=int, default=3)
    parser.add_argument("--team-factory", help="Optional 'module:function' returning (Leader, list[Agent]).")
    parser.add_argument("--agent-temperature", type=float, default=0.7)
    parser.add_argument("--leader-temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--completions-per-prompt", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    leader, agents = build_team(args.team_size, args.team_factory)
    tasks = load_tasks(args.input, limit=args.limit)
    prompt_records = build_sft_prompt_records(
        tasks,
        agents,
        generation=GenerationConfig(
            n=1,
            temperature=args.agent_temperature,
            max_tokens=args.max_tokens,
        ),
    )
    sft_examples = build_sft_examples(
        prompt_records,
        leader,
        completions_per_prompt=args.completions_per_prompt,
        generation=GenerationConfig(
            n=args.completions_per_prompt,
            temperature=args.leader_temperature,
            max_tokens=args.max_tokens,
        ),
    )
    write_jsonl(args.output, [example.to_dict() for example in sft_examples])


if __name__ == "__main__":
    main()
