from __future__ import annotations

import argparse

from ..data import load_tasks, write_jsonl
from ..schemas import GenerationConfig
from ..training.datasets import build_mlpo_prompt_records, collect_agent_samples
from .common import build_team


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate offline agent samples and MLPO prompt records.")
    parser.add_argument("--input", required=True, help="JSONL task file.")
    parser.add_argument("--output", required=True, help="Output JSONL path for MLPO leader prompts.")
    parser.add_argument("--agent-samples-output", help="Optional JSONL path for raw agent samples.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--team-size", type=int, default=3)
    parser.add_argument("--team-factory", help="Optional 'module:function' returning (Leader, list[Agent]).")
    parser.add_argument("--samples-per-agent", type=int, default=4)
    parser.add_argument("--easy-threshold", type=float, default=0.75)
    parser.add_argument("--keep-easy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, agents = build_team(args.team_size, args.team_factory)
    tasks = load_tasks(args.input, limit=args.limit)
    agent_samples = collect_agent_samples(
        tasks,
        agents,
        samples_per_agent=args.samples_per_agent,
        generation=GenerationConfig(n=args.samples_per_agent),
    )
    if args.agent_samples_output:
        write_jsonl(args.agent_samples_output, [record.to_dict() for record in agent_samples])

    prompt_records = build_mlpo_prompt_records(
        agent_samples,
        samples_per_task=args.samples_per_agent,
        easy_threshold=args.easy_threshold,
        drop_easy_tasks=not args.keep_easy,
    )
    write_jsonl(args.output, [record.to_dict() for record in prompt_records])


if __name__ == "__main__":
    main()
