from __future__ import annotations

import argparse
import json

from ..data import load_tasks, write_jsonl
from ..inference import HierarchicalMAS
from ..schemas import GenerationConfig, Task
from .common import build_team


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hierarchical leader-agent MAS inference.")
    parser.add_argument("--question", type=str, help="Single question to solve.")
    parser.add_argument("--input", type=str, help="JSONL file containing problem/question fields.")
    parser.add_argument("--output", type=str, help="Optional JSONL output path.")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--team-size", type=int, default=3)
    parser.add_argument("--team-factory", help="Optional 'module:function' returning (Leader, list[Agent]).")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.question and not args.input:
        raise SystemExit("Provide --question or --input.")

    leader, agents = build_team(args.team_size, args.team_factory)
    mas = HierarchicalMAS(
        leader=leader,
        agents=agents,
        agent_generation=GenerationConfig(n=1),
        leader_generation=GenerationConfig(n=1),
    )

    if args.question:
        tasks = [Task(problem=args.question, task_id="manual")]
    else:
        tasks = load_tasks(args.input, limit=args.limit)

    runs = [mas.run(task, rounds=args.rounds) for task in tasks]
    if args.output:
        write_jsonl(args.output, [run.to_dict() for run in runs])
    else:
        print(json.dumps([run.to_dict() for run in runs], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
