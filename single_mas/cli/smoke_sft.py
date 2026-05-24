from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from ..data import load_tasks, write_jsonl
from ..parsing import answers_match
from ..schemas import GenerationConfig
from ..training.datasets import build_sft_examples, build_sft_prompt_records
from .common import build_team


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one-task SFT data smoke test: agents answer, leader samples 16 completions, then build SFT example."
    )
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "math_split" / "clients" / "client_0" / "train.jsonl"),
        help="Task JSONL path. Defaults to client_0 train split.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "single_mas" / "outputs" / "sft_smoke_client0.jsonl"),
    )
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
    task = load_tasks(args.input, limit=1)[0]

    prompt_record = build_sft_prompt_records(
        [task],
        agents,
        generation=GenerationConfig(
            n=1,
            temperature=args.agent_temperature,
            max_tokens=args.max_tokens,
        ),
    )[0]

    leader_generation = GenerationConfig(
        n=args.completions_per_prompt,
        temperature=args.leader_temperature,
        max_tokens=args.max_tokens,
    )
    leader_completions = leader.generate(prompt_record.prompt, leader_generation)
    correctness = [
        answers_match(completion, task.reference_answer)
        for completion in leader_completions
    ]

    # Build the actual backtracking-style SFT record using the same pipeline.
    replay_backend = _ReplayLeaderBackend(leader_completions, leader.backend)
    replay_leader = type(leader)(name=leader.name, backend=replay_backend)
    sft_examples = build_sft_examples(
        [prompt_record],
        replay_leader,
        completions_per_prompt=args.completions_per_prompt,
        generation=leader_generation,
    )

    write_jsonl(
        args.output,
        [
            {
                "task": asdict(task),
                "agent_responses": prompt_record.agent_responses,
                "leader_prompt": prompt_record.prompt,
                "leader_completions": [
                    {"text": completion, "is_correct": is_correct}
                    for completion, is_correct in zip(leader_completions, correctness)
                ],
                "sft_examples": [example.to_dict() for example in sft_examples],
            }
        ],
    )
    print(f"Wrote SFT smoke output to {args.output}")
    print(f"Leader completions: {len(leader_completions)}")
    print(f"Correct completions: {sum(correctness)}")
    print(f"SFT examples: {len(sft_examples)}")


class _ReplayLeaderBackend:
    """Reuse the already sampled leader outputs for build_sft_examples."""

    def __init__(self, first_outputs: list[str], fallback_backend) -> None:
        self.name = "replay"
        self._first_outputs = first_outputs
        self._fallback_backend = fallback_backend
        self._used_first = False

    def generate(self, prompt: str, config: GenerationConfig) -> list[str]:
        if not self._used_first:
            self._used_first = True
            return self._first_outputs[: config.n]
        return self._fallback_backend.generate(prompt, config)


if __name__ == "__main__":
    main()
