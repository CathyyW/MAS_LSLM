from __future__ import annotations

from dataclasses import dataclass

from .parsing import answers_match, has_required_leader_format


@dataclass(slots=True)
class RewardConfig:
    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    format_reward: float = 0.1


def correctness_reward(completion: str, reference_solution: str | None, config: RewardConfig) -> float:
    return config.correct_reward if answers_match(completion, reference_solution) else config.incorrect_reward


def format_reward(completion: str, config: RewardConfig) -> float:
    return config.format_reward if has_required_leader_format(completion) else 0.0


def leader_reward(completion: str, reference_solution: str | None, config: RewardConfig | None = None) -> float:
    cfg = config or RewardConfig()
    return correctness_reward(completion, reference_solution, cfg) + format_reward(completion, cfg)
