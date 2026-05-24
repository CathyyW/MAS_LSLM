from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Task:
    """A single reasoning problem."""

    problem: str
    solution: str | None = None
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reference_answer(self) -> str | None:
        return self.solution


@dataclass(slots=True)
class GenerationConfig:
    n: int = 1
    temperature: float = 0.7
    max_tokens: int = 2048
    stop: list[str] | None = None


@dataclass(slots=True)
class AgentResponse:
    agent_name: str
    round_index: int
    prompt: str
    text: str


@dataclass(slots=True)
class LeaderResponse:
    round_index: int
    prompt: str
    text: str
    answer: str | None = None


@dataclass(slots=True)
class RoundTrace:
    round_index: int
    agent_responses: list[AgentResponse]
    leader_response: LeaderResponse


@dataclass(slots=True)
class MASRun:
    task: Task
    rounds: list[RoundTrace]
    final_answer: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentSampleRecord:
    """Offline agent responses for one task."""

    task: Task
    samples_by_agent: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LeaderPromptRecord:
    """One MLPO/GRPO prompt: task plus one response from each agent."""

    group_id: str
    variant_index: int
    task: Task
    agent_responses: dict[str, str]
    prompt: str
    reference_answer: str | None = None
    easy_task_filtered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SFTExample:
    prompt: str
    completion: str
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
