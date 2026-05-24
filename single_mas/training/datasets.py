from __future__ import annotations

import random
from collections.abc import Iterable

from ..models import Agent, Leader
from ..parsing import answers_match, extract_boxed_answer, extract_think_block
from ..prompts import build_agent_round0_prompt, build_backtracking_prompt, build_leader_prompt
from ..schemas import (
    AgentSampleRecord,
    GenerationConfig,
    LeaderPromptRecord,
    SFTExample,
    Task,
)


def collect_agent_samples(
    tasks: Iterable[Task],
    agents: list[Agent],
    samples_per_agent: int = 4,
    generation: GenerationConfig | None = None,
) -> list[AgentSampleRecord]:
    """Generate the paper's 4K round-0 agent responses per task."""

    if samples_per_agent < 1:
        raise ValueError("samples_per_agent must be >= 1")
    generation_config = generation or GenerationConfig(n=samples_per_agent, temperature=0.7)
    records: list[AgentSampleRecord] = []

    for task in tasks:
        samples_by_agent: dict[str, list[str]] = {}
        for agent_number, agent in enumerate(agents, start=1):
            prompt = build_agent_round0_prompt(task.problem, agent_number, len(agents))
            cfg = GenerationConfig(
                n=samples_per_agent,
                temperature=generation_config.temperature,
                max_tokens=generation_config.max_tokens,
                stop=generation_config.stop,
            )
            samples_by_agent[agent.name] = agent.generate(prompt, cfg)
        records.append(AgentSampleRecord(task=task, samples_by_agent=samples_by_agent))

    return records


def build_mlpo_prompt_records(
    agent_sample_records: Iterable[AgentSampleRecord],
    samples_per_task: int = 4,
    easy_threshold: float = 0.75,
    drop_easy_tasks: bool = True,
) -> list[LeaderPromptRecord]:
    """Create grouped MLPO prompts from one response per agent.

    A task is considered easy when at least `easy_threshold` of all sampled agent
    responses match the reference solution. The paper filters those tasks before
    leader optimization.
    """

    records: list[LeaderPromptRecord] = []
    for sample_record in agent_sample_records:
        task = sample_record.task
        all_agent_outputs = [
            output for outputs in sample_record.samples_by_agent.values() for output in outputs
        ]
        correctness = [
            answers_match(output, task.reference_answer) for output in all_agent_outputs
        ]
        easy_ratio = sum(correctness) / len(correctness) if correctness else 0.0
        is_easy = easy_ratio >= easy_threshold
        if drop_easy_tasks and is_easy:
            continue

        max_variants = min(samples_per_task, *(len(v) for v in sample_record.samples_by_agent.values()))
        group_id = task.task_id or task.problem[:80]
        for variant_index in range(max_variants):
            agent_responses = {
                agent_name: outputs[variant_index]
                for agent_name, outputs in sample_record.samples_by_agent.items()
            }
            prompt = build_leader_prompt(task.problem, agent_responses)
            records.append(
                LeaderPromptRecord(
                    group_id=group_id,
                    variant_index=variant_index,
                    task=task,
                    agent_responses=agent_responses,
                    prompt=prompt,
                    reference_answer=extract_boxed_answer(task.reference_answer or ""),
                    easy_task_filtered=is_easy,
                )
            )
    return records


def build_sft_prompt_records(
    tasks: Iterable[Task],
    agents: list[Agent],
    generation: GenerationConfig | None = None,
) -> list[LeaderPromptRecord]:
    """Create SFT leader prompts from one round of agent responses per task."""

    agent_sample_records = collect_agent_samples(
        tasks,
        agents,
        samples_per_agent=1,
        generation=generation or GenerationConfig(n=1, temperature=0.7),
    )
    records: list[LeaderPromptRecord] = []
    for sample_record in agent_sample_records:
        task = sample_record.task
        agent_responses = {
            agent_name: outputs[0]
            for agent_name, outputs in sample_record.samples_by_agent.items()
        }
        records.append(
            LeaderPromptRecord(
                group_id=task.task_id or task.problem[:80],
                variant_index=0,
                task=task,
                agent_responses=agent_responses,
                prompt=build_leader_prompt(task.problem, agent_responses),
                reference_answer=extract_boxed_answer(task.reference_answer or ""),
                easy_task_filtered=False,
            )
        )
    return records


def build_sft_examples(
    prompt_records: Iterable[LeaderPromptRecord],
    leader: Leader,
    completions_per_prompt: int = 16,
    generation: GenerationConfig | None = None,
    seed: int = 42,
) -> list[SFTExample]:
    """Construct SFT examples with optional synthetic backtracking completions."""

    rng = random.Random(seed)
    generation_config = generation or GenerationConfig(n=completions_per_prompt, temperature=0.8)
    examples: list[SFTExample] = []

    for record in prompt_records:
        cfg = GenerationConfig(
            n=completions_per_prompt,
            temperature=generation_config.temperature,
            max_tokens=generation_config.max_tokens,
            stop=generation_config.stop,
        )
        completions = leader.generate(record.prompt, cfg)
        correct = [c for c in completions if answers_match(c, record.task.reference_answer)]
        incorrect = [c for c in completions if not answers_match(c, record.task.reference_answer)]

        if not correct:
            continue
        if not incorrect:
            examples.append(
                SFTExample(
                    prompt=record.prompt,
                    completion=rng.choice(correct),
                    task_id=record.task.task_id,
                    metadata={"source": "all_correct_leader_completion"},
                )
            )
            continue

        correct_completion = rng.choice(correct)
        incorrect_completion = rng.choice(incorrect)
        correct_reasoning = extract_think_block(correct_completion) or correct_completion
        incorrect_reasoning = extract_think_block(incorrect_completion) or incorrect_completion
        correct_answer = extract_boxed_answer(correct_completion) or extract_boxed_answer(record.task.reference_answer or "") or ""
        backtracking_prompt = build_backtracking_prompt(
            question=record.task.problem,
            agent_responses=record.agent_responses,
            incorrect_reasoning=incorrect_reasoning,
            correct_reasoning=correct_reasoning,
            correct_answer=correct_answer,
        )
        backtracked_completion = leader.generate(
            backtracking_prompt,
            GenerationConfig(n=1, temperature=generation_config.temperature, max_tokens=generation_config.max_tokens),
        )[0]
        examples.append(
            SFTExample(
                prompt=record.prompt,
                completion=backtracked_completion,
                task_id=record.task.task_id,
                metadata={"source": "synthetic_backtracking"},
            )
        )

    return examples
