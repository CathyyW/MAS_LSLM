from __future__ import annotations

from dataclasses import dataclass

from .models import Agent, Leader
from .parsing import extract_boxed_answer
from .prompts import build_agent_revision_prompt, build_agent_round0_prompt, build_leader_prompt
from .schemas import AgentResponse, GenerationConfig, LeaderResponse, MASRun, RoundTrace, Task


@dataclass(slots=True)
class HierarchicalMAS:
    """Paper-style leader-agent inference loop."""

    leader: Leader
    agents: list[Agent]
    agent_generation: GenerationConfig
    leader_generation: GenerationConfig

    def run(self, task: Task, rounds: int = 5) -> MASRun:
        if rounds < 1:
            raise ValueError("rounds must be >= 1")
        if not self.agents:
            raise ValueError("at least one agent is required")

        traces: list[RoundTrace] = []
        previous_agent_texts: dict[str, str] = {}
        previous_leader_text = ""

        for round_index in range(rounds):
            agent_responses: list[AgentResponse] = []
            current_agent_texts: dict[str, str] = {}

            for agent_number, agent in enumerate(self.agents, start=1):
                if round_index == 0:
                    prompt = build_agent_round0_prompt(task.problem, agent_number, len(self.agents))
                else:
                    prompt = build_agent_revision_prompt(
                        task.problem,
                        agent_number,
                        len(self.agents),
                        previous_agent_texts[agent.name],
                        previous_leader_text,
                    )
                text = agent.generate(prompt, self.agent_generation)[0]
                current_agent_texts[agent.name] = text
                agent_responses.append(
                    AgentResponse(agent_name=agent.name, round_index=round_index, prompt=prompt, text=text)
                )

            leader_prompt = build_leader_prompt(task.problem, current_agent_texts)
            leader_text = self.leader.generate(leader_prompt, self.leader_generation)[0]
            leader_response = LeaderResponse(
                round_index=round_index,
                prompt=leader_prompt,
                text=leader_text,
                answer=extract_boxed_answer(leader_text),
            )
            traces.append(
                RoundTrace(
                    round_index=round_index,
                    agent_responses=agent_responses,
                    leader_response=leader_response,
                )
            )
            previous_agent_texts = current_agent_texts
            previous_leader_text = leader_text

        final_answer = traces[-1].leader_response.answer if traces else None
        return MASRun(task=task, rounds=traces, final_answer=final_answer)
