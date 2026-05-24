from __future__ import annotations


def build_agent_round0_prompt(question: str, agent_number: int, team_size: int) -> str:
    return f"""You are part of a team of {team_size} LLMs responsible for solving a math problem. You are Agent_{agent_number}. Strictly follow the given plan to solve the following math problem by thinking step by step through the plan. Make sure to first define the premises of the question, and make sure that the answer is consistent with the premises. Strictly stick to the facts and question provided.

*** Question: {question} ***

*** Plan: Solve the given math problem. Think step-by-step, providing detailed calculations and reasoning for your steps. ***

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{{[answer]}}$. Where [answer] is just the final number or expression that solves the problem.
"""


def build_agent_revision_prompt(
    question: str,
    agent_number: int,
    team_size: int,
    previous_solution: str,
    leader_output: str,
) -> str:
    return f"""You are part of a team of {team_size} LLMs collaborating to solve a math problem. You are Agent_{agent_number}. Your goal is to improve your previous response using feedback in the form of questions. Strictly follow the plan and think step-by-step through the math problem. Make sure your answer is consistent with the premises and addresses the raised questions thoroughly.

*** Question: {question} ***

Your previous solution: {previous_solution}

Additionally, the aggregator has evaluated agent responses from the previous round and has also raised questions about your previous response.

The aggregator's output: {leader_output}

*** Plan:

1. Carefully reflect on the aggregator's feedback and your previous solution.
2. Revise your answer step-by-step to improve its correctness and clarity. Address each question raised where relevant.
3. Double-check for any logical, calculation, or reasoning errors.

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{{[answer]}}$. Where [answer] is just the final number or expression that solves the problem.
"""


def build_leader_prompt(question: str, agent_responses: dict[str, str]) -> str:
    response_blocks = "\n\n".join(
        f"{agent_name} Response: {response}" for agent_name, response in agent_responses.items()
    )
    return f"""You are an expert aggregator LLM tasked with evaluating multiple agents' responses to a math problem. Your goal is to critically analyze all agent responses, identify correct reasoning or errors, and then provide a unified answer.

Question: {question}

{response_blocks}

Please complete the following two blocks in order and you must write your response in exactly the following two-block format and include no text outside these tags:

1. <think>...</think>: A long, detailed reasoning process.
2. <answer>...</answer>: Your final answer should be aggregated from the best elements of the agents' responses.

- End the answer with: Therefore, the final answer is: $\\boxed{{[answer]}}$.
"""


def build_backtracking_prompt(
    question: str,
    agent_responses: dict[str, str],
    incorrect_reasoning: str,
    correct_reasoning: str,
    correct_answer: str,
) -> str:
    response_blocks = "\n".join(
        f"{agent_name} response: {response}" for agent_name, response in agent_responses.items()
    )
    return f"""You are an AI aggregator tasked with evaluating multiple agent responses to a question and aggregating them into a coherent and accurate final answer. You will be provided with the original prompt of the aggregator, a correct aggregation, and an incorrect aggregation. Your goal is to mimic an aggregator which tries to aggregate the agent responses, first makes mistakes in its reasoning process by going down the wrong path and then backtracks to the correct reasoning.

Question: {question}
{response_blocks}
Incorrect reasoning from Previous Aggregator: {incorrect_reasoning}
Correct reasoning from Previous Aggregator: {correct_reasoning}

Your solution should feel like a continuous, thoughtful monologue that includes:
- One or more initial realistic mistakes in reasoning.
- Natural moments of self-correction and backtracking.
- Accurate evaluation of agent correctness as provided in the correct response.
- A correct final solution as provided in the correct response.

At the end of your response you should include the final answer as: Final answer: $\\boxed{{{correct_answer}}}$
"""
