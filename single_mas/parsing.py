from __future__ import annotations

import re


_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)


def extract_answer_block(text: str) -> str | None:
    match = _ANSWER_BLOCK_RE.search(text)
    if not match:
        return None
    return match.group(1).strip()


def extract_think_block(text: str) -> str | None:
    match = _THINK_BLOCK_RE.search(text)
    if not match:
        return None
    return match.group(1).strip()


def extract_boxed_answer(text: str) -> str | None:
    matches = _BOXED_RE.findall(text)
    if matches:
        return matches[-1].strip()
    answer_block = extract_answer_block(text)
    if answer_block:
        return answer_block.strip()
    return None


def normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    normalized = answer.strip()
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace("$", "")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.rstrip(".")
    return normalized.lower()


def answers_match(candidate_text: str, reference_text: str | None) -> bool:
    if reference_text is None:
        return False
    candidate = normalize_answer(extract_boxed_answer(candidate_text) or candidate_text)
    reference = normalize_answer(extract_boxed_answer(reference_text) or reference_text)
    return candidate is not None and reference is not None and candidate == reference


def has_required_leader_format(text: str) -> bool:
    return extract_think_block(text) is not None and extract_answer_block(text) is not None
