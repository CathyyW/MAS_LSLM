from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import Agent, HuggingFaceCausalLM, Leader, LLMBackend, StubLLM


@dataclass(frozen=True, slots=True)
class ModelConfig:
    role: str
    name: str
    model_id: str
    revision: str | None = None
    use_chat_template: bool = True
    trust_remote_code: bool = False
    model_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PaperReproductionConfig:
    leader: ModelConfig
    agents: tuple[ModelConfig, ...]
    sft_completions_per_prompt: int = 16
    agent_samples_per_task: int = 4
    easy_threshold: float = 0.75


DEFAULT_REPRO_CONFIG = PaperReproductionConfig(
    leader=ModelConfig(
        role="leader",
        name="DeepSeek-R1-Distill-Qwen-1.5B",
        model_id="models/DeepSeek-R1-Distill-Qwen-1.5B",
    ),
    agents=(
        ModelConfig(
            role="agent",
            name="Gemma-2-2B-it",
            model_id="models/gemma-2-2b-it",
        ),
        ModelConfig(
            role="agent",
            name="Sheared-LLaMA-1.3B-ShareGPT",
            model_id="models/Sheared-LLaMA-1.3B-ShareGPT",
        ),
        ModelConfig(
            role="agent",
            name="Qwen3-1.7B",
            model_id="models/Qwen3-1.7B-FP8",
        ),
    ),
)


def build_backend(config: ModelConfig, *, backend: str = "hf") -> LLMBackend:
    if backend == "stub":
        return StubLLM(name=config.name)
    if backend != "hf":
        raise ValueError(f"Unknown backend {backend!r}; expected 'hf' or 'stub'.")
    model_id = _resolve_model_id(config.model_id)
    model_kwargs = dict(config.model_kwargs)
    if Path(model_id).exists():
        model_kwargs.setdefault("local_files_only", True)
    return HuggingFaceCausalLM(
        model_id=model_id,
        name=config.name,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
        use_chat_template=config.use_chat_template,
        model_kwargs=model_kwargs,
    )


def _resolve_model_id(model_id: str) -> str:
    candidate = Path(model_id)
    if candidate.is_absolute():
        return str(candidate)
    workspace_root = Path(__file__).resolve().parents[1]
    resolved = (workspace_root / candidate).resolve()
    if resolved.exists():
        return str(resolved)
    return model_id


def build_reproduction_team(team_size: int = 3, *, backend: str = "hf") -> tuple[Leader, list[Agent]]:
    if team_size != len(DEFAULT_REPRO_CONFIG.agents):
        raise ValueError(
            f"DEFAULT_REPRO_CONFIG defines {len(DEFAULT_REPRO_CONFIG.agents)} agents; "
            f"got team_size={team_size}."
        )
    leader = Leader(
        name=DEFAULT_REPRO_CONFIG.leader.name,
        backend=build_backend(DEFAULT_REPRO_CONFIG.leader, backend=backend),
    )
    agents = [
        Agent(name=config.name, backend=build_backend(config, backend=backend))
        for config in DEFAULT_REPRO_CONFIG.agents
    ]
    return leader, agents


def build_stub_reproduction_team(team_size: int = 3) -> tuple[Leader, list[Agent]]:
    return build_reproduction_team(team_size, backend="stub")
