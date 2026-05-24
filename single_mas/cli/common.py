from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from ..models import Agent, Leader, StubLLM


def build_stub_team(team_size: int) -> tuple[Leader, list[Agent]]:
    leader = Leader(name="leader", backend=StubLLM(name="leader"))
    agents = [
        Agent(name=f"Agent {i}", backend=StubLLM(name=f"agent_{i}"))
        for i in range(1, team_size + 1)
    ]
    return leader, agents


def load_object(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError("Factory spec must look like 'module.submodule:function_name'.")
    module_name, object_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def build_team(team_size: int, factory_spec: str | None = None) -> tuple[Leader, list[Agent]]:
    if factory_spec is None:
        return build_stub_team(team_size)
    factory: Callable[[int], tuple[Leader, list[Agent]]] = load_object(factory_spec)
    return factory(team_size)
