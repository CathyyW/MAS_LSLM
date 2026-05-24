from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .schemas import GenerationConfig


class LLMBackend(Protocol):
    """Small interface all leader and agent model adapters must implement."""

    name: str

    def generate(self, prompt: str, config: GenerationConfig) -> list[str]:
        """Return `config.n` completions for a prompt."""


@dataclass(slots=True)
class Agent:
    name: str
    backend: LLMBackend

    def generate(self, prompt: str, config: GenerationConfig) -> list[str]:
        return self.backend.generate(prompt, config)


@dataclass(slots=True)
class Leader:
    name: str
    backend: LLMBackend

    def generate(self, prompt: str, config: GenerationConfig) -> list[str]:
        return self.backend.generate(prompt, config)


@dataclass(slots=True)
class StubLLM:
    """Deterministic placeholder for pipeline tests.

    Replace this class with a backend that calls your local vLLM server,
    HuggingFace model, OpenAI-compatible endpoint, or another inference stack.
    """

    name: str = "stub"

    def generate(self, prompt: str, config: GenerationConfig) -> list[str]:
        outputs: list[str] = []
        for i in range(config.n):
            outputs.append(
                "<think>\n"
                f"{self.name} placeholder completion {i + 1}. "
                "Plug a real LLM backend into single_mas.models.LLMBackend.\n"
                "</think>\n"
                "<answer>\n"
                "Therefore, the final answer is: $\\boxed{UNKNOWN}$.\n"
                "</answer>"
            )
        return outputs


@dataclass(slots=True)
class HuggingFaceCausalLM:
    """Local Transformers backend for Hugging Face causal language models.

    Dependencies are imported lazily so the lightweight scaffold can still run
    with StubLLM when Transformers/PyTorch are not installed.
    """

    model_id: str
    name: str | None = None
    revision: str | None = None
    device_map: str | dict[str, Any] | None = "auto"
    torch_dtype: str | None = "auto"
    trust_remote_code: bool = False
    use_chat_template: bool = True
    model_kwargs: dict[str, Any] | None = None

    _tokenizer: Any = None
    _model: Any = None

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = self.model_id

    def generate(self, prompt: str, config: GenerationConfig) -> list[str]:
        tokenizer, model = self._load()
        encoded_prompt = self._format_prompt(tokenizer, prompt)
        inputs = tokenizer(encoded_prompt, return_tensors="pt")
        if hasattr(model, "device"):
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": config.max_tokens,
            "num_return_sequences": config.n,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if config.temperature and config.temperature > 0:
            generation_kwargs.update({"do_sample": True, "temperature": config.temperature})
        else:
            generation_kwargs.update({"do_sample": False})

        outputs = model.generate(**inputs, **generation_kwargs)
        prompt_length = inputs["input_ids"].shape[-1]
        completions = [
            tokenizer.decode(output[prompt_length:], skip_special_tokens=True).strip()
            for output in outputs
        ]
        if config.stop:
            completions = [self._apply_stop(text, config.stop) for text in completions]
        return completions

    def _load(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Install Transformers and PyTorch to use HuggingFaceCausalLM, "
                "for example: pip install transformers torch accelerate"
            ) from exc

        kwargs = dict(self.model_kwargs or {})
        if self.device_map is not None:
            kwargs["device_map"] = self.device_map
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
            **kwargs,
        )
        return self._tokenizer, self._model

    def _format_prompt(self, tokenizer: Any, prompt: str) -> str:
        if self.use_chat_template and getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    @staticmethod
    def _apply_stop(text: str, stop: list[str]) -> str:
        earliest: int | None = None
        for marker in stop:
            index = text.find(marker)
            if index != -1 and (earliest is None or index < earliest):
                earliest = index
        return text[:earliest].strip() if earliest is not None else text
