"""Naive Hugging Face model executor for the synchronous EngineCore."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.scheduler import ScheduledRequest, SchedulerOutput
from .base import Executor, ModelForwardOutput


@dataclass(frozen=True, slots=True)
class NaiveHFExecutorConfig:
    """Hugging Face model loading settings for the no-KV-cache executor."""

    model: str
    device: str = "auto"
    dtype: str = "auto"
    local_files_only: bool = False
    pad_token_id: int | None = None
    eos_token_ids: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must not be empty")


class NaiveHFExecutor(Executor):
    """Run full-context, no-cache HF causal-LM forward passes for each step."""

    def __init__(
        self,
        config: NaiveHFExecutorConfig,
        *,
        model: Any | None = None,
        torch_module: Any | None = None,
    ) -> None:
        self.config = config
        self._torch = torch_module
        if model is None:
            self._torch, model = self._load_model(config)
        if self._torch is None:
            raise RuntimeError("torch_module is required with an injected model")
        self.model = model
        if config.pad_token_id is None:
            raise ValueError("pad_token_id must be provided by the input processor")
        self._pad_token_id = config.pad_token_id
        self._eos_token_ids = config.eos_token_ids
        self._input_device = self._get_input_device()

    @staticmethod
    def _load_model(config: NaiveHFExecutorConfig) -> tuple[Any, Any]:
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError as error:
            raise RuntimeError(
                "NaiveHFExecutor requires x-tokens[hf] (torch and transformers)"
            ) from error

        dtype = None if config.dtype == "auto" else getattr(torch, config.dtype)
        model_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "local_files_only": config.local_files_only,
        }
        if config.device == "auto":
            model_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)
        if config.device not in {"auto", "cpu"}:
            model.to(config.device)
        model.eval()
        return torch, model

    @property
    def eos_token_ids(self) -> frozenset[int]:
        return self._eos_token_ids

    def execute_model(self, batch: SchedulerOutput) -> ModelForwardOutput:
        if not batch.requests:
            return ModelForwardOutput(logits=None)
        contexts = [request.context_token_ids for request in batch.requests]
        max_length = max(len(context) for context in contexts)
        input_ids = [
            [self._pad_token_id] * (max_length - len(context)) + list(context)
            for context in contexts
        ]
        attention_mask = [
            [0] * (max_length - len(context)) + [1] * len(context)
            for context in contexts
        ]
        torch = self._torch
        inputs = {
            "input_ids": torch.tensor(input_ids, device=self._input_device),
            "attention_mask": torch.tensor(attention_mask, device=self._input_device),
        }
        with torch.inference_mode():
            logits = self.model(**inputs, use_cache=False).logits[:, -1, :]
        return ModelForwardOutput(logits=logits)

    def sample_tokens(
        self, output: ModelForwardOutput, batch: SchedulerOutput
    ) -> tuple[int, ...]:
        if not batch.requests:
            return ()
        if output.logits is None:
            raise ValueError("model output logits must not be None")
        logits = output.logits
        return tuple(
            self._sample(logits[index], request)
            for index, request in enumerate(batch.requests)
        )

    def _get_input_device(self) -> Any:
        embeddings = self.model.get_input_embeddings()
        return embeddings.weight.device

    def _sample(self, logits: Any, request: ScheduledRequest) -> int:
        sampling = request.request.sampling
        torch = self._torch
        if sampling.temperature == 0:
            return int(torch.argmax(logits).item())
        scaled = logits / sampling.temperature
        if sampling.top_k is not None:
            top_k = min(sampling.top_k, scaled.shape[-1])
            cutoff = torch.topk(scaled, top_k).values[-1]
            scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))
        if sampling.top_p < 1:
            sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            remove = torch.cumsum(sorted_probs, dim=-1) - sorted_probs > sampling.top_p
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            scaled = torch.full_like(scaled, float("-inf"))
            scaled.scatter_(0, sorted_indices, sorted_logits)
        probabilities = torch.softmax(scaled, dim=-1)
        return int(torch.multinomial(probabilities, 1).item())
