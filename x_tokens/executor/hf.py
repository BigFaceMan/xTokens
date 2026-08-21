"""Hugging Face executor for the naive, full-context decode loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.scheduler import ScheduledRequest, SchedulingBatch
from .base import Executor


@dataclass(frozen=True, slots=True)
class HFExecutorConfig:
    """Hugging Face-specific model loading configuration."""

    model: str
    device: str = "auto"
    dtype: str = "auto"
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must not be empty")


class HFExecutor(Executor):
    """Run one no-cache causal-LM forward pass for each scheduled sequence."""

    def __init__(
        self,
        config: HFExecutorConfig,
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
        torch_module: Any | None = None,
    ) -> None:
        self.config = config
        self._torch = torch_module
        if model is None or tokenizer is None:
            self._torch, model, tokenizer = self._load_model(config)
        if self._torch is None:
            raise RuntimeError("torch_module is required with an injected model")
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.tokenizer.pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id or eos_token_id")
        self._eos_token_ids = self._normalize_eos_token_ids(self.tokenizer.eos_token_id)
        self._input_device = self._get_input_device()

    @staticmethod
    def _load_model(config: HFExecutorConfig) -> tuple[Any, Any, Any]:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "HF EngineCore requires x-tokens[hf] (torch and transformers)"
            ) from error

        dtype = None if config.dtype == "auto" else getattr(torch, config.dtype)
        tokenizer = AutoTokenizer.from_pretrained(
            config.model, local_files_only=config.local_files_only
        )
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
        return torch, model, tokenizer

    @staticmethod
    def _normalize_eos_token_ids(value: int | list[int] | None) -> frozenset[int]:
        if value is None:
            return frozenset()
        if isinstance(value, int):
            return frozenset((value,))
        return frozenset(value)

    @property
    def eos_token_ids(self) -> frozenset[int]:
        return self._eos_token_ids

    def encode(self, prompt: str | tuple[int, ...]) -> tuple[int, ...]:
        if isinstance(prompt, tuple):
            return prompt
        encoded = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        return tuple(int(token_id) for token_id in encoded)

    def execute(self, batch: SchedulingBatch) -> tuple[int, ...]:
        if not batch.requests:
            return ()
        contexts = [request.context_token_ids for request in batch.requests]
        max_length = max(len(context) for context in contexts)
        pad_token_id = self.tokenizer.pad_token_id
        input_ids = [
            [pad_token_id] * (max_length - len(context)) + list(context)
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
        return tuple(
            self._sample(logits[index], request)
            for index, request in enumerate(batch.requests)
        )

    def decode_delta(self, request: ScheduledRequest) -> str:
        decoded = self.tokenizer.decode(
            request.output_token_ids, skip_special_tokens=True
        )
        if decoded.startswith(request.decoded_text):
            delta = decoded[len(request.decoded_text) :]
        else:
            # Some tokenizers can rewrite whitespace while decoding. Returning the
            # full new text is safer than dropping user-visible output.
            delta = decoded
        request.decoded_text = decoded
        return delta

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
