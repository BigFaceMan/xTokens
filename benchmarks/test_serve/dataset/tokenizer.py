"""Optional tokenizer integration with a dependency-free fallback."""

from __future__ import annotations

from typing import Any


class BasicTokenizer:
    """Small deterministic tokenizer used when transformers is unavailable.

    It is intentionally only a length estimator. The benchmark never needs a
    local model or tokenizer to talk to an already-running inference server.
    """

    def encode(self, text: str) -> list[str]:
        return text.split()

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)


def load_tokenizer(name: str | None, trust_remote_code: bool = False) -> Any:
    """Load a tokenizer when requested, otherwise return ``BasicTokenizer``.

    A tokenizer is optional because the server can report usage in its final
    streaming chunk. Importing transformers is delayed to keep this package
    useful in minimal benchmark environments.
    """
    if not name:
        return BasicTokenizer()
    try:
        from transformers.models.auto.tokenization_auto import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "--tokenizer was supplied but transformers is not installed"
        ) from exc
    return AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)


def count_tokens(tokenizer: Any, text: str) -> int:
    if hasattr(tokenizer, "count"):
        return tokenizer.count(text)
    encoded = tokenizer(text, add_special_tokens=False)
    ids = getattr(encoded, "input_ids", encoded)
    return len(ids)


def make_prompt(tokenizer: Any, length: int, seed: int) -> str:
    """Create a deterministic prompt with approximately ``length`` tokens.

    ``BasicTokenizer`` has no vocabulary, so it uses one whitespace-delimited
    word per estimated token. Hugging Face tokenizers can decode token IDs;
    generate from their non-special vocabulary instead. This prevents strings
    such as ``token_123`` from expanding into multiple subword tokens and
    makes random-workload lengths comparable with the server's tokenizer.
    """
    if length <= 0:
        return ""
    if isinstance(tokenizer, BasicTokenizer):
        return " ".join(f"token_{(seed + i) % 100000}" for i in range(length))

    vocab_size = getattr(tokenizer, "vocab_size", None)
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    if not isinstance(vocab_size, int) or vocab_size <= len(special_ids):
        return " ".join(f"token_{(seed + i) % 100000}" for i in range(length))

    # Start away from byte/control-token-heavy low vocabulary IDs. Re-encode
    # after decoding and extend/truncate until the prompt reaches the target.
    # A deterministic offset makes all generated workloads reproducible.
    token_ids: list[int] = []
    candidate = (vocab_size // 2 + seed) % vocab_size
    for _ in range(max(1, length * 4)):
        if candidate not in special_ids:
            token_ids.append(candidate)
            prompt = tokenizer.decode(token_ids, skip_special_tokens=True)
            encoded = tokenizer.encode(prompt, add_special_tokens=False)
            if len(encoded) >= length:
                return tokenizer.decode(encoded[:length], skip_special_tokens=True)
        candidate = (candidate + 1) % vocab_size

    # Retain a usable deterministic fallback for unusual tokenizer APIs.
    return " ".join(f"token_{(seed + i) % 100000}" for i in range(length))
