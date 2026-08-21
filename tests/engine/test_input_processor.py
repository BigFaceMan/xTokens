from __future__ import annotations

import pytest

from x_tokens.engine.input_processor import TokenizerInputProcessor
from x_tokens.engine.output_processor import TokenizerOutputProcessor
from x_tokens.engine.types import GenerateRequest, SamplingParams


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert add_special_tokens
        return {"input_ids": [ord(character) for character in text]}

    def decode(self, token_ids: tuple[int, ...], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids if token_id != 9)


def test_processes_text_and_pre_tokenized_prompts() -> None:
    processor = TokenizerInputProcessor(FakeTokenizer())
    text_request = GenerateRequest("text", "model", "hi", SamplingParams())
    ids_request = GenerateRequest("ids", "model", (1, 2), SamplingParams())

    assert processor.process(text_request).prompt == (104, 105)
    assert processor.process(ids_request).prompt == (1, 2)


def test_rejects_empty_prompt() -> None:
    processor = TokenizerInputProcessor(FakeTokenizer())
    empty_request = GenerateRequest("empty", "model", "", SamplingParams())

    with pytest.raises(ValueError, match="prompt must contain"):
        processor.process(empty_request)


def test_output_processor_decodes_incrementally_and_cleans_up() -> None:
    processor = TokenizerOutputProcessor(FakeTokenizer())

    assert processor.process_token("request", 104) == "h"
    assert processor.process_token("request", 105) == "i"
    processor.finish("request")
    assert processor.process_token("request", 104) == "h"
