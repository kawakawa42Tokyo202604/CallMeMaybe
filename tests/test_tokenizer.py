"""Unit tests for CustomTokenizer and
 constrained decoding utility functions.
"""
from __future__ import annotations

import json
from pathlib import Path
import string
from typing import TYPE_CHECKING, cast

import pytest
from src.tokenizer import CustomTokenizer
from src.run import mask_logits, resolve_param_value

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model


class LocalVocabulary:
    """Expose a local vocabulary through the SDK's tokenizer interface."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def get_path_to_vocab_file(self) -> str:
        """Return the vocabulary fixture without loading any model."""
        return str(self.path)


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory: pytest.TempPathFactory) -> CustomTokenizer:
    """Exercise the real tokenizer with a complete local ASCII vocabulary."""
    tokens = list(dict.fromkeys(
        string.ascii_letters + string.digits + string.punctuation + "Ġ🌟"
    )) + ["<|unk|>", "<|im_end|>", "<|im_start|>", "<|endoftext|>"]
    path = tmp_path_factory.mktemp("vocabulary") / "vocab.json"
    path.write_text(json.dumps(dict(zip(tokens, range(len(tokens))))),
                    encoding="utf-8")
    return CustomTokenizer(cast("Small_LLM_Model", LocalVocabulary(path)))


class TestCustomTokenizer:
    """Test suite for CustomTokenizer encode and decode methods."""

    def test_encode_decode_roundtrip(self, tokenizer: CustomTokenizer) -> None:
        """Test that encoding and decoding returns the original string."""
        sample_text = "What is the sum of 2 and 3?"
        encoded = tokenizer.encode(sample_text)

        assert isinstance(encoded, list)
        assert len(encoded) > 0
        assert all(isinstance(tid, int) for tid in encoded)

        decoded = tokenizer.decode(encoded)
        assert decoded == sample_text

    def test_encode_space_handling(self, tokenizer: CustomTokenizer) -> None:
        """Test space handling in BPE encoding."""
        text_with_spaces = "Hello World"
        encoded = tokenizer.encode(text_with_spaces)
        decoded = tokenizer.decode(encoded)

        assert decoded == "Hello World"
        assert "!" not in decoded

    def test_special_tokens_filtered_in_decode(
        self, tokenizer: CustomTokenizer
    ) -> None:
        """Test that special control tokens are filtered during decoding."""
        eos_id = tokenizer.eos_token_id
        hello_ids = tokenizer.encode("Hello")

        combined_ids = hello_ids + [eos_id]
        decoded = tokenizer.decode(combined_ids)

        assert "<|im_end|>" not in decoded
        assert "<|endoftext|>" not in decoded
        assert decoded == "Hello"

    def test_empty_string(self, tokenizer: CustomTokenizer) -> None:
        """Test encoding and decoding of an empty string."""
        encoded = tokenizer.encode("")
        assert encoded == []
        decoded = tokenizer.decode([])
        assert decoded == ""

    def test_unknown_or_fallback_character(
        self, tokenizer: CustomTokenizer
    ) -> None:
        """Test handling of characters not directly in vocabulary."""
        rare_text = "Hello 🌟 World"
        encoded = tokenizer.encode(rare_text)
        assert isinstance(encoded, list)
        assert len(encoded) > 0


class TestRunUtilities:
    """Test suite for helper functions in src/run.py."""

    def test_mask_logits(self) -> None:
        """Test that allowed token logits are boosted significantly."""
        logits = [0.1, 0.5, -0.2, 1.0]
        allowed_ids = {1, 3}

        masked = mask_logits(logits, allowed_ids)

        assert masked[0] == 0.1
        assert masked[1] == 0.5 + 100000.0
        assert masked[2] == -0.2
        assert masked[3] == 1.0 + 100000.0

    def test_resolve_param_value_number(self) -> None:
        """Test numeric extraction in parameter resolution."""
        prompt = "What is the sum of 265 and 345?"

        val1 = resolve_param_value("a", "float", prompt, 0)
        val2 = resolve_param_value("b", "float", prompt, 1)

        assert val1 == "265.0"
        assert val2 == "345.0"

    def test_resolve_param_value_string(self) -> None:
        """Test string extraction from quoted prompts."""
        prompt = "Reverse the string 'hello'"

        val = resolve_param_value("s", "string", prompt, 0)
        assert val == '"hello"'
