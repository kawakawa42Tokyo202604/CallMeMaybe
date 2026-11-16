"""Optional model download and forward-pass check for make test-model."""

import os

import pytest


@pytest.mark.model
def test_real_model_logits_and_tokenizer() -> None:
    """Verify a vocabulary-sized forward pass using the actual SDK model."""
    from llm_sdk import Small_LLM_Model
    from src.tokenizer import CustomTokenizer

    model = Small_LLM_Model(
        model_name=os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B"),
        device="cpu",
    )
    tokenizer = CustomTokenizer(model)
    tokens = tokenizer.encode("What is 2 + 3?")
    assert tokenizer.decode(tokens) == "What is 2 + 3?"
    logits = model.get_logits_from_input_ids(tokens)
    assert len(logits) > max(tokens)
    assert all(isinstance(value, float) for value in logits)
