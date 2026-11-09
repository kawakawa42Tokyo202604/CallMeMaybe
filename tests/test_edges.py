"""Basic tokenizer and input-mask checks without a model download."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, cast
import unittest

from src.mask import mask
from src.tokenizer import CustomTokenizer

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model


class LocalVocabulary:
    """Supply a small local vocabulary through the SDK's file interface."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def get_path_to_vocab_file(self) -> str:
        return str(self.path)


class BasicTests(unittest.TestCase):
    def setUp(self) -> None:
        workspace = tempfile.TemporaryDirectory(prefix="callmemaybe-test-")
        self.addCleanup(workspace.cleanup)
        self.work = Path(workspace.name)
        self.vocab = {
            "h": 1, "e": 2, "l": 3, "o": 4, "hello": 5,
            "Ġ": 6, "world": 7, "!": 8, "<|unk|>": 9,
            "<|im_end|>": 10, "<|im_start|>": 11,
            "<|endoftext|>": 12, "🌟": 13,
        }
        path = self.work / "vocab.json"
        path.write_text(json.dumps(self.vocab), encoding="utf-8")
        self.tokenizer = CustomTokenizer(
            cast("Small_LLM_Model", LocalVocabulary(path)))
        self.args = argparse.Namespace(
            functions_definition=str(self.work / "functions.json"),
            input=str(self.work / "input.json"))

    def write_inputs(self, definitions: object, prompts: object) -> None:
        Path(self.args.functions_definition).write_text(
            json.dumps(definitions), encoding="utf-8")
        Path(self.args.input).write_text(json.dumps(prompts), encoding="utf-8")

    def test_empty_and_roundtrip(self) -> None:
        for text in ("", "hello", "hello world!", "  hello  ", "hello 🌟"):
            with self.subTest(text=text):
                ids = self.tokenizer.encode(text)
                self.assertEqual(self.tokenizer.decode(ids), text)
        self.assertEqual(self.tokenizer.encode(""), [])

    def test_longest_match_and_unknown_character(self) -> None:
        self.assertEqual(self.tokenizer.encode("hello"), [5])
        self.assertEqual(self.tokenizer.encode("?"), [9])

    def test_control_and_unknown_ids(self) -> None:
        self.assertEqual(self.tokenizer.decode([11, 5, 10, 12]), "hello")
        self.assertEqual(self.tokenizer.decode([-1, 999999]), "")

    def test_input_mask_contains_function_prompt_and_eos(self) -> None:
        self.write_inputs([
            {"name": "hello", "description": "greeting",
             "parameters": {"text": {"type": "string"}}}
        ], [{"prompt": "world"}])
        allowed = mask(self.args, self.tokenizer)
        self.assertTrue({5, 7, self.tokenizer.eos_token_id} <= allowed)
        self.assertTrue(allowed <= set(self.vocab.values()))

    def test_empty_input_lists(self) -> None:
        self.write_inputs([], [])
        self.assertIn(self.tokenizer.eos_token_id,
                      mask(self.args, self.tokenizer))

    def test_missing_and_malformed_json(self) -> None:
        for name in ("functions_definition", "input"):
            for malformed in (False, True):
                with self.subTest(file=name, malformed=malformed):
                    self.write_inputs([], [])
                    path = Path(getattr(self.args, name))
                    if malformed:
                        path.write_text("{", encoding="utf-8")
                    else:
                        path.unlink()
                    with redirect_stdout(io.StringIO()):
                        with self.assertRaises(SystemExit) as error:
                            mask(self.args, self.tokenizer)
                    self.assertEqual(error.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
