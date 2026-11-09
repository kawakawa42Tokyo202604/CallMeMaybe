from __future__ import annotations

import json
from typing import Dict, TYPE_CHECKING
from functools import lru_cache

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model


class CustomTokenizer:
    def __init__(self, llm: Small_LLM_Model) -> None:
        vocab_path = llm.get_path_to_vocab_file()
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab: Dict[str, int] = json.load(f)

        self.inv_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}

        self.eos_token_id: int = self.vocab.get(
            "<|im_end|>", self.vocab.get("<|endoftext|>", 151643)
        )

    @lru_cache(maxsize=None)
    def encode(self, text: str) -> list[int]:
        """テキストを自作の最長一致アルゴリズムでトークンIDリストに変換する。"""
        prepared_text = text.replace(" ", "Ġ")

        res: list[int] = []
        i = 0
        n = len(prepared_text)

        while i < n:
            match_found = False

            for j in range(n, i, -1):
                sub_str = prepared_text[i:j]

                if sub_str in self.vocab:
                    res.append(self.vocab[sub_str])
                    i = j
                    match_found = True
                    break

            if not match_found:
                char = prepared_text[i]
                if char in self.vocab:
                    res.append(self.vocab[char])
                else:
                    unk_id = self.vocab.get(
                        "<|unk|>", self.vocab.get("<|endoftext|>", 0)
                    )
                    res.append(unk_id)
                i += 1

        return res

    def decode(self, token_ids: list[int]) -> str:
        """トークンIDのリストを元の文字列に復元する（自作デコード）。"""
        decoded_tokens: list[str] = []

        for tid in token_ids:
            if tid in self.inv_vocab:
                token_str = self.inv_vocab[tid]

                if (
                    token_str in ["<|im_start|>",
                                  "<|im_end|>", "<|endoftext|>"]
                        ):
                    continue

                decoded_tokens.append(token_str)

        raw_text = "".join(decoded_tokens)

        cleaned_text = raw_text.replace("Ġ", " ").replace(" ", " ")

        return cleaned_text
