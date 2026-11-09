import json
import sys
import argparse
from typing import Any
from src.tokenizer import CustomTokenizer


def mask(args: argparse.Namespace, tokenizer: CustomTokenizer) -> set[int]:
    try:
        allowed_list: set[str] = set()

        func_path = args.functions_definition
        with open(func_path, "r", encoding="utf-8") as f1:
            functions_list: list[dict[str, Any]] = json.load(f1)

        for f in functions_list:
            keys = f.keys()
            for key in keys:
                allowed_list.add(key)
                if key == "parameters" or key == "returns":
                    for v in f[key]:
                        allowed_list.add(v)
                elif key == "description":
                    pass
                else:
                    allowed_list.add(str(f[key]))

        input_path = args.input
        with open(input_path, "r", encoding="utf-8") as f2:
            input_list: list[dict[str, Any]] = json.load(f2)
        for inp in input_list:
            for key, value in inp.items():
                allowed_list.add(key)
                allowed_list.add(value)
        a_list = (
            "\"}{:,_- 0123456789"
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )
        for c in a_list:
            allowed_list.add(c)

        zxc = sorted(allowed_list, key=lambda s: (len(s), s))
        allowed_token: set[int] = set()

        for va in zxc:
            token_ids = tokenizer.encode(va)
            allowed_token.update(token_ids)

        if tokenizer.eos_token_id is not None:
            allowed_token.add(tokenizer.eos_token_id)

        return allowed_token

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)
