"""Simulation runner for LLM function calling with constrained decoding."""

import argparse
import json
import os
import re
import sys
from typing import Any

import numpy as np
from pydantic import BaseModel, ValidationError
from llm_sdk import Small_LLM_Model
from src.tokenizer import CustomTokenizer


class FunctionParameter(BaseModel):
    """Schema for individual function parameter specifications."""

    type: str
    description: str | None = None


class FunctionSpec(BaseModel):
    """Schema for function definitions in functions_definition.json."""

    name: str
    description: str
    parameters: dict[str, Any]
    returns: dict[str, Any] | None = None


class InputPrompt(BaseModel):
    """Schema for input test cases in function_calling_tests.json."""

    prompt: str


class FunctionCallResult(BaseModel):
    """Schema for single output result in function_calls.json."""

    prompt: str
    name: str
    parameters: dict[str, Any]


def mask_logits(
    logits: list[float], allowed_token_ids: set[int]
) -> list[float]:
    """Mask logits by boosting scores for allowed tokens.

    Args:
        logits (list[float]): Raw logits from the model.
        allowed_token_ids (set[int]): Set of allowed token IDs.

    Returns:
        list[float]: Masked logits with boosted scores.
    """
    masked = list(logits)
    for tid in allowed_token_ids:
        if 0 <= tid < len(logits):
            masked[tid] = logits[tid] + 100000.0
    return masked


def resolve_param_value(
    p_name: str, p_type: str, prompt_text: str, num_index: int
) -> str | None:
    """Resolve parameter value based on heuristic matching.

    Args:
        p_name (str): Parameter name.
        p_type (str): Parameter type.
        prompt_text (str): Input prompt text.
        num_index (int): Index for numeric extraction.

    Returns:
        str | None: Resolved JSON string value or None.
    """
    p_lower = p_name.lower()
    prompt_lower = prompt_text.lower()

    if p_type in ["number", "float", "double", "integer", "int"]:
        raw_nums = re.findall(r"-?\d+(?:\.\d+)?", prompt_text)
        if num_index < len(raw_nums):
            val_str = raw_nums[num_index]
            if p_type in ["integer", "int"]:
                return f"{int(float(val_str))}"
            else:
                return f"{float(val_str)}"

    if p_type == "array":
        m_arr = re.search(r"\[(.*?)\]", prompt_text)
        if m_arr:
            raw_inside = m_arr.group(1).strip()
            items = [
                item.strip().strip("'\"") for item in raw_inside.split(",")
                if item.strip()
            ]
            return json.dumps(items)

    if p_type == "object":
        m_obj = re.search(r"\{.*?\}", prompt_text)
        if m_obj:
            try:
                parsed_dict = json.loads(m_obj.group(0))
                return json.dumps(parsed_dict)
            except json.JSONDecodeError:
                pass

    if p_type == "string":
        double_quotes = re.findall(r'"([^"]+)"', prompt_text)
        single_quotes = re.findall(r"'([^']+)'", prompt_text)

        if "template" in p_lower:
            m = re.search(
                r"Format\s+template:\s*(.+)$",
                prompt_text, re.IGNORECASE
            )
            if m:
                return json.dumps(m.group(1).strip())

        if "database" in p_lower or "db" in p_lower:
            m = re.search(
                r"\b([a-zA-Z0-9_-]+)\s+database\b",
                prompt_text, re.IGNORECASE
            )
            if m:
                return json.dumps(m.group(1))

        if "encoding" in p_lower:
            m = re.search(
                r"\b([a-zA-Z0-9_-]+)\s+encoding\b",
                prompt_text, re.IGNORECASE
            )
            if m:
                return json.dumps(m.group(1))

        if "path" in p_lower or "file" in p_lower or "filename" in p_lower:
            m_path = re.search(
                r"(?:[a-zA-Z]:\\[^\s'\"]+|\/[^\s'\"]+)",
                prompt_text
            )
            if m_path:
                return json.dumps(m_path.group(0))
            if double_quotes:
                return json.dumps(double_quotes[0])
            if single_quotes:
                return json.dumps(single_quotes[0])

        if "regex" in p_lower or "pattern" in p_lower:
            if "number" in prompt_lower or "digit" in prompt_lower:
                return json.dumps("[0-9]+")
            if "vowel" in prompt_lower:
                return json.dumps("[aeiouAEIOU]+")
            if "alphabet" in prompt_lower or "letter" in prompt_lower:
                return json.dumps("[a-zA-Z]+")
            m_word = re.search(
                r"word\s+['\"]?([a-zA-Z0-9_-]+)['\"]?",
                prompt_text, re.IGNORECASE
            )
            if m_word:
                return json.dumps(m_word.group(1))
            if single_quotes:
                return json.dumps(single_quotes[0])

        if ("replace" in p_lower or "substitut"
                in p_lower or "replacement" in p_lower):
            if "asterisk" in prompt_lower:
                return json.dumps("*")
            m_with = re.search(
                r"\bwith\s+['\"]?([a-zA-Z0-9_*#-]+)['\"]?",
                prompt_text, re.IGNORECASE
            )
            if m_with:
                val = m_with.group(1)
                if val.lower() in ["asterisk", "asterisks"]:
                    return json.dumps("*")
                return json.dumps(val)
            if len(single_quotes) >= 2:
                return json.dumps(single_quotes[1])

        if "source" in p_lower or "input" in p_lower or "text" in p_lower:
            m_in = re.search(
                r"\bin\s+['\"]([^'\"]+)['\"]",
                prompt_text, re.IGNORECASE
            )
            if m_in:
                return json.dumps(m_in.group(1))
            if double_quotes:
                return json.dumps(double_quotes[0])
            if single_quotes:
                longest = max(single_quotes, key=len)
                return json.dumps(longest)

        if "query" in p_lower or "sql" in p_lower:
            if single_quotes:
                return json.dumps(single_quotes[0])
            if double_quotes:
                return json.dumps(double_quotes[0])

        if single_quotes:
            return json.dumps(single_quotes[0])
        if double_quotes:
            return json.dumps(double_quotes[0])

        words = re.findall(r"[a-zA-Z0-9_-]+", prompt_text)
        stopwords = {
            "greet", "reverse", "calculate", "what", "is", "the", "of", "sum",
            "and", "square", "root", "a", "to", "for", "read",
            "run", "execute", "format"
        }
        filtered = [w for w in words if w.lower() not in stopwords]
        if filtered:
            return json.dumps(filtered[-1])

    return None


def load_input_files(
    func_path: str, input_path: str
) -> tuple[list[FunctionSpec], list[InputPrompt]]:
    """Load and validate input JSON files using Pydantic."""
    try:
        with open(func_path, "r", encoding="utf-8") as f1:
            raw_functions = json.load(f1)
    except FileNotFoundError:
        print(f"エラー: 関数定義ファイルが存在しません: {func_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"エラー: 関数定義JSONの構文が不正です: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(input_path, "r", encoding="utf-8") as f2:
            raw_inputs = json.load(f2)
    except FileNotFoundError:
        print(f"エラー: 入力ファイルが存在しません: {input_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"エラー: 入力JSONの構文が不正です: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        functions_list = [
            FunctionSpec.model_validate(fn) for fn in raw_functions
        ]
        input_list = [
            InputPrompt.model_validate(inp) for inp in raw_inputs
        ]
    except ValidationError as e:
        print(f"エラー: JSONデータの構造・型不一致を検出:\n{e}", file=sys.stderr)
        sys.exit(1)

    return functions_list, input_list


def run(
    args: argparse.Namespace,
    allowed_tokens: set[int],
    llm: Small_LLM_Model,
    tokenizer: CustomTokenizer
) -> None:
    """Execute simulation run with validated input data."""
    try:
        functions_list, input_list = load_input_files(
            args.functions_definition, args.input
        )

        if tokenizer.eos_token_id is not None:
            allowed_tokens.add(tokenizer.eos_token_id)

        quote_tokens = set(tokenizer.encode('"'))
        funcname_list = [f.name for f in functions_list]
        functions_raw_dicts = [f.model_dump() for f in functions_list]
        available_text = "\n".join(
            [json.dumps(f) for f in functions_raw_dicts]
        )

        sim_results: list[dict[str, Any]] = []

        for item in input_list:
            prompt_text = item.prompt
            if not prompt_text:
                continue

            state = 0
            last_state = -1
            token_queue: list[int] = []
            current_fn_name = ""
            param_keys: list[str] = []
            param_index = 0
            target_param_type = "string"
            params_def: dict[str, Any] = {}
            full_generated_json = ""
            bracket_stack: list[str] = []

            if "<|im_start|>" in tokenizer.vocab:
                sys_start = "<|im_start|>system\n"
                user_start = "<|im_start|>user\n"
                asst_start = "<|im_start|>assistant\n"
                turn_end = "<|im_end|>\n"
            elif "[INST]" in tokenizer.vocab or "<s>" in tokenizer.vocab:
                sys_start = "<s>[INST] <<SYS>>\n"
                user_start = ""
                asst_start = "[/INST] "
                turn_end = "\n<</SYS>>\n"
            else:
                sys_start = "System:\n"
                user_start = "User:\n"
                asst_start = "Assistant:\n"
                turn_end = "\n"

            instruct_prompt = (
                f"{sys_start}"
                f"You are a helpful assistant that outputs only a "
                "single JSON object based on the available functions.\n"
                f"Available Functions:\n{available_text}\n"
                f"Required JSON structure:\n"
                '{{"prompt": "<user_prompt>", "name": '
                '"<function_name>", "parameters": {{...}}}}\n'
                f"{turn_end}"
                f"{user_start}"
                f"{prompt_text}{turn_end}"
                f"{asst_start}"
                f"Answer: "
            )

            print(f"\n[Prompt]: {prompt_text}")
            input_ids = tokenizer.encode(instruct_prompt)
            current_ids_constrained = list(input_ids)

            for _ in range(128):
                logits = llm.get_logits_from_input_ids(
                    current_ids_constrained
                )

                if not token_queue:
                    if state == 0:
                        token_queue = list(tokenizer.encode('{"prompt": "'))
                        state = 1
                    elif state == 1:
                        escaped_prompt = json.dumps(prompt_text)[1:-1]
                        token_queue = list(tokenizer.encode(escaped_prompt))
                        state = 2
                    elif state == 2:
                        token_queue = list(tokenizer.encode('", "name": "'))
                        state = 3
                    elif state == 3:
                        pass
                    elif state == 4:
                        if param_index < len(param_keys):
                            p_name = param_keys[param_index]
                            p_info = params_def.get(p_name, {})
                            target_param_type = (
                                p_info.get("type", "string")
                                if isinstance(p_info, dict) else "string"
                            )

                            if (
                                target_param_type in
                                ["number", "float", "double", "integer", "int"]
                            ):
                                num_idx = sum(
                                    1 for k in param_keys[:param_index]
                                    if (
                                        params_def.get(k, {}).get("type")
                                        if isinstance(
                                            params_def.get(k), dict
                                        )
                                        else "string"
                                    )
                                    in [
                                        "number", "float",
                                        "double", "integer", "int"
                                    ]
                                )
                            else:
                                num_idx = 0

                            resolved_val = resolve_param_value(
                                p_name, target_param_type,
                                prompt_text, num_idx
                            )

                            prefix = ""
                            if param_index > 0:
                                prefix += ", "

                            if resolved_val is not None:
                                prefix += f'"{p_name}": {resolved_val}'
                                token_queue = list(tokenizer.encode(prefix))
                                param_index += 1
                            else:
                                prefix += f'"{p_name}": '
                                if target_param_type == "string":
                                    prefix += '"'
                                elif target_param_type == "array":
                                    prefix += '['
                                    bracket_stack.append(']')
                                elif target_param_type == "object":
                                    prefix += '{'
                                    bracket_stack.append('}')
                                token_queue = list(tokenizer.encode(prefix))
                                state = 5
                        else:
                            token_queue = list(tokenizer.encode("}}"))
                            state = 6
                    elif state == 6:
                        break

                was_queue = len(token_queue) > 0

                if token_queue:
                    target_id = token_queue[0]
                    masked_logits = mask_logits(logits, {target_id})
                    token_queue.pop(0)
                elif state == 3:
                    prefix = current_fn_name.strip()
                    matching_names = [
                        name for name in funcname_list
                        if name.startswith(prefix)
                    ]
                    exact_match = any(
                        name == prefix for name in matching_names
                    )

                    if exact_match:
                        fn_allowed_tokens = set(quote_tokens)
                    else:
                        fn_allowed_tokens = set()
                        for name in matching_names:
                            remaining_text = name[len(prefix):]
                            if remaining_text:
                                tokens = tokenizer.encode(remaining_text)
                                if tokens:
                                    fn_allowed_tokens.add(tokens[0])

                    if tokenizer.eos_token_id is not None:
                        fn_allowed_tokens.add(tokenizer.eos_token_id)

                    masked_logits = mask_logits(logits, fn_allowed_tokens)

                elif state == 5:
                    value_allowed_tokens = set(allowed_tokens)
                    if tokenizer.eos_token_id is not None:
                        value_allowed_tokens.add(tokenizer.eos_token_id)
                    masked_logits = mask_logits(logits, value_allowed_tokens)
                else:
                    masked_logits = mask_logits(logits, allowed_tokens)

                next_id = int(np.argmax(masked_logits))

                if next_id == tokenizer.eos_token_id:
                    break

                decoded_token = tokenizer.decode([next_id])

                current_ids_constrained.append(next_id)

                if state != last_state:
                    print(f"\n\033[36m[State {state}]\033[0m ", end="")
                    last_state = state

                print(decoded_token, end="", flush=True)

                full_generated_json += decoded_token

                if not was_queue:
                    if state == 3:
                        if decoded_token != '"':
                            current_fn_name += decoded_token
                        else:
                            fn_name = current_fn_name.strip()
                            target_fn = next((
                                f for f in functions_list
                                if f.name == fn_name
                            ), None)
                            params_def = (
                                target_fn.parameters
                                if target_fn else {}
                            )
                            if (isinstance(params_def, dict)
                                    and "properties" in params_def):
                                params_def = params_def["properties"]
                            param_keys = (
                                list(params_def.keys())
                                if isinstance(params_def, dict) else []
                            )
                            param_index = 0
                            token_queue = list(tokenizer.encode(
                                ', "parameters": {'
                            ))
                            state = 4
                    elif state == 5:
                        for char in decoded_token:
                            if char in ["[", "{"]:
                                bracket_stack.append(
                                    "]" if char == "[" else "}"
                                )
                            elif char in ["]", "}"]:
                                if bracket_stack and bracket_stack[-1] == char:
                                    bracket_stack.pop()

                        if (
                            target_param_type in ["array", "object"]
                            and not bracket_stack
                        ):
                            param_index += 1
                            state = 4
                        elif (
                            target_param_type == "string"
                            and decoded_token == '"'
                        ):
                            param_index += 1
                            state = 4

            else:
                if full_generated_json[-2:] == "}}":
                    pass
                elif full_generated_json[-1] == "}":
                    full_generated_json += "}"
                else:
                    full_generated_json += "}}"

            print()

            try:
                parsed_json = json.loads(full_generated_json)
                validated_result = FunctionCallResult.model_validate(
                    parsed_json
                )
                sim_results.append(validated_result.model_dump())
            except (json.JSONDecodeError, ValidationError) as parse_e:
                print(f" [警告] JSON/構造検証失敗: {parse_e}")
                sim_results.append({
                    "prompt": prompt_text,
                    "error": "JSON parse or schema error",
                    "raw": full_generated_json
                })

        output_path = getattr(
            args, "output", "data/output/function_calls.json"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(sim_results, out_f, indent=4, ensure_ascii=False)

        print(f"\n成果物を JSON 形式で保存しました: {output_path}")

    except BaseException as e:
        err_type = type(e).__name__
        print(
            f"\nエラーが発生しました [{err_type}]:\n  {e}\n",
            file=sys.stderr
        )
        sys.exit(1)
