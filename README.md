# Call Me Maybe

## English

A Python tool that turns a written request such as “add 2 and 3” into a function name and arguments that software can use. It reads function definitions and requests, then uses a small language model (LLM) to write JSON results.

For example, defining `fn_add_numbers` allows it to select that function with arguments such as `a=2` and `b=3` for “What is the sum of 2 and 3?”. The output is data for another program to use when calling a function.

### Example result format

The request “What is the sum of 2 and 3?” is represented in this form. `name` identifies the function and `parameters` contains its arguments.

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  }
]
```

### Run

Requires Python 3.12 or newer and uv.

```sh
make install
make run
# Override input and output paths
make run ARGS="--functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calls.json"
```

On first use, the SDK downloads the model, which defaults to Qwen/Qwen3-0.6B. Results are saved to `data/output/function_calls.json` by default.

| Input | Contents |
| --- | --- |
| `functions_definition.json` | Available function names, descriptions, and argument types |
| `function_calling_tests.json` | Written requests to convert |

### Tests

```sh
make test        # Basic tests without additional libraries
make test-model  # Existing tests using dependencies and a real model
make lint       # flake8 and mypy
```

Basic tests use a small local vocabulary to check empty strings, spaces, longest matching, unknown characters, special tokens, input masks, and JSON loading errors. `make test` does not measure real-model inference accuracy.

### Implementation

- Reads the vocabulary file and splits text using longest matching.
- Uses state transitions and token restrictions to construct JSON.
- Uses Pydantic for input validation and helper rules to extract numbers and strings.

## 日本語

「2と3を足して」のような文章を、プログラムが扱える関数名と引数のデータに変換するPythonのツールです。呼び出せる関数の一覧と指示文を読み込み、小型の言語モデル（LLM）を使ってJSON形式で結果を書き出します。

例えば加算関数 `fn_add_numbers` を定義しておくと、「What is the sum of 2 and 3?」から関数名と `a=2`・`b=3` のような引数を選びます。生成するのは、別のプログラムが関数を呼び出すためのデータです。

### 変換するデータの例

「What is the sum of 2 and 3?」という指示を、次の形式で表します。`name` が呼び出す関数、`parameters` がその関数へ渡す値です。

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  }
]
```

### 実行

Python 3.12以降とuvを使います。

```sh
make install
make run
# 入出力を指定する場合
make run ARGS="--functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calls.json"
```

初回実行時はSDKがモデル（既定：Qwen/Qwen3-0.6B）をダウンロードします。結果は既定で `data/output/function_calls.json` に保存します。

| 入力 | 内容 |
| --- | --- |
| `functions_definition.json` | 関数名・説明・引数の型などの定義 |
| `function_calling_tests.json` | 関数呼び出しに変換する指示文 |

### テスト

```sh
make test        # 追加ライブラリ不要の基本テスト
make test-model  # 依存関係と実モデルを使う既存テスト
make lint        # flake8・mypy
```

基本テストではローカルの小さな語彙を使い、空文字・空白・最長一致・未知文字・特殊トークン・入力マスク・JSON読み込みエラーを確認します。実モデルの推論精度は `make test` の対象外です。

### 実装メモ

- 語彙ファイルを読み込み、最長一致でトークンを分割します。
- 状態遷移とトークン候補の制御でJSONを生成します。
- 入力検証にPydanticを使い、数値や文字列の抽出には補助的なルールも使います。
