import argparse
from llm_sdk import Small_LLM_Model
from src.tokenizer import CustomTokenizer
from src.mask import mask
from src.run import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Function Calling Tool")

    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json"
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json"
    )

    args = parser.parse_args()

    llm = Small_LLM_Model()
    tokenizer = CustomTokenizer(llm)

    allowed_token: set[int] = mask(args, tokenizer)
    run(args, allowed_token, llm, tokenizer)


if __name__ == "__main__":
    main()
