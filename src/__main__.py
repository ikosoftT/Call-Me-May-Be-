from __future__ import annotations

import argparse
import sys

from llm_sdk.llm_sdk import Small_LLM_Model

from .decoder import choose_function_name
from .models import FunctionCallResult
from .parameter_decoder import extract_parameters
from .parser import (
    ParserError,
    load_project_inputs,
    write_results,
)


DEFAULT_FUNCTIONS_PATH = (
    "data/input/functions_definition.json"
)

DEFAULT_INPUT_PATH = (
    "data/input/function_calling_tests.json"
)

DEFAULT_OUTPUT_PATH = (
    "data/output/function_calling_results.json"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Call Me Maybe function-calling tool"
    )

    parser.add_argument(
        "--functions_definition",
        default=DEFAULT_FUNCTIONS_PATH,
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
    )

    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    try:
        project_input = load_project_inputs(
            args.functions_definition,
            args.input,
        )
    except ParserError as exc:
        print(
            f"Error: {exc}",
            file=sys.stderr,
        )
        return 1

    model = Small_LLM_Model()

    results: list[FunctionCallResult] = []

    for item in project_input.prompts:
        try:
            function_name = choose_function_name(
                model,
                item.prompt,
                project_input.functions,
            )

            selected_function = next(
                function
                for function in project_input.functions
                if function.name == function_name
            )

            parameters = extract_parameters(
                model,
                item.prompt,
                selected_function,
            )

            result = FunctionCallResult(
                prompt=item.prompt,
                name=function_name,
                parameters=parameters,
            )

            results.append(result)

        except (
            ValueError,
            RuntimeError,
            NotImplementedError,
        ) as exc:
            print(
                f"Error while processing "
                f"{item.prompt!r}: {exc}",
                file=sys.stderr,
            )

            return 1

    try:
        write_results(
            args.output,
            results,
        )
    except ParserError as exc:
        print(
            f"Error: {exc}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())