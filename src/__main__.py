from __future__ import annotations

import argparse
import sys

from llm_sdk.llm_sdk import Small_LLM_Model

from .decoder import choose_function_name
from .models import FunctionCallResult, FunctionDefinition
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
    """Build the command-line argument parser.

    Returns:
        Configured `argparse.ArgumentParser` with input/output path options.
    """
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


def find_function_definition(
    function_name: str,
    functions: list[FunctionDefinition],
) -> FunctionDefinition:
    """Find a function definition by name.

    Args:
        function_name: Name selected by the decoder.
        functions: Available function definitions.

    Returns:
        The matching function definition.

    Raises:
        ValueError: If no function has the requested name.
    """
    for function in functions:
        if function.name == function_name:
            return function

    raise ValueError(f"selected function {function_name!r} is not defined")


def main() -> int:
    """Run the complete function-calling pipeline.

    Returns:
        Process exit code. `0` means success, `1` means a handled error.
    """
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

    try:
        model = Small_LLM_Model()
    except Exception as exc:
        print(
            f"Error: failed to load the LLM model: {exc}",
            file=sys.stderr,
        )
        return 1

    results: list[FunctionCallResult] = []

    for item in project_input.prompts:
        try:
            function_name = choose_function_name(
                model,
                item.prompt,
                project_input.functions,
            )
            selected_function = find_function_definition(
                function_name,
                project_input.functions,
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
        except Exception as exc:
            print(
                f"Unexpected error while processing {item.prompt!r}: {exc}",
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
