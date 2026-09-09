"""CLI entrypoint for Call Me Maybe.

For now this only validates parsing. Later we will replace the placeholder
results with constrained-decoding-generated results.
"""

from __future__ import annotations # Idid for the py detection cycle

import argparse # for Auto Parsing
import sys # for sys stderr etc

from .models import FunctionCallResult # output file
from .parser import ParserError, load_project_inputs, write_results

DEFAULT_FUNCTIONS_PATH = "data/input/functions_definition.json"
DEFAULT_INPUT_PATH = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT_PATH = "data/output/function_calling_results.json"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser matching the subject usage."""
    parser = argparse.ArgumentParser(description="Call Me Maybe function-calling tool")
    parser.add_argument("--functions_definition", default=DEFAULT_FUNCTIONS_PATH)
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    return parser


def main() -> int:
    """Run the parsing stage and write temporary placeholder output."""
    args = build_arg_parser().parse_args()

    try:
        project_input = load_project_inputs(args.functions_definition, args.input)
    except ParserError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    
    print(f"Loaded {len(project_input.functions)} functions")
    print(f"Loaded {len(project_input.prompts)} prompts")

    # Temporary placeholder so the pipeline creates a valid output file now.
    # Next step: replace this with LLM + constrained decoding.
    fallback_name = project_input.functions[0].name if project_input.functions else ""
    results = [
        FunctionCallResult(prompt=item.prompt, name=fallback_name, parameters={})
        for item in project_input.prompts
    ]
    # write the data to output file
    try:
        write_results(args.output, results)
    except ParserError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote placeholder output to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
