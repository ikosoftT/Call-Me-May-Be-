# ==== MAIN CLI PROJECT RUNING POINT === 
from __future__ import annotations # Idid for the py detection cycle

import argparse # for Auto Parsing
import sys # for sys stderr etc

from models import FunctionCallResult # output file
from parser import ParserError, load_project_inputs, write_results

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
    args = build_arg_parser().parse_args()

    try:
        project_input = load_project_inputs(args.functions_definition, args.input)
    except ParserError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
