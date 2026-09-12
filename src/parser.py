from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError  # type:ignore

from models import FunctionCallResult, FunctionDefinition, ProjectInput, PromptInput

# its a Template based but for pydantic.
T = TypeVar("T", bound=BaseModel)


class ParserError(Exception):
    """Raised when input/output parsing fails with a user-friendly message."""


class DuplicateKeyErr(Exception):
    """raised when found  JSON key duplication"""
    pass


def read_json_file(path: str | Path) -> Any:
    """Read a JSON file and return raw Python data.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed Python object.

    Raises:
        ParserError: If the file is missing, unreadable, or invalid JSON.
    """
    file_path = Path(path)
    try:
        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file,
                             object_pairs_hook=reject_duplicated_keys)
    except FileNotFoundError as exc:
        raise ParserError(f"file not found: {file_path}") from exc
    except PermissionError as exc:
        raise ParserError(
            f"permission denied while reading: {file_path}") from exc
    except json.JSONDecodeError as exc:
        raise ParserError(
            f"invalid JSON in {file_path}: line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except DuplicateKeyErr as exc:
        raise ParserError(
            f"Duplication detected at {file_path}: {exc}") from exc
    except OSError as exc:
        raise ParserError(f"cannot read {file_path}: {exc}") from exc


def reject_duplicated_keys(pairs) -> dict:

    obj = dict()

    for key, val in pairs:
        if key in obj:
            raise DuplicateKeyErr(f"a JSON key : '{key}' is duplicated")
        obj[key] = val
    return obj


def validate_list(data: Any, model: type[T], source_name: str) -> list[T]:
    """Validate a raw JSON value as a list of Pydantic model objects."""
    if not isinstance(data, list):
        raise ParserError(f"{source_name} must contain a JSON array")

    validated: list[T] = []
    for index, item in enumerate(data):
        try:
            validated.append(model.model_validate(item))
        except ValidationError as exc:
            raise ParserError(
                f"invalid item at index {index} in {source_name}: {exc}") from exc
    return validated


def load_functions(path: str | Path) -> list[FunctionDefinition]:
    """Load and validate function definitions."""
    raw = read_json_file(path)
    return validate_list(raw, FunctionDefinition, "functions definition file")


def load_prompts(path: str | Path) -> list[PromptInput]:
    """Load and validate prompt inputs."""
    raw = read_json_file(path)
    return validate_list(raw, PromptInput, "prompt input file")


def load_project_inputs(functions_path: str | Path, prompts_path: str | Path) -> ProjectInput:
    """Load both project input files and validate cross-file constraints."""
    functions = load_functions(functions_path)
    prompts = load_prompts(prompts_path)

    for p in prompts:
        if not p.prompt or not p.prompt.strip():
            raise ParserError("Prompts can't be empty")

    try:
        return ProjectInput(functions=functions, prompts=prompts)
    except ValidationError as exc:
        raise ParserError(f"invalid project input: {exc}") from exc


def write_results(path: str | Path, results: list[FunctionCallResult]) -> None:
    """Write final function-call results as pretty valid JSON."""
    output_path = Path(path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [result.model_dump() for result in results]
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
            file.write("\n")
    except OSError as exc:
        raise ParserError(
            f"cannot write output file {output_path}: {exc}") from exc
