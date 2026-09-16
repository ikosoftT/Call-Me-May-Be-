from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, TypeVar

from pydantic import BaseModel, ValidationError

from .models import (
    FunctionCallResult,
    FunctionDefinition,
    ProjectInput,
    PromptInput,
)

T = TypeVar("T", bound=BaseModel)


class ParserError(Exception):
    """Raised when input/output parsing fails with a user-friendly message."""


class DuplicateKeyErr(Exception):
    """Raised when a JSON object contains the same key more than once."""


def format_validation_error(exc: ValidationError) -> str:
    """Convert a Pydantic error into a short user-facing message.

    Args:
        exc: Pydantic validation exception.

    Returns:
        A readable string listing every invalid field and the reason.
    """
    messages: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        if not location:
            location = "<root>"
        messages.append(f"{location}: {error['msg']}")
    return "; ".join(messages)


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
            return json.load(
                file,
                object_pairs_hook=reject_duplicated_keys,
            )
    except FileNotFoundError as exc:
        raise ParserError(f"file not found: {file_path}") from exc
    except PermissionError as exc:
        raise ParserError(
            f"permission denied while reading: {file_path}") from exc
    except json.JSONDecodeError as exc:
        raise ParserError(
            f"invalid JSON in {file_path}: line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc
    except DuplicateKeyErr as exc:
        raise ParserError(
            f"duplicate JSON key in {file_path}: {exc}"
        ) from exc
    except OSError as exc:
        raise ParserError(f"cannot read {file_path}: {exc}") from exc


def reject_duplicated_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicated keys while json.load builds each object.

    Args:
        pairs: Key/value pairs from one JSON object.

    Returns:
        A dictionary containing the same key/value pairs.

    Raises:
        DuplicateKeyErr: If the same key appears twice in the object.
    """
    obj: dict[str, Any] = {}

    for key, val in pairs:
        if key in obj:
            raise DuplicateKeyErr(f"key {key!r} appears more than once")
        obj[key] = val
    return obj


def validate_list(data: Any, model: type[T], source_name: str) -> list[T]:
    """Validate a raw JSON value as a list of Pydantic model objects.

    Args:
        data: Raw value loaded from JSON.
        model: Pydantic model used to validate each list item.
        source_name: Human-readable name used in error messages.

    Returns:
        A list of validated Pydantic objects.

    Raises:
        ParserError: If the root value is not a list or one item is invalid.
    """
    if not isinstance(data, list):
        raise ParserError(f"{source_name} must contain a JSON array")

    validated: list[T] = []
    for index, item in enumerate(data):
        try:
            validated.append(model.model_validate(item))
        except ValidationError as exc:
            raise ParserError(
                f"invalid item at index {index} in {source_name}: "
                f"{format_validation_error(exc)}"
            ) from exc
    return validated


def load_functions(path: str | Path) -> list[FunctionDefinition]:
    """Load and validate the function definition file.

    Args:
        path: Path to `functions_definition.json`.

    Returns:
        Validated function definitions.

    Raises:
        ParserError: If reading or validation fails.
    """
    raw = read_json_file(path)
    return validate_list(raw, FunctionDefinition, "functions definition file")


def load_prompts(path: str | Path) -> list[PromptInput]:
    """Load and validate the prompt input file.

    Args:
        path: Path to `function_calling_tests.json`.

    Returns:
        Validated prompt objects.

    Raises:
        ParserError: If reading or validation fails.
    """
    raw = read_json_file(path)
    return validate_list(raw, PromptInput, "prompt input file")


def load_project_inputs(
    functions_path: str | Path,
    prompts_path: str | Path,
) -> ProjectInput:
    """Load both project input files and validate cross-file constraints.

    Args:
        functions_path: Path to the function definitions file.
        prompts_path: Path to the prompt tests file.

    Returns:
        A validated project input object.

    Raises:
        ParserError: If either file is invalid or prompts are empty.
    """
    functions = load_functions(functions_path)
    prompts = load_prompts(prompts_path)

    for index, prompt in enumerate(prompts):
        if not prompt.prompt or not prompt.prompt.strip():
            raise ParserError(
                f"prompt input file item {index}: prompt cannot be empty"
            )

    try:
        return ProjectInput(functions=functions, prompts=prompts)
    except ValidationError as exc:
        raise ParserError(
            f"invalid project input: {format_validation_error(exc)}"
        ) from exc


def write_results(path: str | Path, results: list[FunctionCallResult]) -> None:
    """Write final function-call results as pretty valid JSON.

    Args:
        path: Destination JSON file path.
        results: Validated function-call result objects.

    Raises:
        ParserError: If the directory/file cannot be written or if the payload
            contains a non-standard JSON value such as NaN or Infinity.
    """
    output_path = Path(path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [result.model_dump() for result in results]
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                payload,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            file.write("\n")
    except ValueError as exc:
        raise ParserError(
            f"cannot write non-standard JSON value to {output_path}: {exc}. "
            "JSON output must not contain NaN or Infinity."
        ) from exc
    except OSError as exc:
        raise ParserError(
            f"cannot write output file {output_path}: {exc}"
        ) from exc
