from __future__ import annotations

import re
from typing import Any

from .decoder import constrained_choice
from .models import FunctionDefinition, TypeDefinition


def get_numbers(prompt: str) -> list[str]:
    """
    Extract numbers from the user prompt.
    """

    return re.findall(
        r"-?\d+(?:\.\d+)?",
        prompt,
    )


def get_quoted_strings(prompt: str) -> list[str]:
    result = []
    i = 0

    while i < len(prompt):
        quote = prompt[i]

        if quote not in {"'", '"'}:
            i += 1
            continue

        i += 1
        chars = []
        escaped = False

        while i < len(prompt):
            char = prompt[i]

            if escaped:
                chars.append(char)
                escaped = False

            elif char == "\\":
                escaped = True

            elif char == quote:
                result.append("".join(chars))
                break

            else:
                chars.append(char)

            i += 1

        i += 1

    return result

def get_source_string_candidates(
    user_prompt: str,
) -> list[str]:
    """
    Extract possible source strings.
    """

    quoted = get_quoted_strings(user_prompt)

    if not quoted:
        return []

    # Usually the source string is the longest quoted value.
    return [
        max(quoted, key=len)
    ]


def get_regex_candidates(
    user_prompt: str,
) -> list[str]:
    """
    Build possible regex values from the request.
    """

    lower = user_prompt.lower()

    # Example:
    # Replace all numbers ...
    if "number" in lower or "numbers" in lower:
        return [
            r"\d+",
            r"[0-9]+",
        ]

    # Example:
    # Replace all vowels ...
    if "vowel" in lower or "vowels" in lower:
        return [
            r"[aeiouAEIOU]",
            r"[aeiou]",
        ]

    # Example:
    # Substitute the word 'cat' ...
    word_match = re.search(
        r"\bword\s+['\"]([^'\"]+)['\"]",
        user_prompt,
        re.IGNORECASE,
    )

    if word_match:
        return [
            re.escape(word_match.group(1))
        ]

    # Example:
    # regex '...'
    regex_match = re.search(
        r"\b(?:regex|pattern)\s+['\"]([^'\"]+)['\"]",
        user_prompt,
        re.IGNORECASE,
    )

    if regex_match:
        return [
            regex_match.group(1)
        ]

    return []


def get_replacement_candidates(
    user_prompt: str,
) -> list[str]:
    """
    Extract possible replacement values.
    """

    lower = user_prompt.lower()

    # Example:
    # "with asterisks"
    if "asterisk" in lower:
        return ["*"]

    # Example:
    # with 'dog'
    quoted_match = re.search(
        r"\bwith\s+['\"]([^'\"]+)['\"]",
        user_prompt,
        re.IGNORECASE,
    )

    if quoted_match:
        return [
            quoted_match.group(1)
        ]

    # Example:
    # with NUMBERS
    normal_match = re.search(
        r"\bwith\s+([^'\"\s,.]+)",
        user_prompt,
        re.IGNORECASE,
    )

    if normal_match:
        return [
            normal_match.group(1)
        ]

    return []


def get_string_candidates(
    user_prompt: str,
    param_name: str,
) -> list[str]:
    """
    Build candidates depending on the meaning
    of the string parameter.
    """

    if param_name == "source_string":
        return get_source_string_candidates(
            user_prompt
        )

    if param_name == "regex":
        return get_regex_candidates(
            user_prompt
        )

    if param_name == "replacement":
        return get_replacement_candidates(
            user_prompt
        )

    # Generic string parameter.
    # Example:
    # Greet 'john'
    quoted = get_quoted_strings(user_prompt)

    if quoted:
        return quoted

    # Example:
    # Greet john
    words = re.findall(
        r"[A-Za-z_][A-Za-z0-9_-]*",
        user_prompt,
    )

    return words


def get_candidates(
    user_prompt: str,
    param_name: str,
    param_type: TypeDefinition,
    already_extracted: dict[str, Any],
) -> list[str]:
    """
    Build the allowed values for one parameter.
    """

    kind = param_type.type

    # --------------------
    # NUMBER / INTEGER
    # --------------------

    if kind in {"number", "integer"}:
        numbers = get_numbers(user_prompt)

        used_numbers = {
            float(value)
            for value in already_extracted.values()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
        }

        remaining = [
            number
            for number in numbers
            if float(number) not in used_numbers
        ]

        if remaining:
            return remaining

        return numbers

    # --------------------
    # BOOLEAN
    # --------------------

    if kind == "boolean":
        return [
            "true",
            "false",
        ]

    # --------------------
    # STRING
    # --------------------

    if kind == "string":
        return get_string_candidates(
            user_prompt,
            param_name,
        )

    return []


def build_parameter_prompt(
    user_prompt: str,
    function: FunctionDefinition,
    param_name: str,
    param_type: TypeDefinition,
    already_extracted: dict[str, Any],
) -> str:
    """
    Build the prompt used by the model
    to choose one parameter value.
    """

    result = (
        "You are extracting one parameter "
        "for a function call.\n\n"
        f"Function: {function.name}\n"
        f"Description: {function.description}\n\n"
        f"Parameter: {param_name}\n"
        f"Type: {param_type.type}\n"
    )

    if already_extracted:
        result += "\nAlready extracted:\n"

        for name, value in already_extracted.items():
            result += (
                f"- {name}: {value}\n"
            )

    result += (
        f"\nUser request: {user_prompt}\n"
        "Value:"
    )

    return result


def cast_value(
    value: str,
    param_type: TypeDefinition,
) -> Any:
    """
    Convert the selected string into
    the expected Python type.
    """

    if param_type.type == "integer":
        return int(float(value))

    if param_type.type == "number":
        return float(value)

    if param_type.type == "boolean":
        return value.lower() == "true"

    if param_type.type == "null":
        return None

    return value


def decode_parameter(
    model: Any,
    user_prompt: str,
    function: FunctionDefinition,
    param_name: str,
    param_type: TypeDefinition,
    already_extracted: dict[str, Any],
) -> Any:
    """
    Decode one parameter.
    """

    candidates = get_candidates(
        user_prompt,
        param_name,
        param_type,
        already_extracted,
    )

    if not candidates:
        raise RuntimeError(
            f"No candidates found for "
            f"parameter {param_name!r}"
        )

    prompt = build_parameter_prompt(
        user_prompt,
        function,
        param_name,
        param_type,
        already_extracted,
    )

    selected = constrained_choice(
        model,
        prompt,
        candidates,
    )

    return cast_value(
        selected,
        param_type,
    )


def extract_parameters(
    model: Any,
    user_prompt: str,
    function: FunctionDefinition,
) -> dict[str, Any]:
    """
    Extract all parameters for the selected function.
    """

    result: dict[str, Any] = {}

    for param_name, param_type in function.parameters.items():

        result[param_name] = decode_parameter(
            model,
            user_prompt,
            function,
            param_name,
            param_type,
            result,
        )

    return result