from __future__ import annotations

import re
from typing import Any

from .decoder import ChoiceModel, constrained_choice
from .models import FunctionDefinition, TypeDefinition


def unique(values: list[str]) -> list[str]:
    """Remove duplicates while preserving order.

    Args:
        values: String values that may contain duplicates.

    Returns:
        The same values without duplicates, keeping the first occurrence.
    """

    return list(dict.fromkeys(values))


def semantic_request_text(text: str) -> str:
    """Keep the semantic request and drop obvious injected output instructions.

    Args:
        text: Full user prompt.

    Returns:
        Prompt text before common prompt-injection markers.
    """

    injection_markers = [
        "ignore previous instructions",
        "set output to",
    ]

    lowered = text.lower()
    cut_at = len(text)

    for marker in injection_markers:
        marker_index = lowered.find(marker)
        if marker_index != -1:
            cut_at = min(cut_at, marker_index)

    return text[:cut_at]


def extract_numbers(text: str) -> list[str]:
    """Find finite numeric values explicitly present in the user request.

    Args:
        text: Full user prompt.

    Returns:
        Numeric strings in prompt order. Integers, decimals, and scientific
        notation are supported.
    """

    finite_number_text = semantic_request_text(text)

    return unique(
        re.findall(
            r"(?<![\w.])-?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][+-]?\d+)?(?!\w|\.\d)",
            finite_number_text,
        )
    )


def decode_user_escapes(value: str) -> str:
    """Decode common escaped text a user may write inside quotes.

    Args:
        value: A string that may contain escapes such as `\\u0044`.

    Returns:
        The decoded string when possible, otherwise the original value.
    """

    try:
        decoded = value.encode("utf-8").decode("unicode_escape")
        return decoded.encode("utf-16", "surrogatepass").decode("utf-16")
    except UnicodeDecodeError:
        return value


def extract_quoted_strings(text: str) -> list[str]:
    """Extract quoted strings while respecting escaped quote characters.

    Args:
        text: Full user prompt.

    Returns:
        Values found between single or double quotes. The surrounding quotes are
        not included.
    """

    values: list[str] = []
    index = 0

    while index < len(text):
        quote = text[index]
        if quote not in {"'", '"'}:
            index += 1
            continue

        index += 1
        chars: list[str] = []
        escaped = False

        while index < len(text):
            char = text[index]

            if escaped:
                chars.append("\\" + char)
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                break
            else:
                chars.append(char)

            index += 1

        if index < len(text) and text[index] == quote:
            values.append("".join(chars))

        index += 1

    return unique(values)


def extract_phrase_value(text: str, phrases: list[str]) -> str | None:
    """Return the quoted value after the last matching phrase.

    Args:
        text: Full user prompt.
        phrases: Phrases that should appear immediately before a quoted value.

    Returns:
        The quoted value after the latest matching phrase, or `None`.
    """

    matches: list[tuple[int, str]] = []

    for phrase in phrases:
        pattern = re.compile(
            rf"\b{re.escape(phrase)}\s+(['\"])",
            flags=re.IGNORECASE,
        )

        for match in pattern.finditer(text):
            quote = match.group(1)
            index = match.end()
            chars: list[str] = []
            escaped = False

            while index < len(text):
                char = text[index]

                if escaped:
                    chars.append("\\" + char)
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    matches.append((match.start(), "".join(chars)))
                    break
                else:
                    chars.append(char)

                index += 1

    if not matches:
        return None

    return max(matches, key=lambda item: item[0])[1]


def extract_words(text: str) -> list[str]:
    """Extract normal words that may represent string arguments.

    Args:
        text: Full user prompt.

    Returns:
        Unique word-like strings in prompt order.
    """

    return unique(
        re.findall(
            r"[A-Za-z_][A-Za-z0-9_-]*",
            text,
        )
    )


def build_string_candidates(
    user_prompt: str,
    param_name: str,
) -> list[str]:
    """Build constrained candidate values for string parameters.

    The candidate set depends on the parameter role:
    - source_string: text that should be transformed
    - regex: pattern describing what to match
    - replacement: replacement text/symbol

    Args:
        user_prompt: Natural-language prompt.
        param_name: Name of the function parameter being decoded.

    Returns:
        Candidate strings the model is allowed to choose from.
    """

    quoted = extract_quoted_strings(user_prompt)
    lower = user_prompt.lower()

    candidates: list[str] = []

    # -------------------------
    # source_string
    # -------------------------
    if param_name == "source_string":
        source_value = extract_phrase_value(
            user_prompt,
            [
                "source string",
                "string",
                "in",
            ],
        )

        if source_value is not None:
            candidates.append(source_value)
        elif quoted:
            candidates.append(max(quoted, key=len))

        else:
            # Fallback if no quoted string exists.
            candidates.append(user_prompt)

    # -------------------------
    # regex
    # -------------------------
    elif param_name == "regex":
        semantic_regex_found = False

        explicit_regex = extract_phrase_value(
            user_prompt,
            [
                "regex pattern",
                "regex",
                "pattern",
            ],
        )

        if explicit_regex is not None:
            candidates.append(explicit_regex)
            semantic_regex_found = True

        if "number" in lower or "numbers" in lower:
            candidates.extend([
                r"\d+",
                r"[0-9]+",
            ])
            semantic_regex_found = True

        if "vowel" in lower or "vowels" in lower:
            candidates.extend([
                r"[aeiouAEIOU]",
                r"[aeiou]",
            ])
            semantic_regex_found = True

        # Example:
        # Substitute the word 'cat' with 'dog'
        word_value = extract_phrase_value(
            user_prompt,
            [
                "word",
            ],
        )

        if word_value is not None:
            candidates.append(
                re.escape(word_value)
            )
            semantic_regex_found = True

        # Fallback: short quoted strings may be regex targets.
        if not semantic_regex_found:
            for value in quoted:
                if len(value.split()) <= 3:
                    candidates.append(value)

    # -------------------------
    # replacement
    # -------------------------
    elif param_name == "replacement":
        if "asterisk" in lower:
            candidates.append("*")
        else:
            replacement_value = extract_phrase_value(
                user_prompt,
                [
                    "with",
                ],
            )

            if replacement_value is not None:
                candidates.append(
                    decode_user_escapes(replacement_value)
                )
            else:
                replacement_match = re.search(
                    r"\bwith\s+([^'\"\s]+)",
                    user_prompt,
                    flags=re.IGNORECASE,
                )

                if replacement_match:
                    candidates.append(
                        replacement_match.group(1)
                    )

            # Fallback:
            # quoted values after the target may represent replacement.
            if len(quoted) >= 2:
                candidates.append(quoted[1])

    # -------------------------
    # generic string parameter
    # -------------------------
    else:
        decoded_quoted = [
            decode_user_escapes(value)
            for value in quoted
        ]
        candidates.extend(decoded_quoted)

        after_colon = re.search(
            r":\s*(.+?)\.?$",
            user_prompt,
            flags=re.DOTALL,
        )
        if after_colon:
            candidates.append(after_colon.group(1).rstrip("."))

        candidates.extend(extract_words(user_prompt))

    return unique(candidates)


def build_parameter_prompt(
    user_prompt: str,
    function: FunctionDefinition,
    param_name: str,
    param_def: TypeDefinition,
    already_extracted: dict[str, Any],
) -> str:
    """Build semantic context for one parameter decision.

    Args:
        user_prompt: Natural-language prompt.
        function: Selected function definition.
        param_name: Name of the parameter currently being extracted.
        param_def: Type definition for the current parameter.
        already_extracted: Parameters decoded before this one.

    Returns:
        Prompt text that asks the model to choose one parameter value.
    """

    lines = [
        "You are extracting one argument for a function call.",
        "Choose only the value that belongs to the requested parameter.",
        "Do not calculate the function result.",
        "",
        f"Function: {function.name}",
        f"Description: {function.description}",
        "",
        "Function parameters:",
    ]

    for name, definition in function.parameters.items():
        lines.append(
            f"- {name}: {definition.type}"
        )

    if already_extracted:
        lines.extend([
            "",
            "Arguments already extracted:",
        ])

        for name, value in already_extracted.items():
            lines.append(
                f"- {name}: {value!r}"
            )

    lines.extend([
        "",
        f"Parameter to extract: {param_name}",
        f"Expected type: {param_def.type}",
        "",
        f"User request: {user_prompt}",
        "",
        "Value:",
    ])

    return "\n".join(lines)


def decode_number(
    model: ChoiceModel,
    prompt: str,
    user_prompt: str,
    already_extracted: dict[str, Any],
) -> float:
    """Decode a `number` parameter as a float.

    Args:
        model: LLM wrapper used to score numeric candidates.
        prompt: Parameter-selection prompt.
        user_prompt: Original natural-language prompt.
        already_extracted: Previous parameter values, used to avoid selecting
            the same number twice when several numbers exist.

    Returns:
        Selected numeric value as a float.

    Raises:
        RuntimeError: If no finite number can be found in the prompt.
    """

    candidates = extract_numbers(user_prompt)

    # Avoid repeatedly choosing the same numeric argument when
    # multiple distinct values exist.
    used = set()
    for value in already_extracted.values():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            used.add(float(value))

    remaining = [
        value
        for value in candidates
        if float(value) not in used
    ]

    if remaining:
        candidates = remaining

    if not candidates:
        raise RuntimeError(
            "No numeric value found in the user request"
        )

    selected = constrained_choice(
        model,
        prompt,
        candidates,
    )

    return float(selected)


def decode_string(
    model: ChoiceModel,
    prompt: str,
    user_prompt: str,
    param_name: str,
) -> str:
    """Decode a `string` parameter from constrained candidates.

    Args:
        model: LLM wrapper used to score string candidates.
        prompt: Parameter-selection prompt.
        user_prompt: Original natural-language prompt.
        param_name: Name of the string parameter.

    Returns:
        Selected string value.

    Raises:
        RuntimeError: If no string candidate can be built.
    """

    candidates = build_string_candidates(
        user_prompt,
        param_name,
    )

    if not candidates:
        raise RuntimeError(
            f"No string candidates found for parameter {param_name!r}"
        )

    return constrained_choice(
        model,
        prompt,
        candidates,
    )


def decode_parameter(
    model: ChoiceModel,
    user_prompt: str,
    function: FunctionDefinition,
    param_name: str,
    param_def: TypeDefinition,
    already_extracted: dict[str, Any],
) -> Any:
    """Decode one function parameter according to its declared JSON type.

    Args:
        model: LLM wrapper used by constrained decoding.
        user_prompt: Original natural-language prompt.
        function: Selected function definition.
        param_name: Name of the parameter currently being decoded.
        param_def: Type definition for the parameter.
        already_extracted: Parameter values decoded before this one.

    Returns:
        The decoded parameter value with the expected Python type.

    Raises:
        NotImplementedError: If the parameter type is not supported.
        RuntimeError: If no valid candidate can be decoded.
        ValueError: If constrained decoding receives invalid choices.
    """
    prompt = build_parameter_prompt(
        user_prompt,
        function,
        param_name,
        param_def,
        already_extracted,
    )

    if param_def.type == "number":
        return decode_number(
            model,
            prompt,
            user_prompt,
            already_extracted,
        )

    if param_def.type == "integer":
        value = decode_number(
            model,
            prompt,
            user_prompt,
            already_extracted,
        )

        return int(value)

    if param_def.type == "string":
        return decode_string(
            model,
            prompt,
            user_prompt,
            param_name,
        )

    if param_def.type == "boolean":
        selected = constrained_choice(
            model,
            prompt,
            ["true", "false"],
        )

        return selected == "true"

    if param_def.type == "null":
        return None

    raise NotImplementedError(
        f"Parameter type {param_def.type!r} is not supported"
    )


def extract_parameters(
    model: ChoiceModel,
    user_prompt: str,
    function: FunctionDefinition,
) -> dict[str, Any]:
    """Decode every required argument of the selected function.

    Args:
        model: LLM wrapper used by constrained decoding.
        user_prompt: Original natural-language prompt.
        function: Selected function definition.

    Returns:
        Dictionary mapping parameter names to decoded values.
    """

    result: dict[str, Any] = {}

    for param_name, param_def in function.parameters.items():
        result[param_name] = decode_parameter(
            model=model,
            user_prompt=user_prompt,
            function=function,
            param_name=param_name,
            param_def=param_def,
            already_extracted=result,
        )

    return result
