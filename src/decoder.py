from __future__ import annotations

from typing import TypeVar

from .models import FunctionDefinition


T = TypeVar("T")


def constrained_choice(
    model,
    prompt: str,
    choices: list[str],
) -> str:
    """
    Choose exactly one string from a finite set using
    multi-token prefix-constrained decoding.
    """

    if not choices:
        raise ValueError("No choices available")

    prompt_ids = model.encode(prompt)[0].tolist()

    encoded_choices: list[tuple[str, list[int]]] = []

    for choice in choices:
        ids = model.encode(choice)[0].tolist()

        if not ids:
            raise ValueError(
                f"Choice {choice!r} encoded to zero tokens"
            )

        encoded_choices.append((choice, ids))

    generated: list[int] = []

    while True:
        candidates: list[tuple[str, list[int]]] = []

        for choice, ids in encoded_choices:
            if ids[:len(generated)] == generated:
                candidates.append((choice, ids))

        if not candidates:
            raise RuntimeError(
                "Constrained decoding reached an invalid prefix"
            )

        # Complete choice.
        for choice, ids in candidates:
            if ids == generated:
                return choice

        allowed_next_tokens: set[int] = set()

        for _, ids in candidates:
            if len(ids) > len(generated):
                allowed_next_tokens.add(
                    ids[len(generated)]
                )

        if not allowed_next_tokens:
            raise RuntimeError(
                "No valid next token during constrained decoding"
            )

        logits = model.get_logits_from_input_ids(
            prompt_ids + generated
        )

        constrained_logits = [float("-inf")] * len(logits)

        for token_id in allowed_next_tokens:
            constrained_logits[token_id] = logits[token_id]

        best_token_id = max(
            allowed_next_tokens,
            key=lambda token_id: constrained_logits[token_id],
        )

        generated.append(best_token_id)


def build_function_selection_prompt(
    prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    lines = [
        "You are a function-calling assistant.",
        "Select exactly one function for the user request.",
        "Output only the function name.",
        "",
        "Available functions:",
    ]

    for function in functions:
        lines.append(
            f"- {function.name}: {function.description}"
        )

    lines.extend([
        "",
        f"User request: {prompt}",
        "",
        "Selected function:",
    ])

    return "\n".join(lines)


def choose_function_name(
    model,
    prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    if not functions:
        raise ValueError("No functions available")

    selection_prompt = build_function_selection_prompt(
        prompt,
        functions,
    )

    return constrained_choice(
        model,
        selection_prompt,
        [function.name for function in functions],
    )