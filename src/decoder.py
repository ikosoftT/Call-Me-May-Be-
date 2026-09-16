from __future__ import annotations

from typing import Any, Protocol

from .models import FunctionDefinition


class ChoiceModel(Protocol):
    """Minimal model interface needed by constrained decoding."""

    def encode(self, text: str) -> Any:
        """Return token ids for text."""
        ...

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return next-token logits for the provided token ids."""
        ...


def constrained_choice(
    model: ChoiceModel,
    prompt: str,
    choices: list[str],
) -> str:
    """Choose one string using multi-token prefix-constrained decoding.

    Args:
        model: LLM wrapper exposing `encode` and `get_logits_from_input_ids`.
        prompt: Context sent to the model before generation.
        choices: Exact strings the decoder is allowed to return.

    Returns:
        The selected string from `choices`.

    Raises:
        ValueError: If no choices are provided or a choice cannot be tokenized.
        RuntimeError: If decoding reaches an impossible prefix.
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
    """Build the prompt used to select a function name.

    Args:
        prompt: Natural-language user request.
        functions: Available function definitions.

    Returns:
        A text prompt that lists the available functions and asks for one name.
    """
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
    model: ChoiceModel,
    prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    """Select the best function name for a user prompt.

    Args:
        model: LLM wrapper used to score valid function names.
        prompt: Natural-language user request.
        functions: Available function definitions.

    Returns:
        The selected function name.

    Raises:
        ValueError: If there are no available functions.
    """
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
