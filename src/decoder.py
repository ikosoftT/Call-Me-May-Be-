from typing import List

from .models import FunctionDefinition


def build_function_selection_prompt(
    prompt: str,
    functions: List[FunctionDefinition],
) -> str:
    """Build the semantic context used for function selection."""

    lines = [
        "You are a function-calling assistant.",
        "Choose exactly one function from the available functions.",
        "",
        "Available functions:",
    ]

    for func in functions:
        lines.append(f"- {func.name}: {func.description}")

    lines.extend([
        "",
        f"User request: {prompt}",
        "",
        "Function:",
    ])

    return "\n".join(lines)


def choose_function_name(
    model,
    prompt: str,
    functions: List[FunctionDefinition],
) -> str:
    """Choose one valid function name using prefix-constrained decoding."""

    if not functions:
        raise ValueError("No functions available for decoding")

    selection_prompt = build_function_selection_prompt(
        prompt,
        functions,
    )

    prompt_ids = model.encode(selection_prompt)[0].tolist()

    encoded_functions: dict[str, list[int]] = {}

    for func in functions:
        ids = model.encode(func.name)[0].tolist()

        if not ids:
            raise ValueError(
                f"Function name {func.name!r} encoded to zero tokens"
            )

        encoded_functions[func.name] = ids

    generated: list[int] = []

    while True:
        candidates: list[tuple[str, list[int]]] = []

        for name, ids in encoded_functions.items():
            if ids[:len(generated)] == generated:
                candidates.append((name, ids))

        if not candidates:
            raise RuntimeError(
                "Constrained decoding reached an invalid prefix"
            )

        # If the currently generated token sequence exactly matches
        # one valid function name, decoding is complete.
        for name, ids in candidates:
            if ids == generated:
                return name

        allowed_next_tokens: set[int] = set()

        for _, ids in candidates:
            if len(ids) > len(generated):
                next_token = ids[len(generated)]
                allowed_next_tokens.add(next_token)

        if not allowed_next_tokens:
            raise RuntimeError(
                "No valid next token available during function decoding"
            )

        logits = model.get_logits_from_input_ids(
            prompt_ids + generated
        )

        constrained_logits = [float("-inf")] * len(logits)

        for token_id in allowed_next_tokens:
            constrained_logits[token_id] = logits[token_id]

        best_token_id = max(
            range(len(constrained_logits)),
            key=lambda i: constrained_logits[i],
        )

        generated.append(best_token_id)