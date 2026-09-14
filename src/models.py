"""Pydantic models for the Call Me Maybe project.

This module validates the two input files and the final output shape.
It deliberately stays small: one model for function definitions, one for
input prompts, and one for generated function-call results.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

JsonType = Literal[
    "string",
    "number",
    "boolean",
    "integer",
    "object",
    "array",
    "null",
]


class TypeDefinition(BaseModel):
    """Represents a typed field in a function schema."""

    model_config = ConfigDict(extra="forbid")

    type: JsonType


class FunctionDefinition(BaseModel):
    """Represents one callable function exposed to the LLM.

    Attributes:
        name: Function name to write in the output file.
        description: Natural-language explanation used during selection.
        parameters: Mapping of argument names to JSON type definitions.
        returns: JSON type returned by the function.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, TypeDefinition]
    returns: TypeDefinition

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Ensure function names are not blank after trimming.

        Args:
            value: Raw function name from JSON.

        Returns:
            The stripped function name.

        Raises:
            ValueError: If the stripped function name is empty.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("function name cannot be empty")
        return stripped

    @field_validator("parameters")
    @classmethod
    def validate_parameters(
        cls,
        value: dict[str, TypeDefinition],
    ) -> dict[str, TypeDefinition]:
        """Ensure parameter names are valid and unique after trimming.

        Args:
            value: Raw parameter mapping from JSON.

        Returns:
            Parameter mapping with stripped names.

        Raises:
            ValueError: If a name is empty or duplicated after trimming.
        """
        clean: dict[str, TypeDefinition] = {}
        for key, definition in value.items():
            stripped = key.strip()
            if not stripped:
                raise ValueError("parameter name cannot be empty")
            if stripped in clean:
                raise ValueError(
                    f"duplicate parameter name after trimming: {stripped}")
            clean[stripped] = definition
        return clean


class PromptInput(BaseModel):
    """Represents one natural-language prompt from the input test file.

    Attributes:
        prompt: User request that should be converted to a function call.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(...)


class FunctionCallResult(BaseModel):
    """Final output object required by the subject.

    Attributes:
        prompt: Original prompt copied from the input file.
        name: Selected function name.
        parameters: Extracted arguments for the function.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]


class ProjectInput(BaseModel):
    """Validated in-memory representation of all project inputs.

    Attributes:
        functions: Available function definitions.
        prompts: Prompt test cases to process.
    """

    functions: list[FunctionDefinition]
    prompts: list[PromptInput]

    @model_validator(mode="after")
    def validate_unique_function_names(self) -> "ProjectInput":
        """Reject duplicated function names.

        Returns:
            The current validated object.

        Raises:
            ValueError: If two function definitions use the same name.
        """
        names = [function.name for function in self.functions]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"duplicate function names: {', '.join(duplicates)}")
        return self
