from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict
from pydantic import BaseModel, Field, model_validator, ConfigDict


class ParamDef(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: str = Field(..., description="data type of func")


class ReturnDef(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: str = Field(..., description="return val type")


class FuncDef(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(..., description='name of func')
    description: str = Field(..., description='desc of func')
    parameters: Dict[str, ParamDef] = Field(
        default_factory=dict, description="dict mapps param names")
    returns: ReturnDef = Field(..., description="return type def")

    # Logical Validation
    @model_validator(mode="after")
    def validate_function_schema(self) -> "FuncDef":
        if not self.name or not self.name.strip():
            raise ValueError("function name can't be empty")
        return self


class Prompt(BaseModel):
    model_config = ConfigDict(extra='forbid')

    prompt: str = Field(..., description="given prompt to lm")

    @model_validator(mode="after")
    def validate_prompt(self) -> "Prompt":
        if not self.prompt or not self.prompt.strip():
            raise ValueError("prompt cannot be empty")
        return self

# Logical PARSING : Load, Parse, validate


def load_functions_definition(file_path: Path) -> List[FuncDef]:

    try:
        with open(file_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON schema in func file {file_path}: {e}") from e
    if not isinstance(data, list):
        raise ValueError("Excpeting JSON array")
    functions: List[FuncDef] = []
    seen_names: set = set()

    for idx, item in enumerate(data, start=1):
        try:
            func_def = FuncDef(**item)
        except Exception as e:
            raise ValueError(f"Invalid SCHEMA at function line {idx}")
        if func_def.name in seen_names:
            raise ValueError(
                f"Duplicated func detected {func_def.name} line {idx}")
        seen_names.add(func_def.name)
        functions.append(func_def)
    if not functions:
        raise ValueError("func definition list can't be empty")
    return functions


def load_test_prompts(file_path: Path) -> List[Prompt]:

    try:
        with open(file_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON format in prompt file {file_path}: {e}") from e
    if not isinstance(data, list):
        raise ValueError(f"should be array of prompts at least asmat!")
    prompts: List[Prompt] = []

    for idx, p in enumerate(data, start=1):
        try:
            test_prompt = Prompt(**p)
        except Exception as e:
            raise ValueError(f"Invalid prompt at line {idx}")
        prompts.append(test_prompt)
    if not prompts:
        raise ValueError("prompts list cannot be empty")
    return prompts
