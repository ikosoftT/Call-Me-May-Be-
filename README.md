*This project has been created as part of the 42 curriculum by yikoubaz.*

# Call Me Maybe

## Description

Call Me Maybe is a small function-calling project. The program receives natural language prompts and translates each one into a structured function call.

The goal is not to answer the prompt directly. If the user asks, "What is the sum of 2 and 3?", the expected result is not `5`. The expected result is a JSON object that says which function should be called and which arguments should be passed to it.

The program reads:

- `data/input/functions_definition.json`: the available functions and their parameter types
- `data/input/function_calling_tests.json`: the prompts to process

It writes:

- `data/output/function_calling_results.json`: the generated function calls

Each output object contains only:

- `prompt`
- `name`
- `parameters`

## Instructions

Install dependencies:

```sh
uv sync
```

Run the project with default paths:

```sh
uv run python -m src
```

Run the project with explicit paths:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

The Makefile also provides:

```sh
make install
make run
make debug
make clean
make lint
```

## Example Usage

Input prompt:

```json
{
  "prompt": "What is the sum of 2 and 3?"
}
```

Function definition:

```json
{
  "name": "fn_add_numbers",
  "description": "Add two numbers together and return their sum.",
  "parameters": {
    "a": { "type": "number" },
    "b": { "type": "number" }
  },
  "returns": { "type": "number" }
}
```

Expected output:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2.0,
    "b": 3.0
  }
}
```

## Algorithm Explanation

The project uses constrained decoding in two places.

First, it chooses the function name. The prompt sent to the model describes the available functions, but the model is not allowed to freely generate any text. It must choose one value from the list of valid function names.

Second, it extracts parameters. For each parameter in the selected function, the program builds a candidate list that matches the expected type:

- For `number`, it extracts numeric values from the prompt and stores them as floats.
- For `string`, it extracts quoted text, names, regex patterns, replacements, and source strings depending on the parameter name.
- For `boolean`, it restricts the choice to `true` or `false`.
- For `null`, it returns `None`.

The constrained decoder works token by token. At every generation step, it only allows tokens that can still lead to one of the valid choices. This prevents the model from producing broken JSON, random prose, or values outside the candidate set.

The final JSON file is not generated directly by the model. Python builds it from validated data and writes it with the `json` module.

## Design Decisions

The code is split by responsibility:

- `src/__main__.py`: command-line entry point
- `src/parser.py`: JSON reading, validation errors, and output writing
- `src/models.py`: Pydantic models for input and output structures
- `src/decoder.py`: constrained function-name and choice decoding
- `src/parameter_decoder.py`: parameter extraction and typed value decoding

Pydantic is used because the subject requires it and because it gives clear validation errors for malformed input.

Numbers are converted to floats for `number` parameters, because the subject examples use float representation such as `2.0`.

The output writer uses standard JSON serialization and refuses non-standard values like `NaN` or `Infinity`, since those are not valid strict JSON values.

## Performance Analysis

The first run can be slow if dependencies or model weights need to be downloaded. After setup, the runtime mostly depends on how many prompts are processed and how long the constrained choices are.

Reliability is improved because the model is not asked to freely write JSON. Function names and parameters are selected from controlled candidate lists, and the final output is written by Python.

The parser handles normal examples and harder cases such as escaped quotes, regex patterns, fake JSON inside prompts, tabs, line breaks, and numeric values that must be represented as floats.

## Challenges Faced

One challenge was regex extraction. Regex prompts often contain backslashes, braces, quotes, and text that looks like JSON. A simple regex parser can easily mix up the pattern, replacement, and source string.

Another challenge was number handling. The subject uses `number` as a type, and the examples show float representation. Because of that, values like `2` and `3` from the prompt must become `2.0` and `3.0` in the output.

Prompt-injection-looking text was also tested. The program should treat that text as data from the prompt, not as instructions to add extra JSON keys.

## Testing Strategy

I validated the project with:

- simple addition prompts
- greeting prompts
- string reversal prompts
- square-root prompts
- regex substitution prompts
- escaped quotes
- tabs and line breaks in strings
- fake JSON inside a prompt
- very small and very large numbers

I also checked that the output remains valid JSON and that each result object keeps only the required keys.

## Resources

- Python JSON documentation: https://docs.python.org/3/library/json.html
- Python regular expressions documentation: https://docs.python.org/3/library/re.html
- Pydantic documentation: https://docs.pydantic.dev/
- Hugging Face Transformers documentation: https://huggingface.co/docs/transformers
- Qwen model page: https://huggingface.co/Qwen

AI was used to help review the subject requirements, find edge cases, and improve README wording.