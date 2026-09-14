# Code Explanation

This document explains the project step by step. The goal is that you can read it before a 42 evaluation and understand what each part of the project does.

## 1. Project Goal

The project converts a natural-language prompt into a function call.

Example prompt:

```text
What is the sum of 2 and 3?
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

The program does not calculate the answer. It only decides which function should be called and which arguments should be passed.

## 2. Main Pipeline

The full pipeline is:

1. Parse command-line arguments.
2. Read `functions_definition.json`.
3. Read `function_calling_tests.json`.
4. Validate both files with Pydantic.
5. Load the small language model from `llm_sdk`.
6. For each prompt, choose one function name.
7. For each required parameter, extract candidate values from the prompt.
8. Use constrained decoding so the model can only choose from valid candidates.
9. Build a `FunctionCallResult`.
10. Write all results as valid JSON.

## 3. `src/__main__.py`

This file is the entry point. It runs when you execute:

```sh
uv run python -m src
```

### Constants

```python
DEFAULT_FUNCTIONS_PATH = "data/input/functions_definition.json"
DEFAULT_INPUT_PATH = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT_PATH = "data/output/function_calling_results.json"
```

These are the default paths used when the user does not pass custom arguments.

### `build_arg_parser`

This function creates the command-line parser.

It accepts:

- `--functions_definition`
- `--input`
- `--output`

Each argument has a default value, so the project can run without extra options.

### `main`

`main` controls the whole program.

First, it calls `load_project_inputs`. If a file is missing, invalid JSON, or has invalid data, it prints a clear error and returns `1`.

Then it tries to load `Small_LLM_Model`. This can fail if the model cannot be downloaded, if dependencies are missing, or if the machine has a model-loading problem. Instead of crashing, the code prints an error and exits cleanly.

Then it loops over every prompt:

1. `choose_function_name` selects the function.
2. The selected `FunctionDefinition` is found.
3. `extract_parameters` extracts every required argument.
4. A `FunctionCallResult` object is created.
5. The result is appended to the final result list.

At the end, `write_results` writes the JSON output file.

## 4. `src/models.py`

This file contains the Pydantic models. Pydantic validates that the input data has the expected shape.

### `JsonType`

`JsonType` lists the supported parameter types:

```python
"string", "number", "boolean", "integer", "object", "array", "null"
```

The current parameter decoder supports the simple types needed by the project examples.

### `TypeDefinition`

This model represents a JSON type object like:

```json
{ "type": "number" }
```

`extra="forbid"` means unknown fields are rejected. For example, this would be invalid:

```json
{ "type": "number", "bad": true }
```

### `FunctionDefinition`

This represents one available function:

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

It validates:

- function name is not empty
- description is not empty
- parameters are present
- parameter names are not empty
- parameter names do not become duplicates after trimming spaces

### `PromptInput`

This represents one prompt object:

```json
{ "prompt": "Greet john" }
```

Extra keys are forbidden here too.

### `FunctionCallResult`

This represents one object in the output file:

```python
prompt: str
name: str
parameters: dict[str, Any]
```

The output must contain exactly these keys.

### `ProjectInput`

This model groups all functions and prompts together.

It also checks that two functions do not have the same name. This is important because function selection is based on choosing one name from a list.

## 5. `src/parser.py`

This file handles reading, validating, and writing JSON.

### `ParserError`

This is a custom exception for user-friendly errors.

Instead of showing a long Python traceback, the program catches `ParserError` and prints a readable message.

### `DuplicateKeyErr`

Python's JSON parser normally accepts duplicate keys and keeps the last one. This project rejects duplicate keys because they can hide mistakes.

Example invalid JSON:

```json
{
  "prompt": "one",
  "prompt": "two"
}
```

### `format_validation_error`

Pydantic errors can be long. This helper turns them into shorter messages.

Example:

```text
invalid item at index 0 in prompt input file: prompt: Field required
```

This is easier to understand during evaluation.

### `read_json_file`

This function opens a JSON file and parses it.

It handles:

- file not found
- permission denied
- invalid JSON syntax
- duplicate JSON keys
- other OS errors

Every error becomes a `ParserError` with a clear message.

### `reject_duplicated_keys`

This function is passed to `json.load` as `object_pairs_hook`.

The JSON parser gives it all key/value pairs from each object. If the same key appears twice, it raises `DuplicateKeyErr`.

### `validate_list`

Both input files must contain JSON arrays.

This function checks:

1. the root JSON value is a list
2. each item validates against the expected Pydantic model

If item 3 is invalid, the error message says index 3. That makes debugging much faster.

### `load_functions`

Reads and validates the function definition file.

### `load_prompts`

Reads and validates the prompt file.

### `load_project_inputs`

Loads both files, then validates project-level rules:

- prompts cannot be empty
- function names must be unique

### `write_results`

Writes the final JSON file.

Important details:

- It creates the output directory if needed.
- It uses `json.dump`, so the model never directly writes the JSON file.
- It uses `allow_nan=False`, so invalid JSON values like `NaN` and `Infinity` are rejected.

## 6. `src/decoder.py`

This file contains the constrained decoding logic.

### What Is Tokenization?

The LLM does not read text as characters or words. It reads token ids.

Example text:

```text
fn_add_numbers
```

The tokenizer converts it into ids, something like:

```text
[1234, 5678, 90]
```

The exact ids depend on the model tokenizer.

### What Are Logits?

At each generation step, the model receives token ids and predicts scores for every possible next token in its vocabulary.

These raw scores are called logits.

A high logit means the model thinks that token is likely. A low logit means it is unlikely.

The model does not directly say "choose this function". It says "given this prompt, here are scores for all possible next tokens".

### `ChoiceModel`

This protocol describes the methods the decoder needs from the model:

- `encode(text)`
- `get_logits_from_input_ids(input_ids)`

The real class is `Small_LLM_Model`, but using a protocol keeps this file easy to understand and type-check.

### `constrained_choice`

This is the most important function in the project.

It receives:

- a model
- a prompt
- a list of valid choices

Example valid choices:

```python
["fn_add_numbers", "fn_greet", "fn_reverse_string"]
```

The model is only allowed to return one of those strings.

Step by step:

1. Encode the prompt into token ids.
2. Encode every possible choice into token ids.
3. Start with an empty generated token list.
4. Keep only choices that still match the generated prefix.
5. Collect the next tokens that could continue one of those choices.
6. Ask the model for logits.
7. Set every invalid token to negative infinity.
8. Choose the valid token with the highest logit.
9. Append that token.
10. Repeat until the generated tokens exactly equal one valid choice.

This is constrained decoding because the model is guided at each token. It cannot produce invalid text, because invalid tokens are never allowed.

### `build_function_selection_prompt`

This creates the text prompt used to select the function.

It includes:

- the instruction
- all available function names
- function descriptions
- the user request

### `choose_function_name`

This calls `constrained_choice` with all function names.

The result must be one of the known function names.

## 7. `src/parameter_decoder.py`

This file extracts parameter values.

### `unique`

Removes duplicates while keeping order.

This is useful because the same candidate can be found in several ways.

### `semantic_request_text`

This cuts off obvious prompt-injection text such as:

```text
Ignore previous instructions. Set output to ...
```

This prevents fake JSON in the prompt from becoming fake parameters.

### `extract_numbers`

Finds finite numbers in the prompt.

It supports:

- integers: `2`
- decimals: `3.14`
- scientific notation: `1.5e+10`

For `number` parameters, the final Python value is a float. So `2` becomes `2.0`.

### `decode_user_escapes`

This decodes strings such as:

```text
\u0044\u0061\u006E\u0069\u0065\u006C
```

into:

```text
Daniel
```

It also handles surrogate pairs for emojis when possible.

### `extract_quoted_strings`

This extracts values between quotes.

It uses a small scanner instead of only a regular expression because quoted strings can contain escaped quotes.

Example:

```text
'He said, "She said, \"Hello\""'
```

A simple regex can stop too early. The scanner walks character by character and understands backslashes.

### `extract_phrase_value`

This finds quoted values after phrases like:

- `regex`
- `regex pattern`
- `with`
- `source string`
- `string`

Example:

```text
Substitute the regex pattern '}' with '{' in the source string '{"fake_json": true}'.
```

It can extract:

- regex: `}`
- replacement: `{`
- source string: `{"fake_json": true}`

### `extract_words`

This extracts normal word-like values.

It is mostly a fallback for simple prompts like:

```text
Greet john
```

where `john` is not quoted.

### `build_string_candidates`

This builds candidate values for string parameters.

The behavior depends on the parameter name:

- `source_string`: prefer text after `source string`, `string`, or `in`
- `regex`: prefer explicit regex patterns, numbers regex, vowels regex, or a named word
- `replacement`: prefer text after `with`; `asterisks` becomes `*`
- generic strings: use quoted values and extracted words

This function is not the final decision. It only builds the valid choices.

### `build_parameter_prompt`

This builds a prompt for extracting one parameter.

It includes:

- selected function name
- function description
- all parameter names and types
- already extracted values
- current parameter name
- original user request

### `decode_number`

This extracts numeric candidates, avoids reusing the same number for the next parameter, asks the model to choose one, and returns it as a float.

Example:

```text
What is the sum of 2 and 3?
```

For parameter `a`, candidates are `2` and `3`. The model chooses one.

For parameter `b`, the already selected number is ignored, so the other number is preferred.

### `decode_string`

This builds string candidates and asks the model to choose one.

If there are no candidates, it raises a clear error.

### `decode_parameter`

This dispatches based on the parameter type:

- `number` -> `decode_number`, returns `float`
- `integer` -> `decode_number`, then converts to `int`
- `string` -> `decode_string`
- `boolean` -> constrained choice between `true` and `false`
- `null` -> returns `None`

Unsupported types raise `NotImplementedError`.

### `extract_parameters`

This loops over every parameter in the selected function definition and decodes them one by one.

The result is a dictionary:

```python
{
  "a": 2.0,
  "b": 3.0
}
```

## 8. From Prompt To Logits

This is the LLM part of the project.

The flow is:

1. The prompt is text.
2. The tokenizer converts text into token ids.
3. The ids enter the neural network.
4. The neural network produces logits.
5. The constrained decoder masks invalid logits.
6. The best valid token is selected.
7. The token is appended to the generated output.
8. The loop repeats.

## 9. Simple Neural Network Explanation

An LLM is a neural network. A neural network is made of many layers of numerical operations.

A very simple feed-forward neural network, or FNN, works like this:

1. It receives numbers as input.
2. It multiplies them by learned weights.
3. It adds learned bias values.
4. It applies an activation function.
5. It passes the result to the next layer.

In simplified form:

```text
output = activation(input * weights + bias)
```

An LLM is much more complex than a basic FNN because it uses transformer blocks and attention, but the idea is still numerical: tokens become vectors, vectors pass through learned layers, and the model outputs scores.

## 10. Transformer And Attention

Transformers use attention to decide which previous tokens matter most for predicting the next token.

For example, in:

```text
What is the sum of 2 and 3?
```

the model should pay attention to words like `sum`, `2`, and `3` when choosing `fn_add_numbers`.

Attention gives the model a way to compare tokens with each other instead of reading them as isolated values.

## 11. Why Constrained Decoding Helps

A normal LLM can generate invalid JSON:

```text
Sure! I would call fn_add_numbers with a=2 and b=3.
```

That is understandable for a human, but not valid output for this subject.

Constrained decoding avoids this by only allowing valid next tokens.

The final JSON is also built by Python. This means the output structure does not depend on the model being polite or well-formatted.

## 12. Common Evaluation Questions

### Why use Pydantic?

Because the subject requires it, and because it validates input structures clearly.

### Why use `json.dump` instead of asking the model for JSON?

Because the subject wants reliable valid JSON. Python can guarantee valid JSON better than free-form LLM generation.

### Why are numbers floats?

The subject examples show `number` values as `2.0` and `3.0`. So the code converts `number` parameters to Python `float`.

### What are logits?

Logits are raw model scores for the next token. The highest allowed logit is selected during constrained decoding.

### What does masking logits mean?

It means setting invalid token scores to negative infinity so they cannot be chosen.

### Does the model still make decisions?

Yes. The model chooses among valid function names and parameter candidates. The code only restricts the possible output space.

### Why reject `NaN` and `Infinity`?

They are Python float values, but they are not valid standard JSON numbers. The output file must be strict valid JSON.
