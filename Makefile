install:
	-pip install uv
	uv sync

run:
	uv run python3 -m src

debug:
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__ src/__pycache__ .mypy_cache .pytest_cache data/output 
	rm -rf .flake8

lint:
	uv run flake8 . --exclude=llm_sdk && uv run mypy . --exclude=llm_sdk --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 . --exclude=llm_sdk && uv run mypy . --exclude=llm_sdk --strict
