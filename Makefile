.DEFAULT_GOAL := all
PYTHON ?= python3
UV ?= uv
TEST_PYTHON ?= 3.12
TEST_ARGS ?=
ARGS ?=

.PHONY: all install run debug test test-model clean lint lint-strict

all:
	$(PYTHON) -m compileall -q src

install:
	uv sync

run:
	uv run --no-sync python -m src $(ARGS)

debug:
	uv run --no-sync python -m pdb -m src $(ARGS)

test:
	$(UV) run --no-project --python $(TEST_PYTHON) \
		--with-requirements requirements-test.txt \
		python -m pytest tests -m 'not model' -v $(TEST_ARGS)

test-model:
	$(UV) run python -m pytest tests/test_model.py -m model -v $(TEST_ARGS)

clean:
	rm -rf .pytest_cache .mypy_cache __pycache__ src/__pycache__ src/.mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

lint:
	uv run --no-sync flake8 src tests
	uv run --no-sync mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs src tests

lint-strict:
	uv run --no-sync flake8 src tests
	uv run --no-sync mypy --strict src tests
