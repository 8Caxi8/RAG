MAIN = src
PY = python3
USER := $(shell whoami)
UV := $(shell which uv 2>/dev/null || echo "$$HOME/.local/bin/uv")

UV_CACHE_DIR = /sgoinfre/$(USER)/.cache/uv
export UV_CACHE_DIR

install:
	@which uv > /dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	@$$HOME/.local/bin/uv sync

run:
	clear
	$(UV) run $(PY) -m $(MAIN)

debug:
	$(UV) run $(PY) -m pdb -m $(MAIN) --verbose

clean:
	find . -name "__pycache__" -print -exec rm -rf {} +
	find . -name ".mypy_cache" -print -exec rm -rf {} +
	find . -name "*.pyc" -print -delete

lint:
	$(UV) run flake8 --exclude=.venv,llm_sdk .
	$(UV) run mypy . --exclude '\.venv|llm_sdk' --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV) run flake8 --exclude=.venv,llm_sdk .
	$(UV) run mypy . --exclude '\.venv|llm_sdk' --strict

.PHONY: install run debug clean lint lint-strict verbose