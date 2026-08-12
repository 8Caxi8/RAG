MAIN = src
PY = python3
USER := $(shell whoami)
UV := $(shell which uv 2>/dev/null || echo "$$HOME/.local/bin/uv")
 
ifneq ($(wildcard /sgoinfre/.),)
UV_CACHE_DIR := /sgoinfre/$(USER)/.cache/uv
HF_HOME := /sgoinfre/$(USER)/.cache/huggingface
export UV_CACHE_DIR
export HF_HOME
endif
 
install:
	@which uv > /dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	@$$HOME/.local/bin/uv sync
 
run:
	$(UV) run $(PY) -m $(MAIN) $(ARGS)
 
debug:
	$(UV) run $(PY) -m pdb -m $(MAIN) $(ARGS)
 
clean:
	find . -name "__pycache__" -print -exec rm -rf {} +
	find . -name ".mypy_cache" -print -exec rm -rf {} +
	find . -name "*.pyc" -print -delete
 
lint:
	$(UV) run flake8 --exclude=.venv,data,moulinette .
	$(UV) run mypy . --exclude '\.venv|data|moulinette' --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
 
.PHONY: install run debug clean lint