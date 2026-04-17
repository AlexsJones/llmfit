# Makefile for llmfit-python
# Convenience commands for formatting, linting, type-checking, and testing

.PHONY: help fmt lint test typecheck check

# Default target
help:
	@echo "llmfit-python development targets:"
	@echo ""
	@echo "  make fmt         - Format Python code with ruff"
	@echo "  make lint        - Lint Python code with ruff"
	@echo "  make test        - Run Python tests with pytest"
	@echo "  make typecheck   - Type-check Python code with ty"
	@echo "  make check       - Run all Python quality checks"
	@echo ""

# Format Python code
fmt:
	uv run ruff format .

# Lint Python code
lint:
	uv run ruff check .

# Run Python tests
test:
	uv run pytest -vv

# Type-check Python code
typecheck:
	uv run ty check .

# Run all Python quality checks
check: fmt lint typecheck test
