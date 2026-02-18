# Makefile for llmfit
# Convenience commands for building, testing, and updating the model database

.PHONY: help build release clean run test update-models check fmt clippy install

# Default target
help:
	@echo "llmfit - LLM Model Fit Analyzer"
	@echo ""
	@echo "Available targets:"
	@echo "  make build          - Build debug binary"
	@echo "  make release        - Build release binary"
	@echo "  make run            - Run in TUI mode (debug)"
	@echo "  make test           - Run all unit tests"
	@echo "  make update-models  - Fetch latest model data from HuggingFace"
	@echo "  make check          - Run cargo check"
	@echo "  make fmt            - Format code with rustfmt"
	@echo "  make clippy         - Run clippy linter"
	@echo "  make clean          - Remove build artifacts"
	@echo "  make install        - Install release binary to ~/.cargo/bin"
	@echo ""

# Build debug version
build:
	cargo build

# Build release version
release:
	cargo build --release

# Clean build artifacts
clean:
	cargo clean

# Run in TUI mode
run:
	cargo run

# Run tests
test:
	cargo test

# Update model database from HuggingFace
update-models:
	@./scripts/update_models.sh

# Check compilation without building
check:
	cargo check

# Format code
fmt:
	cargo fmt

# Run clippy
clippy:
	cargo clippy -- -D warnings

# Install to ~/.cargo/bin
install:
	cargo install --path .
