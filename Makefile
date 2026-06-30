# Self-documenting: `make` lists mainline targets (b3_core CLI wrappers).
# Viz and offline extras: `b3_core viz …` and examples/offline/.

UV ?= uv
RUN ?= $(UV) run
B3 ?= $(RUN) b3_core

CASE ?= examples/simple.yaml
SWEEP_ROOT ?= examples/param_sweeps

.PHONY: help install test lint run sweep

.DEFAULT_GOAL := help

help: ## List targets (default goal)
	@printf '%b\n' \
		'b3_core Makefile — mainline CLI wrappers' \
		'' \
		'Variables:' \
		'  UV=$(UV)  RUN=$(RUN)' \
		'  CASE=$(CASE)  SWEEP_ROOT=$(SWEEP_ROOT)' \
		'' \
		'Targets:'
	@grep -E '^[a-zA-Z0-9_.-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@printf '%b\n' '' 'Viz: b3_core viz --help' 'Offline: examples/offline/README.md'

install: ## uv sync
	$(UV) sync

test: ## pytest
	$(RUN) pytest

lint: ## ruff check
	$(RUN) ruff check src tests examples

run: ## b3_core run — homogenise one case
	$(B3) run $(CASE)

sweep: ## b3_core sweep homogenise
	$(B3) sweep homogenise --root $(SWEEP_ROOT)