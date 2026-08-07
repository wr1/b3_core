# Self-documenting: `make` (or `make help`) lists targets.
# Viz and offline extras: `b3_core viz …` and examples/offline/.
# DocKB: `make selfdoc` (tree) · `make docs` (serve) · `make docs-build` (static)
#        PORT=3011 DOCKB=dockb  DOCS_BASE=/b3_core  PREVIEW_PORT=4173

UV ?= uv
RUN ?= $(UV) run
B3 ?= $(RUN) b3_core
DOCKB ?= dockb
DOCKB_EXPORT ?= dockb-export
PORT ?= 3000
PREVIEW_PORT ?= 4173
# GitHub Pages project site: https://wr1.github.io/b3_core/
DOCS_BASE ?= /b3_core
SITE_DIR ?= site

CASE ?= examples/simple.yaml
SWEEP_ROOT ?= examples/param_sweeps

.PHONY: help install test lint format pre-commit run sweep \
	selfdoc docs docs-serve docs-open docs-static docs-build docs-preview

.DEFAULT_GOAL := help

help: ## List targets (default goal)
	@printf '%b\n' \
		'b3_core Makefile — mainline CLI wrappers + DocKB' \
		'' \
		'Variables:' \
		'  UV=$(UV)  RUN=$(RUN)  DOCKB=$(DOCKB)  PORT=$(PORT)' \
		'  DOCKB_EXPORT=$(DOCKB_EXPORT)  DOCS_BASE=$(DOCS_BASE)  SITE_DIR=$(SITE_DIR)' \
		'  PREVIEW_PORT=$(PREVIEW_PORT)  CASE=$(CASE)  SWEEP_ROOT=$(SWEEP_ROOT)' \
		'' \
		'Targets:'
	@grep -E '^[a-zA-Z0-9_.-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@printf '%b\n' \
		'' \
		'Dev: make install | make lint | make format | make pre-commit | make test' \
		'DocKB: make selfdoc | make docs | make docs-build | make docs-preview' \
		'Viz: b3_core viz --help' \
		'Offline: examples/offline/README.md'

install: ## uv sync (+ dev extras)
	$(UV) sync --extra dev
	$(RUN) pre-commit install

test: ## pytest
	$(RUN) pytest

lint: ## ruff check (src + tests)
	$(RUN) ruff check src tests

format: ## ruff format (src + tests)
	$(RUN) ruff format src tests
	$(RUN) ruff check --fix src tests

pre-commit: ## run pre-commit hooks on all files
	$(RUN) pre-commit run --all-files

run: ## b3_core run — homogenise one case
	$(B3) run $(CASE)

sweep: ## b3_core sweep homogenise
	$(B3) sweep homogenise --root $(SWEEP_ROOT)

# ---------------------------------------------------------------------------
# DocKB (fumano) — docs/*.mdx (+ optional kb/) via shared dockb runtime
# ---------------------------------------------------------------------------

selfdoc: docs-static ## list docs tree + serve hints (no server)

docs: docs-serve ## serve DocKB site with dockb (default PORT=3000)

docs-serve: ## same as docs — dockb from project root
	@command -v $(DOCKB) >/dev/null 2>&1 || { \
		echo "dockb not found on PATH."; \
		echo "Install shared runtime:"; \
		echo "  ln -s \$$HOME/apps/dockb-runtime/bin/dockb ~/.local/bin/dockb"; \
		echo "  (or set DOCKB=/path/to/dockb)"; \
		exit 1; \
	}
	@test -d docs || { echo "missing docs/ — run from repo root"; exit 1; }
	@test -f dockb.json || echo "warning: missing dockb.json (site identity defaults apply)"
	@port=$(PORT); \
	while command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -qE ":$$port\\s"; do \
		echo "port $$port in use, trying $$((port+1))"; port=$$((port+1)); \
	done; \
	echo "DocKB: http://localhost:$$port/docs"; \
	echo "  Dev KB (if kb/ present): http://localhost:$$port/dev"; \
	echo "  Guides: getting-started, homogenize, visualization"; \
	echo "  Concepts: method, resin-halo, curvature"; \
	echo "  Reference: input-schema, outputs, backends, cli"; \
	$(DOCKB) $$port

docs-open: ## open browser to /docs (uses PORT; run make docs first)
	@url="http://localhost:$(PORT)/docs"; \
	echo "opening $$url"; \
	if command -v xdg-open >/dev/null 2>&1; then xdg-open "$$url"; \
	elif command -v sensible-browser >/dev/null 2>&1; then sensible-browser "$$url"; \
	else echo "open $$url in a browser (no xdg-open)"; fi

docs-static: ## list docs/ MDX tree (no server)
	@if [ ! -d docs ]; then echo "missing docs/"; exit 1; fi
	@echo "=== docs/ (static MDX — no live server) ==="
	@find docs -type f \( -name '*.mdx' -o -name '*.md' -o -name 'meta.json' \) | sort | sed 's|^|  |'
	@echo ""
	@if [ -d kb/dev ]; then \
		echo "=== kb/dev/ (rich mode) ==="; \
		find kb/dev -type f \( -name '*.md' -o -name 'README.md' \) 2>/dev/null | sort | sed 's|^|  |'; \
		echo ""; \
	fi
	@echo "Live server:          make docs              # needs dockb"
	@echo "Static export:        make docs-build        # → $(SITE_DIR)/ for Pages"
	@echo "Preview export:       make docs-preview      # http://localhost:$(PREVIEW_PORT)$(DOCS_BASE)/"
	@echo "Published:            https://wr1.github.io/b3_core/"
	@echo "Raw entry:            docs/index.mdx  docs/guides/  docs/concepts/  docs/reference/"

docs-build: ## static HTML export → site/ (basePath=$(DOCS_BASE); needs dockb-export)
	@command -v $(DOCKB_EXPORT) >/dev/null 2>&1 || { \
		echo "dockb-export not found on PATH."; \
		echo "  ln -s \$$HOME/apps/dockb-runtime/bin/dockb-export ~/.local/bin/dockb-export"; \
		echo "  (or set DOCKB_EXPORT=/path/to/dockb-export)"; \
		exit 1; \
	}
	@test -d docs || { echo "missing docs/ — run from repo root"; exit 1; }
	$(DOCKB_EXPORT) $(SITE_DIR) $(DOCS_BASE)
	@touch $(SITE_DIR)/.nojekyll
	@test -f $(SITE_DIR)/index.html && test -d $(SITE_DIR)/docs && test -d $(SITE_DIR)/figures || { \
		echo "docs-build: incomplete export under $(SITE_DIR)/"; exit 1; \
	}
	@echo "docs-build: ready — make docs-preview, or commit $(SITE_DIR)/ for GitHub Pages"
	@echo "  live URL: https://wr1.github.io/b3_core/"

docs-preview: ## serve prebuilt site/ with basePath (default port $(PREVIEW_PORT))
	@test -f $(SITE_DIR)/index.html || { \
		echo "missing $(SITE_DIR)/index.html — run: make docs-build"; exit 1; \
	}
	@root=$$(mktemp -d /tmp/b3_core-pages.XXXXXX); \
	ln -sfn "$$(cd $(SITE_DIR) && pwd)" "$$root/b3_core"; \
	url="http://localhost:$(PREVIEW_PORT)$(DOCS_BASE)/docs/"; \
	echo "docs-preview: $$url"; \
	echo "  (static root $$root — Ctrl-C to stop)"; \
	if command -v xdg-open >/dev/null 2>&1; then (sleep 0.4; xdg-open "$$url") >/dev/null 2>&1 & fi; \
	python3 -m http.server $(PREVIEW_PORT) --directory "$$root"