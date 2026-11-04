# Promotion_ROI_Analysis — pipeline targets.
# The project path contains a space ("Coding Projects"), so every path is quoted and
# commands are invoked through the venv interpreter via $(PY).

PY := ./.venv/bin/python
PIP := ./.venv/bin/pip
RUFF := ./.venv/bin/ruff
PYTEST := ./.venv/bin/pytest

.DEFAULT_GOAL := help

.PHONY: help setup download select data panel its synth impact warning test dashboard lint all

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(firstword $(MAKEFILE_LIST)) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup: ## Create/refresh the venv dependencies
	"$(PY)" -m venv .venv || true
	"$(PIP)" install --upgrade pip
	"$(PIP)" install -r requirements.txt

download: ## Data is already provided in data/raw (see docs/data_sources.md)
	@echo "Raw Kaggle CSVs are expected in data/raw/ (git-ignored)."
	@echo "To re-download: install the Kaggle CLI, accept the competition rules, then run"
	@echo "  kaggle competitions download -c favorita-grocery-sales-forecasting -p data/raw"
	@echo "and extract the archives into data/raw/. See docs/data_sources.md for details."

select: ## Detect promotion events and write data/processed/selected_events.parquet
	"$(PY)" -m src.load

data: select ## Alias for 'select' (the Unit 1 data-layer build)

panel: ## Build per-event donor panels (data/interim) + analyzable/dropped tables
	"$(PY)" -m src.build_panel

its: ## Fit interrupted time-series counterfactuals (needs: panel)
	"$(PY)" -m src.interrupted_ts

synth: ## Fit synthetic-control counterfactuals + ITS/SC comparison (needs: its)
	"$(PY)" -m src.synthetic_control

impact: ## Cannibalization + net-impact + classification (needs: synth)
	"$(PY)" -m src.cannibalization
	"$(PY)" -m src.net_impact

warning: ## Early-warning rule + backtest (needs: impact)
	"$(PY)" -m src.early_warning

test: ## Run the test suite
	"$(PYTEST)" -q

dashboard: ## Build the precomputed Plotly dashboard -> docs_site/index.html (needs: impact warning)
	"$(PY)" -m src.build_dashboard

lint: ## Lint the source tree
	"$(RUFF)" check src

all: select panel its synth impact warning dashboard ## Run the full pipeline through the published dashboard
	@echo "Pipeline complete: selected_events, panels, ITS + SC, cannibalization + net impact, early warning, dashboard."
