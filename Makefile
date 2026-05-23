## ─────────────────────────────────────────────────────────────────────────────
## Stellar Spectra Analyzer — developer commands
##
## All Python tasks run inside the uv-managed virtual environment (.venv).
## Run `make` or `make help` to list available targets.
## ─────────────────────────────────────────────────────────────────────────────

PYTHON     := uv run python
UV         := uv
VENV       := .venv
PYTEST     := uv run pytest
SRC        := src
TESTS      := tests
OUTPUT_DIR := models

.DEFAULT_GOAL := help

# ── Environment ───────────────────────────────────────────────────────────────

.PHONY: install
install: ## Create .venv and install all non-TF dependencies
	$(UV) sync --extra dev

.PHONY: install-tf
install-tf: ## Install TensorFlow into the uv environment (Python 3.11/3.12 only)
	$(UV) pip install "tensorflow==2.13.0"

.PHONY: venv
venv: ## Print the path to the active uv virtual environment
	@$(UV) run python -c "import sys; print(sys.prefix)"

# ── Testing ───────────────────────────────────────────────────────────────────

.PHONY: test
test: ## Run the full test suite
	$(PYTEST) $(TESTS)/

.PHONY: test-cov
test-cov: ## Run tests with HTML coverage report (opens in browser)
	$(PYTEST) $(TESTS)/ --cov=$(SRC) --cov-report=term-missing --cov-report=html
	@echo "Coverage report → htmlcov/index.html"

.PHONY: test-fast
test-fast: ## Run tests, skip markers: slow, gpu, integration
	$(PYTEST) $(TESTS)/ -m "not slow and not gpu and not integration"

.PHONY: test-gpu
test-gpu: ## Run only GPU-marked tests
	$(PYTEST) $(TESTS)/ -m gpu

# ── Training ──────────────────────────────────────────────────────────────────

.PHONY: train-dense
train-dense: ## Train Dense model on synthetic dummy data
	$(PYTHON) -m src.train \
		--model-type dense \
		--data-source dummy \
		--epochs 50 \
		--batch-size 64 \
		--output-dir $(OUTPUT_DIR)

.PHONY: train-conv1d
train-conv1d: ## Train Conv1D model on synthetic dummy data
	$(PYTHON) -m src.train \
		--model-type conv1d \
		--data-source dummy \
		--epochs 50 \
		--batch-size 128 \
		--output-dir $(OUTPUT_DIR)

.PHONY: train-sdss
train-sdss: ## Train Conv1D model on real SDSS data (requires internet)
	$(PYTHON) -m src.train \
		--model-type conv1d \
		--data-source sdss \
		--n-stars 2000 \
		--min-snr 15 \
		--epochs 100 \
		--batch-size 128 \
		--output-dir $(OUTPUT_DIR)

# ── Smoke-tests (quick sanity checks) ────────────────────────────────────────

.PHONY: smoke
smoke: ## Quick 2-epoch training smoke-test for both architectures
	$(PYTHON) -m src.train --model-type dense  --data-source dummy --epochs 2 --batch-size 64
	$(PYTHON) -m src.train --model-type conv1d --data-source dummy --epochs 2 --batch-size 64

.PHONY: gpu-info
gpu-info: ## Print TensorFlow GPU / device information
	@$(PYTHON) -c "from src.gpu_config import get_device_info; import json; print(json.dumps(get_device_info(), indent=2))" \
		|| echo "TensorFlow not found in .venv — run 'make install-tf' first"

# ── Data ──────────────────────────────────────────────────────────────────────

.PHONY: fetch-sdss
fetch-sdss: ## Download a small SDSS sample into data/raw/ (n=200 stars)
	$(PYTHON) -c "\
from src.sdss_loader import SDSSLoader; \
import numpy as np; \
loader = SDSSLoader(); \
spectra, labels = loader.build_dataset(n_stars=200, cache_dir='data/raw'); \
np.save('data/processed/spectra.npy', spectra); \
print(f'Saved {spectra.shape[0]} spectra to data/processed/')"

# ── Cleanup ───────────────────────────────────────────────────────────────────

.PHONY: clean
clean: ## Remove compiled Python files and test caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc"     -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage

.PHONY: clean-models
clean-models: ## Remove saved model checkpoints
	rm -rf $(OUTPUT_DIR)

.PHONY: clean-all
clean-all: clean clean-models ## Remove everything including the virtual environment
	rm -rf $(VENV)

# ── Help ──────────────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z0-9_-]+:.*## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
