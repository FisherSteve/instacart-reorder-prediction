# Instacart Reorder Prediction - Makefile Orchestration
# Als Lernprojekt bewusst ausführlich kommentiert und strukturiert
#
# Usage:
#   make build          - Run data validation and feature engineering
#   make train MODEL=xgb - Train specified model with configuration
#   make report         - Generate leaderboard and HTML reports
#   make validate       - Run end-to-end pipeline testing
#   make clean          - Remove intermediate and output files
#
# Examples:
#   make build
#   make train MODEL=logreg
#   make train MODEL=xgb
#   make train MODEL=lgbm
#   make report
#   make validate
#   make clean

# Default Python interpreter (can be overridden)
PYTHON := python

# Default model for training (can be overridden with MODEL=...)
MODEL := logreg

# Configuration file (can be overridden with CONFIG=...)
CONFIG := config.yaml

# Check if virtual environment is activated
VENV_CHECK := $(shell $(PYTHON) -c "import sys; print('venv' if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 'system')")

# Colors for output formatting
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

# Help target - shows available commands
.PHONY: help
help:
	@echo "$(GREEN)Instacart Reorder Prediction - Makefile Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Main Commands:$(NC)"
	@echo "  make build          - Run data validation and feature engineering"
	@echo "  make train MODEL=x  - Train model (logreg, xgb, lgbm)"
	@echo "  make report         - Generate performance reports"
	@echo "  make validate       - Run end-to-end pipeline validation"
	@echo "  make clean          - Clean intermediate and output files"
	@echo ""
	@echo "$(YELLOW)Setup Commands:$(NC)"
	@echo "  make install        - Install Python dependencies"
	@echo "  make setup          - Complete setup (venv + install + validate)"
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make build"
	@echo "  make train MODEL=logreg"
	@echo "  make train MODEL=xgb"
	@echo "  make train MODEL=lgbm"
	@echo "  make report"
	@echo ""
	@echo "$(YELLOW)Configuration:$(NC)"
	@echo "  PYTHON=$(PYTHON)"
	@echo "  MODEL=$(MODEL)"
	@echo "  CONFIG=$(CONFIG)"
	@echo "  Virtual Environment: $(VENV_CHECK)"

# Check if config file exists
check-config:
	@if [ ! -f "$(CONFIG)" ]; then \
		echo "$(RED)ERROR: Configuration file $(CONFIG) not found!$(NC)"; \
		echo "$(YELLOW)Create config.yaml or specify CONFIG=path/to/config.yaml$(NC)"; \
		exit 1; \
	fi

# Check if virtual environment is activated
check-venv:
	@if [ "$(VENV_CHECK)" = "system" ]; then \
		echo "$(YELLOW)WARNING: No virtual environment detected$(NC)"; \
		echo "$(YELLOW)Consider activating venv: source venv/bin/activate (Unix) or venv\\Scripts\\activate (Windows)$(NC)"; \
	else \
		echo "$(GREEN)✓ Virtual environment active$(NC)"; \
	fi

# Install Python dependencies
.PHONY: install
install: check-venv
	@echo "$(GREEN)Installing Python dependencies...$(NC)"
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

# Complete setup: create venv, install dependencies, validate
.PHONY: setup
setup:
	@echo "$(GREEN)Setting up Instacart Reorder Prediction pipeline...$(NC)"
	@if [ ! -d "venv" ]; then \
		echo "$(YELLOW)Creating virtual environment...$(NC)"; \
		$(PYTHON) -m venv venv; \
		echo "$(GREEN)✓ Virtual environment created$(NC)"; \
		echo "$(YELLOW)Activate with: source venv/bin/activate (Unix) or venv\\Scripts\\activate (Windows)$(NC)"; \
	else \
		echo "$(GREEN)✓ Virtual environment already exists$(NC)"; \
	fi
	@echo "$(YELLOW)Please activate the virtual environment and run 'make install' to continue$(NC)"

# Build target - data validation and feature engineering
.PHONY: build
build: check-config check-venv
	@echo "$(GREEN)Building feature dataset...$(NC)"
	@echo "$(YELLOW)Step 1: Data validation and feature engineering$(NC)"
	$(PYTHON) src/build_dataset.py --config $(CONFIG)
	@echo "$(GREEN)✓ Feature dataset built successfully$(NC)"

# Train target - train specified model
.PHONY: train
train: check-config check-venv
	@echo "$(GREEN)Training model: $(MODEL)$(NC)"
	@if [ ! -f "data/features.parquet" ]; then \
		echo "$(YELLOW)Features not found. Running build first...$(NC)"; \
		$(MAKE) build; \
	fi
	@echo "$(YELLOW)Training $(MODEL) model with configuration...$(NC)"
	$(PYTHON) src/train.py --model $(MODEL) --config $(CONFIG)
	@echo "$(GREEN)✓ Model $(MODEL) trained successfully$(NC)"

# Train all models sequentially
.PHONY: train-all
train-all: check-config check-venv
	@echo "$(GREEN)Training all models...$(NC)"
	@if [ ! -f "data/features.parquet" ]; then \
		echo "$(YELLOW)Features not found. Running build first...$(NC)"; \
		$(MAKE) build; \
	fi
	@echo "$(YELLOW)Training LogisticRegression...$(NC)"
	$(PYTHON) src/train.py --model logreg --config $(CONFIG)
	@echo "$(YELLOW)Training XGBoost...$(NC)"
	$(PYTHON) src/train.py --model xgb --config $(CONFIG)
	@echo "$(YELLOW)Training LightGBM...$(NC)"
	$(PYTHON) src/train.py --model lgbm --config $(CONFIG)
	@echo "$(GREEN)✓ All models trained successfully$(NC)"

# Report target - generate leaderboard and HTML reports
.PHONY: report
report: check-config check-venv
	@echo "$(GREEN)Generating performance reports...$(NC)"
	@if [ ! -d "reports" ] || [ -z "$$(ls -A reports/metrics_*.json 2>/dev/null)" ]; then \
		echo "$(YELLOW)No model metrics found. Training a model first...$(NC)"; \
		$(MAKE) train MODEL=$(MODEL); \
	fi
	@echo "$(YELLOW)Collecting metrics and generating reports...$(NC)"
	$(PYTHON) src/report.py --config $(CONFIG)
	@echo "$(GREEN)✓ Reports generated successfully$(NC)"
	@echo "$(YELLOW)Open reports/report.html in your browser$(NC)"

# Validate target - end-to-end pipeline testing
.PHONY: validate
validate: check-config check-venv
	@echo "$(GREEN)Running end-to-end pipeline validation...$(NC)"
	@if [ -f "src/validate_pipeline.py" ]; then \
		echo "$(YELLOW)Running pipeline validation script...$(NC)"; \
		$(PYTHON) src/validate_pipeline.py --config $(CONFIG); \
	else \
		echo "$(YELLOW)Running basic validation workflow...$(NC)"; \
		echo "$(YELLOW)1. Building features...$(NC)"; \
		$(MAKE) build; \
		echo "$(YELLOW)2. Training test model...$(NC)"; \
		$(MAKE) train MODEL=logreg; \
		echo "$(YELLOW)3. Generating report...$(NC)"; \
		$(MAKE) report; \
	fi
	@echo "$(GREEN)✓ Pipeline validation completed$(NC)"

# Clean target - remove intermediate and output files
.PHONY: clean
clean:
	@echo "$(GREEN)Cleaning intermediate and output files...$(NC)"
	@echo "$(YELLOW)Removing feature datasets...$(NC)"
	@rm -f data/features.parquet
	@rm -rf data/intermediate/
	@echo "$(YELLOW)Removing model outputs...$(NC)"
	@rm -f reports/model_*.joblib
	@rm -f reports/metrics_*.json
	@rm -f reports/metrics_leaderboard.csv
	@rm -f reports/report.html
	@echo "$(YELLOW)Removing Python cache...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup completed$(NC)"

# Deep clean - also remove virtual environment
.PHONY: clean-all
clean-all: clean
	@echo "$(GREEN)Deep cleaning (including virtual environment)...$(NC)"
	@echo "$(YELLOW)Removing virtual environment...$(NC)"
	@rm -rf venv/
	@echo "$(GREEN)✓ Deep cleanup completed$(NC)"

# Development targets
.PHONY: lint
lint: check-venv
	@echo "$(GREEN)Running code quality checks...$(NC)"
	@if command -v flake8 >/dev/null 2>&1; then \
		echo "$(YELLOW)Running flake8...$(NC)"; \
		flake8 src/ --max-line-length=120 --ignore=E501,W503; \
	else \
		echo "$(YELLOW)flake8 not installed, skipping lint$(NC)"; \
	fi

# Test configuration loading
.PHONY: test-config
test-config: check-config check-venv
	@echo "$(GREEN)Testing configuration loading...$(NC)"
	$(PYTHON) src/config_utils.py
	@echo "$(GREEN)✓ Configuration test completed$(NC)"

# Show current status
.PHONY: status
status: check-config check-venv
	@echo "$(GREEN)Pipeline Status:$(NC)"
	@echo ""
	@echo "$(YELLOW)Configuration:$(NC)"
	@if [ -f "$(CONFIG)" ]; then \
		echo "  ✓ Config file: $(CONFIG)"; \
	else \
		echo "  ✗ Config file: $(CONFIG) (missing)"; \
	fi
	@echo ""
	@echo "$(YELLOW)Data:$(NC)"
	@if [ -f "data/features.parquet" ]; then \
		echo "  ✓ Features: data/features.parquet"; \
	else \
		echo "  ✗ Features: data/features.parquet (run 'make build')"; \
	fi
	@echo ""
	@echo "$(YELLOW)Models:$(NC)"
	@for model in logreg xgb lgbm; do \
		if [ -f "reports/model_$$model.joblib" ]; then \
			echo "  ✓ Model: $$model"; \
		else \
			echo "  ✗ Model: $$model (run 'make train MODEL=$$model')"; \
		fi; \
	done
	@echo ""
	@echo "$(YELLOW)Reports:$(NC)"
	@if [ -f "reports/report.html" ]; then \
		echo "  ✓ HTML Report: reports/report.html"; \
	else \
		echo "  ✗ HTML Report: reports/report.html (run 'make report')"; \
	fi

# Quick start - complete workflow
.PHONY: quickstart
quickstart: check-config check-venv
	@echo "$(GREEN)Running complete pipeline workflow...$(NC)"
	@echo "$(YELLOW)This will: build features → train all models → generate reports$(NC)"
	@read -p "Continue? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	$(MAKE) build
	$(MAKE) train-all
	$(MAKE) report
	@echo "$(GREEN)✓ Complete pipeline finished!$(NC)"
	@echo "$(YELLOW)Open reports/report.html to see results$(NC)"

# Show pipeline workflow
.PHONY: workflow
workflow:
	@echo "$(GREEN)Instacart Reorder Prediction - Pipeline Workflow$(NC)"
	@echo ""
	@echo "$(YELLOW)1. Setup (one-time):$(NC)"
	@echo "   make setup          # Create virtual environment"
	@echo "   source venv/bin/activate  # Activate venv (Unix)"
	@echo "   make install        # Install dependencies"
	@echo ""
	@echo "$(YELLOW)2. Build Features:$(NC)"
	@echo "   make build          # SQL-based feature engineering"
	@echo ""
	@echo "$(YELLOW)3. Train Models:$(NC)"
	@echo "   make train MODEL=logreg  # Train LogisticRegression"
	@echo "   make train MODEL=xgb     # Train XGBoost"
	@echo "   make train MODEL=lgbm    # Train LightGBM"
	@echo "   # OR: make train-all     # Train all models"
	@echo ""
	@echo "$(YELLOW)4. Generate Reports:$(NC)"
	@echo "   make report         # Create HTML performance report"
	@echo ""
	@echo "$(YELLOW)5. Validation & Cleanup:$(NC)"
	@echo "   make validate       # End-to-end pipeline test"
	@echo "   make clean          # Remove intermediate files"
	@echo ""
	@echo "$(YELLOW)Quick Start:$(NC)"
	@echo "   make quickstart     # Complete workflow in one command"

# Phony targets (targets that don't create files)
.PHONY: help check-config check-venv install setup build train train-all report validate clean clean-all lint test-config status quickstart workflow