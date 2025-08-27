.PHONY: help install install-dev test lint format clean build run-web run-demo train download-data

# Default target
help:
	@echo "Ultimate MNIST Digit Recognition - Development Commands"
	@echo "======================================================"
	@echo ""
	@echo "Setup:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  download-data Download MNIST dataset"
	@echo ""
	@echo "Development:"
	@echo "  test         Run tests"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black and isort"
	@echo "  clean        Clean up cache and build files"
	@echo ""
	@echo "Application:"
	@echo "  run-web      Launch Streamlit web application"
	@echo "  run-demo     Run feature demonstration"
	@echo "  train        Train all models"
	@echo ""
	@echo "Build:"
	@echo "  build        Build package for distribution"
	@echo ""

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

# Data
download-data:
	python quick_start.py

# Testing
test:
	pytest tests/ -v --cov=. --cov-report=html

test-fast:
	pytest tests/ -v

# Code quality
lint:
	flake8 .
	mypy . --ignore-missing-imports
	black --check .
	isort --check-only --diff .

format:
	black .
	isort .

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .mypy_cache/

# Application
run-web:
	streamlit run interface/streamlit_app.py

run-demo:
	python demo.py

train:
	python main.py --train

# Build
build:
	python -m build

# Quick setup for new developers
setup: install-dev download-data
	@echo "Setup complete! Run 'make run-web' to start the application."

# Full development cycle
dev-cycle: format lint test
	@echo "Development cycle complete!"

# Production deployment check
deploy-check: lint test build
	@echo "All checks passed! Ready for deployment."
