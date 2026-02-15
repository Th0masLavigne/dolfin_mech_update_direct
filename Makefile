# --- Variables ---
DOC_SRC = docs/src/
DOC_OUT = ../docs/build_docs/
SRC_DIR = src/

.PHONY: install test cov clean docs help linter format

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install   Install dependencies and configure the environment"
	@echo "  linter     Execute ruff check"
	@echo "  format     Execute ruff format"
	@echo "  test      Run tests using pytest"
	@echo "  cov       Run tests and generate coverage reports (terminal + HTML)"
	@echo "  docs      Build documentation using Sphinx"
	@echo "  clean     Remove temporary files and build artifacts"
install:
	pip install -e '.[dev,docs]'
	git config core.editor "nano"

linter:
	ruff check

format:
	ruff format

test:
	pytest -v --tb=short

cov:
	pytest -v --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html

docs:
	@if [ -d "$(DOC_OUT)" ]; then rm -rf $(DOC_OUT)*; fi # folder cannot be removed if bind
	mkdir -p $(DOC_OUT)
	sphinx-build -W -v -P -E -a -b html $(DOC_SRC) $(DOC_OUT)

clean:
	rm -rf .pytest_cache .ruff_cache .venv build/ dist/ htmlcov/ 
	if [ -d "$(DOC_OUT)" ]; then rm -rf $(DOC_OUT)*; fi 
	rm -rf $(DOC_SRC)api/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "outputs" -exec rm -rf {} +