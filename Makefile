PYTHON ?= python

.PHONY: install format lint test run docker-build docker-run
install:
	$(PYTHON) -m pip install -r requirements-dev.txt
format:
	$(PYTHON) -m black .
lint:
	$(PYTHON) -m black --check .
	$(PYTHON) -m ruff check .
test:
	$(PYTHON) -m pytest
run:
	$(PYTHON) analysis.py
docker-build:
	docker build -t wine-project:week4 .
docker-run:
	mkdir -p container-output
	docker run --rm -v "$(CURDIR)/container-output:/app/figures" wine-project:week4
