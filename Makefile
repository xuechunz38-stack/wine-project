IMAGE ?= wine-quality

.PHONY: install format lint test run benchmark docker-build docker-run docker-test clean

install:
	python -m pip install -r requirements-dev.txt

format:
	python -m black .

lint:
	python -m black --check .
	python -m flake8 .

test:
	python -m pytest

run:
	python analysis.py

benchmark:
	python analysis_polars.py

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	mkdir -p output
	docker run --rm -v "$(CURDIR)/output:/app/output" $(IMAGE)

docker-test:
	docker run --rm $(IMAGE) python -m pytest -q

clean:
	rm -rf .pytest_cache .bench_tmp output **/__pycache__
