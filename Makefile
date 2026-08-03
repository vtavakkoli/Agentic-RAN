.PHONY: install format lint typecheck test coverage data train run compose compose-test clean

install:
	python -m pip install -e ".[dev]"

format:
	ruff format .
	ruff check --fix .

lint:
	python -m compileall -q agentic_ran tests
	ruff check --select E9,F63,F7,F82 .

typecheck:
	python -m compileall -q agentic_ran tests

test:
	pytest -q

coverage:
	pytest --cov=agentic_ran --cov-report=term-missing --cov-report=html

data:
	python -m agentic_ran generate-data --output data/runtime/ran_policy_sample.csv

train: data
	python -m agentic_ran train --data data/runtime/ran_policy_sample.csv

run: train
	python -m agentic_ran serve

compose:
	docker compose up --build

compose-test:
	docker compose --profile test up --build --abort-on-container-exit --exit-code-from test test

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	rm -f artifacts/* results/* data/runtime/*
