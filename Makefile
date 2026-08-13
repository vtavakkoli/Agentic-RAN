.PHONY: install test coverage lint typecheck serve bridge simulate compose-test compose-control

install:
	python -m pip install -e ".[dev,oran]"

test:
	pytest -q

coverage:
	pytest --cov=agentic_ran --cov-report=term-missing

lint:
	ruff check --select E9,F63,F7,F82 .

typecheck:
	python -m compileall -q agentic_ran tests

serve:
	agentic-ran serve --port 8080

bridge:
	agentic-ran serve-bridge --port 8090

simulate:
	agentic-ran control-step --mode simulated --intent green-ran

compose-test:
	docker compose --profile test up --build --abort-on-container-exit --exit-code-from test test

compose-control:
	docker compose --profile control up --build
