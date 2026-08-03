# Contributing

Thank you for improving Agentic-RAN.

## Development workflow

1. Create a focused branch.
2. Install `python -m pip install -e ".[dev]"`.
3. Add or update tests for every behavioral change.
4. Run `make lint`, `make typecheck`, and `make coverage`.
5. Keep policy actions bounded and document every new guardrail.
6. Open a pull request with the motivation, risk assessment, and validation results.

## Design rules

- The learned model may propose; it must not bypass the safety critic.
- Every policy requires a description, bounded action, impact parameters, and tests.
- Avoid heavyweight ML dependencies unless a measured benefit justifies them.
- Do not commit operator data, secrets, subscriber identifiers, trained production models, or generated artifacts.
- Preserve deterministic tests and seeded data generation.

## Commit style

Use concise imperative messages such as `add handover safety guard` or `improve batch decision validation`.
