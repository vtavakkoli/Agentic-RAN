# syntax=docker/dockerfile:1.7
FROM python:3.13-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY agentic_ran ./agentic_ran
RUN python -m pip install --upgrade pip build \
    && python -m build --wheel --outdir /dist

FROM python:3.13-slim AS runtime

ARG APP_UID=10001
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    AGENTIC_RAN_DATASET=/workspace/data/runtime/ran_policy_sample.csv \
    AGENTIC_RAN_MODEL=/workspace/artifacts/policy_selector.joblib \
    AGENTIC_RAN_POLICIES=/workspace/configs/policies.yaml \
    AGENTIC_RAN_RESULTS=/workspace/results \
    AGENTIC_RAN_AUDIT=/workspace/results/audit/decisions.jsonl \
    AGENTIC_RAN_REGISTRY=/workspace/artifacts/registry \
    AGENTIC_RAN_EXECUTION_MODE=recommend \
    AGENTIC_RAN_PLANNING_HORIZON=3

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${APP_UID} agentic \
    && useradd --uid ${APP_UID} --gid agentic --create-home --shell /usr/sbin/nologin agentic
WORKDIR /workspace
COPY --from=builder /dist/*.whl /tmp/
RUN python -m pip install /tmp/*.whl "websocket-client>=1.8,<2" \
    && rm -f /tmp/*.whl
COPY configs ./configs
COPY data/bootstrap ./data/bootstrap
RUN mkdir -p data/runtime artifacts/registry results/audit \
    && chown -R agentic:agentic /workspace
USER agentic

EXPOSE 8080
HEALTHCHECK --interval=20s --timeout=3s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2)" || exit 1

ENTRYPOINT ["agentic-ran"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]

FROM runtime AS test
USER root
RUN python -m pip install "pytest>=8.3,<10" "pytest-cov>=6,<8" "httpx>=0.27,<1"
COPY tests ./tests
USER agentic
