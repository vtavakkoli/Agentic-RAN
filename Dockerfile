FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy only runtime code required by the pipeline.
COPY agentic_ran /app/agentic_ran
COPY scripts /app/scripts
COPY src /app/src

# Runtime writable/mountable paths.
RUN mkdir -p /app/results /app/shared_data
VOLUME ["/app/results", "/app/shared_data"]

CMD ["python", "scripts/run_all.py"]
