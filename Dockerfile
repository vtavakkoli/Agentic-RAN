FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY agentic_ran /app/agentic_ran
COPY scripts /app/scripts
COPY src /app/src
COPY models /app/models
COPY policies /app/policies
COPY configs /app/configs

RUN mkdir -p /app/results /app/shared_data

CMD ["python", "-m", "src.benchmark", "--benchmark-scope", "main"]
