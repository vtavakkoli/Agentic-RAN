FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY agentic_ran /app/agentic_ran
COPY scripts /app/scripts

RUN mkdir -p /app/results /app/shared_data

CMD ["python", "-m", "scripts.run_scenario", "--scenario", "lightweight-32"]
