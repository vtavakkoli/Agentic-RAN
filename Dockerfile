FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy runtime sources used by scenario runner and report aggregator.
COPY xapp_agent.py /app/xapp_agent.py
COPY generate_data.py /app/generate_data.py
COPY oran_sim /app/oran_sim
COPY scripts /app/scripts

CMD ["python", "xapp_agent.py", "--model_type", "lightweight-32"]
