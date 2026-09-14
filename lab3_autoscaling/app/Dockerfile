FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
      "fastapi==0.115.6" \
      "uvicorn[standard]==0.34.0" \
      "httpx==0.28.1"

COPY broker.py worker.py ./

ENV PYTHONUNBUFFERED=1

# Overridden per deployment: broker:app or worker:app.
CMD ["uvicorn", "broker:app", "--host", "0.0.0.0", "--port", "8000"]
