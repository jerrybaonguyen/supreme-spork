FROM python:3.11-slim

# Speed up installs
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps for torch/sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "app:api", "--host", "0.0.0.0", "--port", "8000"]