FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /home/agent

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    unzip \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

CMD ["sleep", "infinity"]
