FROM python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93

ARG PIP_VERSION=26.1.1
ARG SETUPTOOLS_VERSION=80.9.0
ARG WHEEL_VERSION=0.46.3

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

RUN python3 -m pip install --no-cache-dir \
    "pip==${PIP_VERSION}" \
    "setuptools==${SETUPTOOLS_VERSION}" \
    "wheel==${WHEEL_VERSION}"

# Task setup runs as root, while the agent's shell and Python tools explicitly
# run as this unprivileged account. The integrity scorer remains under /root.
RUN useradd --create-home --home-dir /home/agent --shell /bin/bash agent \
  && chown -R agent:agent /home/agent

CMD ["sleep", "infinity"]
