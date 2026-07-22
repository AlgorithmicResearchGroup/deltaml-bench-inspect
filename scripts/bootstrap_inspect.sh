#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/."
  exit 1
fi

cd "$ROOT_DIR"

DELTAML_ENVIRONMENT="$ROOT_DIR/.inspect-venv"
UV_PROJECT_ENVIRONMENT="$DELTAML_ENVIRONMENT" uv sync --locked --python 3.12

cat <<EOF
Inspect environment ready.

Activate:
  source .inspect-venv/bin/activate

List tasks:
  inspect list tasks deltamlbench_inspect/tasks/pwc.py

Check local readiness:
  ./run_benchmark.sh doctor

Smoke run:
  ./run_benchmark.sh smoke
EOF
