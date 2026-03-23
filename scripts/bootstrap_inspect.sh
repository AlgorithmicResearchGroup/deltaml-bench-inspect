#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/."
  exit 1
fi

cd "$ROOT_DIR"

if [[ ! -d .inspect-venv ]]; then
  uv venv .inspect-venv --python 3.12
fi

uv pip install --python .inspect-venv/bin/python -e .

cat <<EOF
Inspect environment ready.

Activate:
  source .inspect-venv/bin/activate

List tasks:
  inspect list tasks deltamlbench_inspect/tasks/pwc.py

Smoke run:
  ./run_benchmark.sh smoke
EOF
