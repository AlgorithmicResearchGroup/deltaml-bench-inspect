#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_FILE="$ROOT_DIR/deltamlbench_inspect/tasks/pwc.py"
DEFAULT_MODEL="${INSPECT_EVAL_MODEL:-${OPENAI_MODEL:-openai/gpt-4.1-mini}}"
DEFAULT_SOLVER="$ROOT_DIR/deltamlbench_inspect/solvers.py@modular_public_solver"
INSPECT_BIN="${INSPECT_BIN:-$ROOT_DIR/.inspect-venv/bin/inspect}"

if [[ ! -x "$INSPECT_BIN" ]]; then
  if command -v inspect >/dev/null 2>&1; then
    INSPECT_BIN="$(command -v inspect)"
  else
    echo "Inspect is not installed. Run ./scripts/bootstrap_inspect.sh first."
    exit 1
  fi
fi

usage() {
  cat <<EOF
Usage:
  ./run_benchmark.sh list
  ./run_benchmark.sh smoke
  ./run_benchmark.sh run <task_name> [model] [solver]

Examples:
  ./run_benchmark.sh list
  ./run_benchmark.sh smoke
  ./run_benchmark.sh run pwc_cnn_main anthropic/claude-sonnet-4-6
EOF
}

cmd="${1:-}"
case "$cmd" in
  list)
    "$INSPECT_BIN" list tasks "$TASK_FILE"
    ;;
  smoke)
    "$INSPECT_BIN" eval "$TASK_FILE@pwc_cnn_main" --solver "$ROOT_DIR/deltamlbench_inspect/solvers.py@baseline_submit" --limit 1 --no-sandbox-cleanup
    ;;
  run)
    task_name="${2:-}"
    if [[ -z "$task_name" ]]; then
      usage
      exit 1
    fi
    model="${3:-$DEFAULT_MODEL}"
    solver="${4:-$DEFAULT_SOLVER}"
    "$INSPECT_BIN" eval "$TASK_FILE@$task_name" --model "$model" --solver "$solver" --limit 1
    ;;
  *)
    usage
    exit 1
    ;;
esac
