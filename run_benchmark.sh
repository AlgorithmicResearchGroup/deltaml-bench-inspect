#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_FILE="deltamlbench_inspect/tasks/pwc.py"
DEFAULT_MODEL="${INSPECT_EVAL_MODEL:-${OPENAI_MODEL:-openai/gpt-4.1-mini}}"
DEFAULT_SOLVER="deltamlbench_inspect/solvers.py@modular_public_solver"
INSPECT_BIN="${INSPECT_BIN:-$ROOT_DIR/.inspect-venv/bin/inspect}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.inspect-venv/bin/python}"

cd "$ROOT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

require_inspect() {
  if [[ -x "$INSPECT_BIN" ]]; then
    return
  fi
  if command -v inspect >/dev/null 2>&1; then
    INSPECT_BIN="$(command -v inspect)"
  else
    echo "Inspect is not installed. Run ./scripts/bootstrap_inspect.sh first."
    exit 1
  fi
}

usage() {
  cat <<EOF
Usage:
  ./run_benchmark.sh list
  ./run_benchmark.sh validate
  ./run_benchmark.sh doctor
  ./run_benchmark.sh smoke
  ./run_benchmark.sh run [--allow-unready] <task_name> [model] [solver]

Examples:
  ./run_benchmark.sh list
  ./run_benchmark.sh smoke
  ./run_benchmark.sh run pwc_cnn_main anthropic/claude-sonnet-4-6
EOF
}

cmd="${1:-}"
case "$cmd" in
  list)
    require_inspect
    "$INSPECT_BIN" list tasks "$TASK_FILE"
    ;;
  validate)
    "$PYTHON_BIN" -m deltamlbench_inspect.cli validate "${@:2}"
    ;;
  doctor)
    "$PYTHON_BIN" -m deltamlbench_inspect.cli doctor "${@:2}"
    ;;
  smoke)
    require_inspect
    DELTAML_HERMETIC_SMOKE=1 DELTAML_JUDGE_STUB=1 "$INSPECT_BIN" eval "$TASK_FILE@pwc_cnn_main" --solver "deltamlbench_inspect/solvers.py@baseline_submit" --limit 1
    ;;
  run)
    require_inspect
    gate_args=()
    shift
    if [[ "${1:-}" == "--allow-unready" ]]; then
      gate_args+=("--allow-unready")
      shift
    fi
    task_name="${1:-}"
    if [[ -z "$task_name" ]]; then
      usage
      exit 1
    fi
    model="${2:-$DEFAULT_MODEL}"
    solver="${3:-$DEFAULT_SOLVER}"
    "$PYTHON_BIN" -m deltamlbench_inspect.cli check-task "$task_name" "${gate_args[@]}"
    "$INSPECT_BIN" eval "$TASK_FILE@$task_name" --model "$model" --solver "$solver" --limit 1
    ;;
  *)
    usage
    exit 1
    ;;
esac
