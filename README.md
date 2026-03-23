# DeltaMLBench (Inspect)

DeltaMLBench is an Inspect-native benchmark for the complete `pwc_*` task families in [`deltamlbench/`](/Users/arg/Desktop/PUBLIC/deltaml-bench-public/deltamlbench). This branch is the Inspect runtime branch and is intended to become the standalone `deltaml-bench-inspect` repo.

## Supported Scope

- Supported runtime: Inspect
- Supported benchmark families: 49 complete `pwc_*` families, exposed as 98 task variants (`main` and `hidden_score`)
- Archived from the Inspect runtime: `pwc_fb15k_237_dabr`, `pwc_food_101_mano_tiny`, `pwc_hme100k_ical`, `pwc_imagenet_10_dpac`, `pwc_istd_rasm`
- Not supported by the new runtime: `ai_rd_*`

## Prerequisites

- Python 3.12
- `uv`
- Docker
- Optional provider key for real agent runs:
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`
  - `GOOGLE_API_KEY`

## Quick Start

1. Create the repo-local Inspect environment.

```bash
./scripts/bootstrap_inspect.sh
```

2. List the registered PWC tasks.

```bash
./run_benchmark.sh list
```

3. Run a no-model smoke test that validates task import, Docker sandboxing, setup, and scoring.

```bash
.inspect-venv/bin/inspect eval \
  deltamlbench_inspect/tasks/pwc.py@pwc_cnn_main \
  --solver deltamlbench_inspect/solvers.py@baseline_submit \
  --limit 1 \
  --no-sandbox-cleanup
```

4. Run a real `modular-public` task.

```bash
export ANTHROPIC_API_KEY=...
export MODULAR_PUBLIC_SETTINGS_PACK=t_context_and_usage_awarep_claude_legacy_1xc3.5sgda
export MODULAR_PUBLIC_ANTHROPIC_MODEL=claude-sonnet-4-6
./run_benchmark.sh run pwc_cnn_main anthropic/claude-sonnet-4-6
```

Inspect log viewer:

```bash
source .inspect-venv/bin/activate
inspect view start --host 127.0.0.1 --port 7575 --log-dir .inspect-logs
```

## Repo Layout

- [`deltamlbench/`](/Users/arg/Desktop/PUBLIC/deltaml-bench-public/deltamlbench): source task families and assets
- [`deltamlbench_inspect/`](/Users/arg/Desktop/PUBLIC/deltaml-bench-public/deltamlbench_inspect): Inspect-native runtime, scorers, task wrappers, sandbox image, and the real `modular-public` solver integration
- [`metr/`](/Users/arg/Desktop/PUBLIC/deltaml-bench-public/metr): compatibility shim for legacy task/scoring imports
- [`run_benchmark.sh`](/Users/arg/Desktop/PUBLIC/deltaml-bench-public/run_benchmark.sh): simple launcher for listing and running Inspect tasks
- [`scripts/bootstrap_inspect.sh`](/Users/arg/Desktop/PUBLIC/deltaml-bench-public/scripts/bootstrap_inspect.sh): first-run local setup

Use [`deltamlbench/README.md`](/Users/arg/Desktop/PUBLIC/deltaml-bench-public/deltamlbench/README.md) for task-source documentation. The root README is the only user-facing quickstart for running the benchmark.

## Notes

- `main` task variants expose a score tool to the agent.
- `hidden_score` variants run the same scorer but do not expose intermediate score feedback.
- Most PWC tasks still require substantial compute and may require GPUs to be practical; the smoke solver is intended only to validate the runtime path.
- The supported agent in this branch is `modular-public`.
- The new runtime keeps the existing `assets/score.py`, `anti_cheat_validation`, and task setup logic rather than rewriting each task by hand.

## Troubleshooting

- `inspect` not found: rerun `./scripts/bootstrap_inspect.sh`
- Docker build errors: verify `docker ps` works before launching a task
- Provider auth failures: export the matching provider key in your shell before running `inspect eval`
- A task directory is missing from `./run_benchmark.sh list`: it is probably one of the five archived incomplete `pwc_*` imports
