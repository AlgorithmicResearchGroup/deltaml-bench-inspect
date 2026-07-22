# DeltaMLBench (Inspect)

DeltaMLBench is an Inspect-native benchmark for the active `pwc_*` task families in [`deltamlbench/`](deltamlbench/).

## Supported Scope

- Supported runtime: Inspect
- Supported benchmark families: 35 `pwc_*` families, exposed as 70 task variants (`main` and `hidden_score`)
- Archived from the Inspect runtime: five incomplete imports and fourteen tasks removed for insufficient headroom or redundant coverage; see [`archive/README.md`](archive/README.md) and [`docs/TASK_SUITABILITY_AUDIT.md`](docs/TASK_SUITABILITY_AUDIT.md)
- Not supported by the new runtime: `ai_rd_*`

## Evaluation Model

Agents compute and report task metrics through `solution.evaluate()`. The
runtime executes that code as the unprivileged agent, audits the reported values
against code and declared logs, submits a redacted code-and-trajectory bundle to
the configured integrity judge, and only then accepts the paper comparison.
Hidden variants suppress feedback rather than using separate hidden labels. See
[`docs/SCORER_PROTOCOL.md`](docs/SCORER_PROTOCOL.md).

Task papers, repository snapshots, and datasets are fetched from the public
[`AlgorithmicResearchGroup/dmlb`](https://huggingface.co/buckets/AlgorithmicResearchGroup/dmlb)
Hugging Face storage bucket. `deltamlbench doctor` verifies every object declared
by the active task manifests before a run is launched.

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

3. Check the suite contract and local runtime before launching work.

```bash
./run_benchmark.sh validate
./run_benchmark.sh doctor
```

4. Run a no-model smoke test that validates task import, Docker sandboxing, setup, and scoring.

```bash
.inspect-venv/bin/inspect eval \
  deltamlbench_inspect/tasks/pwc.py@pwc_cnn_main \
  --solver deltamlbench_inspect/solvers.py@baseline_submit \
  --limit 1 \
  --no-sandbox-cleanup
```

5. Run a real `modular-public` task.

```bash
export ANTHROPIC_API_KEY=...
export DELTAML_JUDGE_MODEL=anthropic/<judge-model>
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

- [`deltamlbench/`](deltamlbench/): source task families and assets
- [`deltamlbench_inspect/`](deltamlbench_inspect/): Inspect-native runtime, policies, scorers, task wrappers, sandbox image, and the `modular-public` solver integration
- [`metr/`](metr/): compatibility shim for legacy task/scoring imports
- [`run_benchmark.sh`](run_benchmark.sh): simple launcher for listing and running Inspect tasks
- [`scripts/bootstrap_inspect.sh`](scripts/bootstrap_inspect.sh): first-run local setup
- [`docs/SCORER_PROTOCOL.md`](docs/SCORER_PROTOCOL.md): scorer-owned submission, trust-boundary, aggregation, and migration contract
- [`docs/PRODUCTION_READINESS.md`](docs/PRODUCTION_READINESS.md): release gates, operational checks, and current blockers

Use [`deltamlbench/README.md`](deltamlbench/README.md) for task-source documentation. The root README is the only user-facing quickstart for running the benchmark.

## Notes

- `main` task variants expose a score tool to the agent.
- `hidden_score` variants run the same scorer but do not expose intermediate score feedback.
- Most PWC tasks still require substantial compute and may require GPUs to be practical; the smoke solver is intended only to validate the runtime path.
- The supported agent in this branch is `modular-public`.
- The runtime preserves legacy score scripts for task context, while the shared scorer validates agent-reported metrics and applies the authoritative paper-baseline policies.

## Troubleshooting

- `inspect` not found: rerun `./scripts/bootstrap_inspect.sh`
- Docker build errors: verify `docker ps` works before launching a task
- Provider auth failures: export the matching provider key in your shell before running `inspect eval`
- A task directory is missing from `./run_benchmark.sh list`: it is probably one of the five archived incomplete `pwc_*` imports
