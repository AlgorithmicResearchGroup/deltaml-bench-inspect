# DeltaMLBench Task Sources

This directory contains the source task families used by the Inspect-native DeltaMLBench runtime. It is task-source documentation, not a second bootstrap path.

## Current Status

- `pwc_*` families are the supported benchmark surface.
- The Inspect runtime discovers the complete `pwc_*` families automatically and exposes `main` and `hidden_score` variants.
- `ai_rd_*` families remain in the repo as source material but are not part of the supported Inspect workflow.

## Task Layout

Most complete `pwc_*` families contain:

- `README.md`
- `manifest.yaml`
- `build_steps.json`
- `requirements.txt`
- `<task_name>.py`
- `assets/score.py`
- `assets/for_agent/solution.py`
- `anti_cheat_validation/`

The Inspect runtime reuses the existing task assets, setup commands, and scoring code rather than maintaining separate handwritten task ports.

## Running Tasks

Use the repo root workflow from [README.md](/Users/arg/Desktop/PUBLIC/deltaml-bench-public/README.md):

```bash
./scripts/bootstrap_inspect.sh
./run_benchmark.sh list
./run_benchmark.sh run pwc_cnn_main anthropic/claude-sonnet-4-6
```

Individual task READMEs under `pwc_*` now use the same root-level Inspect workflow.
