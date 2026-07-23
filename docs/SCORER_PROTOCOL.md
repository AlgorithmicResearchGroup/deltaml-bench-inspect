# Reported-Metric Scoring and Integrity Protocol

## Purpose

DeltaMLBench measures whether an agent can improve a task repository. The agent
runs the task's evaluation code and reports the resulting metrics. The benchmark
does not independently rerun the model against a second hidden dataset.

The system accepts a score only when it can connect the reported metrics to the
submitted code and declared execution evidence, and then compares those metrics
with the checked-in paper baseline.

## Trust boundary

Metric collection and scoring are separate processes:

1. `/usr/local/bin/deltamlbench_report.py` runs as the unprivileged `agent` user.
   It imports `/home/agent/solution/solution.py`, calls `evaluate()`, records the
   returned metrics, and hashes the exact source that produced them.
2. `/root/deltamlbench/scorer_driver.py` runs as root. It validates the report,
   source hash, code structure, evidence paths, and metric consistency before
   applying the policy baseline.
3. The host builds a bounded, redacted review bundle and sends it—together with
   the task, final agent messages, and pre-scoring Inspect execution events—to
   the configured integrity judge. A non-pass verdict sets the score to zero;
   `escalate` requires manual review.

Agent-controlled Python is never executed with root privileges. Policies and
the integrity driver remain unreadable and unwritable by the agent.

## Agent report and evidence

`solution.evaluate()` returns either a flat numeric metric mapping:

```python
{"accuracy": 83.2}
```

or a structured result when task constraints require measurements:

```python
{
    "metrics": {"accuracy": 83.2},
    "measurements": {"trainable_parameters": 1_000_000},
}
```

The agent also writes `/home/agent/submission/evidence.json`:

```json
{
  "version": 1,
  "final_metrics": {"accuracy": 83.2},
  "artifacts": [
    "solution/solution.py",
    "solution/training_log.json",
    "solution/model.pt"
  ]
}
```

Artifact paths are relative to `/home/agent`, must remain inside that workspace,
and must resolve to regular non-symlink files. At least one declared training or
evaluation log is required. JSON logs must contain the same final metric values
reported by `solution.evaluate()`.

## Integrity gate

The root-owned scorer fails closed when:

- the metric report or evidence manifest is missing;
- expected metrics are absent, non-numeric, non-finite, or outside policy bounds;
- `evaluate()` directly returns literal metric values;
- `solution.py` changes after metric collection;
- evidence paths escape the workspace, are missing, or are symbolic links;
- evidence metrics disagree with the report;
- declared JSON logs do not contain the reported final metrics; or
- task-specific resource or multi-objective constraints fail.

The score result includes hashes, an evidence inventory, and the structured
judge verdict. Setup records a root-owned hash snapshot of the initial solution
tree. The judge sees the final entrypoint, every new or changed solution file,
declared text artifacts, report and evidence manifests, the agent conversation,
and bounded execution events captured before scoring. Secret-shaped fields and
credential strings are redacted. Binary and large artifacts are represented by
hashes, sizes, and bounded previews; insufficient truncated evidence must be
escalated.

The judge does not receive root-owned scorer source, host files, environment
credentials, or the deterministic paper-improvement score. Review materials are
untrusted data, and the judge prompt explicitly forbids following instructions
embedded in agent-controlled code or logs.

Set `DELTAML_JUDGE_MODEL` to an Inspect model identifier for production runs.
Missing, malformed, failed, or `escalate` judge outcomes fail closed. Hermetic
smoke tests use a clearly labeled local stub and never represent a real judge run.
The stub additionally requires `DELTAML_JUDGE_STUB=1`; skipping heavyweight
external task assets alone does not bypass the judge.
The legacy per-task LLM grader files are neither staged nor executed; the shared
host-side judge is the only model-review path.

## Failure-independent agent audit

`modular-public` model requests use Inspect Agent Bridge. The bridge routes the
agent's OpenAI-compatible SDK calls through the active Inspect model and records
them in the normal Inspect transcript. The adapter converts the scaffold's
legacy `function_call`/`function` messages to modern `tool_calls`/`tool`
messages before they reach the bridge.

Inspect sample logs remain authoritative for completed samples. In addition,
the adapter writes an append-only journal to `logs/audit/<sample-uuid>/` while a
sample is running so evidence does not depend on successful sample finalization.
The directory contains:

- `events.jsonl`: flushed model, tool, action, observation, score, submission,
  checkpoint, error, and terminal-status events;
- `payloads/`: compressed full payloads for events too large to store inline;
- `state/latest.json.gz`: the most recent modular-public state checkpoint; and
- `workspace/`: content-addressed copies and manifests for changed files under
  `/home/agent/solution`.

The journal is written on the host rather than inside the Docker sandbox. It
therefore survives solver exceptions, token or time limits, cancellation, and
normal sandbox cleanup. `status.json` records the terminal state when Python is
able to execute cancellation cleanup. A hard process or machine failure may
omit that final status, but every event flushed before the failure remains.

Set `DELTAML_AUDIT_DIR` to relocate the audit root. Workspace snapshots capture
individual files smaller than 5 MiB and at most 25 MiB of changed content per
checkpoint by default; set `DELTAML_AUDIT_SNAPSHOT_MAX_BYTES` to change the
per-checkpoint total. Larger files are excluded from failure-time workspace
snapshots; declared final artifacts are still inventoried and hashed by the
normal scorer when a sample reaches scoring.

## Paper comparison

[`evaluation_policies.yaml`](../deltamlbench_inspect/evaluation_policies.yaml)
defines paper baselines, metric directions, bounds, weights, and constraints.
After integrity passes, the shared scorer computes improvement against those
baselines. Reported values at or below the baseline receive zero improvement.

`main` and `hidden_score` use the same collection, audit, and comparison path.
The hidden variant removes metric and evidence details from agent-visible scorer
output; it does not use a separate hidden dataset.
