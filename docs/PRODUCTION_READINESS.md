# Production Readiness

The 35-family catalog is frozen for the current release cycle. Catalog quality
work is no longer part of the platform-readiness gate unless a task is found to
be unsafe or invalid.

## Release gates

A release candidate must satisfy all of the following:

1. `deltamlbench validate` passes the catalog, variant, policy, packaging, and
   Python-version checks.
2. `deltamlbench doctor` passes on each execution environment, including Docker
   daemon, Inspect availability, and complete access to the canonical
   `AlgorithmicResearchGroup/dmlb` Hugging Face bucket. A real
   `DELTAML_JUDGE_MODEL` must be configured.
3. CI passes unit tests, source compilation, locked package builds, and the
   sandbox image build.
4. Every released family has a paper baseline, `solution.evaluate()` reporting
   contract, and integrity-review configuration.
5. At least one baseline and one agent run per variant complete the report,
   evidence audit, and paper comparison on the target hardware class.
6. Hidden-score runs are checked for leakage, and scorer failures remain
   fail-closed.
7. Agent Bridge transcript capture and the host-side durable audit journal are
   validated for success, exception, cancellation, and configured usage-limit
   termination.

Policy status records task-design maturity; it does not replace the integrity
gate. Scores are accepted only when the individual run's code-and-evidence audit
passes.

## Commands

```bash
# Structural checks; suitable for CI and does not require Docker.
./run_benchmark.sh validate

# Structural and local runtime checks.
./run_benchmark.sh doctor

# Strict release gate for the platform and all family scoring contracts.
./run_benchmark.sh doctor --release --tasks
```

## Current state

The platform contract, locked environment, CI workflow, unprivileged agent
runtime, two-stage scorer trust boundary, reported-metric contracts, integrity
audit, and preflight commands are implemented. Full-scale baseline and agent
reproducibility runs remain operational release work. The readiness command is
the canonical machine-readable source for structural counts and blockers; use
`--json` when integrating it with release automation.
