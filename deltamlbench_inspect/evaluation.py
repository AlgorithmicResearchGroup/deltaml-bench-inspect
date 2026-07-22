from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

POLICY_FILE = Path(__file__).with_name("evaluation_policies.yaml")


class EvaluationPolicyError(ValueError):
    """Raised when an evaluation policy or reported result is invalid."""


_ALLOWED_STATUSES = {"pilot", "revised", "calibration_required", "redesign_required"}
_ALLOWED_AGGREGATIONS = {
    "relative_improvement",
    "weighted_mean_relative_improvement",
    "constrained_relative_improvement",
    "all_metrics_must_improve",
    "worst_group_relative_improvement",
}


def _validate_policy(task_name: str, policy: Any) -> dict[str, Any]:
    if not isinstance(policy, dict):
        raise EvaluationPolicyError(f"Policy for {task_name} must be a mapping")
    if policy.get("status") not in _ALLOWED_STATUSES:
        raise EvaluationPolicyError(f"Policy for {task_name} has invalid status")
    if policy.get("aggregation") not in _ALLOWED_AGGREGATIONS:
        raise EvaluationPolicyError(f"Policy for {task_name} has invalid aggregation")
    metrics = policy.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise EvaluationPolicyError(f"Policy for {task_name} must define metrics")
    for metric_name, config in metrics.items():
        if not isinstance(config, dict):
            raise EvaluationPolicyError(f"Metric {task_name}.{metric_name} must be a mapping")
        if config.get("direction") not in {"maximize", "minimize"}:
            raise EvaluationPolicyError(f"Metric {task_name}.{metric_name} has invalid direction")
        bounds = config.get("bounds")
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise EvaluationPolicyError(f"Metric {task_name}.{metric_name} needs two bounds")
        _finite_number(config.get("weight", 1.0), f"{task_name}.{metric_name} weight")
    required = policy.get("required_outputs")
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        raise EvaluationPolicyError(f"Policy for {task_name} needs required_outputs")
    return policy


@lru_cache(maxsize=1)
def load_evaluation_policies() -> dict[str, dict[str, Any]]:
    import yaml

    document = yaml.safe_load(POLICY_FILE.read_text(encoding="utf-8")) or {}
    if document.get("version") != 1:
        raise EvaluationPolicyError("Unsupported evaluation policy version")
    tasks = document.get("tasks")
    if not isinstance(tasks, dict):
        raise EvaluationPolicyError("Evaluation policy file must contain a tasks mapping")
    return {name: _validate_policy(name, policy) for name, policy in tasks.items()}


def evaluation_policy(task_name: str) -> dict[str, Any] | None:
    policy = load_evaluation_policies().get(task_name)
    return dict(policy) if policy is not None else None


def format_evaluation_contract(task_name: str, policy: Mapping[str, Any]) -> str:
    metrics = policy.get("metrics", {})
    metric_lines = []
    for name, config in metrics.items():
        direction = config.get("direction")
        baseline = config.get("baseline")
        weight = config.get("weight", 1.0)
        baseline_text = "pending calibration" if baseline is None else str(baseline)
        metric_lines.append(
            f"- `{name}`: {direction}; baseline {baseline_text}; weight {weight}"
        )

    required = ", ".join(
        f"`{name}`" for name in policy.get("required_outputs", [])
    ) or "task-specific evidence"
    constraints = policy.get("constraints", {})
    constraint_text = (
        "\nConstraints:\n```json\n"
        + json.dumps(constraints, indent=2, sort_keys=True)
        + "\n```"
        if constraints
        else ""
    )
    artifact_formats = policy.get("artifact_formats", {})
    artifact_text = (
        "\nArtifact formats:\n```json\n"
        + json.dumps(artifact_formats, indent=2, sort_keys=True)
        + "\n```"
        if artifact_formats
        else ""
    )
    return "\n".join(
        [
            "## Benchmark evaluation contract",
            "",
            "This contract supersedes earlier scoring and deliverable text in the task",
            "description wherever they conflict.",
            "Your implementation's `solution.evaluate()` must compute and return the metrics",
            "below. Report real measured values; hardcoded or fabricated metrics receive zero.",
            "Also write `/home/agent/submission/evidence.json` with version 1, a",
            "`final_metrics` mapping, and an `artifacts` list of workspace-relative code,",
            "training-log, evaluation-log, and checkpoint paths used to support the report.",
            f"Expected task outputs or evidence: {required}.",
            f"Aggregation: `{policy.get('aggregation')}`.",
            "Metrics:",
            *metric_lines,
            constraint_text,
            artifact_text,
        ]
    ).rstrip()


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationPolicyError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise EvaluationPolicyError(f"{label} must be finite")
    return number


def _metric_improvement(name: str, config: Mapping[str, Any], value: Any) -> float:
    current = _finite_number(value, name)
    baseline_raw = config.get("baseline")
    if baseline_raw is None:
        raise EvaluationPolicyError(f"{name} requires baseline calibration")
    baseline = _finite_number(baseline_raw, f"{name} baseline")

    bounds = config.get("bounds") or [None, None]
    lower, upper = bounds
    if lower is not None and current < float(lower):
        raise EvaluationPolicyError(f"{name}={current} is below its lower bound {lower}")
    if upper is not None and current > float(upper):
        raise EvaluationPolicyError(f"{name}={current} exceeds its upper bound {upper}")

    direction = config.get("direction")
    if config.get("normalization") == "absolute_points":
        delta = current - baseline if direction == "maximize" else baseline - current
        return max(0.0, delta)
    if baseline == 0:
        raise EvaluationPolicyError(f"{name} cannot use relative improvement at baseline zero")
    if direction == "maximize":
        improvement = (current - baseline) / abs(baseline) * 100.0
    elif direction == "minimize":
        improvement = (baseline - current) / abs(baseline) * 100.0
    else:
        raise EvaluationPolicyError(f"{name} has invalid direction {direction!r}")
    return max(0.0, improvement)


def _constraint_failures(
    constraints: Mapping[str, Any], measurements: Mapping[str, Any]
) -> list[str]:
    failures: list[str] = []
    for name, config in constraints.items():
        if not isinstance(config, Mapping):
            if config == "required" and name not in measurements:
                failures.append(f"missing required measurement: {name}")
            continue
        if config.get("calibration_required"):
            failures.append(f"constraint requires calibration: {name}")
            continue
        if config.get("measurement") == "scorer_owned" and name not in measurements:
            failures.append(f"missing scorer-owned measurement: {name}")
            continue
        if name not in measurements:
            failures.append(f"missing constraint measurement: {name}")
            continue
        value = _finite_number(measurements[name], name)
        maximum = config.get("maximum")
        minimum = config.get("minimum")
        if maximum is not None and value > float(maximum):
            failures.append(f"{name}={value} exceeds maximum {maximum}")
        if minimum is not None and value < float(minimum):
            failures.append(f"{name}={value} is below minimum {minimum}")
    return failures


def score_evaluator_result(
    policy: Mapping[str, Any],
    metrics: Mapping[str, Any],
    measurements: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate reported metrics and compare them with the paper baseline."""

    measurements = measurements or {}

    configs = policy.get("metrics")
    if not isinstance(configs, Mapping) or not configs:
        raise EvaluationPolicyError("Policy must define at least one metric")

    improvements: dict[str, float] = {}
    weights: dict[str, float] = {}
    for name, config in configs.items():
        if name not in metrics:
            raise EvaluationPolicyError(f"Report did not contain required metric {name}")
        improvements[name] = _metric_improvement(name, config, metrics[name])
        weights[name] = _finite_number(config.get("weight", 1.0), f"{name} weight")

    failures = _constraint_failures(policy.get("constraints", {}), measurements)
    if failures:
        return {
            "score": 0.0,
            "status": "constraint_failed",
            "metrics": dict(metrics),
            "improvements": improvements,
            "constraint_failures": failures,
        }

    aggregation = policy.get("aggregation")
    if aggregation == "all_metrics_must_improve" and any(
        value <= 0 for value in improvements.values()
    ):
        score = 0.0
    elif aggregation == "worst_group_relative_improvement":
        group_scores = measurements.get("group_improvements")
        if not isinstance(group_scores, Mapping) or not group_scores:
            raise EvaluationPolicyError("Report did not contain group_improvements")
        score = min(_finite_number(value, str(name)) for name, value in group_scores.items())
        score = max(0.0, score)
    else:
        weight_sum = sum(weights.values())
        if weight_sum <= 0:
            raise EvaluationPolicyError("Metric weights must sum to a positive number")
        score = sum(improvements[name] * weights[name] for name in improvements) / weight_sum

    return {
        "score": float(score),
        "status": "scored",
        "metrics": dict(metrics),
        "improvements": improvements,
        "measurements": dict(measurements),
    }
