from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

try:
    from .evaluation import EvaluationPolicyError, score_evaluator_result
except ImportError:  # staged as a standalone root-owned script in the sandbox
    from evaluation import EvaluationPolicyError, score_evaluator_result

_LOG_SUFFIXES = {".json", ".jsonl", ".log", ".txt"}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationPolicyError(f"{path} must contain a JSON object")
    return value


def _finite_metrics(value: Any, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise EvaluationPolicyError(f"{label} must be a mapping")
    metrics: dict[str, float] = {}
    for name, raw in value.items():
        if not isinstance(name, str) or isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise EvaluationPolicyError(f"{label}.{name} must be numeric")
        number = float(raw)
        if not math.isfinite(number):
            raise EvaluationPolicyError(f"{label}.{name} must be finite")
        metrics[name] = number
    return metrics


def _measurements(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationPolicyError("report.measurements must be a mapping")
    result: dict[str, Any] = {}
    for name, raw in value.items():
        if not isinstance(name, str):
            raise EvaluationPolicyError("measurement names must be strings")
        if isinstance(raw, Mapping):
            result[name] = _measurements(raw)
        elif not isinstance(raw, bool) and isinstance(raw, (int, float)) and math.isfinite(float(raw)):
            result[name] = float(raw)
        elif raw is None or isinstance(raw, (str, bool)):
            result[name] = raw
        else:
            raise EvaluationPolicyError(f"report.measurements.{name} is invalid")
    return result


def _inside_workspace(workspace: Path, relative: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute():
        raise EvaluationPolicyError(f"Evidence path must be relative: {relative}")
    root = workspace.resolve()
    candidate = root / raw
    current = root
    for part in raw.parts:
        current = current / part
        if current.is_symlink():
            raise EvaluationPolicyError(f"Evidence artifact traverses a symbolic link: {relative}")
    path = candidate.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise EvaluationPolicyError(f"Evidence path escapes workspace: {relative}") from error
    if not path.is_file():
        raise EvaluationPolicyError(f"Evidence artifact is not a regular file: {relative}")
    return path


def _literal_metric_returns(source: str, metric_names: set[str]) -> list[str]:
    tree = ast.parse(source)
    evaluate_functions = [
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "evaluate"
    ]
    if not evaluate_functions:
        return ["solution does not define evaluate()"]
    issues: list[str] = []
    for function in evaluate_functions:
        for node in ast.walk(function):
            if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Dict):
                continue
            returned: dict[str, ast.AST] = {}
            for key, value in zip(node.value.keys, node.value.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    returned[key.value] = value
            literal = [
                name
                for name in metric_names
                if name in returned
                and isinstance(returned[name], ast.Constant)
                and isinstance(returned[name].value, (int, float))
            ]
            if literal:
                issues.append("evaluate() directly returns literal metric values: " + ", ".join(sorted(literal)))
    return issues


def _find_metric_values(value: Any, names: set[str]) -> dict[str, list[float]]:
    found: dict[str, list[float]] = {name: [] for name in names}

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if key in names and not isinstance(child, bool) and isinstance(child, (int, float)):
                    found[str(key)].append(float(child))
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return found


def _log_metrics(paths: list[Path], names: set[str]) -> dict[str, list[float]]:
    found: dict[str, list[float]] = {name: [] for name in names}
    for path in paths:
        if path.suffix.lower() not in {".json", ".jsonl"} or path.stat().st_size > 20_000_000:
            continue
        try:
            if path.suffix.lower() == ".jsonl":
                values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            else:
                values = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        current = _find_metric_values(values, names)
        for name, metric_values in current.items():
            found[name].extend(metric_values)
    return found


def audit_integrity(
    *,
    workspace: Path,
    solution_path: Path,
    evidence_path: Path,
    report: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = _finite_metrics(report.get("metrics"), "report.metrics")
    expected = set(policy.get("metrics", {}))
    missing = sorted(expected - set(metrics))
    issues: list[str] = []
    if missing:
        issues.append("missing reported metrics: " + ", ".join(missing))

    if not solution_path.is_file() or solution_path.is_symlink():
        issues.append("solution.py is missing or is a symbolic link")
        source = ""
    else:
        source = solution_path.read_text(encoding="utf-8")
        actual_hash = hashlib.sha256(solution_path.read_bytes()).hexdigest()
        if report.get("solution_sha256") != actual_hash:
            issues.append("solution changed after metric collection")
        try:
            issues.extend(_literal_metric_returns(source, expected))
        except SyntaxError as error:
            issues.append(f"solution.py cannot be parsed: {error}")

    evidence_files: list[Path] = []
    evidence_rows: list[dict[str, Any]] = []
    if not evidence_path.is_file():
        issues.append("submission/evidence.json is required")
    else:
        evidence = _load_json(evidence_path)
        if evidence.get("version") != 1:
            issues.append("unsupported evidence manifest version")
        final_metrics = _finite_metrics(evidence.get("final_metrics", {}), "evidence.final_metrics")
        for name in expected:
            if name not in final_metrics or name not in metrics:
                continue
            tolerance = max(1e-6, abs(metrics[name]) * 1e-4)
            if abs(final_metrics[name] - metrics[name]) > tolerance:
                issues.append(f"evidence final_metrics.{name} does not match the reported value")
        artifacts = evidence.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts or not all(isinstance(item, str) for item in artifacts):
            issues.append("evidence.artifacts must be a non-empty list of relative paths")
        else:
            for relative in artifacts[:100]:
                try:
                    path = _inside_workspace(workspace, relative)
                except EvaluationPolicyError as error:
                    issues.append(str(error))
                    continue
                evidence_files.append(path)
                evidence_rows.append(
                    {
                        "path": path.relative_to(workspace.resolve()).as_posix(),
                        "size": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )

    log_files = [path for path in evidence_files if path.suffix.lower() in _LOG_SUFFIXES]
    if not log_files:
        issues.append("at least one declared training or evaluation log is required")
    logged = _log_metrics(log_files, expected)
    for name in sorted(expected & set(metrics)):
        values = logged.get(name, [])
        tolerance = max(1e-6, abs(metrics[name]) * 1e-4)
        if not any(abs(value - metrics[name]) <= tolerance for value in values):
            issues.append(f"reported metric {name} was not found with the same value in declared JSON logs")

    return {
        "status": "passed" if not issues else "failed",
        "issues": issues,
        "solution_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest() if source else None,
        "evidence": evidence_rows,
    }


def run_scorer(
    policy_path: Path,
    report_path: Path,
    evidence_path: Path,
    workspace: Path,
    solution_path: Path,
) -> dict[str, Any]:
    policy = _load_json(policy_path)
    if not report_path.is_file():
        return {
            "score": 0.0,
            "status": "report_missing",
            "message": {"error": "solution.evaluate() did not produce a metric report"},
        }
    report = _load_json(report_path)
    if report.get("version") != 1 or report.get("source") != "solution.evaluate":
        raise EvaluationPolicyError("Unsupported or untrusted metric report")
    metrics = _finite_metrics(report.get("metrics"), "report.metrics")
    measurements = _measurements(report.get("measurements", {}))
    audit = audit_integrity(
        workspace=workspace,
        solution_path=solution_path,
        evidence_path=evidence_path,
        report=report,
        policy=policy,
    )
    if audit["status"] != "passed":
        return {
            "score": 0.0,
            "status": "integrity_failed",
            "reported_metrics": metrics,
            "integrity": audit,
            "message": {"error": "Reported metrics did not pass the code-and-evidence audit"},
        }

    payload = score_evaluator_result(policy, metrics, measurements)
    payload["policy_status"] = policy.get("status")
    payload["integrity"] = audit
    payload["reported_metrics"] = metrics
    payload["message"] = {
        "interpretation": f"Integrity-validated reported metrics produced {payload['score']:.4f}"
    }
    return payload


def _hidden_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "score": payload.get("score", 0.0),
        "status": payload.get("status", "unknown"),
        "message": {"interpretation": "Score details hidden for this variant"},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--strip-score-information", action="store_true")
    args = parser.parse_args()
    try:
        payload = run_scorer(args.policy, args.report, args.evidence, args.workspace, args.solution)
    except Exception as error:  # failures remain machine-readable and fail closed
        payload = {
            "score": 0.0,
            "status": "scoring_error",
            "message": {"error": str(error), "exception": repr(error)},
        }
    if args.strip_score_information:
        payload = _hidden_payload(payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
