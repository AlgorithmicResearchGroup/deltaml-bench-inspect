from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import numbers
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


def _load_solution(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("deltamlbench_agent_solution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load solution: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "evaluate", None)):
        raise RuntimeError("Solution must define evaluate()")
    return module


def _numeric_mapping(value: Any, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a mapping")
    result: dict[str, float] = {}
    for name, raw in value.items():
        if isinstance(name, str) and not isinstance(raw, bool) and isinstance(raw, numbers.Real):
            number = float(raw)
            if math.isfinite(number):
                result[name] = number
    return result


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(name): _json_value(child) for name, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(child) for child in value]
    if not isinstance(value, bool) and isinstance(value, numbers.Real):
        number = float(value)
        if not math.isfinite(number):
            raise RuntimeError("measurements must be finite")
        return number
    if value is None or isinstance(value, (str, bool)):
        return value
    raise RuntimeError(f"measurements contain a non-JSON value: {type(value).__name__}")


def collect_report(solution_path: Path, output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_path.parent / "evaluation.log"
    source_bytes = solution_path.read_bytes()
    module = _load_solution(solution_path)
    with log_path.open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                raw_result = module.evaluate()
            except Exception:
                traceback.print_exc()
                raise

    if not isinstance(raw_result, Mapping):
        raise RuntimeError("solution.evaluate() must return a mapping")
    raw_metrics = raw_result.get("metrics", raw_result)
    metrics = _numeric_mapping(raw_metrics, "reported metrics")
    measurements = _json_value(raw_result.get("measurements", {}))
    if not isinstance(measurements, dict):
        raise RuntimeError("measurements must be a mapping")
    if not metrics:
        raise RuntimeError("solution.evaluate() did not report any numeric metrics")

    report = {
        "version": 1,
        "source": "solution.evaluate",
        "solution_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "reported_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "measurements": measurements,
        "evaluation_log": log_path.name,
    }
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect_report(args.solution, args.output)
    print(json.dumps({"status": "reported", "metrics": sorted(report["metrics"])}))


if __name__ == "__main__":
    main()
