from __future__ import annotations

import importlib.util
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

SCORING_SCRIPT_PATH = "/home/agent/score.py"
SCORE_LOG_PATH = Path("/home/agent/score_log.jsonl")


class IntermediateScoreResult(dict):
    """Dict-like compatibility container used by legacy TaskFamily code."""

    def __init__(
        self,
        *,
        score: float | None,
        message: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            score=score,
            message=message or {},
            details=details or {},
        )

    @property
    def score(self) -> float | None:
        value = self.get("score")
        return None if value is None else float(value)


def setup_scoring() -> None:
    SCORE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def check_scoring_group() -> None:
    return None


def get_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_score(timestamp: str, **kwargs: Any) -> None:
    setup_scoring()
    payload = {"timestamp": timestamp, **kwargs}
    with SCORE_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _extract_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Could not find JSON object in scoring output")


def intermediate_score(scoring_script_path: str = SCORING_SCRIPT_PATH) -> IntermediateScoreResult:
    parts = shlex.split(scoring_script_path)
    if not parts:
        raise ValueError("Empty scoring script path")
    if parts[0].endswith(".py"):
        parts.insert(0, "python3")
    result = subprocess.run(
        parts,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = _extract_json(result.stdout)
    return IntermediateScoreResult(
        score=payload.get("score"),
        message=payload.get("message"),
        details=payload.get("details"),
    )


def get_best_score(
    score_log: list[IntermediateScoreResult],
    select_best_fn=max,
) -> float:
    numeric_scores = [
        float(entry["score"])
        for entry in score_log
        if entry.get("score") is not None
    ]
    if not numeric_scores:
        return float("nan")
    return float(select_best_fn(numeric_scores))


def load_module_from_path(
    module_path: str | Path,
    *,
    add_to_sys_modules: bool = False,
) -> ModuleType:
    module_path = Path(module_path)
    module_name = f"deltamlbench_dynamic_{module_path.stem}_{abs(hash(module_path))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    if add_to_sys_modules:
        sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
