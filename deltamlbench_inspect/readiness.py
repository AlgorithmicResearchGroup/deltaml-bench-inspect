from __future__ import annotations

import shutil
import subprocess
import sys
import json
import os
import re
from importlib.metadata import PackageNotFoundError, version
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from deltamlbench_inspect.evaluation import load_evaluation_policies
from deltamlbench_inspect.runtime import REPO_ROOT, discover_pwc_specs

ASSET_BUCKET = "AlgorithmicResearchGroup/dmlb"

EXPECTED_FAMILY_COUNT = 35
EXPECTED_VARIANTS = {"main", "hidden_score"}


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    message: str


def _runtime_check(command: str, args: list[str]) -> Check:
    executable = shutil.which(command)
    if executable is None:
        return Check(command, "error", f"{command} is not installed or is not on PATH")
    try:
        completed = subprocess.run(
            [executable, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Check(command, "error", f"{command} check failed: {exc}")
    output = (completed.stdout or completed.stderr).strip().splitlines()
    detail = output[0] if output else f"exit code {completed.returncode}"
    status = "pass" if completed.returncode == 0 else "error"
    return Check(command, status, detail)


def _inspect_check() -> Check:
    try:
        installed = version("inspect-ai")
    except PackageNotFoundError:
        return Check("inspect", "error", "inspect-ai is not installed in the active Python environment")
    return Check("inspect", "pass", installed)


def _installation_check() -> Check:
    try:
        from importlib.metadata import distribution

        direct_url = distribution("deltamlbench-inspect").read_text("direct_url.json")
        if not direct_url:
            return Check("installation", "pass", "installed distribution")
        source = json.loads(direct_url).get("url", "")
        expected = REPO_ROOT.resolve().as_uri()
        if source.rstrip("/") != expected.rstrip("/"):
            return Check("installation", "error", f"environment points to another checkout: {source}")
        return Check("installation", "pass", REPO_ROOT.as_posix())
    except (PackageNotFoundError, json.JSONDecodeError):
        return Check("installation", "error", "deltamlbench-inspect distribution metadata is unavailable")


def _judge_check() -> Check:
    model = os.environ.get("DELTAML_JUDGE_MODEL")
    if not model:
        return Check(
            "integrity_judge",
            "error",
            "DELTAML_JUDGE_MODEL is required for production runs",
        )
    return Check("integrity_judge", "pass", model)


def _asset_check(specs: list[Any]) -> Check:
    from huggingface_hub import list_bucket_tree

    pattern = re.compile(
        r"hf://buckets/AlgorithmicResearchGroup/dmlb/([^'\"\s]+)"
    )
    expected = {
        match.group(1)
        for spec in specs
        for step in spec.build_steps
        for command in step.get("commands", [])
        for match in pattern.finditer(command)
    }
    if not expected:
        return Check("task_assets", "error", "no Hugging Face bucket assets were declared")
    try:
        available = {
            item.path
            for item in list_bucket_tree(ASSET_BUCKET, recursive=True)
            if hasattr(item, "size")
        }
    except Exception as error:  # network/client failures are reported as readiness errors
        return Check("task_assets", "error", f"could not list {ASSET_BUCKET}: {error}")
    missing = sorted(expected - available)
    if missing:
        preview = ", ".join(missing[:3])
        return Check(
            "task_assets",
            "error",
            f"{len(missing)}/{len(expected)} declared bucket objects are missing: {preview}",
        )
    return Check(
        "task_assets",
        "pass",
        f"{len(expected)}/{len(expected)} declared objects present in {ASSET_BUCKET}",
    )


def build_readiness_report(*, check_runtime: bool = True) -> dict[str, Any]:
    checks: list[Check] = []
    specs = discover_pwc_specs()
    policies = load_evaluation_policies()
    active_names = {spec.name for spec in specs}
    policy_names = set(policies)

    if len(specs) == EXPECTED_FAMILY_COUNT:
        checks.append(Check("catalog_size", "pass", f"{len(specs)} active families"))
    else:
        checks.append(
            Check(
                "catalog_size",
                "error",
                f"expected {EXPECTED_FAMILY_COUNT} active families, found {len(specs)}",
            )
        )

    missing_policies = sorted(active_names - policy_names)
    orphaned_policies = sorted(policy_names - active_names)
    if not missing_policies and not orphaned_policies:
        checks.append(Check("policy_coverage", "pass", "every active family has one policy"))
    else:
        checks.append(
            Check(
                "policy_coverage",
                "error",
                f"missing={missing_policies}; orphaned={orphaned_policies}",
            )
        )

    bad_variants = {
        spec.name: sorted(variant.name for variant in spec.variants)
        for spec in specs
        if {variant.name for variant in spec.variants} != EXPECTED_VARIANTS
    }
    variant_count = sum(len(spec.variants) for spec in specs)
    if not bad_variants:
        checks.append(Check("variants", "pass", f"{variant_count} registered variants"))
    else:
        checks.append(Check("variants", "error", f"unexpected variants: {bad_variants}"))

    task_rows: list[dict[str, Any]] = []
    for spec in specs:
        policy = policies.get(spec.name, {})
        score_script = spec.task_dir / "assets" / "score.py"
        starter = spec.task_dir / "assets" / "for_agent" / "solution.py"
        judge_review_policy = bool(policy.get("metrics")) and isinstance(
            policy.get("required_outputs"), list
        )
        status = str(policy.get("status", "missing"))
        blockers: list[str] = []
        metrics = policy.get("metrics", {})
        if not metrics or any(config.get("baseline") is None for config in metrics.values()):
            blockers.append("paper baseline is missing")
        if not score_script.is_file():
            blockers.append("reported-metric contract is missing")
        if not starter.is_file() or "def evaluate" not in starter.read_text(encoding="utf-8"):
            blockers.append("starter solution does not define evaluate()")
        if not judge_review_policy:
            blockers.append("judge review policy is missing")
        release_ready = not blockers
        task_rows.append(
            {
                "name": spec.name,
                "status": status,
                "reported_metric_contract": score_script.is_file(),
                "judge_review_policy": judge_review_policy,
                "release_ready": release_ready,
                "blockers": blockers,
            }
        )

    ready_count = sum(bool(row["release_ready"]) for row in task_rows)
    contract_count = sum(bool(row["reported_metric_contract"]) for row in task_rows)
    judge_policy_count = sum(bool(row["judge_review_policy"]) for row in task_rows)
    checks.append(
        Check(
            "task_release_gate",
            "pass" if ready_count == len(task_rows) else "warning",
            f"{ready_count}/{len(task_rows)} release-ready reported-metric contracts",
        )
    )

    supported_python = (3, 11) <= sys.version_info[:2] < (3, 13)
    checks.append(
        Check(
            "python",
            "pass" if supported_python else "error",
            f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )
    )

    required_files = [
        REPO_ROOT / "deltamlbench_inspect" / "sandbox" / "Dockerfile",
        REPO_ROOT / "deltamlbench_inspect" / "sandbox" / "compose.yaml",
        REPO_ROOT / "deltamlbench_inspect" / "report_collector.py",
        REPO_ROOT / "deltamlbench_inspect" / "review_bundle.py",
        REPO_ROOT / "deltamlbench_inspect" / "integrity_judge.py",
        REPO_ROOT / "deltamlbench_inspect" / "scorer_driver.py",
    ]
    if (REPO_ROOT / "pyproject.toml").is_file():
        required_files.extend([REPO_ROOT / "pyproject.toml", REPO_ROOT / "uv.lock"])
    absent = [path.relative_to(REPO_ROOT).as_posix() for path in required_files if not path.is_file()]
    checks.append(
        Check(
            "release_files",
            "pass" if not absent else "error",
            "required release files present" if not absent else f"missing: {absent}",
        )
    )

    if check_runtime:
        checks.extend(
            [
                _runtime_check("docker", ["info", "--format", "{{.ServerVersion}}"]),
                _inspect_check(),
                _installation_check(),
                _judge_check(),
                _asset_check(specs),
            ]
        )

    platform_ready = not any(check.status == "error" for check in checks)
    production_ready = platform_ready and bool(task_rows) and ready_count == len(task_rows)
    return {
        "schema_version": 1,
        "platform_ready": platform_ready,
        "production_ready": production_ready,
        "suite": {
            "families": len(specs),
            "variants": variant_count,
            "policies": len(policies),
            "reported_metric_contracts": contract_count,
            "judge_review_policies": judge_policy_count,
            "release_ready_families": ready_count,
            "policy_statuses": dict(sorted(Counter(row["status"] for row in task_rows).items())),
        },
        "checks": [asdict(check) for check in checks],
        "tasks": task_rows,
    }


def registered_task(report: dict[str, Any], task_name: str) -> tuple[dict[str, Any], str] | None:
    for task in report["tasks"]:
        for variant in EXPECTED_VARIANTS:
            if task_name == f"{task['name']}_{variant}":
                return task, variant
    return None
