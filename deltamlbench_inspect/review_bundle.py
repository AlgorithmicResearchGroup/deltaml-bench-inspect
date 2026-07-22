from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}
MAX_ARTIFACT_BYTES = 40_000
MAX_TOTAL_TEXT_BYTES = 240_000


def snapshot_tree(root: Path) -> dict[str, str]:
    root = root.resolve()
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _workspace_file(workspace: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise ValueError(f"absolute artifact path: {relative}")
    root = workspace.resolve()
    candidate = root / relative
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symbolic-link artifact path: {relative}")
    path = candidate.resolve()
    path.relative_to(root)
    if not path.is_file():
        raise ValueError(f"artifact is not a regular file: {relative}")
    return path


def build_review_bundle(
    *,
    task_name: str,
    workspace: Path,
    solution_path: Path,
    report_path: Path,
    evidence_path: Path,
    baseline_manifest_path: Path | None = None,
) -> dict[str, Any]:
    report = _json_object(report_path)
    evidence = _json_object(evidence_path)
    remaining = MAX_TOTAL_TEXT_BYTES
    artifacts: list[dict[str, Any]] = []
    for relative in evidence.get("artifacts", [])[:100]:
        if not isinstance(relative, str):
            continue
        try:
            path = _workspace_file(workspace, relative)
        except (OSError, ValueError):
            continue
        raw = path.read_bytes()
        row: dict[str, Any] = {
            "path": relative,
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "mtime_ns": path.stat().st_mtime_ns,
        }
        if path.suffix.lower() in TEXT_SUFFIXES and remaining > 0:
            limit = min(MAX_ARTIFACT_BYTES, remaining)
            text = raw[:limit].decode("utf-8", errors="replace")
            row["content"] = text
            row["truncated"] = len(raw) > limit
            remaining -= len(text.encode("utf-8"))
        artifacts.append(row)

    baseline = _json_object(baseline_manifest_path) if baseline_manifest_path else {}
    solution_root = workspace.resolve() / "solution"
    changed_files: list[dict[str, Any]] = []
    changed_text_remaining = MAX_TOTAL_TEXT_BYTES
    for path in sorted(solution_root.rglob("*")):
        if (
            not path.is_file()
            or path.is_symlink()
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
        ):
            continue
        relative = path.relative_to(solution_root).as_posix()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if baseline.get(relative) == digest:
            continue
        row: dict[str, Any] = {
            "path": f"solution/{relative}",
            "size": len(raw),
            "sha256": digest,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        if path.suffix.lower() in TEXT_SUFFIXES and changed_text_remaining > 0:
            limit = min(MAX_ARTIFACT_BYTES, changed_text_remaining)
            text = raw[:limit].decode("utf-8", errors="replace")
            row["content"] = text
            row["truncated"] = len(raw) > limit
            changed_text_remaining -= len(text.encode("utf-8"))
        changed_files.append(row)

    source = solution_path.read_bytes()
    return {
        "version": 1,
        "task_name": task_name,
        "report": report,
        "evidence_manifest": evidence,
        "solution": {
            "path": solution_path.resolve().relative_to(workspace.resolve()).as_posix(),
            "size": len(source),
            "sha256": hashlib.sha256(source).hexdigest(),
            "content": source[:MAX_ARTIFACT_BYTES].decode("utf-8", errors="replace"),
            "truncated": len(source) > MAX_ARTIFACT_BYTES,
        },
        "changed_solution_files": changed_files,
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--snapshot-output", type=Path)
    parser.add_argument("--task-name")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--solution", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--baseline-manifest", type=Path)
    args = parser.parse_args()
    if args.snapshot_root:
        if not args.snapshot_output:
            parser.error("--snapshot-output is required with --snapshot-root")
        args.snapshot_output.write_text(
            json.dumps(snapshot_tree(args.snapshot_root), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    if not all((args.task_name, args.workspace, args.solution, args.report, args.evidence)):
        parser.error("review mode requires task name, workspace, solution, report, and evidence")
    bundle = build_review_bundle(
        task_name=args.task_name,
        workspace=args.workspace,
        solution_path=args.solution,
        report_path=args.report,
        evidence_path=args.evidence,
        baseline_manifest_path=args.baseline_manifest,
    )
    print(json.dumps(bundle, sort_keys=True))


if __name__ == "__main__":
    main()
