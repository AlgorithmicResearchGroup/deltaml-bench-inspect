from __future__ import annotations

import importlib.util
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_ROOT = REPO_ROOT / "deltamlbench"
SANDBOX_DOCKERFILE = (Path(__file__).resolve().parent / "sandbox" / "compose.yaml").as_posix()
REQUIRED_TASK_FILES = (
    "README.md",
    "manifest.yaml",
    "build_steps.json",
    "requirements.txt",
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    visible_score: bool


@dataclass(frozen=True)
class TaskSpec:
    name: str
    title: str
    task_dir: Path
    instructions: str
    build_steps: list[dict[str, Any]]
    manifest: dict[str, Any]
    variants: tuple[VariantSpec, ...]


def _task_module(task_dir: Path):
    task_file = task_dir / f"{task_dir.name}.py"
    module_name = f"deltamlbench_legacy_{task_dir.name}"
    spec = importlib.util.spec_from_file_location(module_name, task_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import task module {task_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _task_instructions(task_dir: Path) -> str:
    module = _task_module(task_dir)
    task_family = module.TaskFamily
    tasks = task_family.get_tasks()
    task = tasks["main"] if "main" in tasks else next(iter(tasks.values()))
    return task_family.get_instructions(task)


def _complete_pwc_dirs() -> Iterable[Path]:
    for task_dir in sorted(TASKS_ROOT.glob("pwc_*")):
        if all((task_dir / rel).exists() for rel in REQUIRED_TASK_FILES) and (task_dir / f"{task_dir.name}.py").exists():
            yield task_dir


def discover_pwc_specs() -> list[TaskSpec]:
    specs: list[TaskSpec] = []
    for task_dir in _complete_pwc_dirs():
        manifest = yaml.safe_load((task_dir / "manifest.yaml").read_text(encoding="utf-8"))
        build_steps = json.loads((task_dir / "build_steps.json").read_text(encoding="utf-8"))
        variants = tuple(
            VariantSpec(
                name=variant_name,
                visible_score=bool(variant_cfg.get("scoring", {}).get("visible_to_agent", False)),
            )
            for variant_name, variant_cfg in manifest.get("tasks", {}).items()
        )
        specs.append(
            TaskSpec(
                name=task_dir.name,
                title=manifest.get("meta", {}).get("name", task_dir.name),
                task_dir=task_dir,
                instructions=_task_instructions(task_dir),
                build_steps=build_steps,
                manifest=manifest,
                variants=variants or (VariantSpec(name="main", visible_score=True),),
            )
        )
    return specs


def archived_pwc_dirs() -> list[str]:
    complete = {path.name for path in _complete_pwc_dirs()}
    return sorted(path.name for path in TASKS_ROOT.glob("pwc_*") if path.name not in complete)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sanitized_requirements_text(path: Path) -> str:
    lines = []
    for raw_line in _read_text(path).splitlines():
        line = raw_line.strip()
        if not line:
            lines.append(raw_line)
            continue
        lower = line.lower()
        if "task-protected-scoring" in lower:
            continue
        lines.append(raw_line)
    return "\n".join(lines).rstrip() + "\n"


def sample_files_for_task(spec: TaskSpec) -> dict[str, str]:
    files: dict[str, str] = {}
    stage_root = "/tmp/deltamlbench_assets"

    def stage_file(relative_path: str) -> None:
        source = spec.task_dir / relative_path
        if source.exists():
            if relative_path == "requirements.txt":
                files[f"{stage_root}/{relative_path}"] = _sanitized_requirements_text(source)
            else:
                files[f"{stage_root}/{relative_path}"] = _read_text(source)

    for relative_path in [
        "requirements.txt",
        "assets/score.py",
        "assets/for_agent/solution.py",
        "assets/for_agent/training_log_example.json",
    ]:
        stage_file(relative_path)

    for anti_path in sorted((spec.task_dir / "anti_cheat_validation").rglob("*")):
        if anti_path.is_file():
            rel = anti_path.relative_to(spec.task_dir).as_posix()
            files[f"{stage_root}/{rel}"] = _read_text(anti_path)

    compat_root = REPO_ROOT / "metr"
    for compat_file in [compat_root / "__init__.py", compat_root / "task_protected_scoring.py"]:
        rel = compat_file.relative_to(REPO_ROOT).as_posix()
        files[f"{stage_root}/{rel}"] = _read_text(compat_file)

    return files


def setup_script_for_task(spec: TaskSpec) -> str:
    stage_root = "/tmp/deltamlbench_assets"
    commands: list[str] = [
        "set -euo pipefail",
        "mkdir -p /home/agent /home/agent/solution /home/agent/dataset /root",
    ]

    for step in spec.build_steps:
        step_type = step.get("type")
        if step_type == "file":
            src = step["source"]
            dest = step["destination"]
            commands.extend(
                [
                    f"mkdir -p {shlex.quote(str(Path(dest).parent))}",
                    f"cp -R {shlex.quote(f'{stage_root}/{src}')} {shlex.quote(dest)}",
                ]
            )
        elif step_type == "shell":
            commands.extend(step.get("commands", []))

    training_log = spec.task_dir / "assets" / "for_agent" / "training_log_example.json"
    if training_log.exists():
        commands.extend(
            [
                "mkdir -p /home/agent",
                f"cp {shlex.quote(f'{stage_root}/assets/for_agent/training_log_example.json')} /home/agent/training_log_example.json",
            ]
        )

    commands.extend(
        [
            "mkdir -p /home/agent/metr",
            f"cp {shlex.quote(f'{stage_root}/metr/__init__.py')} /home/agent/metr/__init__.py",
            f"cp {shlex.quote(f'{stage_root}/metr/task_protected_scoring.py')} /home/agent/metr/task_protected_scoring.py",
        ]
    )

    return "\n".join(commands) + "\n"


def task_sample(spec: TaskSpec, variant: VariantSpec):
    from inspect_ai.dataset import Sample

    return Sample(
        id=f"{spec.name}:{variant.name}",
        input=spec.instructions,
        target="",
        metadata={
            "task_name": spec.name,
            "variant": variant.name,
            "title": spec.title,
            "visible_score": variant.visible_score,
            "resource_profile": spec.manifest.get("tasks", {}).get(variant.name, {}).get("resources", {}),
            "manifest_meta": spec.manifest.get("meta", {}),
        },
        files=sample_files_for_task(spec),
        setup=setup_script_for_task(spec),
    )


def parse_score_output(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("No JSON score payload found in output")


def score_command(visible_score: bool) -> str:
    flag = "" if visible_score else " --strip_score_information"
    return f"python3 /home/agent/score.py{flag}"


def summarize_manifest_meta(spec: TaskSpec) -> dict[str, Any]:
    meta = spec.manifest.get("meta", {})
    return {
        "default_provider": meta.get("default_provider"),
        "expertise": meta.get("expertise", []),
        "canary": meta.get("canary"),
    }


def normalize_task_name(raw_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", raw_name)
