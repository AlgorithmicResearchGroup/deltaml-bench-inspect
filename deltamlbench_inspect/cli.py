from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from deltamlbench_inspect.readiness import build_readiness_report, registered_task


def _print_report(report: dict[str, Any], *, as_json: bool, include_tasks: bool) -> None:
    if as_json:
        payload = report if include_tasks else {key: value for key, value in report.items() if key != "tasks"}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    suite = report["suite"]
    print(
        f"Suite: {suite['families']} families, {suite['variants']} variants, "
        f"{suite['policies']} policies"
    )
    for check in report["checks"]:
        print(f"[{check['status'].upper():7}] {check['name']}: {check['message']}")
    print(f"Platform ready: {'yes' if report['platform_ready'] else 'no'}")
    print(f"Production ready: {'yes' if report['production_ready'] else 'no'}")
    if include_tasks:
        blocked = [task for task in report["tasks"] if not task["release_ready"]]
        if blocked:
            print("Blocked families:")
            for task in blocked:
                print(f"- {task['name']}: {'; '.join(task['blockers'])}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deltamlbench", description="DeltaMLBench operations and release gates")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate the checked-in suite contract")
    validate.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    validate.add_argument("--tasks", action="store_true", help="include per-family details")

    doctor = subparsers.add_parser("doctor", help="validate the suite and local runtime")
    doctor.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    doctor.add_argument("--tasks", action="store_true", help="include per-family details")
    doctor.add_argument(
        "--release",
        action="store_true",
        help="exit nonzero unless both the platform and every family are release-ready",
    )

    check_task = subparsers.add_parser("check-task", help="gate an individual registered task before launch")
    check_task.add_argument("task_name")
    check_task.add_argument("--allow-unready", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command in {"validate", "doctor"}:
        report = build_readiness_report(check_runtime=args.command == "doctor")
        _print_report(report, as_json=args.json, include_tasks=args.tasks)
        if args.command == "doctor" and args.release:
            return 0 if report["production_ready"] else 1
        return 0 if report["platform_ready"] else 1

    report = build_readiness_report(check_runtime=False)
    match = registered_task(report, args.task_name)
    if match is None:
        print(f"Unknown task: {args.task_name}", file=sys.stderr)
        return 2
    task, variant = match
    if not task["release_ready"] and not args.allow_unready:
        print(
            f"Refusing to launch {args.task_name}: {'; '.join(task['blockers'])}. "
            "Use --allow-unready only for scoring-contract development.",
            file=sys.stderr,
        )
        return 1
    print(f"Task accepted: {task['name']} ({variant})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
