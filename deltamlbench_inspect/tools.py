from __future__ import annotations

import json
from typing import Any

from inspect_ai.tool import Tool, tool
from inspect_ai.util import sandbox

from .runtime import parse_score_output, report_command, score_command


@tool(name="score_task")
def score_task(*, visible_score: bool = True, scorer_owned: bool = False) -> Tool:
    """Run the task scorer inside the sandbox."""

    async def execute() -> str:
        """Run the current task scorer and return the parsed score payload."""
        await sandbox().exec(
            cmd=["bash", "--login", "-c", report_command()],
            user="agent",
            timeout=7200,
        )
        result = await sandbox().exec(
            cmd=[
                "bash",
                "--login",
                "-c",
                score_command(visible_score, scorer_owned=scorer_owned),
            ],
            user="root",
            timeout=7200,
        )
        output = f"{result.stderr or ''}\n{result.stdout or ''}".strip()
        try:
            parsed = parse_score_output(output)
            return json.dumps(parsed, indent=2, sort_keys=True)
        except Exception:
            return output

    return execute


@tool(name="show_task_metadata")
def show_task_metadata(metadata: dict[str, Any]) -> Tool:
    """Return static task metadata to the agent."""

    async def execute() -> str:
        """Return the task metadata JSON for the current sample."""
        return json.dumps(metadata, indent=2, sort_keys=True)

    return execute
