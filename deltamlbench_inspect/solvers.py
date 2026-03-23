from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path

from inspect_ai._util.json import json_changes
from inspect_ai.agent import agent_bridge
from inspect_ai.event import StateEvent
from inspect_ai.log._samples import sample_active, set_active_sample_total_messages
from inspect_ai.log._transcript import transcript
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ModelOutput,
)
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.solver._task_state import state_jsonable
from inspect_ai.tool import ToolCall
from inspect_ai.util import sandbox

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULAR_PUBLIC_ROOT = REPO_ROOT / "modular-public"
def _maybe_parse_json(raw: object) -> object:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _tool_arguments(raw: object) -> dict[str, object]:
    parsed = _maybe_parse_json(raw)
    if isinstance(parsed, dict):
        return parsed
    return {"input": parsed}


def _modular_settings(pack_name: str | None = None) -> dict[str, object]:
    manifest = json.loads((MODULAR_PUBLIC_ROOT / "manifest.json").read_text(encoding="utf-8"))
    pack = pack_name or os.environ.get("MODULAR_PUBLIC_SETTINGS_PACK") or manifest["defaultSettingsPack"]
    settings = manifest["settingsPacks"][pack]
    return {"pack": pack, "settings": settings}


def _load_modular_modules():
    modular_path = MODULAR_PUBLIC_ROOT.as_posix()
    if modular_path not in sys.path:
        sys.path.insert(0, modular_path)

    inspect_runtime = importlib.import_module("inspect_runtime")
    base = importlib.import_module("base")
    main = importlib.import_module("main")
    runtime_types = importlib.import_module("runtime_types")
    return inspect_runtime, base, main, runtime_types


class _ModularPublicInProcessBackend:
    def __init__(
        self,
        *,
        task_state: TaskState,
        settings: dict[str, object],
        runtime_types,
        inspect_runtime,
    ) -> None:
        self.task_state = task_state
        self.settings = settings
        self.runtime_types = runtime_types
        self.inspect_runtime = inspect_runtime
        self.base_messages = list(task_state.messages)
        self.submission: str = ""
        self.score_log: list[object] = []
        self.saved_state: dict[str, object] | None = None
        self.log_lines: list[str] = []

    def _emit_state_update(self, before: dict[str, object]) -> None:
        after = state_jsonable(self.task_state)
        changes = json_changes(before, after)
        if changes:
            transcript()._event(StateEvent(changes=changes))
        set_active_sample_total_messages(len(self.task_state.messages))

    def _messages_from_live_state(self, agent_state) -> list[ChatMessage]:
        nodes = agent_state.nodes
        last_node_id = agent_state.last_node_id
        if not nodes or last_node_id < 0:
            return self.base_messages

        path_ids: list[int] = []
        current = int(last_node_id)
        while current >= 0:
            path_ids.append(current)
            parent = nodes[current].parent
            if parent == current:
                break
            current = int(parent)
        path_ids.reverse()

        messages: list[ChatMessage] = list(self.base_messages)
        pending_tool_call_id: str | None = None
        for node_id in path_ids:
            node = nodes[node_id]
            message = node.message
            content = message.content or ""
            if message.role == "assistant":
                if message.function_call:
                    pending_tool_call_id = f"tool_{node_id}"
                    messages.append(
                        ChatMessageAssistant(
                            content=content,
                            source="generate",
                            model="modular-public",
                            tool_calls=[
                                ToolCall(
                                    id=pending_tool_call_id,
                                    function=str(message.function_call.get("name", "tool")),
                                    arguments=_tool_arguments(
                                        message.function_call.get("arguments")
                                    ),
                                )
                            ],
                        )
                    )
                else:
                    messages.append(
                        ChatMessageAssistant(
                            content=content,
                            source="generate",
                            model="modular-public",
                        )
                    )
            elif message.role in {"function", "tool"}:
                messages.append(
                    ChatMessageTool(
                        content=content,
                        source="generate",
                        function=message.name,
                        tool_call_id=pending_tool_call_id or f"tool_{node_id}",
                    )
                )
                pending_tool_call_id = None
            elif message.role == "user":
                messages.append(ChatMessageUser(content=content, source="generate"))
            elif message.role == "system":
                messages.append(ChatMessageSystem(content=content, source="generate"))
        return messages

    def sync_messages(self, agent_state) -> None:
        before = state_jsonable(self.task_state)
        messages = self._messages_from_live_state(agent_state)
        if messages == self.task_state.messages:
            return
        self.task_state.messages = messages
        self._emit_state_update(before)

    def load_settings(self, path: str | Path) -> dict[str, object]:
        del path
        return self.settings

    def log(self, *content: object) -> None:
        line = " ".join(str(item) for item in content)
        self.log_lines.append(line)
        transcript().info(line)

    def log_with_attributes(self, attributes: dict | None, *content: object) -> None:
        line = " ".join(str(item) for item in content)
        self.log_lines.append(line)
        transcript().info({"content": line, "attributes": attributes})

    def log_image(self, image_url: str, description: str | None = None) -> None:
        transcript().info({"image_url": image_url, "description": description})

    async def getTask(self):
        return self.runtime_types.TaskInfo(
            instructions=self.task_state.input_text,
            scoring=self.runtime_types.ScoringInfo(
                intermediate=bool(self.task_state.metadata.get("visible_score", False))
            ),
        )

    async def get_usage(self):
        active = sample_active()
        elapsed = int(active.running_time) if active is not None else 0
        return self.runtime_types.UsageInfo(
            usage=self.runtime_types.UsageSnapshot(
                tokens=self.task_state.token_usage,
                total_seconds=elapsed,
            ),
            usageLimits=self.runtime_types.UsageLimits(
                tokens=self.task_state.token_limit or 10_000_000,
                total_seconds=8 * 60 * 60,
            ),
        )

    def save_state(self, state: dict[str, object]) -> None:
        self.saved_state = state

    async def action(self, action: dict[str, object]) -> None:
        transcript().info({"action": action})

    async def observation(self, observation: dict[str, object]) -> None:
        transcript().info({"observation": observation})

    async def submit(self, submission: str) -> None:
        self.submission = submission or ""
        raise self.inspect_runtime.SubmitRequested(self.submission)

    async def score(self):
        result = await sandbox().exec(
            cmd=["bash", "--login", "-c", "python3 /home/agent/score.py"],
            timeout=7200,
        )
        payload = self.inspect_runtime.Hooks()._parse_score_output(
            result.stdout or "", result.stderr or ""
        )
        score_result = self.runtime_types.ScoreResult(
            status="scoringSucceeded" if result.success else "invalidSubmission",
            score=payload.get("score"),
            message=payload.get("message", {}),
            execResult=self.runtime_types.ExecResult(
                stdout=result.stdout or "",
                stderr=result.stderr or "",
                exitStatus=result.returncode,
            ),
        )
        self.score_log.append(
            self.runtime_types.ScoreLogEntry(
                score=score_result.score,
                message=score_result.message,
                timestamp=time.time(),
            )
        )
        return score_result

    async def scoreLog(self):
        return list(self.score_log)

    async def run_bash(self, script: str, timeout: float) -> str:
        result = await sandbox().exec(
            cmd=["bash", "--login", "-c", script],
            cwd="/home/agent",
            timeout=int(timeout),
        )
        return json.dumps(
            {
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
                "status": result.returncode,
            }
        )

    async def run_python(self, script: str, timeout: float) -> str:
        result = await sandbox().exec(
            cmd=["python3", "-c", script],
            cwd="/home/agent",
            timeout=int(timeout),
        )
        output = result.stdout or ""
        errors = result.stderr or ""
        if errors:
            return f"{output}\n{errors}".strip()
        return output
@solver
def baseline_submit():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        del generate
        state.output = ModelOutput.from_content("baseline", "baseline")
        state.completed = True
        return state

    return solve


@solver(name="modular_public_solver")
def modular_public_solver(settings_pack: str | None = None):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        del generate
        inspect_runtime, _, modular_main, runtime_types = _load_modular_modules()
        settings_info = _modular_settings(settings_pack)
        backend = _ModularPublicInProcessBackend(
            task_state=state,
            settings=settings_info["settings"],
            runtime_types=runtime_types,
            inspect_runtime=inspect_runtime,
        )

        try:
            async with agent_bridge():
                with inspect_runtime.bind_runtime_backend(backend):
                    await modular_main.main()
        except inspect_runtime.SubmitRequested as submit:
            backend.submission = submit.submission

        state.output = ModelOutput.from_content(
            "modular-public",
            backend.submission or "modular-public completed without explicit submission",
        )
        state.metadata["agent_name"] = "modular-public"
        state.metadata["settings_pack"] = str(settings_info["pack"])
        state.metadata["model_override"] = os.environ.get("MODULAR_PUBLIC_MODEL")
        state.metadata["anthropic_model_override"] = os.environ.get(
            "MODULAR_PUBLIC_ANTHROPIC_MODEL"
        )
        state.metadata["openai_model_override"] = os.environ.get(
            "MODULAR_PUBLIC_OPENAI_MODEL"
        )
        if backend.log_lines:
            state.metadata["agent_log_tail"] = "\n".join(backend.log_lines[-80:])
        if backend.saved_state is not None:
            state.store["modular_public_saved_state"] = backend.saved_state
        state.completed = True
        return state

    return solve
