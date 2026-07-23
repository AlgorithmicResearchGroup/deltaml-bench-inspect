from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import anyio
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
from inspect_ai.util import LimitExceededError, sandbox

from deltamlbench_inspect.durable_audit import DurableAuditLog
from deltamlbench_inspect.runtime import report_command, score_command

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
        audit_root = Path(
            os.environ.get("DELTAML_AUDIT_DIR", str(REPO_ROOT / "logs" / "audit"))
        )
        run_id = str(task_state.uuid or f"sample-{time.time_ns()}")
        self.audit = DurableAuditLog(
            root=audit_root,
            run_id=run_id,
            metadata={
                "task_name": task_state.metadata.get("task_name"),
                "variant": task_state.metadata.get("variant"),
                "sample_id": task_state.sample_id,
                "sample_uuid": str(task_state.uuid or ""),
                "settings": settings,
                "model_override": os.environ.get("MODULAR_PUBLIC_MODEL"),
            },
        )
        self._workspace_manifest: dict[str, dict[str, object]] = {}
        self._bridge_token_usage = 0
        self._pending_request_token_estimate = 0

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
        self.audit.append(
            "messages_synchronized",
            {"message_count": len(messages), "last_node_id": agent_state.last_node_id},
        )

    def load_settings(self, path: str | Path) -> dict[str, object]:
        del path
        return self.settings

    def log(self, *content: object) -> None:
        line = " ".join(str(item) for item in content)
        self.log_lines.append(line)
        self.audit.append("log", {"content": line})
        transcript().info(line)

    def log_with_attributes(self, attributes: dict | None, *content: object) -> None:
        line = " ".join(str(item) for item in content)
        self.log_lines.append(line)
        self.audit.append(
            "log_with_attributes", {"content": line, "attributes": attributes}
        )
        transcript().info({"content": line, "attributes": attributes})

    def log_image(self, image_url: str, description: str | None = None) -> None:
        self.audit.append(
            "image",
            {
                "description": description,
                "image_url_prefix": image_url[:1024],
                "image_url_length": len(image_url),
            },
        )
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
        active_tokens = active.total_tokens if active is not None else 0
        token_limit = active.token_limit if active is not None else None
        time_limit = active.time_limit if active is not None else None
        return self.runtime_types.UsageInfo(
            usage=self.runtime_types.UsageSnapshot(
                tokens=max(
                    self.task_state.token_usage,
                    active_tokens,
                    self._bridge_token_usage,
                ),
                total_seconds=elapsed,
            ),
            usageLimits=self.runtime_types.UsageLimits(
                tokens=token_limit or self.task_state.token_limit or 10_000_000,
                total_seconds=time_limit or 8 * 60 * 60,
            ),
        )

    def model_request(self, payload: dict[str, object]) -> None:
        self._pending_request_token_estimate = max(
            1, len(json.dumps(payload, default=str, ensure_ascii=False)) // 4
        )
        self.audit.append("model_request", payload)

    def model_response(self, payload: dict[str, object]) -> None:
        usage = payload.get("usage")
        reported_tokens = 0
        if isinstance(usage, dict):
            reported_tokens = int(usage.get("total_tokens", 0) or 0)
        if reported_tokens == 0:
            reported_tokens = self._pending_request_token_estimate + max(
                1, len(json.dumps(payload.get("outputs", []), default=str)) // 4
            )
        self._bridge_token_usage += reported_tokens
        self._pending_request_token_estimate = 0
        self.audit.append("model_response", payload)

    def model_error(self, error: BaseException) -> None:
        self.audit.append(
            "model_error",
            {"error_type": type(error).__name__, "message": str(error)},
        )

    def save_state(self, state: dict[str, object]) -> None:
        self.saved_state = state
        self.audit.checkpoint_state(state)

    async def action(self, action: dict[str, object]) -> None:
        self.audit.append("action", action)
        transcript().info({"action": action})

    async def observation(self, observation: dict[str, object]) -> None:
        self.audit.append("observation", observation)
        transcript().info({"observation": observation})

    async def submit(self, submission: str) -> None:
        self.submission = submission or ""
        self.audit.append("submission", {"content": self.submission})
        await self.checkpoint_workspace(reason="submission")
        raise self.inspect_runtime.SubmitRequested(self.submission)

    async def score(self):
        self.audit.append("score_started")
        scorer_owned = bool(self.task_state.metadata.get("evaluation_policy"))
        visible_score = bool(self.task_state.metadata.get("visible_score", False))
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
        self.audit.append(
            "score_finished",
            {
                "status": score_result.status,
                "score": score_result.score,
                "message": score_result.message,
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
                "exit_status": result.returncode,
            },
        )
        await self.checkpoint_workspace(reason="score")
        return score_result

    async def scoreLog(self):
        return list(self.score_log)

    async def run_bash(self, script: str, timeout: float) -> str:
        self.audit.append(
            "tool_started", {"tool": "bash", "script": script, "timeout": timeout}
        )
        result = await sandbox().exec(
            cmd=["bash", "--login", "-c", script],
            cwd="/home/agent",
            user="agent",
            timeout=int(timeout),
        )
        payload = {
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
            "status": result.returncode,
        }
        self.audit.append("tool_finished", {"tool": "bash", **payload})
        await self.checkpoint_workspace(reason="bash")
        return json.dumps(payload)

    async def run_python(self, script: str, timeout: float) -> str:
        self.audit.append(
            "tool_started", {"tool": "python", "script": script, "timeout": timeout}
        )
        result = await sandbox().exec(
            cmd=["python3", "-c", script],
            cwd="/home/agent",
            user="agent",
            timeout=int(timeout),
        )
        output = result.stdout or ""
        errors = result.stderr or ""
        self.audit.append(
            "tool_finished",
            {
                "tool": "python",
                "stdout": output,
                "stderr": errors,
                "status": result.returncode,
            },
        )
        await self.checkpoint_workspace(reason="python")
        if errors:
            return f"{output}\n{errors}".strip()
        return output

    async def checkpoint_workspace(self, *, reason: str) -> None:
        max_bytes = int(
            os.environ.get("DELTAML_AUDIT_SNAPSHOT_MAX_BYTES", str(25 * 1024 * 1024))
        )
        listing = await sandbox().exec(
            cmd=[
                "bash",
                "--login",
                "-c",
                "find /home/agent/solution -type f -size -5242881c "
                "-not -path '*/.git/*' -not -path '*/__pycache__/*' -print0 "
                "| sort -z | xargs -0 -r sha256sum",
            ],
            user="root",
            timeout=120,
        )
        if not listing.success:
            self.audit.append(
                "workspace_checkpoint_error",
                {
                    "reason": reason,
                    "stderr": listing.stderr or "",
                    "exit_status": listing.returncode,
                },
            )
            return

        discovered: dict[str, str] = {}
        for line in (listing.stdout or "").splitlines():
            if "  " not in line:
                continue
            digest, absolute_path = line.split("  ", 1)
            prefix = "/home/agent/solution/"
            if not absolute_path.startswith(prefix):
                continue
            relative_path = absolute_path[len(prefix) :]
            path = Path(relative_path)
            if path.is_absolute() or ".." in path.parts:
                continue
            discovered[relative_path] = digest

        files: dict[str, dict[str, object]] = {}
        captured_bytes = 0
        skipped_files = 0
        for relative_path, digest in discovered.items():
            previous = self._workspace_manifest.get(relative_path)
            if previous is not None and previous.get("sha256") == digest:
                files[relative_path] = previous
                continue
            try:
                content = await sandbox().read_file(
                    f"/home/agent/solution/{relative_path}", text=False
                )
            except Exception as error:
                files[relative_path] = {
                    "sha256": digest,
                    "available": False,
                    "error": str(error),
                }
                skipped_files += 1
                continue
            if not isinstance(content, bytes):
                content = content.encode("utf-8")
            if captured_bytes + len(content) > max_bytes:
                files[relative_path] = {
                    "sha256": digest,
                    "size": len(content),
                    "available": False,
                    "reason": "snapshot_byte_limit",
                }
                skipped_files += 1
                continue
            blob = self.audit.store_workspace_blob(content)
            files[relative_path] = {
                "sha256": digest,
                "size": len(content),
                "available": True,
                "blob": blob,
            }
            captured_bytes += len(content)

        self._workspace_manifest = files
        self.audit.checkpoint_workspace(
            {
                "schema_version": 1,
                "timestamp": time.time(),
                "reason": reason,
                "root": "/home/agent/solution",
                "files": files,
                "captured_bytes": captured_bytes,
                "skipped_files": skipped_files,
            }
        )
@solver
def baseline_submit():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        del generate
        smoke_setup = r"""
import json
from pathlib import Path
solution = Path('/home/agent/solution/solution.py')
solution.parent.mkdir(parents=True, exist_ok=True)
solution.write_text(
    "import json\nfrom pathlib import Path\n\ndef evaluate():\n"
    "    record = json.loads(Path('/home/agent/solution/training_log.json').read_text())\n"
    "    return {\n"
    "        'metrics': {'rouge_l': record['final_metrics']['rouge_l']},\n"
    "        'measurements': {\n"
    "            'fixed_inference_budget': 'hermetic-smoke',\n"
    "            'pinned_generation_model': 'hermetic-smoke',\n"
    "        },\n"
    "    }\n",
    encoding='utf-8',
)
log = Path('/home/agent/solution/training_log.json')
log.write_text(json.dumps({'final_metrics': {'rouge_l': 27.4}}), encoding='utf-8')
submission = Path('/home/agent/submission')
submission.mkdir(parents=True, exist_ok=True)
(submission / 'evidence.json').write_text(json.dumps({
    'version': 1,
    'final_metrics': {'rouge_l': 27.4},
    'artifacts': ['solution/solution.py', 'solution/training_log.json'],
}), encoding='utf-8')
"""
        await sandbox().exec(cmd=["python3", "-c", smoke_setup], user="agent", timeout=60)
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

        outcome = "failed"
        outcome_details: dict[str, object] | None = None
        try:
            await backend.checkpoint_workspace(reason="initial")
            async with agent_bridge():
                with inspect_runtime.bind_runtime_backend(backend):
                    await modular_main.main()
        except inspect_runtime.SubmitRequested as submit:
            backend.submission = submit.submission
            outcome = "submitted"
        except BaseException as error:
            if isinstance(error, LimitExceededError):
                outcome = f"{error.type}_exhausted"
            elif isinstance(error, (asyncio.CancelledError, KeyboardInterrupt)):
                outcome = "cancelled"
            elif isinstance(error, TimeoutError):
                outcome = "timed_out"
            else:
                outcome = "failed"
            outcome_details = {
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            if isinstance(error, LimitExceededError):
                outcome_details.update(
                    {"limit_type": error.type, "value": error.value, "limit": error.limit}
                )
            backend.audit.append("solver_error", outcome_details)
            raise
        else:
            outcome = "completed"
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await backend.checkpoint_workspace(reason="terminal")
                except BaseException as checkpoint_error:
                    backend.audit.append(
                        "terminal_checkpoint_error",
                        {
                            "error_type": type(checkpoint_error).__name__,
                            "message": str(checkpoint_error),
                        },
                    )
                backend.audit.finish(outcome, outcome_details)

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
        state.metadata["durable_audit_path"] = str(backend.audit.run_dir)
        if backend.log_lines:
            state.metadata["agent_log_tail"] = "\n".join(backend.log_lines[-80:])
        if backend.saved_state is not None:
            state.store["modular_public_saved_state"] = backend.saved_state
        state.completed = True
        return state

    return solve
