from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import aiohttp
from pydantic import BaseModel

from runtime_types import (
    ExecResult,
    MiddlemanOutput,
    MiddlemanResult,
    MiddlemanSettings,
    OpenaiChatMessage,
    RatedOption,
    RatingOption,
    ScoreLogEntry,
    ScoreResult,
    TaskInfo,
    UsageInfo,
    UsageLimits,
    UsageSnapshot,
)


DEFAULT_PROXY_PORT = 13131
_runtime_backend: ContextVar[Any | None] = ContextVar(
    "modular_public_runtime_backend", default=None
)


class SubmitRequested(SystemExit):
    def __init__(self, submission: str):
        super().__init__(0)
        self.submission = submission


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


@contextlib.contextmanager
def bind_runtime_backend(backend: Any):
    token = _runtime_backend.set(backend)
    try:
        yield
    finally:
        _runtime_backend.reset(token)


def _current_backend() -> Any | None:
    return _runtime_backend.get()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _message_payload(message: OpenaiChatMessage | dict[str, Any]) -> dict[str, Any]:
    """Normalize messages emitted by both legacy and Pydantic-based modules."""
    return OpenaiChatMessage.model_validate(message).model_dump(exclude_none=True)


def _bridge_compatible_openai_messages(
    messages: list[OpenaiChatMessage] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Translate legacy function messages to the modern tool-call wire format."""
    converted: list[dict[str, Any]] = []
    pending_tool_call_id: str | None = None
    for index, raw_message in enumerate(messages):
        message = _message_payload(raw_message)
        role = message.get("role")
        function_call = message.pop("function_call", None)
        if role == "assistant" and isinstance(function_call, dict):
            pending_tool_call_id = f"modular_tool_{index}"
            converted.append(
                {
                    **message,
                    "tool_calls": [
                        {
                            "id": pending_tool_call_id,
                            "type": "function",
                            "function": {
                                "name": str(function_call.get("name", "tool")),
                                "arguments": str(function_call.get("arguments", "{}")),
                            },
                        }
                    ],
                }
            )
        elif role == "function":
            converted.append(
                {
                    "role": "tool",
                    "content": message.get("content", ""),
                    "tool_call_id": pending_tool_call_id or f"modular_tool_{index}",
                }
            )
            pending_tool_call_id = None
        else:
            converted.append(message)
    return converted


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _normalize_model_name(model: str) -> str:
    model = _resolve_runtime_model(model)
    if model.startswith("inspect"):
        return model
    lowered = model.lower()
    if "claude" in lowered:
        return f"inspect/anthropic/{model}"
    if lowered.startswith(("gpt", "o1", "o3")):
        return f"inspect/openai/{model}"
    return f"inspect/{model}"


def _resolve_runtime_model(model: str) -> str:
    generic_override = os.environ.get("MODULAR_PUBLIC_MODEL")
    if generic_override:
        return generic_override

    lowered = model.lower()
    if "claude" in lowered:
        return os.environ.get("MODULAR_PUBLIC_ANTHROPIC_MODEL", model)
    if lowered.startswith(("gpt", "o1", "o3")):
        return os.environ.get("MODULAR_PUBLIC_OPENAI_MODEL", model)
    return model


def _proxy_port() -> int:
    return int(os.environ.get("INSPECT_MODEL_PROXY_PORT", str(DEFAULT_PROXY_PORT)))


def _proxy_host() -> str:
    return os.environ.get("INSPECT_MODEL_PROXY_HOST", "127.0.0.1")


def _openai_base_url() -> str:
    configured = os.environ.get("OPENAI_BASE_URL")
    if configured:
        return configured
    if _current_backend() is not None:
        return ""
    return f"http://{_proxy_host()}:{_proxy_port()}/v1"


def _anthropic_base_url() -> str:
    configured = os.environ.get("ANTHROPIC_BASE_URL")
    if configured:
        return configured
    if _current_backend() is not None:
        return ""
    return f"http://{_proxy_host()}:{_proxy_port()}"


def _is_retryable_connection_error(ex: Exception) -> bool:
    names = {
        "APIConnectionError",
        "APIStatusError",
        "ConnectError",
        "ConnectTimeout",
        "ReadError",
        "RemoteProtocolError",
    }
    return ex.__class__.__name__ in names


async def _retry_model_request(request, attempts: int = 6):
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await request()
        except Exception as ex:
            last_error = ex
            if not _is_retryable_connection_error(ex) or attempt == attempts:
                raise
            await asyncio.sleep(min(0.5 * attempt, 2.0))
    if last_error is not None:
        raise last_error
    raise RuntimeError("Model request failed without returning an error")


class Hooks(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    def __init__(self):
        super().__init__()
        self._task_path = _env_path("INSPECT_TASK_FILE", "/home/agent/inspect_task.json")
        self._submission_path = _env_path(
            "INSPECT_SUBMISSION_PATH", "/home/agent/inspect_submission.txt"
        )
        self._state_path = _env_path(
            "INSPECT_STATE_PATH", "/home/agent/inspect_saved_state.json"
        )
        self._score_log_path = _env_path(
            "INSPECT_SCORE_LOG_PATH", "/home/agent/inspect_score_log.jsonl"
        )
        self._event_log_path = _env_path(
            "INSPECT_EVENT_LOG_PATH", "/home/agent/inspect_events.jsonl"
        )
        self._started_at = time.time()
        self._token_usage = 0
        self._token_limit = int(os.environ.get("INSPECT_TOKEN_LIMIT", "10000000"))
        self._time_limit = int(os.environ.get("INSPECT_TIME_LIMIT", str(8 * 60 * 60)))

    def main(self, main_function):
        backend = _current_backend()
        if backend is not None and hasattr(backend, "main"):
            return backend.main(main_function)

        async def runner():
            try:
                await main_function(self)
            except SubmitRequested:
                return 0
            except SystemExit as ex:
                return int(ex.code or 0)
            except Exception as ex:
                self.log("fatal error:", repr(ex))
                raise
            return 0

        raise SystemExit(asyncio.run(runner()))

    def log(self, *content: Any):
        backend = _current_backend()
        if backend is not None and hasattr(backend, "log"):
            backend.log(*content)
            return
        line = " ".join(str(item) for item in content)
        print(line, flush=True)
        _append_jsonl(
            self._event_log_path,
            {"type": "log", "content": line, "timestamp": time.time()},
        )

    def log_with_attributes(self, attributes: dict | None, *content: Any):
        backend = _current_backend()
        if backend is not None and hasattr(backend, "log_with_attributes"):
            backend.log_with_attributes(attributes, *content)
            return
        line = " ".join(str(item) for item in content)
        print(line, flush=True)
        _append_jsonl(
            self._event_log_path,
            {
                "type": "log_with_attributes",
                "content": line,
                "attributes": attributes,
                "timestamp": time.time(),
            },
        )

    def log_image(self, image_url: str, description: str | None = None):
        backend = _current_backend()
        if backend is not None and hasattr(backend, "log_image"):
            backend.log_image(image_url, description)
            return
        _append_jsonl(
            self._event_log_path,
            {
                "type": "image",
                "image_url": image_url,
                "description": description,
                "timestamp": time.time(),
            },
        )

    async def getTask(self) -> TaskInfo:
        backend = _current_backend()
        if backend is not None and hasattr(backend, "getTask"):
            return await backend.getTask()
        return TaskInfo(**_read_json(self._task_path))

    async def get_usage(self) -> UsageInfo:
        backend = _current_backend()
        if backend is not None and hasattr(backend, "get_usage"):
            return await backend.get_usage()
        elapsed = int(time.time() - self._started_at)
        return UsageInfo(
            usage=UsageSnapshot(tokens=self._token_usage, total_seconds=elapsed),
            usageLimits=UsageLimits(
                tokens=self._token_limit,
                total_seconds=self._time_limit,
            ),
        )

    def save_state(self, state: dict[str, Any]):
        backend = _current_backend()
        if backend is not None and hasattr(backend, "save_state"):
            backend.save_state(state)
            return
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    async def action(self, action: dict[str, Any]):
        backend = _current_backend()
        if backend is not None and hasattr(backend, "action"):
            await backend.action(action)
            return
        _append_jsonl(
            self._event_log_path,
            {"type": "action", "action": action, "timestamp": time.time()},
        )

    async def observation(self, observation: dict[str, Any]):
        backend = _current_backend()
        if backend is not None and hasattr(backend, "observation"):
            await backend.observation(observation)
            return
        _append_jsonl(
            self._event_log_path,
            {"type": "observation", "observation": observation, "timestamp": time.time()},
        )

    async def submit(self, submission: str):
        backend = _current_backend()
        if backend is not None and hasattr(backend, "submit"):
            await backend.submit(submission)
            return
        self._submission_path.parent.mkdir(parents=True, exist_ok=True)
        self._submission_path.write_text(submission or "", encoding="utf-8")
        raise SubmitRequested(submission or "")

    async def score(self) -> ScoreResult:
        backend = _current_backend()
        if backend is not None and hasattr(backend, "score"):
            return await backend.score()
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "/home/agent/score.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await proc.communicate()
        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        payload = self._parse_score_output(stdout, stderr)
        result = ScoreResult(
            status="scoringSucceeded" if proc.returncode == 0 else "invalidSubmission",
            score=payload.get("score"),
            message=payload.get("message", {}),
            execResult=ExecResult(
                stdout=stdout,
                stderr=stderr,
                exitStatus=int(proc.returncode or 0),
            ),
        )
        _append_jsonl(
            self._score_log_path,
            {
                "score": result.score,
                "message": result.message,
                "timestamp": time.time(),
            },
        )
        return result

    async def scoreLog(self) -> list[ScoreLogEntry]:
        backend = _current_backend()
        if backend is not None and hasattr(backend, "scoreLog"):
            return await backend.scoreLog()
        if not self._score_log_path.exists():
            return []
        entries: list[ScoreLogEntry] = []
        for line in self._score_log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                entries.append(ScoreLogEntry(**json.loads(line)))
        return entries

    async def generate(
        self,
        settings: MiddlemanSettings,
        template: str | None = None,
        templateValues: dict[str, Any] | None = None,
        prompt: str | None = None,
        messages: list[OpenaiChatMessage] | None = None,
        description: str | None = None,
        functions: Any = None,
        extraParameters: dict[str, Any] | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> MiddlemanResult:
        backend = _current_backend()
        if backend is not None and hasattr(backend, "generate"):
            return await backend.generate(
                settings=settings,
                template=template,
                templateValues=templateValues,
                prompt=prompt,
                messages=messages,
                description=description,
                functions=functions,
                extraParameters=extraParameters,
                session=session,
            )
        del template, templateValues, prompt, description, extraParameters
        if messages is None:
            raise ValueError("messages are required")

        normalized_model = _normalize_model_name(settings.model)
        if backend is not None and hasattr(backend, "model_request"):
            backend.model_request(
                {
                    "model": normalized_model,
                    "settings": settings.model_dump(),
                    "messages": [_message_payload(message) for message in messages],
                    "functions": functions or [],
                }
            )
        try:
            if "anthropic" in normalized_model:
                result = await self._generate_anthropic(
                    model=normalized_model,
                    settings=settings,
                    messages=messages,
                    session=session,
                )
            else:
                result = await self._generate_openai(
                    model=normalized_model,
                    settings=settings,
                    messages=messages,
                    functions=functions,
                    session=session,
                )
        except BaseException as error:
            if backend is not None and hasattr(backend, "model_error"):
                backend.model_error(error)
            raise
        if backend is not None and hasattr(backend, "model_response"):
            backend.model_response(result.model_dump(exclude_none=True))

        usage = result.usage or {}
        self._token_usage += int(usage.get("total_tokens", 0))
        return result

    async def generate_one(
        self,
        settings: MiddlemanSettings,
        messages: list[OpenaiChatMessage],
        **kwargs,
    ) -> str:
        result = await self.generate(settings=settings, messages=messages, **kwargs)
        if result.outputs:
            return result.outputs[0].completion
        return ""

    async def rate_options(
        self,
        rating_template: str,
        transcript: str,
        options: list[RatingOption],
        rating_model: str,
    ) -> RatedOption:
        backend = _current_backend()
        if backend is not None and hasattr(backend, "rate_options"):
            return await backend.rate_options(
                rating_template, transcript, options, rating_model
            )
        del rating_template, transcript, rating_model
        if not options:
            raise ValueError("No rating options provided")
        best = max(options, key=lambda option: option.fixedRating or 0.0)
        return RatedOption(action=best.action, rating=best.fixedRating)

    async def _generate_openai(
        self,
        model: str,
        settings: MiddlemanSettings,
        messages: list[OpenaiChatMessage],
        functions: Any = None,
        session: aiohttp.ClientSession | None = None,
    ) -> MiddlemanResult:
        from openai import AsyncOpenAI

        client_kwargs: dict[str, Any] = {"api_key": os.environ.get("OPENAI_API_KEY", "inspect")}
        base_url = _openai_base_url()
        if base_url:
            client_kwargs["base_url"] = base_url
        client = AsyncOpenAI(**client_kwargs)
        payload: dict[str, Any] = {
            "model": model,
            "messages": _bridge_compatible_openai_messages(messages),
            "n": settings.n,
        }
        if model.split("/")[-1].startswith(("o1", "o3")):
            payload["max_completion_tokens"] = settings.max_tokens
        else:
            payload["max_tokens"] = settings.max_tokens
            payload["temperature"] = settings.temp
        if functions:
            payload["tools"] = [{"type": "function", "function": fn} for fn in functions]
        response = await _retry_model_request(
            lambda: client.chat.completions.create(**payload)
        )

        outputs: list[MiddlemanOutput] = []
        for choice in response.choices:
            message = choice.message
            function_call = None
            if message.tool_calls:
                tool = message.tool_calls[0]
                function_call = {
                    "name": tool.function.name,
                    "arguments": tool.function.arguments or "{}",
                }
            outputs.append(
                MiddlemanOutput(
                    completion=message.content or "",
                    function_call=function_call,
                )
            )

        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }
        return MiddlemanResult(outputs=outputs, model=model, usage=usage)

    async def _generate_anthropic(
        self,
        model: str,
        settings: MiddlemanSettings,
        messages: list[OpenaiChatMessage],
        session: aiohttp.ClientSession | None = None,
    ) -> MiddlemanResult:
        from anthropic import AsyncAnthropic

        client_kwargs: dict[str, Any] = {
            "api_key": os.environ.get("ANTHROPIC_API_KEY", "inspect")
        }
        base_url = _anthropic_base_url()
        if base_url:
            client_kwargs["base_url"] = base_url
        client = AsyncAnthropic(**client_kwargs)
        converted_messages: list[dict[str, Any]] = []
        system_prompt: str | None = None
        for message in messages:
            message = OpenaiChatMessage.model_validate(message)
            if message.role == "system":
                system_prompt = str(message.content)
            else:
                converted_messages.append(
                    {
                        "role": "assistant" if message.role == "assistant" else "user",
                        "content": str(message.content),
                    }
                )
        payload: dict[str, Any] = {
            "model": model,
            "messages": converted_messages,
            "max_tokens": settings.max_tokens,
            "temperature": settings.temp,
        }
        if settings.stop:
            payload["stop_sequences"] = list(settings.stop)
        if system_prompt:
            payload["system"] = system_prompt
        response = await _retry_model_request(lambda: client.messages.create(**payload))

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        usage = {
            "input_tokens": response.usage.input_tokens if response.usage else 0,
            "output_tokens": response.usage.output_tokens if response.usage else 0,
            "total_tokens": (
                (response.usage.input_tokens if response.usage else 0)
                + (response.usage.output_tokens if response.usage else 0)
            ),
        }
        return MiddlemanResult(
            outputs=[MiddlemanOutput(completion=text)],
            model=model,
            usage=usage,
        )

    def _parse_score_output(self, stdout: str, stderr: str) -> dict[str, Any]:
        combined = "\n".join(part for part in [stderr, stdout] if part).splitlines()
        for line in reversed(combined):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return {
            "score": 0.0,
            "message": {
                "error": "No JSON score payload found",
                "stdout": stdout[-1000:],
                "stderr": stderr[-1000:],
            },
        }

    def load_settings(self, path: str | Path) -> dict[str, Any]:
        backend = _current_backend()
        if backend is not None and hasattr(backend, "load_settings"):
            return backend.load_settings(path)
        return _read_json(Path(path))

    def sync_messages(self, state: Any) -> None:
        backend = _current_backend()
        if backend is not None and hasattr(backend, "sync_messages"):
            backend.sync_messages(state)


class Actions:
    async def run_bash(self, script: str, timeout: float) -> str:
        backend = _current_backend()
        if backend is not None and hasattr(backend, "run_bash"):
            return await backend.run_bash(script, timeout)
        proc = await asyncio.create_subprocess_shell(
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            status = proc.returncode or 0
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            stdout_b = b""
            stderr_b = f"Command timed out after {timeout} seconds".encode("utf-8")
            status = 124
        payload = {
            "stdout": stdout_b.decode("utf-8", errors="replace"),
            "stderr": stderr_b.decode("utf-8", errors="replace"),
            "status": status,
        }
        return json.dumps(payload)

    async def run_python(self, script: str, timeout: float) -> str:
        backend = _current_backend()
        if backend is not None and hasattr(backend, "run_python"):
            return await backend.run_python(script, timeout)
        temp_path = Path("/tmp") / f"modular_public_{int(time.time() * 1_000_000)}.py"
        temp_path.write_text(script, encoding="utf-8")
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(temp_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            stdout_b = b""
            stderr_b = f"PythonExecTimeoutException: timed out after {timeout} seconds".encode(
                "utf-8"
            )
        finally:
            temp_path.unlink(missing_ok=True)
        output = stdout_b.decode("utf-8", errors="replace")
        errors = stderr_b.decode("utf-8", errors="replace")
        if errors:
            return f"{output}\n{errors}".strip()
        return output
