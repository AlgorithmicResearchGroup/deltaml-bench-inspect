from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import patch

from inspect_ai.agent import agent_bridge
from deltamlbench_inspect.solvers import _load_modular_modules


class ModularRuntimeMessageTests(unittest.TestCase):
    def test_message_payload_accepts_dict_and_model(self) -> None:
        inspect_runtime, _, _, runtime_types = _load_modular_modules()
        expected = {"role": "user", "content": "hello"}

        self.assertEqual(expected, inspect_runtime._message_payload(expected))
        self.assertEqual(
            expected,
            inspect_runtime._message_payload(
                runtime_types.OpenaiChatMessage(role="user", content="hello")
            ),
        )

    def test_legacy_function_messages_are_bridge_compatible(self) -> None:
        inspect_runtime, _, _, _ = _load_modular_modules()
        converted = inspect_runtime._bridge_compatible_openai_messages(
            [
                {"role": "user", "content": "inspect"},
                {
                    "role": "assistant",
                    "content": "",
                    "function_call": {"name": "bash", "arguments": '{"cmd":"pwd"}'},
                },
                {"role": "function", "name": "bash", "content": "/home/agent"},
            ]
        )

        self.assertEqual("tool_calls", next(iter(converted[1].keys() - {"role", "content"})))
        call_id = converted[1]["tool_calls"][0]["id"]
        self.assertEqual("tool", converted[2]["role"])
        self.assertEqual(call_id, converted[2]["tool_call_id"])
        self.assertNotIn("function_call", converted[1])

    def test_legacy_messages_route_through_agent_bridge(self) -> None:
        inspect_runtime, _, _, runtime_types = _load_modular_modules()

        async def exercise_bridge() -> None:
            messages = [
                {"role": "user", "content": "Call bash."},
                {
                    "role": "assistant",
                    "content": "",
                    "function_call": {"name": "bash", "arguments": '{"cmd":"pwd"}'},
                },
                {"role": "function", "name": "bash", "content": "/home/agent"},
                {"role": "user", "content": "Done."},
            ]
            async with agent_bridge():
                result = await inspect_runtime.Hooks()._generate_openai(
                    model="inspect/mockllm/model",
                    settings=runtime_types.MiddlemanSettings(
                        model="gpt-4.1-mini", n=1, max_tokens=32
                    ),
                    messages=messages,
                    functions=[
                        {
                            "name": "bash",
                            "description": "Run bash",
                            "parameters": {
                                "type": "object",
                                "properties": {"cmd": {"type": "string"}},
                                "required": ["cmd"],
                            },
                        }
                    ],
                )
            self.assertEqual("inspect/mockllm/model", result.model)
            self.assertEqual(1, len(result.outputs or []))

        with patch.dict(os.environ, {"OPENAI_API_KEY": "inspect"}):
            asyncio.run(exercise_bridge())


if __name__ == "__main__":
    unittest.main()
