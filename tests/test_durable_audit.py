from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from deltamlbench_inspect.durable_audit import DurableAuditLog


class DurableAuditLogTests(unittest.TestCase):
    def test_events_survive_without_finish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audit = DurableAuditLog(
                root=Path(directory), run_id="sample/one", metadata={"task": "demo"}
            )
            audit.append("tool_started", {"command": "python train.py"})

            events = [
                json.loads(line)
                for line in audit.events_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(["run_started", "tool_started"], [e["type"] for e in events])
            self.assertEqual("sample-one", audit.run_id)
            self.assertFalse((audit.run_dir / "status.json").exists())

    def test_large_payload_and_state_are_written_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audit = DurableAuditLog(
                root=Path(directory),
                run_id="sample",
                metadata={},
                inline_limit=32,
            )
            event = audit.append("model_request", {"prompt": "x" * 200})
            checkpoint = audit.checkpoint_state({"step": 3, "history": ["ok"]})

            payload_path = audit.run_dir / event["payload_ref"]
            payload = json.loads(gzip.decompress(payload_path.read_bytes()))
            state = json.loads(
                gzip.decompress((audit.run_dir / checkpoint["path"]).read_bytes())
            )
            self.assertEqual("x" * 200, payload["prompt"])
            self.assertEqual(3, state["step"])
            self.assertFalse(any(audit.run_dir.rglob("*.tmp")))

    def test_workspace_blobs_and_terminal_status_persist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audit = DurableAuditLog(
                root=Path(directory), run_id="sample", metadata={}
            )
            blob = audit.store_workspace_blob(b"print('hello')\n")
            manifest = {
                "files": {
                    "solution.py": {
                        "available": True,
                        "blob": blob,
                    }
                },
                "captured_bytes": 15,
                "skipped_files": 0,
            }
            audit.checkpoint_workspace(manifest)
            audit.finish("failed", {"error_type": "RuntimeError"})

            self.assertEqual(b"print('hello')\n", (audit.run_dir / blob).read_bytes())
            self.assertEqual(
                "failed",
                json.loads((audit.run_dir / "status.json").read_text())["status"],
            )
            self.assertEqual(
                manifest,
                json.loads((audit.workspace_dir / "latest.json").read_text()),
            )


if __name__ == "__main__":
    unittest.main()
