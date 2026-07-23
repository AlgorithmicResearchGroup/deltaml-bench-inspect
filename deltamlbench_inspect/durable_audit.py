from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


def _json_default(value: object) -> object:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _safe_component(value: str) -> str:
    sanitized = _SAFE_COMPONENT.sub("-", value).strip("-.")
    return sanitized[:160] or "unknown"


class DurableAuditLog:
    """Failure-independent, append-only audit artifacts for one agent sample."""

    def __init__(
        self,
        *,
        root: Path,
        run_id: str,
        metadata: dict[str, Any],
        inline_limit: int = 64 * 1024,
    ) -> None:
        self.run_id = _safe_component(run_id)
        self.run_dir = root.expanduser().resolve() / self.run_id
        self.events_path = self.run_dir / "events.jsonl"
        self.payload_dir = self.run_dir / "payloads"
        self.state_dir = self.run_dir / "state"
        self.workspace_dir = self.run_dir / "workspace"
        self.blob_dir = self.workspace_dir / "blobs"
        self.manifest_dir = self.workspace_dir / "manifests"
        self.inline_limit = inline_limit
        self._sequence = 0

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.payload_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_json(
            self.run_dir / "run.json",
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "created_at": time.time(),
                "metadata": metadata,
            },
        )
        self.append("run_started", metadata)

    def _atomic_bytes(self, path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def _atomic_json(self, path: Path, value: object) -> None:
        self._atomic_bytes(path, _json_bytes(value) + b"\n")

    def append(self, event_type: str, payload: object | None = None) -> dict[str, Any]:
        self._sequence += 1
        event: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "sequence": self._sequence,
            "timestamp": time.time(),
            "type": event_type,
        }
        if payload is not None:
            payload_bytes = _json_bytes(payload)
            if len(payload_bytes) > self.inline_limit:
                digest = hashlib.sha256(payload_bytes).hexdigest()
                filename = f"{self._sequence:08d}-{_safe_component(event_type)}.json.gz"
                compressed = gzip.compress(payload_bytes, compresslevel=6)
                self._atomic_bytes(self.payload_dir / filename, compressed)
                event["payload_ref"] = f"payloads/{filename}"
                event["payload_sha256"] = digest
                event["payload_size"] = len(payload_bytes)
            else:
                event["payload"] = json.loads(payload_bytes)

        encoded = _json_bytes(event) + b"\n"
        with self.events_path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return event

    def checkpoint_state(self, state: object) -> dict[str, Any]:
        encoded = _json_bytes(state)
        digest = hashlib.sha256(encoded).hexdigest()
        compressed = gzip.compress(encoded, compresslevel=6)
        self._atomic_bytes(self.state_dir / "latest.json.gz", compressed)
        checkpoint = {
            "sha256": digest,
            "size": len(encoded),
            "path": "state/latest.json.gz",
        }
        self._atomic_json(self.state_dir / "latest.meta.json", checkpoint)
        self.append("state_checkpoint", checkpoint)
        return checkpoint

    def store_workspace_blob(self, content: bytes) -> str:
        digest = hashlib.sha256(content).hexdigest()
        relative = Path("workspace") / "blobs" / digest[:2] / digest
        destination = self.run_dir / relative
        if not destination.exists():
            self._atomic_bytes(destination, content)
        return relative.as_posix()

    def checkpoint_workspace(self, manifest: dict[str, Any]) -> Path:
        sequence = self._sequence + 1
        relative = Path("workspace") / "manifests" / f"{sequence:08d}.json"
        self._atomic_json(self.run_dir / relative, manifest)
        self._atomic_json(self.workspace_dir / "latest.json", manifest)
        self.append(
            "workspace_checkpoint",
            {
                "path": relative.as_posix(),
                "files": len(manifest.get("files", {})),
                "captured_bytes": manifest.get("captured_bytes", 0),
                "skipped_files": manifest.get("skipped_files", 0),
            },
        )
        return self.run_dir / relative

    def finish(self, status: str, details: object | None = None) -> None:
        payload = {"status": status}
        if details is not None:
            payload["details"] = details
        self.append("run_finished", payload)
        self._atomic_json(
            self.run_dir / "status.json",
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "status": status,
                "finished_at": time.time(),
                "details": details,
            },
        )
