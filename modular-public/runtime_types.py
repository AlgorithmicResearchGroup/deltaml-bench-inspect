from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ExecResult(BaseModel):
    stdout: str = ""
    stderr: str = ""
    exitStatus: int = 0


class ScoreResult(BaseModel):
    status: str = "scoringSucceeded"
    score: float | None = None
    message: dict[str, Any] = Field(default_factory=dict)
    execResult: ExecResult | None = None


class ScoreLogEntry(BaseModel):
    score: float | None = None
    message: dict[str, Any] = Field(default_factory=dict)
    timestamp: float | None = None


class RatingOption(BaseModel):
    action: str
    fixedRating: float | None = None


class RatedOption(BaseModel):
    action: str
    rating: float | None = None


class ScoringInfo(BaseModel):
    intermediate: bool = False


class TaskInfo(BaseModel):
    instructions: str
    scoring: ScoringInfo = Field(default_factory=ScoringInfo)


class UsageSnapshot(BaseModel):
    tokens: int = 0
    total_seconds: int = 0


class UsageLimits(BaseModel):
    tokens: int = 10_000_000
    total_seconds: int = 8 * 60 * 60


class UsageInfo(BaseModel):
    usage: UsageSnapshot = Field(default_factory=UsageSnapshot)
    usageLimits: UsageLimits = Field(default_factory=UsageLimits)


class OpenaiChatMessage(BaseModel):
    role: str
    content: Any
    name: str | None = None
    function_call: dict[str, Any] | None = None


class MiddlemanSettings(BaseModel):
    n: int = 1
    model: str
    temp: float = 1.0
    max_tokens: int = 4096
    stop: list[str] = Field(default_factory=list)


class MiddlemanOutput(BaseModel):
    completion: str = ""
    function_call: dict[str, Any] | None = None


class MiddlemanResult(BaseModel):
    outputs: list[MiddlemanOutput] | None = None
    model: str = ""
    usage: dict[str, Any] = Field(default_factory=dict)


ToolChoice = Literal["auto", "any", "none"]
