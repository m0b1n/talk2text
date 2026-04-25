from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class InputDevice:
    device_id: int
    name: str
    is_default: bool = False


@dataclass(slots=True)
class RecordedAudio:
    path: Path
    sample_rate: int
    duration_seconds: float


@dataclass(slots=True)
class TranscriptCleanup:
    cleaned_text: str
    summary: str
    action_items: list[str]


@dataclass(slots=True)
class TranscriptionOutput:
    raw_text: str
    detected_language: str | None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TranscriptionResult:
    raw_text: str
    cleaned_text: str
    summary: str
    action_items: list[str]
    detected_language: str | None
    notes: list[str]
    audio_path: Path
    duration_seconds: float

