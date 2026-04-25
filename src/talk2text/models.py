from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class InputDevice:
    device_id: str
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
class LiveTranscriptionUpdate:
    session_id: int
    text: str
    detected_language: str | None
    duration_seconds: float


@dataclass(slots=True)
class CleanupUpdate:
    cleanup: TranscriptCleanup
    model_name: str
    elapsed_seconds: float


@dataclass(slots=True)
class TranscriptionResult:
    raw_text: str
    cleaned_text: str
    summary: str
    action_items: list[str]
    detected_language: str | None
    notes: list[str]
    audio_path: Path | None
    duration_seconds: float


@dataclass(slots=True)
class HistoryEntry:
    created_at: str
    result: TranscriptionResult

    @property
    def display_text(self) -> str:
        return self.result.cleaned_text or self.result.raw_text
