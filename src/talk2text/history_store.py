from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .models import HistoryEntry, TranscriptionResult

APP_DIR_NAME = "talk2text"
HISTORY_FILE_NAME = "history.json"


def load_history_entries() -> list[HistoryEntry]:
    path = history_file_path()
    if not path.exists():
        return []

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not isinstance(raw, list):
        return []

    entries: list[HistoryEntry] = []
    for item in raw:
        if isinstance(item, dict):
            entries.append(_history_entry_from_dict(item))
    return entries


def save_history_entries(entries: list[HistoryEntry]) -> None:
    path = history_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = [_history_entry_to_dict(entry) for entry in entries]
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def history_file_path() -> Path:
    return _app_state_dir() / HISTORY_FILE_NAME


def _app_state_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / APP_DIR_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME

    xdg_state_home = os.getenv("XDG_STATE_HOME")
    if xdg_state_home:
        return Path(xdg_state_home) / APP_DIR_NAME
    return Path.home() / ".local" / "state" / APP_DIR_NAME


def _history_entry_to_dict(entry: HistoryEntry) -> dict[str, object]:
    result = entry.result
    return {
        "created_at": entry.created_at,
        "result": {
            "raw_text": result.raw_text,
            "cleaned_text": result.cleaned_text,
            "summary": result.summary,
            "action_items": result.action_items,
            "detected_language": result.detected_language,
            "notes": result.notes,
            "duration_seconds": result.duration_seconds,
        },
    }


def _history_entry_from_dict(data: dict[str, object]) -> HistoryEntry:
    result_data = data.get("result")
    if not isinstance(result_data, dict):
        result_data = {}

    return HistoryEntry(
        created_at=str(data.get("created_at", "")).strip() or "--:--",
        result=TranscriptionResult(
            raw_text=str(result_data.get("raw_text", "")),
            cleaned_text=str(result_data.get("cleaned_text", "")),
            summary=str(result_data.get("summary", "")),
            action_items=_string_list(result_data.get("action_items")),
            detected_language=_optional_string(result_data.get("detected_language")),
            notes=_string_list(result_data.get("notes")),
            audio_path=None,
            duration_seconds=_float_value(result_data.get("duration_seconds")),
        ),
    )


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _optional_string(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _float_value(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
