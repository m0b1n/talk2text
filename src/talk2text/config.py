from __future__ import annotations

from dataclasses import dataclass
import os

DEFAULT_WHISPER_MODEL = "turbo"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_SAMPLE_RATE = 16000


def _as_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(slots=True)
class AppConfig:
    whisper_model: str = DEFAULT_WHISPER_MODEL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    sample_rate: int = DEFAULT_SAMPLE_RATE
    language: str | None = None
    enhance_with_ollama: bool = True
    audio_device: str | None = None

    @classmethod
    def from_env(cls) -> "AppConfig":
        language = os.getenv("TALK2TEXT_LANGUAGE", "").strip() or None
        audio_device = os.getenv("TALK2TEXT_AUDIO_DEVICE", "").strip() or None
        base_url = os.getenv("TALK2TEXT_OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip()
        sample_rate = int(os.getenv("TALK2TEXT_SAMPLE_RATE", str(DEFAULT_SAMPLE_RATE)))

        return cls(
            whisper_model=os.getenv("TALK2TEXT_WHISPER_MODEL", DEFAULT_WHISPER_MODEL).strip(),
            ollama_model=os.getenv("TALK2TEXT_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip(),
            ollama_base_url=base_url.rstrip("/"),
            sample_rate=sample_rate,
            language=language,
            enhance_with_ollama=_as_bool(os.getenv("TALK2TEXT_ENHANCE_WITH_OLLAMA"), True),
            audio_device=audio_device,
        )

