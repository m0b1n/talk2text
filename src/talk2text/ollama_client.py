from __future__ import annotations

import json
from collections.abc import Callable
from importlib import import_module
from typing import Any

from .errors import ProcessingCancelledError
from .models import TranscriptCleanup

CancelCallback = Callable[[], bool]

TRANSCRIPT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "cleaned_text": {"type": "string"},
        "summary": {"type": "string"},
        "action_items": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["cleaned_text", "summary", "action_items"],
}


def build_cleanup_prompt(raw_text: str, language_hint: str | None = None) -> str:
    hint = language_hint or "auto-detect"
    return (
        "You are cleaning a voice transcript.\n"
        "Preserve the original meaning and language.\n"
        "Fix punctuation, casing, spacing and obvious recognition mistakes.\n"
        "Do not invent missing words or facts.\n"
        "If there are no action items, return an empty array.\n"
        f"Language hint: {hint}\n\n"
        "Transcript:\n"
        f"{raw_text}\n"
    )


class OllamaClient:
    def __init__(self, base_url: str, client: Any | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client

    def list_models(self) -> list[str]:
        try:
            payload = self._ensure_client().list()
        except RuntimeError:
            raise
        except Exception as exc:
            if self._is_response_error(exc):
                raise RuntimeError(
                    f"Ollama request failed: {self._get_mapping_value(exc, 'error') or exc}"
                ) from exc
            raise RuntimeError(f"Failed to reach Ollama at {self.base_url}: {exc}") from exc

        models = self._get_mapping_value(payload, "models") or []
        names = []
        for model in models:
            name = self._get_mapping_value(model, "name")
            if name:
                names.append(str(name))
        return sorted(names)

    def cleanup_transcript(
        self,
        raw_text: str,
        model_name: str,
        language_hint: str | None = None,
        cancel_requested: CancelCallback | None = None,
    ) -> TranscriptCleanup:
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return JSON only. Clean the transcript, summarize it in one sentence, "
                        "and extract explicit tasks or commands."
                    ),
                },
                {
                    "role": "user",
                    "content": build_cleanup_prompt(raw_text, language_hint),
                },
            ],
            "format": TRANSCRIPT_SCHEMA,
            "stream": True,
            "options": {"temperature": 0},
        }
        content = self._request_streamed_chat(payload, cancel_requested)
        data = json.loads(content)

        return TranscriptCleanup(
            cleaned_text=str(data.get("cleaned_text", raw_text)).strip() or raw_text,
            summary=str(data.get("summary", "")).strip(),
            action_items=[
                str(item).strip()
                for item in data.get("action_items", [])
                if str(item).strip()
            ],
        )

    def _request_streamed_chat(
        self,
        payload: dict[str, Any],
        cancel_requested: CancelCallback | None = None,
    ) -> str:
        content_parts: list[str] = []
        stream: Any = None
        try:
            stream = self._ensure_client().chat(**payload)
            for chunk in stream:
                if cancel_requested is not None and cancel_requested():
                    raise ProcessingCancelledError()

                message = self._get_mapping_value(chunk, "message")
                part = self._get_mapping_value(message, "content") if message else ""
                if part:
                    content_parts.append(str(part))

                if self._get_mapping_value(chunk, "done"):
                    break
        except ProcessingCancelledError:
            raise
        except RuntimeError:
            raise
        except Exception as exc:
            if self._is_response_error(exc):
                raise RuntimeError(
                    f"Ollama request failed: {self._get_mapping_value(exc, 'error') or exc}"
                ) from exc
            raise RuntimeError(f"Failed to reach Ollama at {self.base_url}: {exc}") from exc
        finally:
            if stream is not None and hasattr(stream, "close"):
                stream.close()

        if cancel_requested is not None and cancel_requested():
            raise ProcessingCancelledError()

        return "".join(content_parts)

    @staticmethod
    def _get_mapping_value(value: Any, key: str) -> Any:
        if isinstance(value, dict):
            return value.get(key)
        return getattr(value, key, None)

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            ollama_module = import_module("ollama")
        except ImportError as exc:
            raise RuntimeError(
                "The Python package 'ollama' is not installed. Run `uv sync` to install it."
            ) from exc

        self._client = ollama_module.Client(host=self.base_url)
        return self._client

    @staticmethod
    def _is_response_error(exc: Exception) -> bool:
        return (
            exc.__class__.__name__ == "ResponseError"
            and exc.__class__.__module__.split(".", 1)[0] == "ollama"
        )
