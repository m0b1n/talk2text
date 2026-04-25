from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any
from urllib import error, request

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
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def list_models(self) -> list[str]:
        payload = self._request_json("/api/tags", None)
        models = payload.get("models", [])
        return sorted(str(model["name"]) for model in models if "name" in model)

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
        content = self._request_streamed_chat("/api/chat", payload, cancel_requested)
        data = json.loads(content)

        return TranscriptCleanup(
            cleaned_text=str(data.get("cleaned_text", raw_text)).strip() or raw_text,
            summary=str(data.get("summary", "")).strip(),
            action_items=[str(item).strip() for item in data.get("action_items", []) if str(item).strip()],
        )

    def _request_streamed_chat(
        self,
        path: str,
        payload: dict[str, Any],
        cancel_requested: CancelCallback | None = None,
    ) -> str:
        req = request.Request(
            url=f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        content_parts: list[str] = []
        try:
            with request.urlopen(req, timeout=30) as response:
                while True:
                    if cancel_requested is not None and cancel_requested():
                        response.close()
                        raise ProcessingCancelledError()

                    raw_line = response.readline()
                    if not raw_line:
                        break

                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue

                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise RuntimeError(str(chunk["error"]))

                    message = chunk.get("message", {})
                    part = message.get("content", "")
                    if part:
                        content_parts.append(str(part))

                    if chunk.get("done"):
                        break
        except error.URLError as exc:
            raise RuntimeError(f"Failed to reach Ollama at {self.base_url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned invalid streaming JSON.") from exc

        if cancel_requested is not None and cancel_requested():
            raise ProcessingCancelledError()

        return "".join(content_parts)

    def _request_json(self, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        body = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")

        req = request.Request(
            url=f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method="POST" if payload is not None else "GET",
        )

        try:
            with request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(f"Failed to reach Ollama at {self.base_url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned invalid JSON.") from exc
