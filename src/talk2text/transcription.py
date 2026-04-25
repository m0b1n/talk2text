from __future__ import annotations

from pathlib import Path
import threading

from .models import TranscriptionOutput


class FasterWhisperTranscriber:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None
        self._backend_note = ""
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def set_model_name(self, model_name: str) -> None:
        if model_name == self._model_name:
            return

        with self._lock:
            self._model_name = model_name
            self._model = None
            self._backend_note = ""

    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionOutput:
        model = self._ensure_model()
        kwargs = {
            "beam_size": 1,
            "condition_on_previous_text": False,
            "vad_filter": True,
        }
        if language:
            kwargs["language"] = language

        segments, info = model.transcribe(str(audio_path), **kwargs)
        segment_list = list(segments)
        raw_text = " ".join(segment.text.strip() for segment in segment_list if segment.text.strip()).strip()

        notes = [self._backend_note]
        if getattr(info, "language_probability", None) is not None and getattr(info, "language", None):
            notes.append(
                f"Detected language {info.language} with confidence {info.language_probability:.2f}"
            )

        return TranscriptionOutput(
            raw_text=raw_text,
            detected_language=getattr(info, "language", None),
            notes=notes,
        )

    def _ensure_model(self):
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            from faster_whisper import WhisperModel

            errors: list[str] = []
            for device, compute_type in (("cuda", "float16"), ("cpu", "int8")):
                try:
                    self._model = WhisperModel(
                        self._model_name,
                        device=device,
                        compute_type=compute_type,
                    )
                    self._backend_note = f"Loaded {self._model_name} on {device} ({compute_type})"
                    return self._model
                except Exception as exc:  # pragma: no cover - hardware dependent
                    errors.append(f"{device}/{compute_type}: {exc}")

        joined = "; ".join(errors)
        raise RuntimeError(f"Failed to initialize faster-whisper model {self._model_name}: {joined}")
