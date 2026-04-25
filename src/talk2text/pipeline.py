from __future__ import annotations

from collections.abc import Callable
import time

from .errors import ProcessingCancelledError
from .models import RecordedAudio, TranscriptionResult
from .ollama_client import OllamaClient
from .transcription import FasterWhisperTranscriber

StatusCallback = Callable[[str], None]
CancelCallback = Callable[[], bool]


class Talk2TextPipeline:
    def __init__(self, transcriber: FasterWhisperTranscriber, ollama_client: OllamaClient) -> None:
        self.transcriber = transcriber
        self.ollama_client = ollama_client

    def process(
        self,
        recorded_audio: RecordedAudio,
        whisper_model: str,
        use_ollama: bool,
        ollama_model: str,
        language: str | None,
        status_callback: StatusCallback | None = None,
        cancel_requested: CancelCallback | None = None,
    ) -> TranscriptionResult:
        def report(message: str) -> None:
            if status_callback is not None:
                status_callback(message)

        def raise_if_cancelled() -> None:
            if cancel_requested is not None and cancel_requested():
                raise ProcessingCancelledError()

        started_at = time.perf_counter()
        raise_if_cancelled()
        self.transcriber.set_model_name(whisper_model)
        report(f"Transcribing with faster-whisper ({whisper_model})...")
        transcription_started_at = time.perf_counter()
        transcription = self.transcriber.transcribe(
            recorded_audio.path,
            language=language,
            cancel_requested=cancel_requested,
        )
        transcription_elapsed = time.perf_counter() - transcription_started_at
        raise_if_cancelled()
        if not transcription.raw_text:
            raise RuntimeError("Whisper returned an empty transcript.")

        cleaned_text = transcription.raw_text
        summary = ""
        action_items: list[str] = []
        notes = list(transcription.notes)
        notes.append(f"Transcription stage: {transcription_elapsed:.2f}s")

        if use_ollama:
            report(f"Enhancing transcript with Ollama ({ollama_model})...")
            ollama_started_at = time.perf_counter()
            try:
                cleanup = self.ollama_client.cleanup_transcript(
                    raw_text=transcription.raw_text,
                    model_name=ollama_model,
                    language_hint=language or transcription.detected_language,
                    cancel_requested=cancel_requested,
                )
                ollama_elapsed = time.perf_counter() - ollama_started_at
                cleaned_text = cleanup.cleaned_text
                summary = cleanup.summary
                action_items = cleanup.action_items
                notes.append(f"Ollama enhancement applied with {ollama_model} in {ollama_elapsed:.2f}s")
            except ProcessingCancelledError:
                raise
            except Exception as exc:
                ollama_elapsed = time.perf_counter() - ollama_started_at
                notes.append(f"Ollama enhancement failed after {ollama_elapsed:.2f}s")
                notes.append(f"Ollama enhancement skipped: {exc}")
        else:
            notes.append("Ollama enhancement disabled.")

        total_elapsed = time.perf_counter() - started_at
        notes.append(f"Total processing time after stop: {total_elapsed:.2f}s")

        return TranscriptionResult(
            raw_text=transcription.raw_text,
            cleaned_text=cleaned_text,
            summary=summary,
            action_items=action_items,
            detected_language=transcription.detected_language,
            notes=notes,
            audio_path=recorded_audio.path,
            duration_seconds=recorded_audio.duration_seconds,
        )
