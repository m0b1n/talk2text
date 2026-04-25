from __future__ import annotations

import time
from multiprocessing.connection import Connection
from typing import Any

from .models import CleanupUpdate, LiveTranscriptionUpdate, RecordedAudio
from .ollama_client import OllamaClient
from .pipeline import Talk2TextPipeline
from .transcription import FasterWhisperTranscriber

WorkerMessage = dict[str, Any]


def worker_process_main(connection: Connection, ollama_base_url: str) -> None:
    transcriber = FasterWhisperTranscriber("turbo")
    ollama_client = OllamaClient(ollama_base_url)
    pipeline = Talk2TextPipeline(transcriber, ollama_client)

    try:
        while True:
            try:
                command = connection.recv()
            except EOFError:
                break

            kind = str(command.get("kind", "")).strip()
            if kind == "shutdown":
                break

            request_id = int(command.get("request_id", 0))
            try:
                if kind == "live":
                    message = _process_live_command(command, transcriber)
                elif kind == "pipeline":
                    message = _process_pipeline_command(command, pipeline, connection)
                elif kind == "cleanup":
                    message = _process_cleanup_command(command, ollama_client)
                else:
                    message = {
                        "type": "error",
                        "request_id": request_id,
                        "message": f"Unsupported worker command: {kind or 'unknown'}",
                    }
            except Exception as exc:
                message = {
                    "type": "error",
                    "request_id": request_id,
                    "message": str(exc),
                }

            if not _safe_send(connection, message):
                break
    finally:
        connection.close()


def _process_live_command(
    command: dict[str, Any],
    transcriber: FasterWhisperTranscriber,
) -> WorkerMessage:
    recorded_audio = _as_recorded_audio(command["recorded_audio"])
    whisper_model = str(command["whisper_model"])
    language = command.get("language")
    session_id = int(command["session_id"])
    request_id = int(command["request_id"])

    transcriber.set_model_name(whisper_model)
    transcription = transcriber.transcribe(recorded_audio.path, language=language)
    return {
        "type": "live_result",
        "request_id": request_id,
        "update": LiveTranscriptionUpdate(
            session_id=session_id,
            text=transcription.raw_text,
            detected_language=transcription.detected_language,
            duration_seconds=recorded_audio.duration_seconds,
        ),
    }


def _process_pipeline_command(
    command: dict[str, Any],
    pipeline: Talk2TextPipeline,
    connection: Connection,
) -> WorkerMessage:
    request_id = int(command["request_id"])
    recorded_audio = _as_recorded_audio(command["recorded_audio"])
    whisper_model = str(command["whisper_model"])
    use_ollama = bool(command["use_ollama"])
    ollama_model = str(command["ollama_model"])
    language = command.get("language")

    def report_progress(message: str) -> None:
        _safe_send(
            connection,
            {
                "type": "progress",
                "request_id": request_id,
                "message": message,
            },
        )

    result = pipeline.process(
        recorded_audio=recorded_audio,
        whisper_model=whisper_model,
        use_ollama=use_ollama,
        ollama_model=ollama_model,
        language=language,
        status_callback=report_progress,
    )
    return {
        "type": "pipeline_result",
        "request_id": request_id,
        "result": result,
    }


def _process_cleanup_command(
    command: dict[str, Any],
    ollama_client: OllamaClient,
) -> WorkerMessage:
    request_id = int(command["request_id"])
    raw_text = str(command["raw_text"])
    model_name = str(command["model_name"])
    language_hint = command.get("language_hint")
    started_at = time.perf_counter()
    cleanup = ollama_client.cleanup_transcript(
        raw_text=raw_text,
        model_name=model_name,
        language_hint=language_hint,
    )
    return {
        "type": "cleanup_result",
        "request_id": request_id,
        "update": CleanupUpdate(
            cleanup=cleanup,
            model_name=model_name,
            elapsed_seconds=time.perf_counter() - started_at,
        ),
    }


def _safe_send(connection: Connection, message: WorkerMessage) -> bool:
    try:
        connection.send(message)
    except (BrokenPipeError, EOFError, OSError):
        return False
    return True


def _as_recorded_audio(value: RecordedAudio | dict[str, Any]) -> RecordedAudio:
    if isinstance(value, RecordedAudio):
        return value
    raise TypeError("Worker received an invalid RecordedAudio payload.")
