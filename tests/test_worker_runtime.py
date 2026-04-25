from pathlib import Path
import unittest

from talk2text.models import RecordedAudio, TranscriptCleanup, TranscriptionOutput, TranscriptionResult
from talk2text.worker_runtime import (
    _process_cleanup_command,
    _process_live_command,
    _process_pipeline_command,
)


class FakeTranscriber:
    def __init__(self) -> None:
        self.model_name = ""

    def set_model_name(self, model_name: str) -> None:
        self.model_name = model_name

    def transcribe(self, audio_path, language=None):
        return TranscriptionOutput(
            raw_text=f"text from {audio_path.name}",
            detected_language=language or "en",
        )


class FakePipeline:
    def process(
        self,
        recorded_audio,
        whisper_model,
        use_ollama,
        ollama_model,
        language,
        status_callback=None,
        cancel_requested=None,
    ):
        if status_callback is not None:
            status_callback("Transcribing...")
        return TranscriptionResult(
            raw_text="raw",
            cleaned_text="clean",
            summary="summary",
            action_items=["task"],
            detected_language=language or "en",
            notes=[f"model={whisper_model}", f"ollama={use_ollama}:{ollama_model}"],
            audio_path=recorded_audio.path,
            duration_seconds=recorded_audio.duration_seconds,
        )


class FakeOllamaClient:
    def cleanup_transcript(self, raw_text, model_name, language_hint=None, cancel_requested=None):
        return TranscriptCleanup(
            cleaned_text=raw_text.upper(),
            summary=f"{model_name}:{language_hint or 'auto'}",
            action_items=["review"],
        )


class FakeConnection:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    def send(self, message: dict[str, object]) -> None:
        self.messages.append(message)


class WorkerRuntimeTests(unittest.TestCase):
    def test_process_live_command_builds_update(self) -> None:
        transcriber = FakeTranscriber()
        audio = RecordedAudio(path=Path("/tmp/live.wav"), sample_rate=16000, duration_seconds=1.5)

        message = _process_live_command(
            {
                "recorded_audio": audio,
                "whisper_model": "turbo",
                "language": "de",
                "session_id": 7,
                "request_id": 11,
            },
            transcriber,
        )

        self.assertEqual(transcriber.model_name, "turbo")
        self.assertEqual(message["type"], "live_result")
        self.assertEqual(message["request_id"], 11)
        self.assertEqual(message["update"].session_id, 7)
        self.assertEqual(message["update"].detected_language, "de")

    def test_process_pipeline_command_reports_progress(self) -> None:
        connection = FakeConnection()
        pipeline = FakePipeline()
        audio = RecordedAudio(path=Path("/tmp/final.wav"), sample_rate=16000, duration_seconds=3.0)

        message = _process_pipeline_command(
            {
                "recorded_audio": audio,
                "whisper_model": "base",
                "use_ollama": False,
                "ollama_model": "qwen3:8b",
                "language": "en",
                "request_id": 5,
            },
            pipeline,
            connection,
        )

        self.assertEqual(connection.messages[0]["type"], "progress")
        self.assertEqual(connection.messages[0]["request_id"], 5)
        self.assertEqual(message["type"], "pipeline_result")
        self.assertEqual(message["result"].audio_path, audio.path)

    def test_process_cleanup_command_wraps_cleanup_update(self) -> None:
        message = _process_cleanup_command(
            {
                "raw_text": "hello there",
                "model_name": "qwen3:8b",
                "language_hint": "en",
                "request_id": 9,
            },
            FakeOllamaClient(),
        )

        self.assertEqual(message["type"], "cleanup_result")
        self.assertEqual(message["request_id"], 9)
        self.assertEqual(message["update"].cleanup.cleaned_text, "HELLO THERE")
        self.assertEqual(message["update"].model_name, "qwen3:8b")


if __name__ == "__main__":
    unittest.main()
