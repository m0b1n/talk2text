from pathlib import Path
import threading
import unittest

from talk2text.errors import ProcessingCancelledError
from talk2text.models import RecordedAudio, TranscriptCleanup, TranscriptionOutput
from talk2text.pipeline import Talk2TextPipeline


class FakeTranscriber:
    def __init__(self, output: TranscriptionOutput):
        self.output = output
        self.model_name = None

    def set_model_name(self, model_name: str) -> None:
        self.model_name = model_name

    def transcribe(self, audio_path, language=None, cancel_requested=None):
        if cancel_requested is not None and cancel_requested():
            raise ProcessingCancelledError()
        return self.output


class FakeOllamaClient:
    def __init__(self, should_cancel: bool = False):
        self.should_cancel = should_cancel

    def cleanup_transcript(self, raw_text, model_name, language_hint=None, cancel_requested=None):
        if self.should_cancel:
            raise ProcessingCancelledError()
        return TranscriptCleanup(
            cleaned_text=raw_text,
            summary="",
            action_items=[],
        )


class Talk2TextPipelineTests(unittest.TestCase):
    def test_process_raises_when_cancelled_before_transcription(self) -> None:
        pipeline = Talk2TextPipeline(
            transcriber=FakeTranscriber(TranscriptionOutput(raw_text="hello", detected_language="en")),
            ollama_client=FakeOllamaClient(),
        )
        audio = RecordedAudio(path=Path("/tmp/fake.wav"), sample_rate=16000, duration_seconds=1.0)
        cancelled = threading.Event()
        cancelled.set()

        with self.assertRaises(ProcessingCancelledError):
            pipeline.process(
                recorded_audio=audio,
                whisper_model="turbo",
                use_ollama=False,
                ollama_model="qwen3:8b",
                language=None,
                cancel_requested=cancelled.is_set,
            )

    def test_process_propagates_cancelled_ollama_cleanup(self) -> None:
        pipeline = Talk2TextPipeline(
            transcriber=FakeTranscriber(TranscriptionOutput(raw_text="hello", detected_language="en")),
            ollama_client=FakeOllamaClient(should_cancel=True),
        )
        audio = RecordedAudio(path=Path("/tmp/fake.wav"), sample_rate=16000, duration_seconds=1.0)

        with self.assertRaises(ProcessingCancelledError):
            pipeline.process(
                recorded_audio=audio,
                whisper_model="turbo",
                use_ollama=True,
                ollama_model="qwen3:8b",
                language=None,
            )


if __name__ == "__main__":
    unittest.main()
