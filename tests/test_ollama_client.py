import unittest
from types import SimpleNamespace

from talk2text.errors import ProcessingCancelledError
from talk2text.ollama_client import (
    TRANSCRIPT_SCHEMA,
    OllamaClient,
    build_cleanup_prompt,
)


class FakeOllamaSdkClient:
    def __init__(self) -> None:
        self.list_response = {"models": []}
        self.chat_chunks = []

    def list(self):
        return self.list_response

    def chat(self, **payload):
        self.last_chat_payload = payload
        return iter(self.chat_chunks)


class OllamaClientHelpersTests(unittest.TestCase):
    def test_cleanup_prompt_includes_transcript_and_hint(self) -> None:
        prompt = build_cleanup_prompt("hello from mic", language_hint="en")

        self.assertIn("hello from mic", prompt)
        self.assertIn("Language hint: en", prompt)

    def test_cleanup_schema_requires_expected_fields(self) -> None:
        required = set(TRANSCRIPT_SCHEMA["required"])
        self.assertEqual(required, {"cleaned_text", "summary", "action_items"})

    def test_list_models_reads_sdk_response_objects(self) -> None:
        client = OllamaClient("http://localhost:11434")
        client._client = FakeOllamaSdkClient()
        client._client.list_response = SimpleNamespace(
            models=[
                SimpleNamespace(name="qwen3:8b"),
                {"name": "gemma3:4b"},
            ]
        )

        self.assertEqual(client.list_models(), ["gemma3:4b", "qwen3:8b"])

    def test_cleanup_transcript_reads_streamed_sdk_chunks(self) -> None:
        client = OllamaClient("http://localhost:11434")
        client._client = FakeOllamaSdkClient()
        client._client.chat_chunks = [
            {"message": {"content": '{"cleaned_text":"hello'}},
            SimpleNamespace(
                message=SimpleNamespace(
                    content=' world","summary":"short","action_items":["x"]}'
                )
            ),
            {"done": True},
        ]

        cleanup = client.cleanup_transcript("hello world", "qwen3:8b", language_hint="en")

        self.assertEqual(cleanup.cleaned_text, "hello world")
        self.assertEqual(cleanup.summary, "short")
        self.assertEqual(cleanup.action_items, ["x"])
        self.assertEqual(client._client.last_chat_payload["model"], "qwen3:8b")
        self.assertTrue(client._client.last_chat_payload["stream"])

    def test_cleanup_transcript_honors_cancel_request(self) -> None:
        client = OllamaClient("http://localhost:11434")
        client._client = FakeOllamaSdkClient()
        client._client.chat_chunks = [{"message": {"content": '{"cleaned_text":"hello"}'}}]

        with self.assertRaises(ProcessingCancelledError):
            client.cleanup_transcript(
                "hello",
                "qwen3:8b",
                cancel_requested=lambda: True,
            )


if __name__ == "__main__":
    unittest.main()
