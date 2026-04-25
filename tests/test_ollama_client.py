import unittest

from talk2text.ollama_client import TRANSCRIPT_SCHEMA, build_cleanup_prompt


class OllamaClientHelpersTests(unittest.TestCase):
    def test_cleanup_prompt_includes_transcript_and_hint(self) -> None:
        prompt = build_cleanup_prompt("hello from mic", language_hint="en")

        self.assertIn("hello from mic", prompt)
        self.assertIn("Language hint: en", prompt)

    def test_cleanup_schema_requires_expected_fields(self) -> None:
        required = set(TRANSCRIPT_SCHEMA["required"])
        self.assertEqual(required, {"cleaned_text", "summary", "action_items"})


if __name__ == "__main__":
    unittest.main()
