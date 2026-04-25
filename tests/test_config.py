import os
import unittest

from talk2text.config import AppConfig


class AppConfigTests(unittest.TestCase):
    def test_config_defaults_from_env(self) -> None:
        original = os.environ.copy()
        try:
            for key in [
                "TALK2TEXT_WHISPER_MODEL",
                "TALK2TEXT_OLLAMA_MODEL",
                "TALK2TEXT_OLLAMA_BASE_URL",
                "TALK2TEXT_LANGUAGE",
                "TALK2TEXT_ENHANCE_WITH_OLLAMA",
                "TALK2TEXT_LIVE_TRANSCRIPTION",
            ]:
                os.environ.pop(key, None)

            config = AppConfig.from_env()

            self.assertEqual(config.whisper_model, "turbo")
            self.assertEqual(config.ollama_model, "qwen3:8b")
            self.assertEqual(config.ollama_base_url, "http://localhost:11434")
            self.assertIsNone(config.language)
            self.assertFalse(config.enhance_with_ollama)
            self.assertFalse(config.live_transcription)
        finally:
            os.environ.clear()
            os.environ.update(original)

    def test_live_transcription_can_be_enabled_from_env(self) -> None:
        original = os.environ.copy()
        try:
            os.environ["TALK2TEXT_LIVE_TRANSCRIPTION"] = "true"

            config = AppConfig.from_env()

            self.assertTrue(config.live_transcription)
        finally:
            os.environ.clear()
            os.environ.update(original)


if __name__ == "__main__":
    unittest.main()
