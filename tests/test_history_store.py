import os
import tempfile
import unittest

from talk2text.history_store import history_file_path, load_history_entries, save_history_entries
from talk2text.models import HistoryEntry, TranscriptionResult


class HistoryStoreTests(unittest.TestCase):
    def test_load_history_returns_empty_when_missing(self) -> None:
        original = os.environ.copy()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                os.environ["XDG_STATE_HOME"] = temp_dir

                self.assertEqual(load_history_entries(), [])
        finally:
            os.environ.clear()
            os.environ.update(original)

    def test_save_and_load_history_round_trip(self) -> None:
        original = os.environ.copy()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                os.environ["XDG_STATE_HOME"] = temp_dir
                entry = HistoryEntry(
                    created_at="04-26 10:30",
                    result=TranscriptionResult(
                        raw_text="raw text",
                        cleaned_text="clean text",
                        summary="summary",
                        action_items=["task"],
                        detected_language="en",
                        notes=["note"],
                        audio_path=None,
                        duration_seconds=1.5,
                    ),
                )

                save_history_entries([entry])
                loaded = load_history_entries()

                self.assertEqual(history_file_path().name, "history.json")
                self.assertEqual(len(loaded), 1)
                self.assertEqual(loaded[0].created_at, "04-26 10:30")
                self.assertEqual(loaded[0].result.cleaned_text, "clean text")
                self.assertEqual(loaded[0].result.action_items, ["task"])
                self.assertEqual(loaded[0].result.detected_language, "en")
        finally:
            os.environ.clear()
            os.environ.update(original)


if __name__ == "__main__":
    unittest.main()
