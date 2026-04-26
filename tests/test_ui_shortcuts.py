import unittest

from PySide6.QtCore import Qt

from talk2text.ui_shortcuts import is_push_to_talk_press, is_push_to_talk_release


class UiShortcutTests(unittest.TestCase):
    def test_space_press_requires_plain_non_repeating_key(self) -> None:
        self.assertTrue(
            is_push_to_talk_press(
                key=Qt.Key.Key_Space,
                modifiers=Qt.KeyboardModifier.NoModifier,
                auto_repeat=False,
            )
        )
        self.assertFalse(
            is_push_to_talk_press(
                key=Qt.Key.Key_Space,
                modifiers=Qt.KeyboardModifier.ShiftModifier,
                auto_repeat=False,
            )
        )
        self.assertFalse(
            is_push_to_talk_press(
                key=Qt.Key.Key_Space,
                modifiers=Qt.KeyboardModifier.NoModifier,
                auto_repeat=True,
            )
        )

    def test_space_release_ignores_other_keys_and_repeats(self) -> None:
        self.assertTrue(is_push_to_talk_release(key=Qt.Key.Key_Space, auto_repeat=False))
        self.assertFalse(is_push_to_talk_release(key=Qt.Key.Key_Return, auto_repeat=False))
        self.assertFalse(is_push_to_talk_release(key=Qt.Key.Key_Space, auto_repeat=True))
