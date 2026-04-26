from __future__ import annotations

from PySide6.QtCore import Qt


def is_push_to_talk_press(
    *,
    key: int,
    modifiers: Qt.KeyboardModifiers,
    auto_repeat: bool,
) -> bool:
    return (
        key == Qt.Key.Key_Space
        and modifiers == Qt.KeyboardModifier.NoModifier
        and not auto_repeat
    )


def is_push_to_talk_release(*, key: int, auto_repeat: bool) -> bool:
    return key == Qt.Key.Key_Space and not auto_repeat
