from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .config import AppConfig

APP_STYLESHEET = """
QMainWindow {
    background: #f5f1ea;
}
QWidget {
    color: #30261d;
    font-size: 11px;
}
QFrame#transcriptCard, QFrame#panelCard {
    background: #fffdf8;
    border: 1px solid #e5ddd1;
    border-radius: 16px;
}
QLabel#statusPill {
    background: #ece4d8;
    border-radius: 10px;
    color: #5f4f41;
    padding: 4px 8px;
    font-size: 10px;
}
QPushButton#recordButton {
    background: #d84c4c;
    border: 4px solid #f7d7d7;
    border-radius: 42px;
    color: white;
    font-size: 13px;
    font-weight: 700;
}
QPushButton#recordButton[mode="recording"] {
    background: #8f2424;
    border-color: #f3b9b9;
}
QPushButton#recordButton[mode="working"] {
    background: #d9a6a6;
    border-color: #f3d8d8;
    color: #fff8f8;
}
QPushButton#recordButton[mode="cancel"] {
    background: #db8c4a;
    border-color: #f0d1b6;
    color: #fff9f3;
}
QPushButton#copyButton, QPushButton#polishButton, QPushButton#applyButton, QPushButton#clearHistoryButton {
    background: #d8efe5;
    border: none;
    border-radius: 12px;
    color: #245843;
    font-size: 10px;
    font-weight: 600;
    padding: 6px 10px;
}
QPushButton#polishButton {
    background: #f2e8d6;
    color: #7a5233;
}
QPushButton#clearHistoryButton {
    background: #f2e6da;
    color: #7a5233;
}
QPushButton#refreshButton {
    background: #efe7dc;
    border: none;
    border-radius: 10px;
    color: #5f4f41;
    font-size: 10px;
    padding: 5px 8px;
}
QComboBox, QLineEdit {
    background: white;
    border: 1px solid #ded4c8;
    border-radius: 10px;
    padding: 6px 8px;
    min-height: 16px;
}
QCheckBox {
    spacing: 8px;
}
QListWidget {
    background: white;
    border: 1px solid #ded4c8;
    border-radius: 12px;
    padding: 4px;
}
QListWidget::item {
    border-radius: 10px;
    padding: 6px;
    margin: 2px 0px;
}
QListWidget::item:selected {
    background: #ece4d8;
    color: #30261d;
}
QToolButton {
    background: transparent;
    border: none;
    border-radius: 12px;
    padding: 4px;
    min-width: 24px;
    min-height: 24px;
}
QToolButton:hover {
    background: #ece4d8;
}
"""


def build_ui(window: Any, config: AppConfig, whisper_models: Sequence[str]) -> None:
    window.setStyleSheet(APP_STYLESHEET)

    root = QWidget(window)
    window.setCentralWidget(root)

    outer = QVBoxLayout(root)
    outer.setContentsMargins(0, 0, 0, 0)

    window.stack = QStackedWidget()
    outer.addWidget(window.stack)

    window.main_page = _build_main_page(window)
    window.config_page = _build_config_page(window, config, whisper_models)
    window.history_page = _build_history_page(window)

    window.stack.addWidget(window.main_page)
    window.stack.addWidget(window.config_page)
    window.stack.addWidget(window.history_page)
    window.stack.setCurrentWidget(window.main_page)


def _build_main_page(window: Any) -> QWidget:
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(8)

    header = QHBoxLayout()
    header.setSpacing(4)
    window.config_button = _make_icon_button(
        nav_buttons=window._nav_buttons,
        theme_name="settings-configure",
        fallback_text="",
        tooltip="Settings",
        handler=window._show_config_page,
        fallback_icon=_gear_icon(),
    )
    window.history_button = _make_icon_button(
        nav_buttons=window._nav_buttons,
        theme_name="document-open-recent",
        fallback_text="H",
        tooltip="History",
        handler=window._show_history_page,
    )
    title = QLabel("Talk2Text")
    title.setAlignment(Qt.AlignmentFlag.AlignCenter)
    title.setStyleSheet("font-size: 12px; font-weight: 700;")
    header.addWidget(window.config_button)
    header.addWidget(title, 1)
    header.addWidget(window.history_button)
    layout.addLayout(header)

    window.main_status_label = QLabel()
    window.main_status_label.setObjectName("statusPill")
    window.main_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(window.main_status_label)

    window.recording_clock = QLabel("00:00")
    window.recording_clock.setAlignment(Qt.AlignmentFlag.AlignCenter)
    window.recording_clock.setStyleSheet("font-size: 10px; color: #8b7560;")
    layout.addWidget(window.recording_clock)

    layout.addStretch(1)

    window.record_button = QPushButton("REC")
    window.record_button.setObjectName("recordButton")
    window.record_button.setFixedSize(84, 84)
    window.record_button.clicked.connect(window._toggle_recording)
    layout.addWidget(window.record_button, 0, Qt.AlignmentFlag.AlignHCenter)

    window.prompt_label = QLabel("Tap once to start")
    window.prompt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    window.prompt_label.setStyleSheet("font-size: 10px; color: #6a5848;")
    layout.addWidget(window.prompt_label)

    transcript_card = QFrame()
    transcript_card.setObjectName("transcriptCard")
    transcript_layout = QVBoxLayout(transcript_card)
    transcript_layout.setContentsMargins(10, 10, 10, 10)
    transcript_layout.setSpacing(6)

    transcript_title = QLabel("Transcript")
    transcript_title.setStyleSheet("font-size: 10px; font-weight: 700; color: #6a5848;")
    transcript_layout.addWidget(transcript_title)

    transcript_scroll = QScrollArea()
    transcript_scroll.setWidgetResizable(True)
    transcript_scroll.setFrameShape(QFrame.Shape.NoFrame)
    transcript_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    transcript_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    transcript_content = QWidget()
    transcript_content_layout = QVBoxLayout(transcript_content)
    transcript_content_layout.setContentsMargins(0, 0, 0, 0)
    window.transcript_label = QLabel()
    window.transcript_label.setWordWrap(True)
    window.transcript_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    window.transcript_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    window.transcript_label.setStyleSheet("font-size: 11px;")
    transcript_content_layout.addWidget(window.transcript_label)
    transcript_content_layout.addStretch(1)
    transcript_scroll.setWidget(transcript_content)
    transcript_layout.addWidget(transcript_scroll, 1)

    layout.addWidget(transcript_card, 1)

    copy_row = QHBoxLayout()
    copy_row.addStretch(1)
    window.polish_button = QPushButton("Polish")
    window.polish_button.setObjectName("polishButton")
    window.polish_button.clicked.connect(window._polish_transcript)
    copy_row.addWidget(window.polish_button)
    window.copy_button = QPushButton("Copy")
    window.copy_button.setObjectName("copyButton")
    copy_icon = QIcon.fromTheme("edit-copy")
    if not copy_icon.isNull():
        window.copy_button.setIcon(copy_icon)
    window.copy_button.clicked.connect(window._copy_transcript_to_clipboard)
    copy_row.addWidget(window.copy_button)
    layout.addLayout(copy_row)

    return page


def _build_config_page(window: Any, config: AppConfig, whisper_models: Sequence[str]) -> QWidget:
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(8)

    header = QHBoxLayout()
    back_button = _make_icon_button(
        nav_buttons=window._nav_buttons,
        theme_name="go-previous",
        fallback_text="<",
        tooltip="Back",
        handler=window._show_main_page,
    )
    header.addWidget(back_button)
    config_title = QLabel("Settings")
    config_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
    config_title.setStyleSheet("font-size: 12px; font-weight: 700;")
    header.addWidget(config_title, 1)
    header.addSpacing(24)
    layout.addLayout(header)

    window.config_status_label = QLabel()
    window.config_status_label.setObjectName("statusPill")
    window.config_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(window.config_status_label)

    card = QFrame()
    card.setObjectName("panelCard")
    card_layout = QVBoxLayout(card)
    card_layout.setContentsMargins(10, 10, 10, 10)
    card_layout.setSpacing(8)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    content = QWidget()
    content_layout = QVBoxLayout(content)
    content_layout.setContentsMargins(0, 0, 0, 0)
    content_layout.setSpacing(8)

    window.device_combo = QComboBox()
    window.refresh_devices_button = QPushButton("Refresh Devices")
    window.refresh_devices_button.setObjectName("refreshButton")
    window.refresh_devices_button.clicked.connect(window._load_input_devices)
    _add_config_field(content_layout, "Input Device", window.device_combo, window.refresh_devices_button)

    window.whisper_model_combo = QComboBox()
    window.whisper_model_combo.setEditable(True)
    window.whisper_model_combo.addItems(list(whisper_models))
    if config.whisper_model not in whisper_models:
        window.whisper_model_combo.addItem(config.whisper_model)
    window.whisper_model_combo.setCurrentText(config.whisper_model)
    _add_config_field(content_layout, "Whisper Model", window.whisper_model_combo)

    window.language_input = QLineEdit(config.language or "")
    window.language_input.setPlaceholderText("Auto-detect")
    _add_config_field(content_layout, "Language Hint", window.language_input)

    window.ollama_model_combo = QComboBox()
    window.ollama_model_combo.setEditable(True)
    window.ollama_model_combo.setCurrentText(config.ollama_model)
    window.refresh_ollama_button = QPushButton("Refresh Models")
    window.refresh_ollama_button.setObjectName("refreshButton")
    window.refresh_ollama_button.clicked.connect(window._load_ollama_models)
    _add_config_field(
        content_layout,
        "Ollama Model",
        window.ollama_model_combo,
        window.refresh_ollama_button,
    )

    window.enhance_checkbox = QCheckBox("Enable manual Ollama polish")
    window.enhance_checkbox.setChecked(config.enhance_with_ollama)
    window.enhance_checkbox.toggled.connect(window._update_action_buttons)
    content_layout.addWidget(window.enhance_checkbox)

    window.live_transcription_checkbox = QCheckBox("Enable live transcription")
    window.live_transcription_checkbox.setChecked(config.live_transcription)
    content_layout.addWidget(window.live_transcription_checkbox)

    content_layout.addStretch(1)
    scroll.setWidget(content)
    card_layout.addWidget(scroll)
    layout.addWidget(card, 1)

    window.apply_button = QPushButton("Apply")
    window.apply_button.setObjectName("applyButton")
    window.apply_button.clicked.connect(window._apply_settings)
    layout.addWidget(window.apply_button)

    return page


def _build_history_page(window: Any) -> QWidget:
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(8)

    header = QHBoxLayout()
    back_button = _make_icon_button(
        nav_buttons=window._nav_buttons,
        theme_name="go-previous",
        fallback_text="<",
        tooltip="Back",
        handler=window._show_main_page,
    )
    header.addWidget(back_button)
    history_title = QLabel("History")
    history_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
    history_title.setStyleSheet("font-size: 12px; font-weight: 700;")
    header.addWidget(history_title, 1)
    window.clear_history_button = QPushButton("Clear")
    window.clear_history_button.setObjectName("clearHistoryButton")
    window.clear_history_button.clicked.connect(window._clear_history)
    header.addWidget(window.clear_history_button)
    layout.addLayout(header)

    window.history_status_label = QLabel("Tap an item to load it on the main page.")
    window.history_status_label.setObjectName("statusPill")
    window.history_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(window.history_status_label)

    window.history_list = QListWidget()
    window.history_list.itemClicked.connect(window._open_history_item)
    window.history_list.itemActivated.connect(window._open_history_item)
    layout.addWidget(window.history_list, 1)

    return page


def _add_config_field(
    parent_layout: QVBoxLayout,
    label_text: str,
    widget: QWidget,
    extra_button: QPushButton | None = None,
) -> None:
    label = QLabel(label_text)
    label.setStyleSheet("font-size: 10px; font-weight: 700; color: #6a5848;")
    parent_layout.addWidget(label)
    parent_layout.addWidget(widget)
    if extra_button is not None:
        parent_layout.addWidget(extra_button, 0, Qt.AlignmentFlag.AlignRight)


def _make_icon_button(
    nav_buttons: list[QToolButton],
    theme_name: str,
    fallback_text: str,
    tooltip: str,
    handler,
    fallback_icon: QIcon | None = None,
) -> QToolButton:
    button = QToolButton()
    button.setToolTip(tooltip)
    icon = QIcon.fromTheme(theme_name)
    if not icon.isNull():
        button.setIcon(icon)
    elif fallback_icon is not None:
        button.setIcon(fallback_icon)
    else:
        button.setText(fallback_text)
    button.clicked.connect(handler)
    nav_buttons.append(button)
    return button


def _gear_icon() -> QIcon:
    size = 18
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.translate(size / 2, size / 2)

    tooth_pen = QPen(QColor("#6a5848"))
    tooth_pen.setWidthF(1.6)
    tooth_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(tooth_pen)

    for angle in range(0, 360, 45):
        painter.save()
        painter.rotate(angle)
        painter.drawLine(0, -7.5, 0, -5.0)
        painter.restore()

    painter.setPen(QPen(QColor("#6a5848"), 1.6))
    painter.setBrush(QColor("#ece4d8"))
    painter.drawEllipse(-4.8, -4.8, 9.6, 9.6)
    painter.setBrush(QColor("#f5f1ea"))
    painter.drawEllipse(-1.7, -1.7, 3.4, 3.4)
    painter.end()

    return QIcon(pixmap)
