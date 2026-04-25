from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import sys
import threading
import time

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .audio import MicrophoneRecorder, list_input_devices
from .config import AppConfig
from .errors import ProcessingCancelledError
from .models import RecordedAudio, TranscriptionResult
from .ollama_client import OllamaClient
from .pipeline import Talk2TextPipeline
from .transcription import FasterWhisperTranscriber

WHISPER_MODELS = [
    "tiny",
    "base",
    "small",
    "medium",
    "turbo",
    "large-v3",
    "distil-large-v3",
]

LIVE_TRANSCRIPTION_INTERVAL_MS = 900
LIVE_TRANSCRIPTION_MIN_DURATION_SECONDS = 0.6
LIVE_TRANSCRIPTION_WINDOW_SECONDS = 8.0
LIVE_LANGUAGE_LOCK_MIN_DURATION_SECONDS = 2.0
LIVE_LANGUAGE_LOCK_MIN_WORDS = 3


@dataclass(slots=True)
class HistoryEntry:
    created_at: str
    result: TranscriptionResult

    @property
    def display_text(self) -> str:
        return self.result.cleaned_text or self.result.raw_text


@dataclass(slots=True)
class LiveTranscriptionUpdate:
    session_id: int
    text: str
    detected_language: str | None
    duration_seconds: float


class PipelineWorker(QObject):
    finished = Signal(object)
    error = Signal(str)
    cancelled = Signal(str)
    progress = Signal(str)

    def __init__(self, pipeline: Talk2TextPipeline) -> None:
        super().__init__()
        self.pipeline = pipeline
        self._cancel_event = threading.Event()

    @Slot()
    def cancel_current(self) -> None:
        self._cancel_event.set()

    @Slot(object, str, bool, str, object)
    def process_recording(
        self,
        recorded_audio: RecordedAudio,
        whisper_model: str,
        use_ollama: bool,
        ollama_model: str,
        language: str | None,
    ) -> None:
        self._cancel_event.clear()
        try:
            result = self.pipeline.process(
                recorded_audio=recorded_audio,
                whisper_model=whisper_model,
                use_ollama=use_ollama,
                ollama_model=ollama_model,
                language=language,
                status_callback=self.progress.emit,
                cancel_requested=self._cancel_event.is_set,
            )
        except ProcessingCancelledError as exc:
            self.cancelled.emit(str(exc))
            return
        except Exception as exc:
            self.error.emit(str(exc))
            return

        self.finished.emit(result)


class LiveTranscriptionWorker(QObject):
    finished = Signal(object)
    error = Signal(str)
    cancelled = Signal()

    def __init__(self, transcriber: FasterWhisperTranscriber) -> None:
        super().__init__()
        self.transcriber = transcriber
        self._cancel_event = threading.Event()

    @Slot()
    def cancel_current(self) -> None:
        self._cancel_event.set()

    @Slot(object, str, object, int)
    def process_snapshot(
        self,
        recorded_audio: RecordedAudio,
        whisper_model: str,
        language: str | None,
        session_id: int,
    ) -> None:
        self._cancel_event.clear()
        try:
            self.transcriber.set_model_name(whisper_model)
            transcription = self.transcriber.transcribe(
                recorded_audio.path,
                language=language,
                cancel_requested=self._cancel_event.is_set,
            )
        except ProcessingCancelledError:
            self.cancelled.emit()
            return
        except Exception as exc:
            self.error.emit(str(exc))
            return
        finally:
            _unlink_file(recorded_audio.path)

        self.finished.emit(
            LiveTranscriptionUpdate(
                session_id=session_id,
                text=transcription.raw_text,
                detected_language=transcription.detected_language,
                duration_seconds=recorded_audio.duration_seconds,
            )
        )


class MainWindow(QMainWindow):
    request_live_transcription = Signal(object, str, object, int)
    cancel_live_transcription = Signal()
    request_pipeline_processing = Signal(object, str, bool, str, object)
    cancel_pipeline_processing = Signal()

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.recorder = MicrophoneRecorder(sample_rate=config.sample_rate)
        self.transcriber = FasterWhisperTranscriber(config.whisper_model)
        self.live_transcriber = FasterWhisperTranscriber(config.whisper_model)
        self.ollama_client = OllamaClient(config.ollama_base_url)
        self.pipeline = Talk2TextPipeline(self.transcriber, self.ollama_client)

        self._recording_started_at: float | None = None
        self._last_result: TranscriptionResult | None = None
        self._history_entries: list[HistoryEntry] = []
        self._current_transcript_text = ""
        self._is_processing = False
        self._live_session_id = 0
        self._live_request_in_flight = False
        self._pipeline_request_in_flight = False
        self._pending_final_audio: RecordedAudio | None = None
        self._session_language_hint: str | None = None
        self._nav_buttons: list[QToolButton] = []

        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.timeout.connect(self._update_recording_clock)
        self.live_timer = QTimer(self)
        self.live_timer.setInterval(LIVE_TRANSCRIPTION_INTERVAL_MS)
        self.live_timer.timeout.connect(self._queue_live_transcription)

        self._live_thread = QThread(self)
        self._live_worker = LiveTranscriptionWorker(self.live_transcriber)
        self._live_worker.moveToThread(self._live_thread)
        self.request_live_transcription.connect(self._live_worker.process_snapshot)
        self.cancel_live_transcription.connect(self._live_worker.cancel_current)
        self._live_worker.finished.connect(self._handle_live_update)
        self._live_worker.error.connect(self._handle_live_error)
        self._live_worker.cancelled.connect(self._handle_live_cancelled)
        self._live_thread.finished.connect(self._live_worker.deleteLater)
        self._live_thread.start()

        self._pipeline_thread = QThread(self)
        self._pipeline_worker = PipelineWorker(self.pipeline)
        self._pipeline_worker.moveToThread(self._pipeline_thread)
        self.request_pipeline_processing.connect(self._pipeline_worker.process_recording)
        self.cancel_pipeline_processing.connect(self._pipeline_worker.cancel_current)
        self._pipeline_worker.progress.connect(self._set_status)
        self._pipeline_worker.finished.connect(self._handle_result)
        self._pipeline_worker.error.connect(self._handle_error)
        self._pipeline_worker.cancelled.connect(self._handle_cancelled)
        self._pipeline_thread.finished.connect(self._pipeline_worker.deleteLater)
        self._pipeline_thread.start()

        self.setWindowTitle("Talk2Text")
        self.setFixedSize(300, 400)
        self._build_ui()
        self._load_input_devices()
        self._load_ollama_models()
        self._show_placeholder_transcript("Tap the red button to record.")
        self._set_status("Ready")
        self._set_idle_state()

    def _build_ui(self) -> None:
        self.setStyleSheet(
            """
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
            QPushButton#copyButton, QPushButton#applyButton, QPushButton#clearHistoryButton {
                background: #d8efe5;
                border: none;
                border-radius: 12px;
                color: #245843;
                font-size: 10px;
                font-weight: 600;
                padding: 6px 10px;
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
        )

        root = QWidget(self)
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget()
        outer.addWidget(self.stack)

        self.main_page = self._build_main_page()
        self.config_page = self._build_config_page()
        self.history_page = self._build_history_page()

        self.stack.addWidget(self.main_page)
        self.stack.addWidget(self.config_page)
        self.stack.addWidget(self.history_page)
        self.stack.setCurrentWidget(self.main_page)

    def _build_main_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(4)
        self.config_button = self._make_icon_button(
            theme_name="settings-configure",
            fallback_text="",
            tooltip="Settings",
            handler=self._show_config_page,
            fallback_icon=self._gear_icon(),
        )
        self.history_button = self._make_icon_button(
            theme_name="document-open-recent",
            fallback_text="H",
            tooltip="History",
            handler=self._show_history_page,
        )
        title = QLabel("Talk2Text")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 12px; font-weight: 700;")
        header.addWidget(self.config_button)
        header.addWidget(title, 1)
        header.addWidget(self.history_button)
        layout.addLayout(header)

        self.main_status_label = QLabel()
        self.main_status_label.setObjectName("statusPill")
        self.main_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.main_status_label)

        self.recording_clock = QLabel("00:00")
        self.recording_clock.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recording_clock.setStyleSheet("font-size: 10px; color: #8b7560;")
        layout.addWidget(self.recording_clock)

        layout.addStretch(1)

        self.record_button = QPushButton("REC")
        self.record_button.setObjectName("recordButton")
        self.record_button.setFixedSize(84, 84)
        self.record_button.clicked.connect(self._toggle_recording)
        layout.addWidget(self.record_button, 0, Qt.AlignmentFlag.AlignHCenter)

        self.prompt_label = QLabel("Tap once to start")
        self.prompt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prompt_label.setStyleSheet("font-size: 10px; color: #6a5848;")
        layout.addWidget(self.prompt_label)

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
        self.transcript_label = QLabel()
        self.transcript_label.setWordWrap(True)
        self.transcript_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.transcript_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.transcript_label.setStyleSheet("font-size: 11px;")
        transcript_content_layout.addWidget(self.transcript_label)
        transcript_content_layout.addStretch(1)
        transcript_scroll.setWidget(transcript_content)
        transcript_layout.addWidget(transcript_scroll, 1)

        layout.addWidget(transcript_card, 1)

        copy_row = QHBoxLayout()
        copy_row.addStretch(1)
        self.copy_button = QPushButton("Copy")
        self.copy_button.setObjectName("copyButton")
        copy_icon = QIcon.fromTheme("edit-copy")
        if not copy_icon.isNull():
            self.copy_button.setIcon(copy_icon)
        self.copy_button.clicked.connect(self._copy_transcript_to_clipboard)
        copy_row.addWidget(self.copy_button)
        layout.addLayout(copy_row)

        return page

    def _build_config_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        header = QHBoxLayout()
        back_button = self._make_icon_button(
            theme_name="go-previous",
            fallback_text="<",
            tooltip="Back",
            handler=self._show_main_page,
        )
        header.addWidget(back_button)
        config_title = QLabel("Settings")
        config_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        config_title.setStyleSheet("font-size: 12px; font-weight: 700;")
        header.addWidget(config_title, 1)
        header.addSpacing(24)
        layout.addLayout(header)

        self.config_status_label = QLabel()
        self.config_status_label.setObjectName("statusPill")
        self.config_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.config_status_label)

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

        self.device_combo = QComboBox()
        self.refresh_devices_button = QPushButton("Refresh Devices")
        self.refresh_devices_button.setObjectName("refreshButton")
        self.refresh_devices_button.clicked.connect(self._load_input_devices)
        self._add_config_field(content_layout, "Input Device", self.device_combo, self.refresh_devices_button)

        self.whisper_model_combo = QComboBox()
        self.whisper_model_combo.setEditable(True)
        self.whisper_model_combo.addItems(WHISPER_MODELS)
        if self.config.whisper_model not in WHISPER_MODELS:
            self.whisper_model_combo.addItem(self.config.whisper_model)
        self.whisper_model_combo.setCurrentText(self.config.whisper_model)
        self._add_config_field(content_layout, "Whisper Model", self.whisper_model_combo)

        self.language_input = QLineEdit(self.config.language or "")
        self.language_input.setPlaceholderText("Auto-detect")
        self._add_config_field(content_layout, "Language Hint", self.language_input)

        self.ollama_model_combo = QComboBox()
        self.ollama_model_combo.setEditable(True)
        self.ollama_model_combo.setCurrentText(self.config.ollama_model)
        self.refresh_ollama_button = QPushButton("Refresh Models")
        self.refresh_ollama_button.setObjectName("refreshButton")
        self.refresh_ollama_button.clicked.connect(self._load_ollama_models)
        self._add_config_field(content_layout, "Ollama Model", self.ollama_model_combo, self.refresh_ollama_button)

        self.enhance_checkbox = QCheckBox("Enhance transcript with Ollama")
        self.enhance_checkbox.setChecked(self.config.enhance_with_ollama)
        content_layout.addWidget(self.enhance_checkbox)

        self.live_transcription_checkbox = QCheckBox("Enable live transcription")
        self.live_transcription_checkbox.setChecked(self.config.live_transcription)
        content_layout.addWidget(self.live_transcription_checkbox)

        content_layout.addStretch(1)
        scroll.setWidget(content)
        card_layout.addWidget(scroll)
        layout.addWidget(card, 1)

        self.apply_button = QPushButton("Apply")
        self.apply_button.setObjectName("applyButton")
        self.apply_button.clicked.connect(self._apply_settings)
        layout.addWidget(self.apply_button)

        return page

    def _build_history_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        header = QHBoxLayout()
        back_button = self._make_icon_button(
            theme_name="go-previous",
            fallback_text="<",
            tooltip="Back",
            handler=self._show_main_page,
        )
        header.addWidget(back_button)
        history_title = QLabel("History")
        history_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        history_title.setStyleSheet("font-size: 12px; font-weight: 700;")
        header.addWidget(history_title, 1)
        self.clear_history_button = QPushButton("Clear")
        self.clear_history_button.setObjectName("clearHistoryButton")
        self.clear_history_button.clicked.connect(self._clear_history)
        header.addWidget(self.clear_history_button)
        layout.addLayout(header)

        self.history_status_label = QLabel("Tap an item to load it on the main page.")
        self.history_status_label.setObjectName("statusPill")
        self.history_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.history_status_label)

        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self._open_history_item)
        self.history_list.itemActivated.connect(self._open_history_item)
        layout.addWidget(self.history_list, 1)

        return page

    def _add_config_field(
        self,
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
        self,
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
        self._nav_buttons.append(button)
        return button

    def _gear_icon(self) -> QIcon:
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

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.live_timer.stop()
        self.cancel_live_transcription.emit()
        self.cancel_pipeline_processing.emit()
        if self.recorder.is_recording:
            try:
                self.recorder.stop()
            except Exception:
                pass
        self._live_thread.quit()
        self._live_thread.wait(2000)
        self._pipeline_thread.quit()
        self._pipeline_thread.wait(2000)
        super().closeEvent(event)

    def _show_main_page(self) -> None:
        self.stack.setCurrentWidget(self.main_page)

    def _show_config_page(self) -> None:
        self.stack.setCurrentWidget(self.config_page)

    def _show_history_page(self) -> None:
        self.stack.setCurrentWidget(self.history_page)

    def _selected_language(self) -> str | None:
        return self.language_input.text().strip() or None

    def _active_language_hint(self) -> str | None:
        return self._session_language_hint or self._selected_language()

    def _live_transcription_enabled(self) -> bool:
        return self.live_transcription_checkbox.isChecked()

    def _selected_ollama_model(self) -> str:
        return self.ollama_model_combo.currentText().strip()

    def _selected_whisper_model(self) -> str:
        return self.whisper_model_combo.currentText().strip()

    def _selected_device_id(self) -> str | None:
        data = self.device_combo.currentData()
        if data is None:
            return None
        return str(data)

    def _load_input_devices(self) -> None:
        try:
            devices = list_input_devices()
        except Exception as exc:
            self._set_status(f"Failed to query microphones: {exc}")
            return

        current_device = self.config.audio_device or self._selected_device_id()
        self.device_combo.clear()
        for device in devices:
            label = device.name
            if device.is_default:
                label = f"{label} (default)"
            self.device_combo.addItem(label, device.device_id)

        if current_device:
            for index in range(self.device_combo.count()):
                if str(self.device_combo.itemData(index)) == current_device:
                    self.device_combo.setCurrentIndex(index)
                    break

        if not devices:
            self._set_status("No input device was found.")
        else:
            self._set_status(f"Loaded {len(devices)} microphone(s).")

    def _load_ollama_models(self) -> None:
        current = self._selected_ollama_model() or self.config.ollama_model
        try:
            models = self.ollama_client.list_models()
        except Exception as exc:
            self.ollama_model_combo.clear()
            self.ollama_model_combo.addItem(current)
            self._set_status(f"Could not load Ollama models: {exc}")
            return

        self.ollama_model_combo.clear()
        self.ollama_model_combo.addItems(models)
        if current in models:
            self.ollama_model_combo.setCurrentText(current)
        elif models:
            self.ollama_model_combo.setCurrentIndex(0)
        else:
            self.ollama_model_combo.setCurrentText(current)

        self._set_status(f"Loaded {len(models)} Ollama model(s).")

    def _apply_settings(self) -> None:
        self.config.audio_device = self._selected_device_id()
        self.config.whisper_model = self._selected_whisper_model()
        self.config.language = self._selected_language()
        self.config.ollama_model = self._selected_ollama_model()
        self.config.enhance_with_ollama = self.enhance_checkbox.isChecked()
        self.config.live_transcription = self._live_transcription_enabled()
        self._set_status("Settings updated.")
        self._show_main_page()

    def _toggle_recording(self) -> None:
        if self._is_processing:
            self._cancel_processing()
            return
        if self.recorder.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _cancel_processing(self) -> None:
        if self._pipeline_request_in_flight:
            self.cancel_pipeline_processing.emit()
        elif self._pending_final_audio is not None:
            self._pending_final_audio = None
            if self._live_request_in_flight:
                self.cancel_live_transcription.emit()
            else:
                self._handle_cancelled("Processing cancelled.")
        else:
            return
        self.prompt_label.setText("Cancelling...")
        self._set_status("Cancelling transcription...")
        self._set_record_button_mode("working", "...")

    def _start_recording(self) -> None:
        try:
            self.recorder.start(device_id=self._selected_device_id())
        except Exception as exc:
            QMessageBox.critical(self, "Recording Error", str(exc))
            return

        self._live_session_id += 1
        self._pending_final_audio = None
        self._session_language_hint = self._selected_language()
        self._last_result = None
        self._recording_started_at = time.monotonic()
        self.elapsed_timer.start(250)
        if self._live_transcription_enabled():
            self.live_timer.start()
            QTimer.singleShot(300, self._queue_live_transcription)
            self._show_placeholder_transcript("Listening...")
            self._set_status("Recording with live transcription...")
        else:
            self._show_placeholder_transcript("Recording... transcript will appear after stop.")
            self._set_status("Recording...")
        self.prompt_label.setText("Tap again to stop")
        self._set_record_button_mode("recording", "STOP")
        self._set_navigation_enabled(False)

    def _stop_recording(self) -> None:
        self.live_timer.stop()
        try:
            recorded_audio = self.recorder.stop()
        except Exception as exc:
            self._set_idle_state()
            QMessageBox.critical(self, "Recording Error", str(exc))
            return

        self.elapsed_timer.stop()
        self.recording_clock.setText("00:00")
        self._live_session_id += 1
        self._is_processing = True
        self._set_record_button_mode("cancel", "ABORT")
        if self._live_transcription_enabled():
            self._pending_final_audio = recorded_audio
            self.prompt_label.setText("Finalizing...")
            self._set_status("Stopping recording and finalizing transcript...")
            if self._live_request_in_flight:
                self.cancel_live_transcription.emit()
                return
            self._start_pending_final_processing()
            return

        self.prompt_label.setText("Transcribing...")
        self._set_status("Working on your transcript...")
        self._run_pipeline(recorded_audio)

    def _queue_live_transcription(self) -> None:
        if (
            not self._live_transcription_enabled()
            or not self.recorder.is_recording
            or self._is_processing
            or self._live_request_in_flight
        ):
            return

        try:
            snapshot = self.recorder.snapshot(max_duration_seconds=LIVE_TRANSCRIPTION_WINDOW_SECONDS)
        except RuntimeError as exc:
            if "No microphone audio" in str(exc):
                return
            self._set_status(f"Live transcription paused: {exc}")
            return
        except Exception as exc:
            self._set_status(f"Live transcription failed: {exc}")
            return

        if snapshot.duration_seconds < LIVE_TRANSCRIPTION_MIN_DURATION_SECONDS:
            _unlink_file(snapshot.path)
            return

        self._live_request_in_flight = True
        self.request_live_transcription.emit(
            snapshot,
            self._selected_whisper_model(),
            self._active_language_hint(),
            self._live_session_id,
        )

    def _start_pending_final_processing(self) -> None:
        if self._pending_final_audio is None:
            return
        if self.recorder.is_recording or self._live_request_in_flight or self._pipeline_request_in_flight:
            return

        recorded_audio = self._pending_final_audio
        self._pending_final_audio = None
        self.prompt_label.setText("Transcribing...")
        self._set_status("Working on your transcript...")
        self._run_pipeline(recorded_audio)

    def _run_pipeline(self, recorded_audio: RecordedAudio) -> None:
        self._pipeline_request_in_flight = True
        self.request_pipeline_processing.emit(
            recorded_audio,
            self._selected_whisper_model(),
            self.enhance_checkbox.isChecked(),
            self._selected_ollama_model(),
            self._active_language_hint(),
        )

    @Slot(object)
    def _handle_live_update(self, update: LiveTranscriptionUpdate) -> None:
        self._live_request_in_flight = False
        if update.session_id != self._live_session_id or not self.recorder.is_recording:
            self._maybe_continue_after_live_worker()
            return
        if not update.text.strip():
            self._maybe_continue_after_live_worker()
            return

        language_locked = self._maybe_lock_session_language(update)
        self._show_transcript(update.text)
        if language_locked and self._session_language_hint is not None:
            self._set_status(f"Recording... locked to {self._session_language_hint}.")
        elif self._session_language_hint is not None:
            self._set_status(f"Recording... using {self._session_language_hint}.")
        elif update.detected_language:
            self._set_status(f"Recording... detected {update.detected_language}.")
        self._maybe_continue_after_live_worker()

    @Slot(str)
    def _handle_live_error(self, message: str) -> None:
        self._live_request_in_flight = False
        if self.recorder.is_recording:
            self._set_status(f"Live transcription skipped: {message}")
        self._maybe_continue_after_live_worker()

    @Slot()
    def _handle_live_cancelled(self) -> None:
        self._live_request_in_flight = False
        self._maybe_continue_after_live_worker()

    def _copy_transcript_to_clipboard(self) -> None:
        if not self._current_transcript_text:
            return
        QApplication.clipboard().setText(self._current_transcript_text)
        self._set_status("Copied to clipboard.")

    def _open_history_item(self, item: QListWidgetItem) -> None:
        index = item.data(Qt.ItemDataRole.UserRole)
        if index is None:
            return
        entry = self._history_entries[int(index)]
        self._last_result = entry.result
        self._show_transcript(entry.display_text)
        self._set_status(f"Loaded transcript from {entry.created_at}.")
        self._show_main_page()

    def _clear_history(self) -> None:
        self._history_entries.clear()
        self.history_list.clear()
        self._set_status("History cleared.")

    @Slot(object)
    def _handle_result(self, result: TranscriptionResult) -> None:
        self._pipeline_request_in_flight = False
        self._last_result = result
        self._show_transcript(result.cleaned_text or result.raw_text)
        self._add_history_entry(result)
        self._is_processing = False
        self._set_idle_state()
        self._set_status(
            f"Done in {result.duration_seconds:.1f}s, {result.detected_language or 'unknown'}."
        )

    @Slot(str)
    def _handle_error(self, message: str) -> None:
        self._pipeline_request_in_flight = False
        self._is_processing = False
        self._set_idle_state()
        self._set_status(f"Processing failed: {message}")
        QMessageBox.critical(self, "Processing Error", message)

    @Slot(str)
    def _handle_cancelled(self, message: str) -> None:
        self._pipeline_request_in_flight = False
        self._is_processing = False
        self._set_idle_state()
        self._set_status(message)
        self.prompt_label.setText("Tap once to start")

    def _add_history_entry(self, result: TranscriptionResult) -> None:
        entry = HistoryEntry(
            created_at=datetime.now().strftime("%H:%M"),
            result=result,
        )
        self._history_entries.insert(0, entry)

        item = QListWidgetItem()
        snippet = entry.display_text.replace("\n", " ").strip()
        if len(snippet) > 52:
            snippet = f"{snippet[:49]}..."
        item.setText(f"{entry.created_at}  {snippet or 'Empty transcript'}")
        item.setToolTip(entry.display_text)
        item.setData(Qt.ItemDataRole.UserRole, 0)
        self.history_list.insertItem(0, item)

        for index in range(self.history_list.count()):
            existing_item = self.history_list.item(index)
            existing_item.setData(Qt.ItemDataRole.UserRole, index)

    def _show_transcript(self, text: str) -> None:
        self._current_transcript_text = text.strip()
        if self._current_transcript_text:
            self.transcript_label.setText(self._current_transcript_text)
        else:
            self.transcript_label.setText("No transcript yet.")
        self.copy_button.setEnabled(bool(self._current_transcript_text))

    def _show_placeholder_transcript(self, text: str) -> None:
        self._current_transcript_text = ""
        self.transcript_label.setText(text)
        self.copy_button.setEnabled(False)

    def _maybe_lock_session_language(self, update: LiveTranscriptionUpdate) -> bool:
        if self._session_language_hint is not None:
            return False
        if not update.detected_language:
            return False
        if update.duration_seconds < LIVE_LANGUAGE_LOCK_MIN_DURATION_SECONDS:
            return False
        if len(update.text.split()) < LIVE_LANGUAGE_LOCK_MIN_WORDS:
            return False

        self._session_language_hint = update.detected_language
        return True

    def _set_idle_state(self) -> None:
        self.elapsed_timer.stop()
        self.live_timer.stop()
        self._recording_started_at = None
        self._pending_final_audio = None
        self._session_language_hint = None
        self.recording_clock.setText("00:00")
        self.prompt_label.setText("Tap once to start")
        self._set_record_button_mode("idle", "REC")
        self._set_navigation_enabled(True)

    def _set_status(self, message: str) -> None:
        self.main_status_label.setText(message)
        self.config_status_label.setText(message)
        self.history_status_label.setText(message)

    def _set_record_button_mode(self, mode: str, text: str) -> None:
        self.record_button.setProperty("mode", mode)
        self.record_button.setText(text)
        self.record_button.setEnabled(mode != "working")
        self.record_button.style().unpolish(self.record_button)
        self.record_button.style().polish(self.record_button)
        self.record_button.update()

    def _set_navigation_enabled(self, enabled: bool) -> None:
        for button in self._nav_buttons:
            button.setEnabled(enabled)
        self.apply_button.setEnabled(enabled)
        self.clear_history_button.setEnabled(enabled)

    @Slot()
    def _maybe_continue_after_live_worker(self) -> None:
        if (
            self._pending_final_audio is not None
            and not self.recorder.is_recording
            and not self._live_request_in_flight
            and not self._pipeline_request_in_flight
        ):
            self._start_pending_final_processing()
            return
        if (
            self._is_processing
            and self._pending_final_audio is None
            and not self.recorder.is_recording
            and not self._live_request_in_flight
            and not self._pipeline_request_in_flight
        ):
            self._handle_cancelled("Processing cancelled.")

    def _update_recording_clock(self) -> None:
        if self._recording_started_at is None:
            self.recording_clock.setText("00:00")
            return

        elapsed = int(time.monotonic() - self._recording_started_at)
        minutes, seconds = divmod(elapsed, 60)
        self.recording_clock.setText(f"{minutes:02d}:{seconds:02d}")


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow(AppConfig.from_env())
    window.show()
    return app.exec()


def _unlink_file(path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass
