from __future__ import annotations

from pathlib import Path
import sys
import time

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .audio import MicrophoneRecorder, list_input_devices
from .config import AppConfig
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


class PipelineWorker(QObject):
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        pipeline: Talk2TextPipeline,
        recorded_audio: RecordedAudio,
        whisper_model: str,
        use_ollama: bool,
        ollama_model: str,
        language: str | None,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.recorded_audio = recorded_audio
        self.whisper_model = whisper_model
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        self.language = language

    @Slot()
    def run(self) -> None:
        try:
            result = self.pipeline.process(
                recorded_audio=self.recorded_audio,
                whisper_model=self.whisper_model,
                use_ollama=self.use_ollama,
                ollama_model=self.ollama_model,
                language=self.language,
                status_callback=self.progress.emit,
            )
        except Exception as exc:
            self.error.emit(str(exc))
            return

        self.finished.emit(result)


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.recorder = MicrophoneRecorder(sample_rate=config.sample_rate)
        self.transcriber = FasterWhisperTranscriber(config.whisper_model)
        self.ollama_client = OllamaClient(config.ollama_base_url)
        self.pipeline = Talk2TextPipeline(self.transcriber, self.ollama_client)

        self._worker_thread: QThread | None = None
        self._worker: PipelineWorker | None = None
        self._recording_started_at: float | None = None
        self._last_result: TranscriptionResult | None = None

        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.timeout.connect(self._update_recording_clock)

        self.setWindowTitle("Talk2Text")
        self.resize(1280, 760)
        self._build_ui()
        self._load_input_devices()
        self._load_ollama_models()
        self._set_idle_state()

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        settings_layout = QFormLayout()
        settings_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.device_combo = QComboBox()
        self.refresh_devices_button = QPushButton("Refresh Devices")
        self.refresh_devices_button.clicked.connect(self._load_input_devices)
        device_row = QHBoxLayout()
        device_row.addWidget(self.device_combo, 1)
        device_row.addWidget(self.refresh_devices_button)
        settings_layout.addRow("Input Device", device_row)

        self.whisper_model_combo = QComboBox()
        self.whisper_model_combo.setEditable(True)
        self.whisper_model_combo.addItems(WHISPER_MODELS)
        if self.config.whisper_model not in WHISPER_MODELS:
            self.whisper_model_combo.addItem(self.config.whisper_model)
        self.whisper_model_combo.setCurrentText(self.config.whisper_model)
        settings_layout.addRow("Whisper Model", self.whisper_model_combo)

        self.language_input = QLineEdit(self.config.language or "")
        self.language_input.setPlaceholderText("Auto-detect")
        settings_layout.addRow("Language Hint", self.language_input)

        self.ollama_model_combo = QComboBox()
        self.ollama_model_combo.setEditable(True)
        self.ollama_model_combo.setCurrentText(self.config.ollama_model)
        self.refresh_ollama_button = QPushButton("Refresh Ollama")
        self.refresh_ollama_button.clicked.connect(self._load_ollama_models)
        ollama_row = QHBoxLayout()
        ollama_row.addWidget(self.ollama_model_combo, 1)
        ollama_row.addWidget(self.refresh_ollama_button)
        settings_layout.addRow("Ollama Model", ollama_row)

        self.enhance_checkbox = QCheckBox("Enhance transcript with Ollama")
        self.enhance_checkbox.setChecked(self.config.enhance_with_ollama)
        settings_layout.addRow("", self.enhance_checkbox)

        outer.addLayout(settings_layout)

        controls_row = QHBoxLayout()
        controls_row.setSpacing(8)

        self.start_button = QPushButton("Start Recording")
        self.start_button.clicked.connect(self._start_recording)
        controls_row.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Recording")
        self.stop_button.clicked.connect(self._stop_recording)
        controls_row.addWidget(self.stop_button)

        self.save_button = QPushButton("Save Transcript")
        self.save_button.clicked.connect(self._save_transcript)
        controls_row.addWidget(self.save_button)

        controls_row.addStretch(1)

        self.recording_clock = QLabel("00:00")
        self.recording_clock.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        controls_row.addWidget(self.recording_clock)

        outer.addLayout(controls_row)

        transcripts_layout = QGridLayout()
        transcripts_layout.setSpacing(12)

        raw_label = QLabel("Raw Transcript")
        transcripts_layout.addWidget(raw_label, 0, 0)
        cleaned_label = QLabel("Processed Transcript")
        transcripts_layout.addWidget(cleaned_label, 0, 1)

        self.raw_transcript = QTextEdit()
        self.raw_transcript.setReadOnly(True)
        transcripts_layout.addWidget(self.raw_transcript, 1, 0)

        self.cleaned_transcript = QTextEdit()
        self.cleaned_transcript.setReadOnly(True)
        transcripts_layout.addWidget(self.cleaned_transcript, 1, 1)

        summary_label = QLabel("Summary")
        transcripts_layout.addWidget(summary_label, 2, 0)
        actions_label = QLabel("Action Items")
        transcripts_layout.addWidget(actions_label, 2, 1)

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setFixedHeight(90)
        transcripts_layout.addWidget(self.summary_text, 3, 0)

        self.action_list = QListWidget()
        transcripts_layout.addWidget(self.action_list, 3, 1)

        outer.addLayout(transcripts_layout, 1)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)

        self.notes_text = QTextEdit()
        self.notes_text.setReadOnly(True)
        self.notes_text.setFixedHeight(100)
        outer.addWidget(self.notes_text)

        save_action = QAction("Save Transcript", self)
        save_action.triggered.connect(self._save_transcript)
        self.addAction(save_action)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.recorder.is_recording:
            try:
                self.recorder.stop()
            except Exception:
                pass
        super().closeEvent(event)

    def _selected_language(self) -> str | None:
        return self.language_input.text().strip() or None

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

        self.device_combo.clear()
        for device in devices:
            label = device.name
            if device.is_default:
                label = f"{label} (default)"
            self.device_combo.addItem(label, device.device_id)

        if self.config.audio_device:
            for index in range(self.device_combo.count()):
                label = self.device_combo.itemText(index)
                if self.config.audio_device.lower() in label.lower():
                    self.device_combo.setCurrentIndex(index)
                    break

        if not devices:
            self._set_status("No input device was found.")
        else:
            self._set_status(f"Loaded {len(devices)} input device(s).")

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

    def _start_recording(self) -> None:
        try:
            self.recorder.start(device_id=self._selected_device_id())
        except Exception as exc:
            QMessageBox.critical(self, "Recording Error", str(exc))
            return

        self._recording_started_at = time.monotonic()
        self.elapsed_timer.start(250)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._set_status("Recording from microphone...")

    def _stop_recording(self) -> None:
        try:
            recorded_audio = self.recorder.stop()
        except Exception as exc:
            self._set_idle_state()
            QMessageBox.critical(self, "Recording Error", str(exc))
            return

        self.elapsed_timer.stop()
        self.recording_clock.setText("00:00")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self._set_status(
            f"Captured {recorded_audio.duration_seconds:.1f}s of audio. Starting transcription..."
        )
        self._run_pipeline(recorded_audio)

    def _run_pipeline(self, recorded_audio: RecordedAudio) -> None:
        self._worker_thread = QThread(self)
        self._worker = PipelineWorker(
            pipeline=self.pipeline,
            recorded_audio=recorded_audio,
            whisper_model=self._selected_whisper_model(),
            use_ollama=self.enhance_checkbox.isChecked(),
            ollama_model=self._selected_ollama_model(),
            language=self._selected_language(),
        )
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._set_status)
        self._worker.finished.connect(self._handle_result)
        self._worker.error.connect(self._handle_error)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.error.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.start()

    def _save_transcript(self) -> None:
        if self._last_result is None:
            QMessageBox.information(self, "Nothing to Save", "Record and transcribe audio first.")
            return

        path, _selected = QFileDialog.getSaveFileName(
            self,
            "Save Transcript",
            str(Path.home() / "talk2text-transcript.txt"),
            "Text Files (*.txt)",
        )
        if not path:
            return

        lines = [
            "Talk2Text Transcript",
            "",
            f"Audio file: {self._last_result.audio_path}",
            f"Duration: {self._last_result.duration_seconds:.1f}s",
            f"Detected language: {self._last_result.detected_language or 'unknown'}",
            "",
            "Raw Transcript",
            self._last_result.raw_text,
            "",
            "Processed Transcript",
            self._last_result.cleaned_text,
            "",
            "Summary",
            self._last_result.summary,
            "",
            "Action Items",
        ]
        lines.extend(self._last_result.action_items or [""])
        lines.extend(["", "Notes"])
        lines.extend(self._last_result.notes)

        Path(path).write_text("\n".join(lines), encoding="utf-8")
        self._set_status(f"Saved transcript to {path}")

    @Slot(object)
    def _handle_result(self, result: TranscriptionResult) -> None:
        self._last_result = result
        self.raw_transcript.setPlainText(result.raw_text)
        self.cleaned_transcript.setPlainText(result.cleaned_text)
        self.summary_text.setPlainText(result.summary)
        self.action_list.clear()
        self.action_list.addItems(result.action_items)
        self.notes_text.setPlainText("\n".join(result.notes))
        self._set_idle_state()
        self._set_status(
            f"Done. {result.duration_seconds:.1f}s audio, language={result.detected_language or 'unknown'}."
        )

    @Slot(str)
    def _handle_error(self, message: str) -> None:
        self._set_idle_state()
        self._set_status(f"Processing failed: {message}")
        QMessageBox.critical(self, "Processing Error", message)

    def _set_idle_state(self) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.save_button.setEnabled(self._last_result is not None)
        self.elapsed_timer.stop()
        self._recording_started_at = None
        self.recording_clock.setText("00:00")

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

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
