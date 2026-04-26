from __future__ import annotations

import sys
import time
from datetime import datetime
from multiprocessing import freeze_support

from PySide6.QtCore import QEvent, Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QApplication,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QToolButton,
)

from .audio import MicrophoneRecorder, list_input_devices
from .config import AppConfig
from .history_store import load_history_entries, save_history_entries
from .models import (
    CleanupUpdate,
    HistoryEntry,
    LiveTranscriptionUpdate,
    RecordedAudio,
    TranscriptionResult,
)
from .ollama_client import OllamaClient
from .ui_shortcuts import is_push_to_talk_press, is_push_to_talk_release
from .ui_layout import build_ui
from .worker_client import ProcessWorkerClient

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
LIVE_TRANSCRIPTION_WINDOW_SECONDS = 5.0
LIVE_VOICE_MEAN_ABS_THRESHOLD = 120
LIVE_VOICE_PEAK_ABS_THRESHOLD = 800
LIVE_LANGUAGE_LOCK_MIN_DURATION_SECONDS = 2.0
LIVE_LANGUAGE_LOCK_MIN_WORDS = 3
IDLE_PROMPT_TEXT = "Tap once or hold Space"
IDLE_PLACEHOLDER_TEXT = "Tap the red button or hold Space to record."


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.recorder = MicrophoneRecorder(
            sample_rate=config.sample_rate,
            rolling_buffer_seconds=LIVE_TRANSCRIPTION_WINDOW_SECONDS,
        )
        self.ollama_client = OllamaClient(config.ollama_base_url)
        self._worker_client = ProcessWorkerClient(config.ollama_base_url, self)

        self._recording_started_at: float | None = None
        self._last_result: TranscriptionResult | None = None
        self._history_entries: list[HistoryEntry] = []
        self._current_transcript_text = ""
        self._is_processing = False
        self._live_session_id = 0
        self._live_request_in_flight = False
        self._pipeline_request_in_flight = False
        self._cleanup_request_in_flight = False
        self._pending_final_audio: RecordedAudio | None = None
        self._active_live_snapshot: RecordedAudio | None = None
        self._active_pipeline_audio: RecordedAudio | None = None
        self._session_language_hint: str | None = None
        self._nav_buttons: list[QToolButton] = []
        self._spacebar_pressed = False
        self._spacebar_recording_active = False

        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.timeout.connect(self._update_recording_clock)
        self.live_timer = QTimer(self)
        self.live_timer.setInterval(LIVE_TRANSCRIPTION_INTERVAL_MS)
        self.live_timer.timeout.connect(self._queue_live_transcription)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        self._worker_client.progress.connect(self._set_status)
        self._worker_client.live_finished.connect(self._handle_live_update)
        self._worker_client.live_error.connect(self._handle_live_error)
        self._worker_client.live_cancelled.connect(self._handle_live_cancelled)
        self._worker_client.pipeline_finished.connect(self._handle_result)
        self._worker_client.pipeline_error.connect(self._handle_error)
        self._worker_client.pipeline_cancelled.connect(self._handle_cancelled)
        self._worker_client.cleanup_finished.connect(self._handle_cleanup_result)
        self._worker_client.cleanup_error.connect(self._handle_cleanup_error)
        self._worker_client.cleanup_cancelled.connect(self._handle_cleanup_cancelled)

        self.setWindowTitle("Talk2Text")
        self.setFixedSize(300, 400)
        build_ui(self, self.config, WHISPER_MODELS)
        self._load_persistent_history()
        self._load_input_devices()
        self._load_ollama_models()
        self._show_placeholder_transcript(IDLE_PLACEHOLDER_TEXT)
        self._set_status("Ready")
        self._set_idle_state()

    def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
        if watched is self and event.type() == QEvent.Type.WindowDeactivate:
            self._stop_spacebar_recording()
            return False
        if event.type() == QEvent.Type.KeyPress and self._handle_spacebar_press(event):
            return True
        if event.type() == QEvent.Type.KeyRelease and self._handle_spacebar_release(event):
            return True
        return super().eventFilter(watched, event)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.live_timer.stop()
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self._reset_spacebar_state()
        if self.recorder.is_recording:
            try:
                _discard_recorded_audio(self.recorder.stop())
            except Exception:
                pass
        self._worker_client.shutdown()
        self._release_active_live_snapshot()
        self._discard_pending_final_audio()
        self._release_active_pipeline_audio()
        super().closeEvent(event)

    def _show_main_page(self) -> None:
        self.stack.setCurrentWidget(self.main_page)

    def _show_config_page(self) -> None:
        self.stack.setCurrentWidget(self.config_page)

    def _show_history_page(self) -> None:
        self.stack.setCurrentWidget(self.history_page)

    def _push_to_talk_available(self) -> bool:
        return self.isActiveWindow() and self.stack.currentWidget() is self.main_page

    def _handle_spacebar_press(self, event) -> bool:
        if not is_push_to_talk_press(
            key=event.key(),
            modifiers=event.modifiers(),
            auto_repeat=event.isAutoRepeat(),
        ):
            return False
        if not self._push_to_talk_available():
            return False

        self._spacebar_pressed = True
        if self.recorder.is_recording or self._is_processing:
            return True

        self._start_recording()
        self._spacebar_recording_active = self.recorder.is_recording
        return True

    def _handle_spacebar_release(self, event) -> bool:
        if not is_push_to_talk_release(key=event.key(), auto_repeat=event.isAutoRepeat()):
            return False
        if not self._spacebar_pressed and not self._spacebar_recording_active:
            return False

        self._spacebar_pressed = False
        if self._spacebar_recording_active:
            self._stop_spacebar_recording()
        return True

    def _stop_spacebar_recording(self) -> None:
        started_by_spacebar = self._spacebar_recording_active
        self._reset_spacebar_state()
        if started_by_spacebar and self.recorder.is_recording and not self._is_processing:
            self._stop_recording()

    def _reset_spacebar_state(self) -> None:
        self._spacebar_pressed = False
        self._spacebar_recording_active = False

    def _selected_language(self) -> str | None:
        return self.language_input.text().strip() or None

    def _active_language_hint(self) -> str | None:
        return self._session_language_hint or self._selected_language()

    def _cleanup_language_hint(self) -> str | None:
        if self._last_result is not None and self._last_result.detected_language:
            return self._last_result.detected_language
        return self._selected_language()

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
        self._update_action_buttons()
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
        if self._cleanup_request_in_flight:
            self._worker_client.cancel_current()
        elif self._pipeline_request_in_flight:
            self._worker_client.cancel_current()
        elif self._pending_final_audio is not None:
            self._discard_pending_final_audio()
            if self._live_request_in_flight:
                self._worker_client.cancel_current()
            else:
                self._handle_cancelled("Processing cancelled.")
        else:
            return

    def _discard_pending_final_audio(self) -> None:
        if self._pending_final_audio is not None:
            _discard_recorded_audio(self._pending_final_audio)
            self._pending_final_audio = None

    def _release_active_live_snapshot(self) -> None:
        if self._active_live_snapshot is not None:
            _unlink_file(self._active_live_snapshot.path)
            self._active_live_snapshot = None

    def _release_active_pipeline_audio(self) -> None:
        if self._active_pipeline_audio is not None:
            _discard_recorded_audio(self._active_pipeline_audio)
            self._active_pipeline_audio = None

    def _start_recording(self) -> None:
        try:
            self.recorder.start(device_id=self._selected_device_id())
        except Exception as exc:
            QMessageBox.critical(self, "Recording Error", str(exc))
            return

        self._live_session_id += 1
        self._pending_final_audio = None
        self._active_live_snapshot = None
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
                self._worker_client.cancel_current()
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
            if not self.recorder.has_voice_activity(
                max_duration_seconds=LIVE_TRANSCRIPTION_WINDOW_SECONDS,
                min_mean_abs=LIVE_VOICE_MEAN_ABS_THRESHOLD,
                min_peak_abs=LIVE_VOICE_PEAK_ABS_THRESHOLD,
            ):
                return
            snapshot = self.recorder.snapshot(
                max_duration_seconds=LIVE_TRANSCRIPTION_WINDOW_SECONDS
            )
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
        self._active_live_snapshot = snapshot
        try:
            self._worker_client.request_live(
                snapshot,
                self._selected_whisper_model(),
                self._active_language_hint(),
                self._live_session_id,
            )
        except Exception as exc:
            self._live_request_in_flight = False
            self._release_active_live_snapshot()
            self._set_status(f"Live transcription failed: {exc}")

    def _start_pending_final_processing(self) -> None:
        if self._pending_final_audio is None:
            return
        if (
            self.recorder.is_recording
            or self._live_request_in_flight
            or self._pipeline_request_in_flight
            or self._cleanup_request_in_flight
        ):
            return

        recorded_audio = self._pending_final_audio
        self._pending_final_audio = None
        self.prompt_label.setText("Transcribing...")
        self._set_status("Working on your transcript...")
        self._run_pipeline(recorded_audio)

    def _run_pipeline(self, recorded_audio: RecordedAudio) -> None:
        self._pipeline_request_in_flight = True
        self._active_pipeline_audio = recorded_audio
        try:
            self._worker_client.request_pipeline(
                recorded_audio,
                self._selected_whisper_model(),
                False,
                self._selected_ollama_model(),
                self._active_language_hint(),
            )
        except Exception as exc:
            self._pipeline_request_in_flight = False
            self._release_active_pipeline_audio()
            self._is_processing = False
            self._set_idle_state()
            self._set_status(f"Processing failed: {exc}")
            QMessageBox.critical(self, "Processing Error", str(exc))

    @Slot(object)
    def _handle_live_update(self, update: LiveTranscriptionUpdate) -> None:
        self._live_request_in_flight = False
        self._release_active_live_snapshot()
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
        self._release_active_live_snapshot()
        if self.recorder.is_recording:
            self._set_status(f"Live transcription skipped: {message}")
        self._maybe_continue_after_live_worker()

    @Slot()
    def _handle_live_cancelled(self) -> None:
        self._live_request_in_flight = False
        self._release_active_live_snapshot()
        self._maybe_continue_after_live_worker()

    def _polish_transcript(self) -> None:
        if self._last_result is None:
            self._set_status("No transcript available to polish.")
            return
        if not self.enhance_checkbox.isChecked():
            self._set_status("Manual Ollama polish is disabled in settings.")
            return
        if self.recorder.is_recording or self._is_processing:
            return

        source_text = self._last_result.raw_text.strip() or self._current_transcript_text
        if not source_text:
            self._set_status("No transcript available to polish.")
            return

        self._cleanup_request_in_flight = True
        self._is_processing = True
        self.prompt_label.setText("Polishing...")
        self._set_status(f"Polishing transcript with Ollama ({self._selected_ollama_model()})...")
        self._set_record_button_mode("cancel", "ABORT")
        self._set_navigation_enabled(False)
        self._update_action_buttons()
        try:
            self._worker_client.request_cleanup(
                source_text,
                self._selected_ollama_model(),
                self._cleanup_language_hint(),
            )
        except Exception as exc:
            self._cleanup_request_in_flight = False
            self._is_processing = False
            self._set_idle_state()
            self._set_status(f"Ollama polish failed: {exc}")
            QMessageBox.critical(self, "Ollama Error", str(exc))

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
        self._persist_history()
        self._set_status("History cleared.")

    @Slot(object)
    def _handle_result(self, result: TranscriptionResult) -> None:
        self._pipeline_request_in_flight = False
        self._active_pipeline_audio = None
        self._discard_result_audio(result)
        self._last_result = result
        self._show_transcript(result.cleaned_text or result.raw_text)
        self._add_history_entry(result)
        self._is_processing = False
        self._set_idle_state()
        self._set_status(
            f"Done in {result.duration_seconds:.1f}s, {result.detected_language or 'unknown'}."
        )

    @Slot(object)
    def _handle_cleanup_result(self, update: CleanupUpdate) -> None:
        self._cleanup_request_in_flight = False
        if self._last_result is not None:
            self._last_result.cleaned_text = update.cleanup.cleaned_text
            self._last_result.summary = update.cleanup.summary
            self._last_result.action_items = update.cleanup.action_items
            self._last_result.notes.append(
                f"Manual Ollama polish with {update.model_name} in {update.elapsed_seconds:.2f}s"
            )

        self._show_transcript(update.cleanup.cleaned_text)
        self._sync_history_list()
        self._persist_history()
        self._is_processing = False
        self._set_idle_state()
        self._set_status(f"Polished with {update.model_name} in {update.elapsed_seconds:.1f}s.")

    @Slot(str)
    def _handle_error(self, message: str) -> None:
        self._pipeline_request_in_flight = False
        self._release_active_pipeline_audio()
        self._is_processing = False
        self._set_idle_state()
        self._set_status(f"Processing failed: {message}")
        QMessageBox.critical(self, "Processing Error", message)

    @Slot(str)
    def _handle_cleanup_error(self, message: str) -> None:
        self._cleanup_request_in_flight = False
        self._is_processing = False
        self._set_idle_state()
        self._set_status(f"Ollama polish failed: {message}")
        QMessageBox.critical(self, "Ollama Error", message)

    @Slot(str)
    def _handle_cancelled(self, message: str) -> None:
        self._pipeline_request_in_flight = False
        self._release_active_pipeline_audio()
        self._is_processing = False
        self._set_idle_state()
        self._set_status(message)
        self.prompt_label.setText(IDLE_PROMPT_TEXT)

    @Slot(str)
    def _handle_cleanup_cancelled(self, message: str) -> None:
        self._cleanup_request_in_flight = False
        self._is_processing = False
        self._set_idle_state()
        self._set_status(message)
        self.prompt_label.setText(IDLE_PROMPT_TEXT)

    def _add_history_entry(self, result: TranscriptionResult) -> None:
        entry = HistoryEntry(
            created_at=datetime.now().strftime("%m-%d %H:%M"),
            result=result,
        )
        self._history_entries.insert(0, entry)
        self._sync_history_list()
        self._persist_history()

    def _sync_history_list(self) -> None:
        self.history_list.clear()
        for index, entry in enumerate(self._history_entries):
            item = QListWidgetItem()
            snippet = entry.display_text.replace("\n", " ").strip()
            if len(snippet) > 52:
                snippet = f"{snippet[:49]}..."
            item.setText(f"{entry.created_at}  {snippet or 'Empty transcript'}")
            item.setToolTip(entry.display_text)
            item.setData(Qt.ItemDataRole.UserRole, index)
            self.history_list.addItem(item)

    def _load_persistent_history(self) -> None:
        self._history_entries = load_history_entries()
        self._sync_history_list()

    def _persist_history(self) -> None:
        try:
            save_history_entries(self._history_entries)
        except Exception as exc:
            self._set_status(f"Could not save history: {exc}")

    def _show_transcript(self, text: str) -> None:
        self._current_transcript_text = text.strip()
        if self._current_transcript_text:
            self.transcript_label.setText(self._current_transcript_text)
        else:
            self.transcript_label.setText("No transcript yet.")
        self._update_action_buttons()

    def _show_placeholder_transcript(self, text: str) -> None:
        self._current_transcript_text = ""
        self.transcript_label.setText(text)
        self._update_action_buttons()

    def _update_action_buttons(self, _checked: bool | None = None) -> None:
        has_text = bool(self._current_transcript_text)
        busy = (
            self.recorder.is_recording
            or self._is_processing
            or self._live_request_in_flight
            or self._pipeline_request_in_flight
            or self._cleanup_request_in_flight
        )
        self.copy_button.setEnabled(has_text)
        self.polish_button.setEnabled(has_text and self.enhance_checkbox.isChecked() and not busy)

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
        self._reset_spacebar_state()
        self.recording_clock.setText("00:00")
        self.prompt_label.setText(IDLE_PROMPT_TEXT)
        self._set_record_button_mode("idle", "REC")
        self._set_navigation_enabled(True)
        self._update_action_buttons()

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

    def _discard_result_audio(self, result: TranscriptionResult) -> None:
        if result.audio_path is None:
            return
        _unlink_file(result.audio_path)
        result.audio_path = None

    @Slot()
    def _maybe_continue_after_live_worker(self) -> None:
        if (
            self._pending_final_audio is not None
            and not self.recorder.is_recording
            and not self._live_request_in_flight
            and not self._pipeline_request_in_flight
            and not self._cleanup_request_in_flight
        ):
            self._start_pending_final_processing()
            return
        if (
            self._is_processing
            and self._pending_final_audio is None
            and not self.recorder.is_recording
            and not self._live_request_in_flight
            and not self._pipeline_request_in_flight
            and not self._cleanup_request_in_flight
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
    freeze_support()
    app = QApplication(sys.argv)
    window = MainWindow(AppConfig.from_env())
    window.show()
    return app.exec()


def _discard_recorded_audio(recorded_audio: RecordedAudio) -> None:
    _unlink_file(recorded_audio.path)


def _unlink_file(path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass
