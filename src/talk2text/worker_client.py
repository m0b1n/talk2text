from __future__ import annotations

from multiprocessing import get_context
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from .models import RecordedAudio
from .worker_runtime import worker_process_main

CANCELLED_MESSAGE = "Processing cancelled."


class ProcessWorkerClient(QObject):
    progress = Signal(str)
    live_finished = Signal(object)
    pipeline_finished = Signal(object)
    cleanup_finished = Signal(object)
    live_error = Signal(str)
    pipeline_error = Signal(str)
    cleanup_error = Signal(str)
    live_cancelled = Signal()
    pipeline_cancelled = Signal(str)
    cleanup_cancelled = Signal(str)

    def __init__(self, ollama_base_url: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._ollama_base_url = ollama_base_url
        self._ctx = get_context("spawn")
        self._connection: Connection | None = None
        self._process: BaseProcess | None = None
        self._retiring_processes: list[BaseProcess] = []
        self._active_kind: str | None = None
        self._active_request_id = 0
        self._next_request_id = 0
        self._termination_expected = False
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(50)
        self._poll_timer.timeout.connect(self._poll_messages)
        self._poll_timer.start()

    def request_live(
        self,
        recorded_audio: RecordedAudio,
        whisper_model: str,
        language: str | None,
        session_id: int,
    ) -> None:
        self._start_request(
            "live",
            {
                "recorded_audio": recorded_audio,
                "whisper_model": whisper_model,
                "language": language,
                "session_id": session_id,
            },
        )

    def request_pipeline(
        self,
        recorded_audio: RecordedAudio,
        whisper_model: str,
        use_ollama: bool,
        ollama_model: str,
        language: str | None,
    ) -> None:
        self._start_request(
            "pipeline",
            {
                "recorded_audio": recorded_audio,
                "whisper_model": whisper_model,
                "use_ollama": use_ollama,
                "ollama_model": ollama_model,
                "language": language,
            },
        )

    def request_cleanup(
        self,
        raw_text: str,
        model_name: str,
        language_hint: str | None,
    ) -> None:
        self._start_request(
            "cleanup",
            {
                "raw_text": raw_text,
                "model_name": model_name,
                "language_hint": language_hint,
            },
        )

    def cancel_current(self) -> None:
        if self._active_kind is None:
            return

        active_kind = self._active_kind
        self._clear_active_request()
        self._terminate_process(expected=True)

        if active_kind == "live":
            self.live_cancelled.emit()
        elif active_kind == "pipeline":
            self.pipeline_cancelled.emit(CANCELLED_MESSAGE)
        else:
            self.cleanup_cancelled.emit(CANCELLED_MESSAGE)

    def shutdown(self) -> None:
        self._poll_timer.stop()
        self._clear_active_request()
        self._terminate_process(expected=True)
        self._reap_processes(force=True)

    def _start_request(self, kind: str, payload: dict[str, object]) -> None:
        if self._active_kind is not None:
            raise RuntimeError("Background worker is already busy.")

        self._next_request_id += 1
        request_id = self._next_request_id
        command = {
            "kind": kind,
            "request_id": request_id,
            **payload,
        }

        self._send_command(command)
        self._active_kind = kind
        self._active_request_id = request_id
        self._termination_expected = False

    def _send_command(self, command: dict[str, object]) -> None:
        self._ensure_process()
        connection = self._connection
        if connection is None:
            raise RuntimeError("Background worker is unavailable.")

        try:
            connection.send(command)
        except (BrokenPipeError, EOFError, OSError):
            self._teardown_process()
            self._ensure_process()
            connection = self._connection
            if connection is None:
                raise RuntimeError("Background worker is unavailable.")
            connection.send(command)

    def _ensure_process(self) -> None:
        self._poll_messages()
        if self._process is not None and self._process.is_alive():
            return

        parent_conn, child_conn = self._ctx.Pipe()
        process = self._ctx.Process(
            target=worker_process_main,
            args=(child_conn, self._ollama_base_url),
            daemon=True,
        )
        process.start()
        child_conn.close()
        self._connection = parent_conn
        self._process = process
        self._termination_expected = False

    @Slot()
    def _poll_messages(self) -> None:
        connection = self._connection
        if connection is not None:
            while connection.poll():
                try:
                    message = connection.recv()
                except EOFError:
                    break
                self._handle_message(message)

        self._handle_process_exit()
        self._reap_processes()

    def _handle_message(self, message: dict[str, object]) -> None:
        request_id = int(message.get("request_id", 0))
        if request_id and request_id != self._active_request_id:
            return

        message_type = str(message.get("type", "")).strip()
        if message_type == "progress":
            self.progress.emit(str(message.get("message", "")))
            return

        active_kind = self._active_kind
        if active_kind is None:
            return

        self._clear_active_request()

        if message_type == "live_result":
            self.live_finished.emit(message.get("update"))
            return
        if message_type == "pipeline_result":
            self.pipeline_finished.emit(message.get("result"))
            return
        if message_type == "cleanup_result":
            self.cleanup_finished.emit(message.get("update"))
            return

        error_message = str(message.get("message", "Background worker failed."))
        self._emit_error(active_kind, error_message)

    def _handle_process_exit(self) -> None:
        process = self._process
        if process is None:
            return

        exit_code = process.exitcode
        if exit_code is None:
            return

        active_kind = self._active_kind
        expected = self._termination_expected
        self._teardown_process()
        if active_kind is None or expected:
            return

        self._clear_active_request()
        self._emit_error(active_kind, f"Background worker exited unexpectedly ({exit_code}).")

    def _emit_error(self, kind: str, message: str) -> None:
        if kind == "live":
            self.live_error.emit(message)
        elif kind == "pipeline":
            self.pipeline_error.emit(message)
        else:
            self.cleanup_error.emit(message)

    def _clear_active_request(self) -> None:
        self._active_kind = None
        self._active_request_id = 0

    def _terminate_process(self, expected: bool) -> None:
        process = self._process
        if process is None:
            self._teardown_connection()
            return

        self._termination_expected = expected
        if process.is_alive():
            process.terminate()
        self._retiring_processes.append(process)
        self._process = None
        self._teardown_connection()

    def _teardown_process(self) -> None:
        process = self._process
        self._process = None
        self._teardown_connection()
        if process is not None:
            self._retire_process(process)

    def _teardown_connection(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except OSError:
                pass
        self._connection = None

    def _retire_process(self, process: BaseProcess) -> None:
        if process.exitcode is None:
            self._retiring_processes.append(process)
            return

        try:
            process.join(timeout=0)
        except Exception:
            pass
        try:
            process.close()
        except ValueError:
            pass

    def _reap_processes(self, force: bool = False) -> None:
        survivors: list[BaseProcess] = []
        for process in self._retiring_processes:
            if force and process.exitcode is None:
                try:
                    process.kill()
                except Exception:
                    pass
            if process.exitcode is None:
                survivors.append(process)
                continue
            try:
                process.join(timeout=0)
            except Exception:
                pass
            try:
                process.close()
            except ValueError:
                pass
        self._retiring_processes = survivors
