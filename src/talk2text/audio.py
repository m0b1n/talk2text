from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import wave

from PySide6.QtCore import QIODevice
from PySide6.QtMultimedia import QAudioDevice, QAudioFormat, QAudioSource, QMediaDevices

from .models import InputDevice, RecordedAudio


class MicrophoneRecorder:
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        rolling_buffer_seconds: float = 6.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.rolling_buffer_seconds = rolling_buffer_seconds
        self._buffer = bytearray()
        self._buffer_lock = threading.Lock()
        self._audio_source: QAudioSource | None = None
        self._io_device: QIODevice | None = None
        self._audio_format: QAudioFormat | None = None
        self._device_id: str | None = None
        self._session_path: Path | None = None
        self._wav_file: wave.Wave_write | None = None
        self._captured_bytes = 0

    @property
    def is_recording(self) -> bool:
        return self._audio_source is not None

    def start(self, device_id: str | None = None) -> None:
        if self._audio_source is not None:
            raise RuntimeError("Recording is already in progress.")

        audio_device = _find_input_device(device_id)
        audio_format = _select_audio_format(
            audio_device=audio_device,
            preferred_sample_rate=self.sample_rate,
            preferred_channels=self.channels,
        )
        audio_source = QAudioSource(audio_device, audio_format)
        io_device = audio_source.start()
        if io_device is None:
            raise RuntimeError("Qt failed to start microphone capture.")

        try:
            session_path, wav_file = _open_recording_file(audio_format)
        except Exception:
            audio_source.stop()
            raise
        with self._buffer_lock:
            self._buffer.clear()
            self._captured_bytes = 0
            self._session_path = session_path
            self._wav_file = wav_file
        self._device_id = device_id
        self._audio_format = audio_format
        self._audio_source = audio_source
        self._io_device = io_device
        self._io_device.readyRead.connect(self._pull_audio)

    def snapshot(self, max_duration_seconds: float | None = None) -> RecordedAudio:
        if self._audio_source is None or self._io_device is None or self._audio_format is None:
            raise RuntimeError("Recording has not started.")

        self._pull_audio()
        with self._buffer_lock:
            payload = _select_snapshot_payload(
                payload=bytes(self._buffer),
                audio_format=self._audio_format,
                max_duration_seconds=max_duration_seconds,
            )

        return _recorded_audio_from_payload(payload, self._audio_format)

    def has_voice_activity(
        self,
        max_duration_seconds: float | None = None,
        min_mean_abs: int = 120,
        min_peak_abs: int = 800,
    ) -> bool:
        if self._audio_source is None or self._io_device is None or self._audio_format is None:
            raise RuntimeError("Recording has not started.")

        self._pull_audio()
        with self._buffer_lock:
            payload = _select_snapshot_payload(
                payload=bytes(self._buffer),
                audio_format=self._audio_format,
                max_duration_seconds=max_duration_seconds,
            )

        return _has_voice_activity(
            payload=payload,
            audio_format=self._audio_format,
            min_mean_abs=min_mean_abs,
            min_peak_abs=min_peak_abs,
        )

    def stop(self) -> RecordedAudio:
        if self._audio_source is None or self._io_device is None or self._audio_format is None:
            raise RuntimeError("Recording has not started.")

        self._pull_audio()
        audio_source = self._audio_source
        io_device = self._io_device
        audio_format = self._audio_format

        self._audio_source = None
        self._io_device = None
        self._audio_format = None

        try:
            io_device.readyRead.disconnect(self._pull_audio)
        except Exception:
            pass

        audio_source.stop()
        self._pull_audio()

        with self._buffer_lock:
            session_path = self._session_path
            wav_file = self._wav_file
            captured_bytes = self._captured_bytes
            self._buffer.clear()
            self._session_path = None
            self._wav_file = None
            self._captured_bytes = 0

        if wav_file is not None:
            wav_file.close()
        if session_path is None:
            raise RuntimeError("Recording output path was not initialized.")

        if captured_bytes <= 0:
            _unlink_file(session_path)
            raise RuntimeError("No microphone audio was captured.")

        return _recorded_audio_from_file(
            path=session_path,
            audio_format=audio_format,
            captured_bytes=captured_bytes,
        )

    def _pull_audio(self) -> None:
        if self._io_device is None:
            return

        chunk = self._io_device.readAll()
        if chunk:
            payload = bytes(chunk)
            with self._buffer_lock:
                wav_file = self._wav_file
                if wav_file is not None:
                    wav_file.writeframesraw(payload)
                self._captured_bytes += len(payload)
                self._buffer.extend(payload)
                _trim_live_buffer(
                    buffer=self._buffer,
                    audio_format=self._audio_format,
                    max_duration_seconds=self.rolling_buffer_seconds,
                )


def list_input_devices() -> list[InputDevice]:
    input_devices: list[InputDevice] = []

    for device in QMediaDevices.audioInputs():
        input_devices.append(
            InputDevice(
                device_id=_device_key(device),
                name=device.description(),
                is_default=device.isDefault(),
            )
        )

    return input_devices


def _device_key(device: QAudioDevice) -> str:
    key = bytes(device.id()).decode("utf-8", errors="ignore").strip()
    return key or device.description()


def _find_input_device(device_id: str | None) -> QAudioDevice:
    devices = QMediaDevices.audioInputs()
    if not devices:
        raise RuntimeError("No audio input device was found.")

    if device_id:
        for device in devices:
            if _device_key(device) == device_id:
                return device
        raise RuntimeError(f"Audio input device {device_id!r} is no longer available.")

    return QMediaDevices.defaultAudioInput()


def _select_audio_format(
    audio_device: QAudioDevice,
    preferred_sample_rate: int,
    preferred_channels: int,
) -> QAudioFormat:
    preferred = audio_device.preferredFormat()
    sample_rates = _unique_ints(
        [
            preferred_sample_rate,
            preferred.sampleRate(),
            48000,
            44100,
            32000,
            16000,
        ]
    )
    channel_counts = _unique_ints(
        [
            preferred_channels,
            preferred.channelCount(),
            1,
            2,
        ]
    )

    for sample_rate in sample_rates:
        for channel_count in channel_counts:
            audio_format = QAudioFormat()
            audio_format.setSampleRate(sample_rate)
            audio_format.setChannelCount(channel_count)
            audio_format.setSampleFormat(QAudioFormat.SampleFormat.Int16)
            if audio_device.isFormatSupported(audio_format):
                return audio_format

    if preferred.sampleFormat() == QAudioFormat.SampleFormat.Int16:
        return preferred

    raise RuntimeError(
        f"Could not find an Int16 recording format for {audio_device.description()}."
    )


def _unique_ints(values: list[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        if value <= 0 or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _bytes_per_sample(audio_format: QAudioFormat) -> int:
    sample_format = audio_format.sampleFormat()
    if sample_format == QAudioFormat.SampleFormat.UInt8:
        return 1
    if sample_format == QAudioFormat.SampleFormat.Int16:
        return 2
    if sample_format in {QAudioFormat.SampleFormat.Int32, QAudioFormat.SampleFormat.Float}:
        return 4
    raise RuntimeError(f"Unsupported audio sample format: {sample_format}")


def _recorded_audio_from_payload(payload: bytes, audio_format: QAudioFormat) -> RecordedAudio:
    if not payload:
        raise RuntimeError("No microphone audio was captured.")

    bytes_per_sample = _bytes_per_sample(audio_format)
    frame_size = max(1, bytes_per_sample * audio_format.channelCount())
    duration_seconds = len(payload) / float(audio_format.sampleRate() * frame_size)
    if duration_seconds <= 0:
        raise RuntimeError("Recorded audio is empty.")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        path = Path(handle.name)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(audio_format.channelCount())
        wav_file.setsampwidth(bytes_per_sample)
        wav_file.setframerate(audio_format.sampleRate())
        wav_file.writeframes(payload)

    return RecordedAudio(
        path=path,
        sample_rate=audio_format.sampleRate(),
        duration_seconds=duration_seconds,
    )


def _recorded_audio_from_file(
    path: Path,
    audio_format: QAudioFormat,
    captured_bytes: int,
) -> RecordedAudio:
    bytes_per_sample = _bytes_per_sample(audio_format)
    frame_size = max(1, bytes_per_sample * audio_format.channelCount())
    duration_seconds = captured_bytes / float(audio_format.sampleRate() * frame_size)
    if duration_seconds <= 0:
        _unlink_file(path)
        raise RuntimeError("Recorded audio is empty.")

    return RecordedAudio(
        path=path,
        sample_rate=audio_format.sampleRate(),
        duration_seconds=duration_seconds,
    )


def _open_recording_file(audio_format: QAudioFormat) -> tuple[Path, wave.Wave_write]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        path = Path(handle.name)

    wav_file = wave.open(str(path), "wb")
    wav_file.setnchannels(audio_format.channelCount())
    wav_file.setsampwidth(_bytes_per_sample(audio_format))
    wav_file.setframerate(audio_format.sampleRate())
    return path, wav_file


def _select_snapshot_payload(
    payload: bytes,
    audio_format: QAudioFormat,
    max_duration_seconds: float | None,
) -> bytes:
    if max_duration_seconds is None:
        return payload

    bytes_per_sample = _bytes_per_sample(audio_format)
    frame_size = max(1, bytes_per_sample * audio_format.channelCount())
    max_frames = int(audio_format.sampleRate() * max_duration_seconds)
    if max_frames <= 0:
        return payload

    max_bytes = max_frames * frame_size
    if len(payload) <= max_bytes:
        return payload

    trimmed = payload[-max_bytes:]
    remainder = len(trimmed) % frame_size
    if remainder:
        trimmed = trimmed[remainder:]
    return trimmed


def _trim_live_buffer(
    buffer: bytearray,
    audio_format: QAudioFormat | None,
    max_duration_seconds: float | None,
) -> None:
    if audio_format is None or max_duration_seconds is None:
        return

    trimmed = _select_snapshot_payload(bytes(buffer), audio_format, max_duration_seconds)
    if len(trimmed) == len(buffer):
        return

    buffer.clear()
    buffer.extend(trimmed)


def _has_voice_activity(
    payload: bytes,
    audio_format: QAudioFormat,
    min_mean_abs: int,
    min_peak_abs: int,
) -> bool:
    if not payload or audio_format.sampleFormat() != QAudioFormat.SampleFormat.Int16:
        return bool(payload)

    sample_width = _bytes_per_sample(audio_format)
    usable_bytes = len(payload) - (len(payload) % sample_width)
    if usable_bytes <= 0:
        return False

    samples = memoryview(payload[:usable_bytes]).cast("h")
    if not samples:
        return False

    step = max(1, len(samples) // 4000)
    total_abs = 0
    count = 0
    peak_abs = 0
    for index in range(0, len(samples), step):
        magnitude = abs(int(samples[index]))
        total_abs += magnitude
        count += 1
        if magnitude > peak_abs:
            peak_abs = magnitude

    mean_abs = total_abs / max(1, count)
    return mean_abs >= min_mean_abs and peak_abs >= min_peak_abs


def _unlink_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass
