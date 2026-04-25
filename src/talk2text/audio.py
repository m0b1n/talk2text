from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import wave

import numpy as np
import sounddevice as sd

from .models import InputDevice, RecordedAudio


class MicrophoneRecorder:
    def __init__(self, sample_rate: int = 16000, channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._device_id: int | None = None

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def start(self, device_id: int | None = None) -> None:
        if self._stream is not None:
            raise RuntimeError("Recording is already in progress.")

        self._frames.clear()
        self._device_id = device_id
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            device=device_id,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> RecordedAudio:
        if self._stream is None:
            raise RuntimeError("Recording has not started.")

        stream = self._stream
        self._stream = None

        stream.stop()
        stream.close()

        with self._lock:
            if not self._frames:
                raise RuntimeError("No microphone audio was captured.")
            audio = np.concatenate(self._frames, axis=0)

        if audio.ndim == 2 and audio.shape[1] == 1:
            audio = audio.reshape(-1)

        duration_seconds = float(audio.shape[0]) / float(self.sample_rate)
        if duration_seconds <= 0:
            raise RuntimeError("Recorded audio is empty.")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            path = Path(handle.name)

        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio.tobytes())

        return RecordedAudio(
            path=path,
            sample_rate=self.sample_rate,
            duration_seconds=duration_seconds,
        )

    def _callback(self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        del frames, time_info
        if status:
            # Preserve the audio frame even when PortAudio reports a recoverable status.
            pass

        with self._lock:
            self._frames.append(indata.copy())


def list_input_devices() -> list[InputDevice]:
    devices = sd.query_devices()
    default_input, _default_output = sd.default.device
    input_devices: list[InputDevice] = []

    for index, device in enumerate(devices):
        max_input_channels = int(device.get("max_input_channels", 0))
        if max_input_channels < 1:
            continue

        input_devices.append(
            InputDevice(
                device_id=index,
                name=str(device["name"]),
                is_default=index == default_input,
            )
        )

    return input_devices

