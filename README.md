# Talk2Text

Local Ubuntu desktop speech-to-text app with:

- microphone capture via `sounddevice`
- transcription via `faster-whisper`
- optional transcript cleanup and action extraction via local Ollama

## What this MVP does

1. Record from your microphone
2. Save the utterance to a temporary WAV file
3. Transcribe locally with Whisper
4. Optionally send the raw transcript to Ollama for:
   - punctuation and cleanup
   - one-line summary
   - extracted action items or commands

The flow is:

```text
microphone -> faster-whisper -> transcript -> Ollama
```

## System dependencies

Ubuntu packages:

```bash
sudo apt update
sudo apt install python3-venv portaudio19-dev
```

If you want GPU acceleration for `faster-whisper`, ensure your NVIDIA CUDA 12 and cuDNN 9 stack is working. CPU fallback is built into the app.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Run

```bash
.venv/bin/talk2text
```

Or:

```bash
PYTHONPATH=src python3 -m talk2text
```

## First-run behavior

- The first Whisper run will download the selected transcription model if it is not already cached.
- Ollama must already be running locally at `http://localhost:11434`.
- The UI will try to list local Ollama models on startup.

## Default choices

- Whisper model: `turbo`
- Ollama model: `qwen3:8b`
- Sample rate: `16000`

## Environment overrides

```bash
export TALK2TEXT_WHISPER_MODEL=large-v3
export TALK2TEXT_OLLAMA_MODEL=qwen3:8b
export TALK2TEXT_OLLAMA_BASE_URL=http://localhost:11434
export TALK2TEXT_LANGUAGE=
export TALK2TEXT_SAMPLE_RATE=16000
export TALK2TEXT_ENHANCE_WITH_OLLAMA=1
```

Leave `TALK2TEXT_LANGUAGE` empty to auto-detect.

## Notes

- Recording uses the default input device unless you pick another one in the UI.
- Transcription uses `vad_filter=True` to trim silence.
- Ollama cleanup is optional. If it fails, the raw transcription is still shown.

