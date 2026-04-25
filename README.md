# Talk2Text

Talk2Text is a local Python app for microphone transcription.
It records audio with Qt, transcribes it locally with `faster-whisper`, and can optionally send the finished transcript to a local Ollama model for cleanup.

## Status

This project is an Ubuntu-focused MVP.

What works today:

- microphone recording through Qt Multimedia
- local Whisper transcription
- optional live transcription
- manual Ollama transcript polish
- session history inside the app

What is still rough:

- no persistent settings/history storage yet
- no global push-to-talk yet
- limited automated coverage around GUI behavior

## Architecture

```text
microphone -> Qt audio capture -> faster-whisper -> transcript
                                             \
                                              -> optional Ollama polish
```

## Requirements

- `uv`
- local Ollama installation if you want transcript polish

The project targets Python `3.12`. `uv` can use an existing Python or manage Python for you.

For GPU acceleration, a working NVIDIA CUDA 12 + cuDNN 9 stack is recommended.
The app can fall back to CPU transcription.

## Quick Start

```bash
uv sync
uv run talk2text
```

Alternative launch command:

```bash
uv run python -m talk2text
```

## Install uv

Install `uv` with the official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Other installation methods are available in the `uv` docs.

## Default Behavior

- Whisper model: `turbo`
- Ollama model: `qwen3:8b`
- Live transcription: disabled by default
- Ollama polish: disabled by default

## Environment Variables

```bash
export TALK2TEXT_WHISPER_MODEL=large-v3
export TALK2TEXT_OLLAMA_MODEL=qwen3:8b
export TALK2TEXT_OLLAMA_BASE_URL=http://localhost:11434
export TALK2TEXT_LANGUAGE=
export TALK2TEXT_SAMPLE_RATE=16000
export TALK2TEXT_ENHANCE_WITH_OLLAMA=0
export TALK2TEXT_LIVE_TRANSCRIPTION=0
```

Leave `TALK2TEXT_LANGUAGE` empty for language auto-detection.

## Development

Sync the project environment:

```bash
uv sync
```

Run checks:

```bash
uv run python -m compileall src tests
uv run python -m unittest discover -s tests
uv run ruff check src tests
```

When dependencies change, refresh the lockfile with:

```bash
uv lock
```

## Project Layout

```text
src/talk2text/
  audio.py           Qt microphone capture
  transcription.py   Whisper integration
  ollama_client.py   Ollama cleanup client
  pipeline.py        transcription pipeline
  ui.py              desktop UI
```

## Troubleshooting

- First Whisper use may download the selected model.
- Ollama polish requires a local Ollama server running at `http://localhost:11434` unless you override it.
- If live transcription feels heavy, keep it disabled and use record-then-transcribe mode.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
