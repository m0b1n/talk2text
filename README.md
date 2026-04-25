# Talk2Text

Talk2Text is a local Ubuntu desktop app for microphone transcription.
It records audio with Qt, transcribes it locally with `faster-whisper`, and can optionally send the finished transcript to a local Ollama model for cleanup.

## Status

This project is an Ubuntu-focused MVP.

What works today:

- microphone recording through Qt Multimedia
- local Whisper transcription
- optional live transcription
- manual Ollama transcript polish
- session history inside the app
- Debian package build helper

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

- Ubuntu
- Python 3.12
- `python3-venv`
- local Ollama installation if you want transcript polish

For GPU acceleration, a working NVIDIA CUDA 12 + cuDNN 9 stack is recommended.
The app can fall back to CPU transcription.

## Quick Start

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/talk2text
```

Alternative launch command:

```bash
PYTHONPATH=src python3 -m talk2text
```

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

## Packaging

The repo includes a Debian packaging path based on `pyside6-deploy`.

```bash
sudo apt install patchelf
chmod +x build-deb.sh
./build-deb.sh
sudo apt install ./talk2text_0.1.0-1_amd64.deb
```

More detail is documented in [docs/deb-packaging.md](docs/deb-packaging.md).

## Development

Install the package in editable mode:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Run checks:

```bash
python3 -m compileall src tests
.venv/bin/python -m unittest discover -s tests
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
