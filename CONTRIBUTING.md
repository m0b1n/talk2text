# Contributing

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Run The App

```bash
.venv/bin/talk2text
```

## Run Checks

```bash
python3 -m compileall src tests
.venv/bin/python -m unittest discover -s tests
.venv/bin/ruff check src tests
```

## Guidelines

- Keep Ubuntu desktop behavior working first. This project is Linux-first.
- Prefer small, reviewable commits.
- Add or update tests when you change behavior.
- Do not commit build artifacts or local model caches.
- Keep Ollama optional and off the critical path unless the change explicitly targets it.

## Pull Requests

- Explain the user-visible change.
- Mention how you tested it.
- Include screenshots or a short recording for UI changes when practical.
