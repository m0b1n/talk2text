# Contributing

## Setup

```bash
uv sync
```

## Run The App

```bash
uv run talk2text
```

## Run Checks

```bash
uv run python -m compileall src tests
uv run python -m unittest discover -s tests
uv run ruff check src tests
```

## Guidelines

- Keep desktop behavior portable across supported platforms.
- Prefer small, reviewable commits.
- Add or update tests when you change behavior.
- Do not commit build artifacts or local model caches.
- Keep Ollama optional and off the critical path unless the change explicitly targets it.

## Pull Requests

- Explain the user-visible change.
- Mention how you tested it.
- Include screenshots or a short recording for UI changes when practical.
