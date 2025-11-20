# Todo UI (NiceGUI + vikunja-mcp)

A NiceGUI-based web UI that lets you log into a Vikunja instance, browse tasks, and ask an OpenRouter-hosted LLM to orchestrate work through the [`vikunja-mcp`](https://github.com/maccam912/vikunja-mcp) server. The agent can break down multi-step requests (for example, designing and building a coat rack) and wire up task dependencies via Vikunja relations.

## Prerequisites
- Python 3.13
- [`uv`](https://github.com/astral-sh/uv) (used for all installs and runtime)
- A Vikunja base URL (usually ends with `/api/v1`) and an API token
- OpenRouter API key in `OPENROUTER_API_KEY` (and optionally `OPENROUTER_MODEL`, `OPENROUTER_SITE_URL`, `OPENROUTER_APP_NAME`)

## Setup
```bash
uv sync
```

## Run the UI
```bash
uv run main.py
```
Open the printed URL, enter your Vikunja base URL and token, then use the agent panel to describe what you want done. The app spawns `vikunja-mcp` via `uv run` under the hood so the LLM can call tools such as `create_task`, `add_relation`, and `complete_task`.

## Linting, formatting, and tests
```bash
uv run ruff format .
uv run ruff check .
uv run -m pytest
```

## Docker
Build and run locally:
```bash
docker build -t todo-ui .
docker run -p 8080:8080 --env-file .env todo-ui
```
The container defaults to `HOST=0.0.0.0` and `PORT=8080`. Provide your Vikunja credentials via env or through the UI.

## Behavior notes
- Tasks are scoped to the selected project/list. Change the selection to refresh the table.
- For dependencies, the agent instructs Vikunja to set `relation_kind="blocked"` on the dependent task (and `relation_kind="blocking"` if you want the inverse).
- Credentials stay in your browser session; API calls go directly from the server to your Vikunja instance.

## Development
- Edit in-place and run locally with `uv run main.py`.
- The project depends on `nicegui`, `openai` (pointed at OpenRouter), `mcp`, and `vikunja-mcp`. Use `uv add ...` to adjust dependencies.
