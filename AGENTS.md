# Agent guidance

- **Use `uv` for everything**: `uv sync` to install, `uv run main.py` to start the UI, and `uv run vikunja-mcp --base-url ... --token ...` if you need to launch the MCP server manually. Avoid `pip` or `python -m venv`.
- **LLM wiring**: models are pulled from OpenRouter via `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` (default `anthropic/claude-3.5-sonnet`), and optional `OPENROUTER_SITE_URL`/`OPENROUTER_APP_NAME` headers. The UI uses `AsyncOpenAI` against `https://openrouter.ai/api/v1`.
- **Vikunja access**: users must supply `VIKUNJA_BASE_URL`-style URLs and API tokens at login. The app starts `vikunja-mcp` with those credentials so tool calls are always routed through the MCP server.
- **Task orchestration expectations**: break goals into tasks with `create_task`, then apply dependencies using `add_relation` (`relation_kind="blocked"` on the dependent task; use `relation_kind="blocking"` for the inverse). Close the loop with `complete_task` or `update_task` as work progresses.
- **Surface area available today** (`vikunja-mcp`): `list_projects`, `find_tasks`, `task_details`, `create_task`, `update_task`, `complete_task`, `add_relation`, `remove_relation`, `add_comment`, `bulk_update`, plus resources for status/projects/task details. Add new tools via the MCP server if you need more coverage.
- **Working test project**: created a throwaway project in the linked Vikunja instance for experiments — id `2`, title `MCP Testbed YYYY-MM-DD HH:MM:SS` (latest: `MCP Testbed 2025-11-20 12:34:20`).
