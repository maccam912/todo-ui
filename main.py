from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from mcp import types as mcp_types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from nicegui import app, ui
from openai import AsyncOpenAI
from vikunja_mcp.client import VikunjaClient
from vikunja_mcp.config import VikunjaConfig
from vikunja_mcp.errors import VikunjaError
from vikunja_mcp.models import Task, TaskRelation

load_dotenv()


# Basic OpenRouter configuration pulled from environment to avoid hard-coding anything
@dataclass
class OpenRouterSettings:
    api_key: str | None = os.getenv("OPENROUTER_API_KEY")
    model: str = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
    base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    referer: str | None = os.getenv("OPENROUTER_SITE_URL")
    app_name: str | None = os.getenv("OPENROUTER_APP_NAME")
    max_tool_rounds: int = int(os.getenv("OPENROUTER_MAX_TOOL_ROUNDS", "6"))


settings = OpenRouterSettings()


@dataclass
class SessionState:
    config: VikunjaConfig | None = None
    client: VikunjaClient | None = None
    projects: list[dict[str, Any]] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    selected_project_id: int | None = None
    agent_history: list[dict[str, str]] = field(default_factory=list)
    busy: bool = False

    async def close(self) -> None:
        if self.client:
            await self.client.aclose()
        self.client = None


session_states: dict[int, SessionState] = {}


def get_state() -> SessionState:
    client_id = ui.context.client.id if ui.context.client else 0
    if client_id not in session_states:
        session_states[client_id] = SessionState()
    return session_states[client_id]


def cleanup_state(client_id: int) -> None:
    session_states.pop(client_id, None)


def format_datetime(value: datetime | None) -> str:
    if not value:
        return "—"
    return value.strftime("%Y-%m-%d %H:%M")


def summarize_relations(relations: list[TaskRelation]) -> str:
    if not relations:
        return "—"
    parts = [f"{rel.relation_kind} #{rel.other_task_id}" for rel in relations]
    return ", ".join(parts)


def task_to_row(task: Task) -> dict[str, Any]:
    return {
        "id": task.id,
        "title": task.title,
        "status": "Done" if task.done else "Open",
        "due": format_datetime(task.due_date),
        "priority": task.priority or "—",
        "relations": summarize_relations(task.related_tasks),
    }


def mcp_tool_to_openai(tool: mcp_types.Tool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or tool.title or "",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


def result_to_text(result: mcp_types.CallToolResult) -> str:
    """Normalize an MCP tool result into a string payload for chat responses."""
    if result.structuredContent is not None:
        try:
            return json.dumps(result.structuredContent, indent=2)
        except Exception:
            return str(result.structuredContent)
    parts: list[str] = []
    for item in result.content:
        if hasattr(item, "text"):
            parts.append(item.text)
        elif hasattr(item, "data"):
            parts.append(str(item.data))
        else:
            parts.append(str(item))
    return "\n".join(parts) if parts else "(empty result)"


SYSTEM_PROMPT = """You are an automation agent that manages tasks in Vikunja using the provided tools.
- Break user requests into clear tasks; prefer using the selected project/list if provided.
- When creating dependencies, call add_relation on the dependent task with relation_kind="blocked"
  where other_task_id is the prerequisite. Use relation_kind="blocking" to show a task blocks another.
- Keep responses concise and mention the task ids you touched."""


class VikunjaMcpAgent:
    def __init__(self, router_settings: OpenRouterSettings) -> None:
        if not router_settings.api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY for LLM access.")
        headers = {}
        if router_settings.referer:
            headers["HTTP-Referer"] = router_settings.referer
        if router_settings.app_name:
            headers["X-Title"] = router_settings.app_name

        self.client = AsyncOpenAI(
            base_url=router_settings.base_url,
            api_key=router_settings.api_key,
            default_headers=headers or None,
        )
        self.model = router_settings.model
        self.max_rounds = router_settings.max_tool_rounds

    async def run(
        self,
        instruction: str,
        config: VikunjaConfig,
        *,
        selected_project_id: int | None,
        projects: list[dict[str, Any]],
        known_tasks: list[Task],
    ) -> dict[str, Any]:
        logs: list[dict[str, str]] = []
        server_args = [
            "run",
            "vikunja-mcp",
            "--base-url",
            config.base_url,
            "--token",
            config.token,
        ]
        if not config.verify_ssl:
            server_args.append("--insecure")
        server = StdioServerParameters(command="uv", args=server_args)

        project_lines = [f"{p.get('id')}: {p.get('title', p.get('name', ''))}" for p in projects]
        context_lines = [
            f"Vikunja base URL: {config.base_url}",
            f"Selected project/list id: {selected_project_id or 'not set'}",
            "Projects:",
        ]
        if project_lines:
            context_lines.extend(project_lines)
        else:
            context_lines.append("(none loaded)")
        if known_tasks:
            for task in known_tasks[:8]:
                context_lines.append(
                    f"Task #{task.id}: {task.title} | status={'done' if task.done else 'open'}"
                )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": "\n".join(context_lines)},
            {"role": "user", "content": instruction},
        ]

        async with stdio_client(server) as (read_stream, write_stream):
            session = ClientSession(read_stream, write_stream)
            await session.initialize()
            tools = await session.list_tools()
            openai_tools = [mcp_tool_to_openai(tool) for tool in tools.tools]
            logs.append({"role": "system", "content": f"Connected MCP with {len(openai_tools)} tools."})

            for _ in range(self.max_rounds):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                )
                choice = response.choices[0].message
                if choice.tool_calls:
                    assistant_msg = {
                        "role": "assistant",
                        "content": choice.content or "",
                        "tool_calls": [],
                    }
                    for call in choice.tool_calls:
                        assistant_msg["tool_calls"].append(
                            {
                                "id": call.id,
                                "type": call.type,
                                "function": {
                                    "name": call.function.name,
                                    "arguments": call.function.arguments,
                                },
                            }
                        )
                    messages.append(assistant_msg)

                    for call in choice.tool_calls:
                        try:
                            args = json.loads(call.function.arguments or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        logs.append(
                            {
                                "role": "assistant",
                                "content": f"Calling {call.function.name} with {args}",
                            }
                        )
                        result = await session.call_tool(call.function.name, args)
                        content = result_to_text(result)
                        logs.append(
                            {
                                "role": "tool",
                                "content": f"{call.function.name} -> {content}",
                            }
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": content,
                            }
                        )
                    continue

                final_text = choice.content or "Done."
                logs.append({"role": "assistant", "content": final_text})
                return {"reply": final_text, "log": logs}

            logs.append(
                {
                    "role": "assistant",
                    "content": "Stopped after max tool iterations; partial results may apply.",
                }
            )
            return {"reply": "Finished early after tool loop.", "log": logs}


async def fetch_projects(state: SessionState) -> None:
    if not state.client:
        return
    state.projects = await state.client.list_projects()
    if state.projects and not state.selected_project_id:
        state.selected_project_id = state.projects[0]["id"]


async def fetch_tasks(state: SessionState) -> None:
    if not state.client:
        return
    params: dict[str, Any] = {}
    if state.selected_project_id:
        params["project_id"] = state.selected_project_id
    state.tasks = await state.client.list_tasks(**params)


def render_history(container: ui.column, state: SessionState) -> None:
    container.clear()
    for entry in state.agent_history:
        with container:
            bg = "bg-slate-800 text-white" if entry["role"] in {"assistant", "system"} else "bg-slate-100"
            ui.markdown(entry["content"]).classes(f"p-3 rounded-md text-sm {bg}")


async def handle_login(
    base_url: str,
    token: str,
    verify_ssl: bool,
    project_select: ui.select,
    task_table: ui.table,
    status_label: ui.label,
) -> None:
    state = get_state()
    await state.close()
    try:
        config = VikunjaConfig.from_options(base_url=base_url, token=token, verify_ssl=verify_ssl)
    except ValueError as exc:
        ui.notify(f"Missing config: {exc}", color="negative")
        return

    try:
        client = VikunjaClient(config)
        state.client = client
        state.config = config
        await fetch_projects(state)
        await fetch_tasks(state)
    except VikunjaError as exc:
        await state.close()
        ui.notify(f"Could not connect to Vikunja: {exc}", color="negative")
        return
    except Exception as exc:
        await state.close()
        ui.notify(f"Unexpected error: {exc}", color="negative")
        return

    project_select.options = {p["id"]: p.get("title", p.get("name", "")) for p in state.projects}
    project_select.update()
    task_table.rows = [task_to_row(t) for t in state.tasks]
    task_table.update()
    status_label.text = f"Connected to {config.base_url}"
    ui.notify("Signed in to Vikunja", color="positive")


async def handle_agent(
    instruction: str,
    history: ui.column,
    task_table: ui.table,
) -> None:
    state = get_state()
    if not state.config or not state.client:
        ui.notify("Login first to let the agent work with Vikunja.", color="warning")
        return
    if not instruction.strip():
        ui.notify("Enter something for the agent to do.", color="warning")
        return
    if not settings.api_key:
        ui.notify("Set OPENROUTER_API_KEY to use the agent.", color="warning")
        return

    state.busy = True
    state.agent_history.append({"role": "user", "content": instruction})
    render_history(history, state)
    try:
        agent = VikunjaMcpAgent(settings)
        result = await agent.run(
            instruction,
            state.config,
            selected_project_id=state.selected_project_id,
            projects=state.projects,
            known_tasks=state.tasks,
        )
        state.agent_history.extend(result["log"])
        render_history(history, state)
        await fetch_tasks(state)
        task_table.rows = [task_to_row(t) for t in state.tasks]
        task_table.update()
    except Exception as exc:
        ui.notify(f"Agent failed: {exc}", color="negative")
    finally:
        state.busy = False


def build_ui() -> None:
    ui.add_head_html(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link
            href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&display=swap"
            rel="stylesheet"
        >
        <style>
            :root { --accent: #3D5AFE; }
            body { background: radial-gradient(circle at 10% 20%, rgba(61,90,254,0.15), transparent 25%),
                            radial-gradient(circle at 90% 10%, rgba(14,165,233,0.15), transparent 20%),
                            linear-gradient(180deg, #0b1120, #0f172a);
                   color: #e2e8f0; font-family: 'Space Grotesk', 'Segoe UI', sans-serif; }
            .card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); }
            .pill { background: rgba(61,90,254,0.12); color: #cbd5ff; padding: 4px 10px;
                    border-radius: 9999px; font-size: 12px; }
        </style>
        """
    )

    with ui.header().classes("justify-between items-center px-6 py-4 card"):
        ui.label("Vikunja LLM Pilot").classes("text-2xl font-semibold")
        ui.label("Build and orchestrate tasks with OpenRouter + MCP").classes("text-sm opacity-80")

    with ui.row().classes("w-full px-6 pt-4 gap-6 items-start"):
        # Left: connection + tasks
        with ui.column().classes("w-3/5 gap-4"):
            with ui.card().classes("card p-4 gap-3"):
                ui.label("Connect to Vikunja").classes("text-lg font-semibold")
                base_url_input = ui.input(
                    label="Vikunja base URL", placeholder="https://vikunja.example/api/v1"
                ).classes("w-full")
                token_input = ui.input(
                    label="API token", password=True, placeholder="Paste a personal access token"
                ).classes("w-full")
                verify_toggle = ui.switch("Verify TLS", value=True)
                status_label = ui.label("Not connected").classes("text-sm opacity-80")
                connect_button = ui.button(
                    "Sign in with uv + MCP",
                    on_click=lambda: handle_login(
                        base_url_input.value,
                        token_input.value,
                        verify_toggle.value,
                        project_select,
                        task_table,
                        status_label,
                    ),
                ).props("color=primary")
                connect_button.classes("w-full")
                ui.label("Credentials are kept in your session only.").classes("text-xs opacity-60")

            with ui.card().classes("card p-4 gap-3"):
                ui.label("Tasks").classes("text-lg font-semibold")
                project_select = ui.select(
                    {},
                    label="Project/List",
                    on_change=lambda e: on_project_change(e, task_table),
                ).classes("w-full")
                task_table = ui.table(
                    columns=[
                        {"name": "title", "label": "Title", "field": "title", "sortable": True},
                        {"name": "status", "label": "Status", "field": "status", "sortable": True},
                        {"name": "due", "label": "Due", "field": "due"},
                        {"name": "priority", "label": "Priority", "field": "priority"},
                        {"name": "relations", "label": "Relations", "field": "relations"},
                    ],
                    rows=[],
                    row_key="id",
                ).classes("w-full")

        # Right: agent
        with ui.column().classes("w-2/5 gap-4"):
            with ui.card().classes("card p-4 gap-3"):
                ui.label("Agent").classes("text-lg font-semibold")
                prompt_box = ui.textarea(
                    label="Describe what you need",
                    placeholder="Design and build a coat rack with dependencies...",
                    auto_resize=True,
                ).classes("w-full")
                history_column = ui.column().classes("gap-2 max-h-[460px] overflow-y-auto")
                ui.button(
                    "Run with LLM and vikunja-mcp",
                    on_click=lambda: handle_agent(prompt_box.value, history_column, task_table),
                ).props("color=secondary").classes("w-full")


async def on_project_change(event: Any, task_table: ui.table) -> None:
    state = get_state()
    try:
        state.selected_project_id = int(event.value)
    except Exception:
        return
    await fetch_tasks(state)
    task_table.rows = [task_to_row(t) for t in state.tasks]
    task_table.update()


@ui.page("/")
async def index_page() -> None:
    build_ui()


@app.on_disconnect
async def _cleanup(client_id: int) -> None:  # type: ignore[override]
    state = session_states.get(client_id)
    if state:
        await state.close()
    cleanup_state(client_id)


if __name__ == "__main__":
    server_host = os.getenv("HOST", "0.0.0.0")
    server_port = int(os.getenv("PORT", "8080"))
    ui.run(title="Vikunja LLM Pilot", host=server_host, port=server_port, reload=False)
