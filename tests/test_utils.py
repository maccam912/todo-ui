from types import SimpleNamespace

from mcp import types as mcp_types
from vikunja_mcp.models import Task, TaskRelation

from main import (
    format_datetime,
    mcp_tool_to_openai,
    result_to_text,
    summarize_relations,
    task_to_row,
)


def test_format_datetime_handles_none() -> None:
    assert format_datetime(None) == "—"


def test_summarize_relations_renders_ids() -> None:
    relation = TaskRelation(id=1, relation_kind="blocked", other_task_id=42)
    assert "blocked #42" in summarize_relations([relation])


def test_task_to_row_maps_fields() -> None:
    relation = TaskRelation(id=1, relation_kind="blocked", other_task_id=2)
    task = Task(id=5, title="Demo", related_tasks=[relation], done=False)
    row = task_to_row(task)
    assert row["status"] == "Open"
    assert "blocked #2" in row["relations"]


def test_result_to_text_prefers_structured_content() -> None:
    result = SimpleNamespace(structuredContent={"ok": True}, content=[])
    text = result_to_text(result)
    assert '"ok": true' in text


def test_result_to_text_falls_back_to_text_chunks() -> None:
    result = SimpleNamespace(structuredContent=None, content=[SimpleNamespace(text="chunk")])
    assert result_to_text(result) == "chunk"


def test_mcp_tool_to_openai_adapts_schema() -> None:
    tool = mcp_types.Tool(name="add_relation", inputSchema={"type": "object", "properties": {}})
    mapped = mcp_tool_to_openai(tool)
    assert mapped["function"]["name"] == "add_relation"
    assert mapped["function"]["parameters"]["type"] == "object"
