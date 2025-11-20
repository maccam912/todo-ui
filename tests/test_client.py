import pytest
import respx
from vikunja_mcp.client import VikunjaClient
from vikunja_mcp.config import VikunjaConfig


@pytest.mark.asyncio
async def test_list_projects_fetches_from_vikunja(respx_mock: respx.MockRouter) -> None:
    base_url = "https://vikunja.example/api/v1"
    respx_mock.get(f"{base_url}/projects").respond(200, json=[{"id": 1, "title": "Inbox"}])
    client = VikunjaClient(VikunjaConfig(base_url=base_url, token="token"))  # noqa: S106
    projects = await client.list_projects()
    assert projects == [{"id": 1, "title": "Inbox"}]
    await client.aclose()
