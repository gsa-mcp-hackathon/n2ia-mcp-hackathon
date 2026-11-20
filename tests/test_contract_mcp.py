"""Verify domain tools, resources and prompts through the MCP protocol."""

import sys
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from mcp_server_n2ia.app import mcp
from tests.test_contract_overlap import award


async def test_domain_capabilities_discovered_and_usable():
    async with Client(mcp) as client:
        tools = {t.name: t for t in await client.list_tools()}
        for name in (
            "search_contract_awards",
            "retrieve_contracts",
            "find_similar_contracts",
            "retrieve_service_category",
            "enrich_contract_context",
        ):
            assert tools[name].annotations.read_only_hint
            assert tools[name].annotations.idempotent_hint
            assert tools[name].input_schema["type"] == "object"
        # Keep Satish's existing integration registered alongside the new tools.
        assert "test_sam_api" in tools
        resources = await client.list_resources()
        assert any(
            str(r.uri) == "contracts://reference/data-dictionary" for r in resources
        )
        content = await client.read_resource("contracts://reference/data-dictionary")
        assert "not probability" in content[0].text
        prompt = await client.get_prompt(
            "review_contract_overlap", {"category": "it_support"}
        )
        assert "retrieve_service_category" in prompt.messages[0].content.text
        assert "it_support" in prompt.messages[0].content.text


async def test_server_process_over_stdio():
    # Launch the real entry point, independent of the in-memory server instance.
    root = Path(__file__).resolve().parents[1]
    transport = StdioTransport(
        command=sys.executable, args=[str(root / "main.py")], cwd=str(root)
    )
    async with Client(transport) as client:
        result = await client.call_tool(
            "find_similar_contracts",
            {"awards": [award("A").model_dump(), award("B", "Agency B").model_dump()]},
        )
        assert result.data["candidate_count"] == 1
        assert "Screening candidates" in result.data["limitations"]


async def test_sam_tool_reports_missing_key_without_network(monkeypatch):
    monkeypatch.delenv("SAM_API_KEY", raising=False)
    async with Client(mcp) as client:
        result = await client.call_tool("test_sam_api", {})
        assert result.data == {"error": "SAM_API_KEY environment variable not found"}
