"""Contract search behavior verified through an in-memory MCP client."""

import json

import httpx
import pytest
from fastmcp import Client

from mcp_server_n2ia.app import mcp

PARAMS = {"start_date": "2024-10-01", "end_date": "2025-09-30"}


def mock_api(monkeypatch, handler):
    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )


async def call(**kwargs):
    async with Client(mcp) as client:
        return await client.call_tool("search_contract_awards", {**PARAMS, **kwargs})


async def test_filters_mapping_and_pagination(monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        assert body["filters"]["award_type_codes"] == ["A", "B", "C", "D"]
        assert body["filters"]["agencies"][0]["name"] == "Department of Defense"
        assert body["filters"]["recipient_search_text"] == ["ACME"]
        assert body["filters"]["keywords"] == ["software"]
        assert body["page"] == 2
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "Award ID": "123",
                        "Recipient Name": "ACME",
                        "Award Amount": 42.5,
                        "generated_internal_id": "CONT_A",
                    }
                ],
                "page_metadata": {"hasNext": True},
            },
        )

    mock_api(monkeypatch, handler)
    result = (
        await call(
            agency="Department of Defense",
            contractor="ACME",
            keyword="software",
            page=2,
        )
    ).data
    assert result["items"][0]["award_obligations_usd"] == 42.5
    assert result["items"][0]["source_url"].endswith("/award/CONT_A")
    assert result["next_page"] == 3
    assert result["count"] == 1


async def test_empty_results(monkeypatch):
    mock_api(monkeypatch, lambda request: httpx.Response(200, json={"results": []}))
    result = (await call()).data
    assert result["items"] == []
    assert result["next_page"] is None


async def test_reversed_dates_do_not_call_api(monkeypatch):
    def unexpected(request):
        pytest.fail("Invalid dates must not call USAspending")

    mock_api(monkeypatch, unexpected)
    assert (await call(end_date="2020-01-01")).data["error"] == "invalid_dates"


@pytest.mark.parametrize("status", [400, 429, 503])
async def test_http_errors(monkeypatch, status):
    mock_api(monkeypatch, lambda request: httpx.Response(status))
    result = (await call()).data
    assert result["status_code"] == status
    assert result["error"] == "upstream_http_error"


async def test_timeout(monkeypatch):
    def handler(request):
        raise httpx.ReadTimeout("timeout", request=request)

    mock_api(monkeypatch, handler)
    assert (await call()).data["error"] == "upstream_unavailable"


async def test_invalid_json(monkeypatch):
    mock_api(monkeypatch, lambda request: httpx.Response(200, text="not JSON"))
    assert (await call()).data["error"] == "invalid_upstream_response"


async def test_limit_is_validated_by_mcp():
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError):
        await call(limit=101)


async def test_classification_objects_and_filters(monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        assert body["filters"]["psc_codes"] == ["D310"]
        assert body["filters"]["naics_codes"] == ["541512"]
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "PSC": {"code": "D310", "description": "Cyber security"},
                        "NAICS": {"code": "541512"},
                        "Recipient UEI": "ABC",
                    }
                ]
            },
        )

    mock_api(monkeypatch, handler)
    row = (await call(psc_codes=["D310"], naics_codes=["541512"])).data["items"][0]
    assert row["psc"] == "D310"
    assert row["naics"] == "541512"
    assert row["recipient_uei"] == "ABC"
