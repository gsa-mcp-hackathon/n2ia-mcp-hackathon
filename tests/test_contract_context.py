"""Category coverage, parent context and matching counterexamples."""

import httpx
import pytest
from fastmcp import Client

from mcp_server_n2ia.app import mcp
from mcp_server_n2ia.services import context as module
from mcp_server_n2ia.tools.contract_overlap import detect_overlap
from tests.test_contract_overlap import award


@pytest.mark.parametrize("category", ["cybersecurity", "it_support"])
async def test_category_terms_deduplicate_and_preserve_coverage(monkeypatch, category):
    calls = []

    async def retrieve(*args, **kwargs):
        calls.append(kwargs["keyword"])
        return {
            "items": [award("same").model_dump()],
            "coverage": [{"agency": "A", "complete": False}],
        }

    monkeypatch.setattr(module, "retrieve_contracts", retrieve)
    async with Client(mcp) as client:
        result = (
            await client.call_tool(
                "retrieve_service_category",
                {
                    "category": category,
                    "start_date": "2023-10-01",
                    "end_date": "2026-09-30",
                },
            )
        ).data
    assert calls == module.CATEGORY_TERMS[category]
    assert result["count"] == 1
    assert result["items"][0]["matched_terms"] == calls
    assert len(result["coverage"]) == 3
    assert not result["complete"]


async def test_parent_context_success_missing_and_failure(monkeypatch):
    def handler(request):
        key = request.url.path.split("/")[-2]
        if key == "failure":
            return httpx.Response(503)
        if key == "unknown":
            return httpx.Response(200, json={"piid": "X"})
        return httpx.Response(
            200,
            json={
                "parent_award": (
                    {
                        "generated_unique_award_id": "CONT_IDV_PARENT_1",
                        "piid": "PARENT",
                        "agency_name": "GSA",
                        "idv_type_description": "GWAC",
                    }
                    if key == "child"
                    else None
                )
            },
        )

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )
    result = await module.enrich_contract_context(
        [award("child"), award("none"), award("failure"), award("unknown")]
    )
    rows = {a["source_award_key"]: a for a in result["items"]}
    assert rows["child"]["parent_award_key"] == "CONT_IDV_PARENT_1"
    assert rows["child"]["parent_source_url"].endswith("/CONT_IDV_PARENT_1")
    assert rows["none"]["context_status"] == "retrieved"
    assert rows["none"]["parent_award_key"] is None
    assert rows["failure"]["context_status"] == "unavailable"
    assert rows["unknown"]["context_status"] == "unavailable"
    assert result["unavailable_count"] == 2


def test_shared_parent_is_context_not_duplication_claim():
    result = detect_overlap(
        [
            award("a", parent_award_key="vehicle", context_status="retrieved"),
            award(
                "b", "Agency B", parent_award_key="vehicle", context_status="retrieved"
            ),
        ]
    )
    pair = result["candidates"][0]
    assert pair["shared_parent_vehicle"]
    assert "distinct task-order scopes" in pair["parent_context_note"]


def test_same_parent_does_not_make_unrelated_scopes_match():
    assert (
        detect_overlap(
            [
                award("a", parent_award_key="vehicle"),
                award(
                    "b",
                    "Agency B",
                    parent_award_key="vehicle",
                    description="desktop printer repair inventory shipping",
                ),
            ]
        )["candidate_count"]
        == 0
    )


def test_reordered_cybersecurity_scope_matches():
    pair = detect_overlap(
        [
            award(
                "a",
                description="enterprise cybersecurity incident detection monitoring",
            ),
            award(
                "b",
                "Agency B",
                description="monitoring incident detection enterprise cybersecurity",
            ),
        ]
    )["candidates"][0]
    assert pair["text_similarity"] == 1
    assert "unknown" in pair["parent_context_note"]


def test_shared_generic_it_words_do_not_match_distinct_work():
    result = detect_overlap(
        [
            award("a", description="IT support services desktop printer repair"),
            award(
                "b",
                "Agency B",
                description="IT support services incident malware detection",
            ),
        ]
    )
    assert result["candidate_count"] == 0
