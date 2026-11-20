"""Screening behavior and retrieval coverage, including counterexamples."""

from datetime import date

import pytest
from fastmcp import Client

from mcp_server_n2ia.app import mcp
from mcp_server_n2ia.services import overlap as module
from mcp_server_n2ia.tools.contract_overlap import Award, detect_overlap


def award(key, agency="Agency A", **updates):
    return Award(
        **{
            "source_award_key": key,
            "awarding_agency": agency,
            "description": "enterprise cloud hosting infrastructure monitoring",
            "performance_start_date": "2024-01-01",
            "performance_end_date": "2025-12-31",
            "psc": "D302",
            **updates,
        }
    )


def test_match_evidence_and_duplicate_removal():
    a, b = award("a"), award("b", "Agency B")
    result = detect_overlap([a, a, b])
    assert result["unique_awards"] == 2
    pair = result["candidates"][0]
    assert pair["text_similarity"] == 1
    assert pair["matching_classifications"] == ["psc"]
    assert pair["overlap_days"] == 731
    assert "do not prove" in result["limitations"]


@pytest.mark.parametrize(
    "updates",
    [
        {"performance_start_date": "2026-01-01", "performance_end_date": "2027-01-01"},
        {"performance_end_date": None},
        {"performance_end_date": "invalid"},
        {"performance_end_date": "2020-01-01"},
        {"description": "construction bridge repair concrete asphalt"},
        {"description": "support services contract"},
    ],
)
def test_nonmatches_and_missing_evidence(updates):
    assert (
        detect_overlap([award("a"), award("b", "Agency B", **updates)])[
            "candidate_count"
        ]
        == 0
    )


def test_same_agency_is_optional():
    awards = [award("a"), award("b")]
    assert detect_overlap(awards)["candidate_count"] == 0
    assert detect_overlap(awards, cross_agency_only=False)["candidate_count"] == 1


def test_inclusive_overlap_and_unknown_classification():
    a = award("a", performance_end_date="2024-01-01", psc=None)
    b = award("b", "Agency B", psc=None)
    pair = detect_overlap([a, b])["candidates"][0]
    assert pair["overlap_days"] == 1
    assert pair["matching_classifications"] == []


def test_vendor_uei_and_candidate_truncation():
    awards = [
        award(str(i), f"Agency {i}", recipient_uei="ONE", contractor=str(i))
        for i in range(3)
    ]
    result = detect_overlap(awards, max_candidates=1)
    assert result["candidate_count"] == 3
    assert len(result["candidates"]) == 1
    assert result["candidates_truncated"]
    assert result["sample_fragmentation"][0]["identified_vendor_count"] == 1


async def test_bounded_retrieval_partial_failure_and_dedup(monkeypatch):
    calls = []

    async def fake(start, end, **kwargs):
        calls.append(kwargs)
        if kwargs["agency"] == "broken":
            return {"error": "upstream_unavailable"}
        return {"items": [{"source_award_key": "one"}], "has_more": True}

    monkeypatch.setattr(module, "search_contract_awards", fake)
    result = await module.retrieve_contracts(
        date(2023, 10, 1),
        date(2026, 9, 30),
        agencies=["A", "broken"],
        pages_per_agency=2,
    )
    assert len(calls) == 3
    assert result["count"] == 1
    assert not result["complete"]
    assert result["coverage"][0]["has_more"]
    assert "error" in result["coverage"][1]


async def test_retrieval_stops_at_final_page(monkeypatch):
    async def fake(*args, **kwargs):
        assert kwargs["page"] == 1
        return {"items": [], "has_more": False}

    monkeypatch.setattr(module, "search_contract_awards", fake)
    result = await module.retrieve_contracts(
        date(2023, 10, 1), date(2026, 9, 30), agencies=["A"], pages_per_agency=2
    )
    assert result["complete"]


async def test_detector_through_mcp_and_bounds():
    from fastmcp.exceptions import ToolError

    async with Client(mcp) as client:
        result = await client.call_tool(
            "find_similar_contracts",
            {"awards": [award("a").model_dump(), award("b", "Agency B").model_dump()]},
        )
        assert result.data["candidate_count"] == 1
        with pytest.raises(ToolError):
            await client.call_tool(
                "retrieve_contracts",
                {
                    "start_date": "2023-10-01",
                    "end_date": "2026-09-30",
                    "pages_per_agency": 200,
                },
            )
