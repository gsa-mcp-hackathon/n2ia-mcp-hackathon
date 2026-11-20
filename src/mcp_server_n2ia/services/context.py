"""Service-category discovery and bounded award/parent context enrichment."""

import asyncio
from datetime import date
from typing import Annotated, Any, Literal
from urllib.parse import quote

import httpx
from pydantic import Field

from mcp_server_n2ia.services.overlap import AGENCIES, Award, retrieve_contracts

# These are discovery queries, not an authoritative classification of IT spend.
# Separate queries provide OR semantics and preserve coverage for every term.
CATEGORY_TERMS = {
    "cloud": ["cloud"],
    "cybersecurity": ["cybersecurity", "cyber security", "information security"],
    "it_support": ["IT support", "help desk", "desktop support"],
}


async def retrieve_service_category(
    category: Literal["cloud", "cybersecurity", "it_support"],
    start_date: date,
    end_date: date,
    page_size: Annotated[int, Field(ge=1, le=50)] = 20,
) -> dict[str, Any]:
    """Search one service category across GSA, DHS and Treasury, one page per term.

    Maximum 450 source rows; deduplication retains all matching category terms.
    Coverage is recorded per term/agency. This is discovery, not all category spend.
    """
    items, coverage = {}, []
    for term in CATEGORY_TERMS[category]:
        result = await retrieve_contracts(
            start_date, end_date, agencies=AGENCIES, keyword=term, page_size=page_size
        )
        if "error" in result:
            return result
        for entry in result["coverage"]:
            coverage.append({**entry, "search_term": term})
        for row in result["items"]:
            key = row["source_award_key"]
            if key not in items:
                items[key] = {**row, "service_category": category, "matched_terms": []}
            items[key]["matched_terms"].append(term)
    return {
        "category": category,
        "items": list(items.values()),
        "count": len(items),
        "coverage": coverage,
        "complete": all(c["complete"] for c in coverage),
        "query": {
            "start_date": str(start_date),
            "end_date": str(end_date),
            "terms": CATEGORY_TERMS[category],
            "agencies": AGENCIES,
        },
        "limitations": "Keyword discovery sample; not all service-category awards. "
        "No spending totals or savings are inferred.",
    }


async def enrich_contract_context(
    awards: Annotated[list[Award], Field(max_length=20)],
) -> dict[str, Any]:
    """Fetch USAspending award details for at most 20 shortlisted awards.

    Preserve unavailable versus no reported parent. Parent source keys are supplied
    by the API, never guessed from a PIID. No SAM.gov credentials are needed.
    """
    unique = {a.source_award_key: a for a in awards}
    semaphore = asyncio.Semaphore(3)
    async with httpx.AsyncClient(timeout=20) as client:

        async def fetch(award):
            row = award.model_dump()
            url = (
                "https://api.usaspending.gov/api/v2/awards/"
                + quote(award.source_award_key, safe="")
                + "/"
            )
            try:
                async with semaphore:
                    response = await client.get(url)
                    response.raise_for_status()
                    data = response.json()
                if not isinstance(data, dict) or "parent_award" not in data:
                    raise ValueError("Missing parent context")
                parent = data["parent_award"]
                if parent is not None and not isinstance(parent, dict):
                    raise ValueError("Malformed parent context")
                parent = parent or {}
                row.update(
                    {
                        "context_status": "retrieved",
                        "parent_award_key": parent.get("generated_unique_award_id"),
                        "parent_piid": parent.get("piid"),
                        "parent_agency": parent.get("agency_name"),
                        "parent_vehicle_type": parent.get("idv_type_description"),
                        "parent_source_url": (
                            "https://www.usaspending.gov/award/"
                            + quote(parent["generated_unique_award_id"], safe="")
                            if parent.get("generated_unique_award_id")
                            else None
                        ),
                        "context_source": url,
                    }
                )
            except (httpx.HTTPError, ValueError, TypeError, KeyError):
                row.update({"context_status": "unavailable", "context_source": url})
            return row

        rows = await asyncio.gather(*(fetch(a) for a in unique.values()))
    return {
        "items": rows,
        "count": len(rows),
        "unavailable_count": sum(a["context_status"] == "unavailable" for a in rows),
    }
