"""Read-only USAspending contract search for backlog item 20282."""

from datetime import date, datetime, timezone
from typing import Annotated, Any
from urllib.parse import quote

import httpx
from pydantic import Field

ENDPOINT = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
FIELDS = [
    "Award ID",
    "Recipient Name",
    "Award Amount",
    "Awarding Agency",
    "Start Date",
    "End Date",
    "Description",
    "generated_internal_id",
    "Last Modified Date",
    "PSC",
    "NAICS",
    "Recipient UEI",
]


def classification_code(value: Any) -> str | None:
    """The live API returns classification objects; older payloads use strings."""
    if isinstance(value, dict):
        value = value.get("code")
    return str(value).strip() or None if value is not None else None


async def search_contract_awards(
    start_date: date,
    end_date: date,
    agency: str | None = None,
    contractor: str | None = None,
    keyword: str | None = None,
    limit: Annotated[int, Field(ge=1, le=100)] = 5,
    page: Annotated[int, Field(ge=1, le=100)] = 1,
    psc_codes: list[str] | None = None,
    naics_codes: list[str] | None = None,
) -> dict[str, Any]:
    """Search public federal prime contracts, sorted by award amount descending.

    Dates filter USAspending award activity; returned start/end dates describe
    performance. Award amounts are award-level obligations, not spending solely
    within the requested interval, payments, or contract ceiling values. Never
    sum a page as an agency/contractor total. Data may lag or be revised; retrieval
    time is not the source's refresh time. No API key or military access needed.
    Agency must be an official top-tier awarding agency name. Contractor is a
    recipient name/UEI/DUNS text search; keyword uses USAspending keyword search.
    Use next_page to retrieve more results; there is no claimed total count.
    """
    # Reject invalid ranges before sending a request to the public service.
    if end_date < start_date:
        return {
            "error": "invalid_dates",
            "hint": "end_date must be on/after start_date.",
        }
    if not 1 <= limit <= 100 or not 1 <= page <= 100:
        return {"error": "invalid_pagination", "hint": "limit and page must be 1–100."}
    # A–D select prime procurement contracts, excluding grants and loans.
    filters: dict[str, Any] = {
        "award_type_codes": ["A", "B", "C", "D"],
        "time_period": [{"start_date": str(start_date), "end_date": str(end_date)}],
    }
    # Omit empty optional filters rather than sending blank search terms.
    if agency and agency.strip():
        filters["agencies"] = [
            {"type": "awarding", "tier": "toptier", "name": agency.strip()}
        ]
    if contractor and contractor.strip():
        filters["recipient_search_text"] = [contractor.strip()]
    if keyword and keyword.strip():
        filters["keywords"] = [keyword.strip()]
    # Classifications narrow service categories more reliably than keywords alone.
    if psc_codes:
        filters["psc_codes"] = psc_codes
    if naics_codes:
        filters["naics_codes"] = naics_codes
    request = {
        "filters": filters,
        "fields": FIELDS,
        "limit": limit,
        "page": page,
        "sort": "Award Amount",
        "order": "desc",
        "subawards": False,
    }
    try:
        # The search endpoint uses POST for read-only queries; no key is required.
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(ENDPOINT, json=request)
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict) or not isinstance(
            payload.get("results"), list
        ):
            raise ValueError("Unexpected response shape")
        # Normalize API labels into stable keys for downstream clients.
        # Keep missing fields as null; never invent amounts or source links.
        items = []
        for row in payload["results"]:
            award_key = row.get("generated_internal_id")
            items.append(
                {
                    "award_id": row.get("Award ID"),
                    "source_award_key": award_key,
                    "psc": classification_code(row.get("PSC")),
                    "naics": classification_code(row.get("NAICS")),
                    "recipient_uei": row.get("Recipient UEI"),
                    "contractor": row.get("Recipient Name"),
                    "award_obligations_usd": row.get("Award Amount"),
                    "awarding_agency": row.get("Awarding Agency"),
                    "performance_start_date": row.get("Start Date"),
                    "performance_end_date": row.get("End Date"),
                    "description": row.get("Description"),
                    "last_modified_date": row.get("Last Modified Date"),
                    "source_url": (
                        "https://www.usaspending.gov/award/"
                        + quote(str(award_key), safe="")
                        if award_key
                        else None
                    ),
                }
            )
        # Use upstream pagination; the current page is not the full result set.
        metadata = payload.get("page_metadata") or {}
        has_more = bool(metadata.get("hasNext", False))
    # Return actionable errors without exposing transport internals.
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        return {
            "error": "upstream_http_error",
            "status_code": status,
            "hint": (
                "Rate limited; retry later."
                if status == 429
                else "Check agency name and filters, or retry later."
            ),
        }
    except httpx.RequestError:
        return {
            "error": "upstream_unavailable",
            "hint": "Check network and retry later.",
        }
    except (ValueError, TypeError, AttributeError):
        return {
            "error": "invalid_upstream_response",
            "hint": "USAspending response changed; retry later.",
        }
    # Record retrieval time separately from each award's modification date.
    return {
        "source": ENDPOINT,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "filters": filters,
        "items": items,
        "count": len(items),
        "page": page,
        "has_more": has_more,
        "next_page": page + 1 if has_more else None,
        "amount_note": (
            "USD award-level obligations; not payments, ceiling values, or totals "
            "limited to the search interval. This page is not a spending total."
        ),
    }
