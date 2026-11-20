"""FastMCP wrapper for search_contract_awards; logic lives in services."""

from datetime import date
from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field

from mcp_server_n2ia.services import usaspending

# Preserve direct imports used by local Python clients.
search_contract_awards = usaspending.search_contract_awards


def register(mcp: FastMCP) -> None:
    """Register one read-only tool on the shared server."""

    @mcp.tool(
        name="search_contract_awards",
        annotations={
            "title": "Search Contract Awards",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
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
        Use next_page to retrieve more results; there is no claimed total count."""
        # Delegate to reusable Python services; do not duplicate business rules.
        return await usaspending.search_contract_awards(
            start_date=start_date,
            end_date=end_date,
            agency=agency,
            contractor=contractor,
            keyword=keyword,
            limit=limit,
            page=page,
            psc_codes=psc_codes,
            naics_codes=naics_codes,
        )
