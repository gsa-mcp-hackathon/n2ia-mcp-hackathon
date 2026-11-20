"""FastMCP wrapper for retrieve_service_category; logic lives in services."""

from datetime import date
from typing import Annotated, Any, Literal

from fastmcp import FastMCP
from pydantic import Field

from mcp_server_n2ia.services import context

# Preserve direct imports used by local Python clients.
retrieve_service_category = context.retrieve_service_category


def register(mcp: FastMCP) -> None:
    """Register one read-only tool on the shared server."""

    @mcp.tool(
        name="retrieve_service_category",
        annotations={
            "title": "Retrieve Service Category",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def retrieve_service_category(
        category: Literal["cloud", "cybersecurity", "it_support"],
        start_date: date,
        end_date: date,
        page_size: Annotated[int, Field(ge=1, le=50)] = 20,
    ) -> dict[str, Any]:
        """Search one service category across GSA, DHS and Treasury, one page per term.

        Maximum 450 source rows; deduplication retains all matching category terms.
        Coverage is recorded per term/agency. This is a discovery sample."""
        # Delegate to reusable Python services; do not duplicate business rules.
        return await context.retrieve_service_category(
            category=category,
            start_date=start_date,
            end_date=end_date,
            page_size=page_size,
        )
