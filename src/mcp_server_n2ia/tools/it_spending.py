"""Read-only live federal IT spending workflow; no persistent data storage."""

from typing import Annotated, Literal

from fastmcp import FastMCP
from pydantic import Field

from mcp_server_n2ia.models import (
    ResponseFormat,
    SpendingCategory,
    SpendingComparisonRequest,
    SpendingComparisonResult,
    SpendingThresholds,
)
from mcp_server_n2ia.services import it_spending


def register(mcp: FastMCP) -> None:
    """Register the live-only fiscal-year report tool."""

    @mcp.tool(
        name="compare_agency_it_spending",
        annotations={
            "title": "Compare Federal IT Spending",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def compare_agency_it_spending(
        categories: Annotated[
            list[SpendingCategory], Field(min_length=1, max_length=6)
        ],
        agencies: Annotated[
            list[str] | None, Field(min_length=2, max_length=50)
        ] = None,
        fiscal_year_count: Annotated[int, Field(ge=3, le=5)] = 3,
        end_fiscal_year: Annotated[int | None, Field(ge=2010, le=9999)] = None,
        agency_basis: Literal["awarding", "funding"] = "awarding",
        include_details: bool = True,
        thresholds: SpendingThresholds | None = None,
        response_format: ResponseFormat = ResponseFormat.JSON,
    ) -> SpendingComparisonResult:
        """Compare live USAspending IT prime-contract obligations over 3-5 fiscal years.

        Provide categories as name, exact psc_codes and optional six-digit
        naics_codes. PSC alternatives are OR; NAICS alternatives refine them
        with AND. These are caller-declared scopes, not certified categories.
        Omit agencies for national classified-contract aggregates, or supply
        official top-tier names. Choose awarding OR funding; never combine them.
        Defaults to three FYs closed at least 90 days. Returns annual obligations,
        YoY/CAGR, whole-period vendor top1/3/5 and HHI, competition/pricing shares,
        material investigation leads, filters, timestamps and coverage. Markdown
        mode adds a readable markdown field to the same structured JSON report.
        Missing/truncated/unreconciled distributions never produce shares.
        include_details=false retrieves aggregate spending only. Caps: 500 HTTP
        requests, 180 seconds, 16 MiB total responses, 1000 recipients per scope.
        Nothing is persisted; retrying may yield revised source figures. Narrow
        scope if detail coverage is incomplete. No API key required. Not all IT
        expenditure, mission-normalized efficiency, waste or estimated savings.
        Modification/vehicle/sole-source/duplication analysis is explicitly
        unassessed; use analyze_contract_overlap separately for overlap screening.
        """
        # Validate and normalize tool inputs using the shared request model.
        request = SpendingComparisonRequest(
            categories=categories,
            agencies=agencies,
            fiscal_year_count=fiscal_year_count,
            end_fiscal_year=end_fiscal_year,
            agency_basis=agency_basis,
            include_details=include_details,
            thresholds=thresholds or SpendingThresholds(),
        )
        # Keep upstream retrieval and calculations in the service layer.
        report = await it_spending.compare_agency_it_spending(request)
        if response_format == ResponseFormat.MARKDOWN:
            # Add the readable view alongside, not instead of, structured results.
            report["markdown"] = it_spending.render_markdown(report)
        return SpendingComparisonResult.model_validate(report)
