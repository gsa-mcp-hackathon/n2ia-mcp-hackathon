"""Static glossary for interpreting contract-overlap and IT-spending tool output.

This resource documents field meanings and evidence limitations; it does not
retrieve, validate, or store the underlying contract or spending data.
"""

from fastmcp import FastMCP


def register(mcp: FastMCP) -> None:
    """Expose the glossary at a stable MCP resource URI for client lookup."""

    @mcp.resource("contracts://reference/data-dictionary")
    def data_dictionary() -> str:
        """Return definitions and caveats for contract and spending report fields."""
        return (
            # Award and overlap fields, including what search completeness means.
            "source_award_key: USAspending global award identifier.\n"
            "award_obligations_usd: award-level obligations, not interval spending.\n"
            "performance_start/end_date: performance dates, not activity filters.\n"
            "PSC/NAICS: service/industry classifications, not proof of equal scope.\n"
            "text_similarity: lexical heuristic from 0 to 1, not probability.\n"
            "parent_award_key: API-reported vehicle; separate orders may be valid.\n"
            "context_status: retrieved, unavailable, or not_requested.\n"
            "complete: pagination finished for the query, not all IT contracts.\n"
            # Spending totals, fiscal-year calculations, and caller-defined scope.
            "total_obligations_usd/annual_spending: net prime-contract action-date "
            "obligations, including deobligations; never summed award lifetimes.\n"
            "fiscal_year: October 1-September 30; default 3 years closed for "
            "90 days, not certified reporting completeness.\n"
            "cagr_pct: endpoint annual-obligation growth across year_count-1 "
            "intervals; unavailable for missing/nonpositive endpoints.\n"
            "category: caller-declared exact PSC OR filters; optional exact "
            "NAICS OR filters refine PSC with AND. No certified mapping.\n"
            # Whole-period vendor and acquisition metric meanings and safeguards.
            "top_vendor/top_three/top_five_share_pct and hhi: whole-period "
            "recipient distribution; incomplete, unidentified, unreconciled "
            "or negative distributions do not produce shares.\n"
            "competition competitive bucket: A/D/F/CDOCiv numerator over "
            "all period obligations; follow-on E Civ is separate, unknown "
            "obligations remain in the denominator.\n"
            # Overall request basis, coverage, and limits on interpreting findings.
            "agency_basis: awarding OR funding; these are separate views.\n"
            "search transaction internal_id: award ID, not unique transaction "
            "identity; modification counts remain unassessed.\n"
            "priority: heuristic review order with dollar/peer thresholds, "
            "not statistical significance or mission-normalized efficiency.\n"
            "coverage: explicit request/page/time/response limits; no spending "
            "storage or cross-call cache. Retrieval time is not refresh time.\n"
            "Findings are screening candidates, not confirmed waste or savings."
        )
