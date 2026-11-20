"""Combined retrieval and contract overlap analysis MCP tool."""

from datetime import date
from typing import Annotated, Any, Literal

from fastmcp import FastMCP
from pydantic import Field

from mcp_server_n2ia.config import settings
from mcp_server_n2ia.services import overlap
from mcp_server_n2ia.services.overlap import AGENCIES, Award


# Contract properties consumed by the Streamlit relationship tables,
# contract detail panels, and graph. Keep service_functions for groups.
DISPLAY_AWARD_FIELDS = frozenset({
    "source_award_key", "award_id", "awarding_agency", "contractor",
    "description", "psc", "naics", "performance_start_date",
    "performance_end_date", "source_url", "service_functions",
})


def _compact_award(award: dict[str, Any]) -> dict[str, Any]:
    """Serialize only display-relevant fields; never mutate analysis data."""
    return {key: value for key, value in award.items()
            if key in DISPLAY_AWARD_FIELDS}

def _compact_combined_analysis(comparison: dict[str, Any]) -> dict[str, Any]:
    """Keep summary metrics first and preserve evidence required by Streamlit."""
    analysis = {
        key: value for key, value in comparison.items()
        if key not in (
            "examples", "examples_truncated",
            "ranked_candidates", "groups",
        )
    }

    analysis["ranked_candidates"] = [
        {**candidate,
         "left": _compact_award(candidate["left"]),
         "right": _compact_award(candidate["right"])}
        for candidate in comparison.get("ranked_candidates", [])
    ]
    analysis["groups"] = [
        {**group,
         "contracts": [_compact_award(contract)
                       for contract in group.get("contracts", [])]}
        for group in comparison.get("groups", [])
    ]
    return analysis


def register(mcp: FastMCP) -> None:

    @mcp.tool(
        name="analyze_contract_overlap",
        annotations={
            "title": "Analyze Contract Overlap",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def analyze_contract_overlap(
        start_date: date,
        end_date: date,
        agencies: Annotated[
            list[str], Field(min_length=1, max_length=3)
        ] = AGENCIES,
        keyword: str = "cybersecurity",
        analysis_method: Literal["lexical", "semantic", "both"] = "both",
        cross_agency_only: bool = True,
        max_candidates: Annotated[
            int, Field(ge=1, le=100)
        ] = 10,
        pages_per_agency: Annotated[
            int,
            Field(
                ge=1,
                le=2,
                description="Number of result pages per agency. Must be 1 or 2; never request more than 2."
            )
        ] = 2,
        page_size: Annotated[
            int, Field(ge=1, le=100)
        ] = 100,
    ) -> dict[str, Any]:
        """Retrieve and screen contracts for potential scope overlap.

        Use this tool for end-to-end contract overlap questions.
        It retrieves awards once. Choose lexical for word matching, semantic
        for Hugging Face embeddings, or both for a side-by-side comparison.
        The lexical mode preserves the original candidate response format.
        Similarity thresholds are configured by the MCP Server.

        Results identify candidates for human review, not proven
        duplication, waste, or savings.
        The selected dates constrain both retrieval and performance-overlap screening.
        In combined mode, ranked_candidates are preliminary text matches;
        groups and grouping_relationship_count reflect stricter screening.
        Never describe preliminary matches as qualified overlap relationships.

        - pages_per_agency must never exceed 2.
        - For multiple fiscal years, call this tool separately for each fiscal year.
        - FY2025: 2024-10-01 through 2025-09-30.
        - FY2026: 2025-10-01 through 2026-09-30.
        - Do not combine fiscal years into one date range.
        """

        retrieved = await overlap.retrieve_contracts(
            start_date=start_date,
            end_date=end_date,
            agencies=agencies,
            keyword=keyword,
            pages_per_agency=pages_per_agency,
            page_size=page_size,
        )

        if "error" in retrieved:
            return retrieved

        awards = [
            Award.model_validate(item)
            for item in retrieved["items"]
        ]

        if analysis_method == "lexical":
            analysis = overlap.detect_overlap(
                awards=awards,
                min_similarity=settings.lexical_similarity_threshold,
                cross_agency_only=cross_agency_only,
                max_candidates=max_candidates,
                analysis_start=start_date,
                analysis_end=end_date,
            )
        else:
            comparison = overlap.compare_awards(
                awards=awards,
                min_lexical_similarity=settings.lexical_similarity_threshold,
                min_semantic_similarity=settings.semantic_similarity_threshold,
                cross_agency_only=cross_agency_only,
                max_examples_per_category=max_candidates,
                analysis_start=start_date,
                analysis_end=end_date,
            )
            if analysis_method == "both":
                analysis = _compact_combined_analysis(comparison)
            else:
                # Semantic-only results include shared and semantic-only matches.
                semantic_examples = (
                    comparison["examples"]["both"]
                    + comparison["examples"]["semantic_only"]
                )
                semantic_examples.sort(
                    key=lambda item: -item["semantic_similarity"]
                )
                analysis = {
                    "model": comparison["model"],
                    "analyzed_count": comparison["analyzed_awards"],
                    "candidate_count": comparison["counts"]["semantic_total"],
                    "candidates": semantic_examples[:max_candidates],
                    "candidates_truncated": (
                        comparison["counts"]["semantic_total"] >
                        len(semantic_examples[:max_candidates])
                    ),
                    "thresholds": comparison["thresholds"],
                    "limitations": comparison["limitations"],
                }

        return {
            "query": retrieved["query"],
            "retrieved_at_utc": retrieved["retrieved_at_utc"],
            "retrieved_count": retrieved["count"],
            "coverage": retrieved["coverage"],
            "complete": retrieved["complete"],
            "retrieval_limitations": retrieved["limitations"],
            "analysis_method": analysis_method,
            "analysis": analysis,
            "cache": retrieved.get("cache"),
        }