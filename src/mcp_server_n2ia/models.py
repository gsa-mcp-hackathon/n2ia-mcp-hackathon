"""Typed request, enums and response schemas used by the MCP tools.

Typed parameters give MCP clients a precise schema, which helps the LLM call
your tools correctly. Prefer enums for closed sets of values and Pydantic
models for structured inputs/outputs.

These are illustrative — replace them with models for your own service.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ResponseFormat(str, Enum):
    """How a tool should shape its return value.

    - JSON     → machine-readable dict/list (best for chaining tool calls)
    - MARKDOWN → human-readable text (best for direct display)
    """

    JSON = "json"
    MARKDOWN = "markdown"


class PaginationParams(BaseModel):
    """Reusable pagination inputs for list-style tools.

    Federal APIs almost always paginate. Expose these to the caller (or handle
    paging internally) and return `has_more` / `next_offset` in your response.
    """

    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of items to return (1–100).",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Number of items to skip from the start of the result set.",
    )


def latest_eligible_fiscal_year(today: date | None = None) -> int:
    """Latest closed federal FY at least 90 days past September 30."""
    today = today or date.today()
    fiscal_year = today.year
    while date(fiscal_year, 9, 30) + timedelta(days=90) > today:
        fiscal_year -= 1
    return fiscal_year


class SpendingCategory(BaseModel):
    """Caller-declared comparison scope; PSCs match alternatives and NAICS narrows them."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=100)
    psc_codes: list[str] = Field(min_length=1, max_length=50)
    naics_codes: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("psc_codes")
    @classmethod
    def validate_psc(cls, codes: list[str]) -> list[str]:
        """Uppercase PSCs and require unique, exact four-character codes."""
        normalized = [code.strip().upper() for code in codes]
        if len(set(normalized)) != len(normalized) or any(
            not re.fullmatch(r"[A-Z0-9]{4}", code) for code in normalized
        ):
            raise ValueError("Provide unique exact four-character PSC codes.")
        return normalized

    @field_validator("naics_codes")
    @classmethod
    def validate_naics(cls, codes: list[str]) -> list[str]:
        """Trim NAICS values and require unique, exact six-digit codes."""
        normalized = [code.strip() for code in codes]
        if len(set(normalized)) != len(normalized) or any(
            not re.fullmatch(r"[0-9]{6}", code) for code in normalized
        ):
            raise ValueError("Provide unique exact six-digit NAICS codes.")
        return normalized


class SpendingThresholds(BaseModel):
    """Material review thresholds, not statistical or regulatory findings."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    materiality_usd: float = Field(default=10_000_000, ge=0, le=1e15)
    growth_pct: float = Field(default=20, ge=0, le=10000)
    growth_peer_gap_pp: float = Field(default=10, ge=0, le=10000)
    top_vendor_share_pct: float = Field(default=50, ge=0, le=100)
    hhi: float = Field(default=2500, ge=0, le=10000)
    acquisition_peer_gap_pp: float = Field(default=10, ge=0, le=100)
    minimum_peers: int = Field(default=3, ge=3, le=50)


class SpendingComparisonRequest(BaseModel):
    """Bounded live comparison request; omitted agencies means national discovery."""

    model_config = ConfigDict(extra="forbid")
    categories: list[SpendingCategory] = Field(min_length=1, max_length=6)
    agencies: list[str] | None = Field(default=None, min_length=2, max_length=50)
    fiscal_year_count: int = Field(default=3, ge=3, le=5)
    end_fiscal_year: int | None = Field(default=None, ge=2010, le=9999)
    agency_basis: Literal["awarding", "funding"] = "awarding"
    thresholds: SpendingThresholds = Field(default_factory=SpendingThresholds)
    include_details: bool = True

    @field_validator("agencies")
    @classmethod
    def validate_agencies(cls, agencies: list[str] | None) -> list[str] | None:
        """Trim agency names and reject blank, overlong, or case-insensitive duplicates."""
        if agencies is None:
            return None
        normalized = [agency.strip() for agency in agencies]
        if any(not name or len(name) > 200 for name in normalized) or len(
            {name.casefold() for name in normalized}
        ) != len(normalized):
            raise ValueError("Provide distinct official top-tier agency names.")
        return normalized

    @model_validator(mode="after")
    def validate_scope(self) -> SpendingComparisonRequest:
        """Choose a mature default end year and validate year and category scope."""
        latest = latest_eligible_fiscal_year()
        if self.end_fiscal_year is None:
            self.end_fiscal_year = latest
        if self.end_fiscal_year > latest:
            raise ValueError("Fiscal years must be closed for at least 90 days.")
        if self.end_fiscal_year - self.fiscal_year_count + 1 < 2008:
            raise ValueError("Search supports complete fiscal years since FY2008.")
        names = [category.name.casefold() for category in self.categories]
        if len(set(names)) != len(names):
            raise ValueError("Category names must be distinct.")
        return self

    @property
    def fiscal_years(self) -> list[int]:
        """Return the inclusive fiscal-year sequence ending at the validated end year."""
        end = self.end_fiscal_year
        assert end is not None
        return list(range(end - self.fiscal_year_count + 1, end + 1))


class SpendingComparisonResult(BaseModel):
    """Typed report envelope; incomplete metrics carry statuses rather than zeros."""

    source: str
    retrieved_at_utc: str
    retrieval_started_at_utc: str
    request: dict[str, Any]
    coverage: dict[str, Any]
    comparisons: list[dict[str, Any]]
    executive_summary: str
    key_findings: list[dict[str, Any]]
    recommended_investigations: list[dict[str, Any]]
    limitations: list[str]
    markdown: str | None = None
