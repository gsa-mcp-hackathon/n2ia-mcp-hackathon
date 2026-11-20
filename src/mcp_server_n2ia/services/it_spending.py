"""Stateless live-API comparisons of federal prime contract obligations."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from statistics import mean, median
from time import monotonic
from typing import Any

import httpx

from mcp_server_n2ia.models import SpendingComparisonRequest
from mcp_server_n2ia.services.Mapping.contract_field_mappings import (
    CODE_DEFINITIONS_BY_FIELD,
)

ORIGIN = "https://api.usaspending.gov"
TOTALS = "/api/v2/search/spending_over_time/"
RECIPIENTS = "/api/v2/search/spending_by_category/recipient/"
AGENCIES = {
    "awarding": "/api/v2/search/spending_by_category/awarding_agency/",
    "funding": "/api/v2/search/spending_by_category/funding_agency/",
}
MAX_PAGES = 10
MAX_REQUESTS = 500
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024
DEADLINE_SECONDS = 180
CENT = Decimal("0.01")
COMPETITION = {
    "competitive": ["A", "D", "F", "CDOCiv"],
    "follow_on": ["E Civ"],
    "noncompetitive": ["B", "C", "G", "NDOCiv"],
}
PRICING = {
    "fixed_price": ["J", "M", "L", "B", "A", "K"],
    "cost_reimbursement": ["S", "R", "U", "V", "T"],
    "time_and_materials_or_labor_hours": ["Y", "Z"],
}
UNSUPPORTED = {
    "modifications": ("Search internal_id identifies awards, not unique transactions."),
    "repeated_sole_source": (
        "Verified transaction reasons and complete history required."
    ),
    "vehicles_and_task_orders": (
        "Verified historical acquisition-vehicle fields required."
    ),
    "capability_duplication": (
        "Use analyze_contract_overlap for bounded award screening."
    ),
    "corporate_parent_concentration": "No verified time-aware ownership crosswalk.",
    "persistent_vendor_dependency": (
        "This report retrieves whole-period vendor shares only."
    ),
}


class SpendingUnavailable(Exception):
    """Allowlisted public failure code; never wraps source content."""

    def __init__(self, code: str, retry_after: float = 0.25):
        super().__init__(code)
        self.retry_after = retry_after


def money(value: Any) -> Decimal:
    """Require finite numeric money while retaining signed deobligations."""
    if value is None or isinstance(value, bool):
        raise ValueError("invalid_amount")
    amount = Decimal(str(value))
    if not amount.is_finite() or abs(amount) > Decimal("1e18"):
        raise ValueError("invalid_amount")
    return amount


def percent_change(before: float | None, after: float | None) -> float | None:
    """Nonpositive baselines do not define a useful growth percentage."""
    if before is None or after is None or before <= 0:
        return None
    return float((money(after) - money(before)) / money(before) * 100)


def annual_metrics(amounts: dict[int, Decimal | None]) -> dict[str, Any]:
    """Annual series, period obligations and CAGR over actual annual intervals."""
    years = sorted(amounts)
    complete = all(amounts[year] is not None for year in years)
    values = [
        float(amounts[year]) if amounts[year] is not None else None for year in years
    ]
    growth = [
        None if index == 0 else percent_change(values[index - 1], value)
        for index, value in enumerate(values)
    ]
    cagr = None
    if (
        complete
        and len(years) >= 2
        and years == list(range(years[0], years[-1] + 1))
        and values[0] > 0
        and values[-1] > 0
    ):
        cagr = ((values[-1] / values[0]) ** (1 / (len(years) - 1)) - 1) * 100
    return {
        "status": "complete" if complete else "incomplete_fiscal_years",
        "annual_spending": [
            {
                "fiscal_year": year,
                "obligations_usd": values[index],
                "yoy_growth_pct": growth[index],
                "growth_reason": None
                if growth[index] is not None
                else (
                    "first_year" if index == 0 else "missing_or_nonpositive_baseline"
                ),
            }
            for index, year in enumerate(years)
        ],
        "total_obligations_usd": float(sum(amounts.values(), Decimal(0)))
        if complete
        else None,
        "latest_yoy_growth_pct": growth[-1],
        "cagr_pct": cagr,
        "cagr_reason": None if cagr is not None else "missing_or_nonpositive_endpoints",
    }


class SpendingAPI:
    """Bounded request-local transport; no file, database or cross-call cache."""

    def __init__(self, client: httpx.AsyncClient):
        """Track request-local budgets and limit concurrent upstream calls."""
        self.client = client
        self.requests = 0
        self.bytes_received = 0
        self.deadline = monotonic() + DEADLINE_SECONDS
        self.semaphore = asyncio.Semaphore(3)

    async def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """Retry eligible transient failures once within the shared time budget."""
        retryable = {
            "upstream_unavailable",
            "rate_limited",
            "upstream_http_502",
            "upstream_http_503",
            "upstream_http_504",
        }
        for attempt in range(2):
            try:
                return await self._post_once(path, body)
            except SpendingUnavailable as exc:
                if (
                    attempt == 1
                    or str(exc) not in retryable
                    or exc.retry_after > 2
                    or monotonic() + exc.retry_after >= self.deadline
                    or self.requests >= MAX_REQUESTS
                ):
                    raise
                await asyncio.sleep(exc.retry_after)
        raise SpendingUnavailable("upstream_unavailable")

    async def _post_once(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """Send one allowlisted POST and validate its bounded JSON response."""
        if path not in {TOTALS, RECIPIENTS, *AGENCIES.values()}:
            raise SpendingUnavailable("destination_not_allowed")
        async with self.semaphore:
            remaining = self.deadline - monotonic()
            if remaining <= 0 or self.requests >= MAX_REQUESTS:
                raise SpendingUnavailable("request_budget_exhausted")
            if self.bytes_received >= MAX_TOTAL_BYTES:
                raise SpendingUnavailable("response_budget_exhausted")
            self.requests += 1
            try:
                async with asyncio.timeout(min(30, remaining)):
                    async with self.client.stream(
                        "POST", ORIGIN + path, json=body
                    ) as response:
                        if response.status_code != 200:
                            retry_after = 0.25
                            if response.status_code == 429:
                                try:
                                    retry_after = max(
                                        0.25,
                                        float(response.headers.get("Retry-After", "1")),
                                    )
                                    if not 0 <= retry_after <= 2:
                                        retry_after = 3
                                except (ValueError, TypeError):
                                    retry_after = 3
                            raise SpendingUnavailable(
                                "rate_limited"
                                if response.status_code == 429
                                else f"upstream_http_{response.status_code}",
                                retry_after=retry_after,
                            )
                        content = bytearray()
                        async for chunk in response.aiter_bytes():
                            self.bytes_received += len(chunk)
                            content.extend(chunk)
                            if len(content) > MAX_RESPONSE_BYTES:
                                raise SpendingUnavailable("response_too_large")
                            if self.bytes_received > MAX_TOTAL_BYTES:
                                raise SpendingUnavailable("response_budget_exhausted")
                payload = json.loads(content, parse_float=Decimal)
                if (
                    not isinstance(payload, dict)
                    or not isinstance(payload.get("results"), list)
                    or not all(isinstance(row, dict) for row in payload["results"])
                    or payload.get("spending_level") != "transactions"
                ):
                    raise ValueError("invalid_shape")
                return payload
            except (httpx.HTTPError, TimeoutError):
                raise SpendingUnavailable("upstream_unavailable") from None
            except (ValueError, TypeError, InvalidOperation):
                raise SpendingUnavailable("invalid_upstream_response") from None

    async def annual(
        self, filters: dict[str, Any], years: list[int]
    ) -> dict[int, Decimal]:
        """Fetch and validate fiscal-year transaction totals for the given filters."""
        payload = await self.post(
            TOTALS,
            {
                "filters": filters,
                "group": "fiscal_year",
                "spending_level": "transactions",
            },
        )
        amounts: dict[int, Decimal] = {}
        try:
            for row in payload["results"]:
                year = int(row["time_period"]["fiscal_year"])
                if year not in years or year in amounts:
                    raise ValueError("unexpected_year")
                amounts[year] = money(row["aggregated_amount"])
        except (ValueError, KeyError, TypeError, InvalidOperation):
            raise SpendingUnavailable("invalid_upstream_response") from None
        return amounts

    async def pages(
        self, path: str, filters: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], str]:
        """Collect a capped page sequence and report whether it ended or hit the cap."""
        rows = []
        for page in range(1, MAX_PAGES + 1):
            payload = await self.post(
                path,
                {
                    "filters": filters,
                    "spending_level": "transactions",
                    "page": page,
                    "limit": 100,
                },
            )
            metadata = payload.get("page_metadata")
            if not isinstance(metadata, dict):
                raise SpendingUnavailable("invalid_upstream_response")
            has_next = metadata.get("hasNext")
            if type(has_next) is not bool or (has_next and not payload["results"]):
                raise SpendingUnavailable("invalid_upstream_response")
            rows.extend(payload["results"])
            if not has_next:
                return rows, "complete"
        return rows, "page_limit"


def filters_for(
    request: SpendingComparisonRequest, category: Any, agency: str | None = None
) -> dict[str, Any]:
    """Build source filters for declared codes, fiscal years, and agency basis."""
    years = request.fiscal_years
    filters: dict[str, Any] = {
        "award_type_codes": ["A", "B", "C", "D"],
        "psc_codes": category.psc_codes,
        "time_period": [
            {
                "start_date": f"{years[0] - 1}-10-01",
                "end_date": f"{years[-1]}-09-30",
            }
        ],
    }
    if category.naics_codes:
        filters["naics_codes"] = {"require": category.naics_codes}
    names = [agency] if agency else request.agencies
    if names:
        filters["agencies"] = [
            {"name": name, "type": request.agency_basis, "tier": "toptier"}
            for name in names
        ]
    return filters


def vendor_metrics(rows: list[dict[str, Any]], total: Decimal) -> dict[str, Any]:
    """Only a complete, identified, reconciled distribution supports shares."""
    amounts: dict[str, Decimal] = {}
    names = {}
    for row in rows:
        identity = row.get("uei") or row.get("recipient_id")
        if not identity:
            return {"status": "missing_vendor_identity"}
        key = str(identity)
        if key in amounts:
            return {"status": "duplicate_vendor_identity"}
        amounts[key] = money(row["amount"])
        names[key] = str(row.get("name") or "Unreported")[:200]
    if abs(sum(amounts.values(), Decimal(0)) - total) > CENT:
        return {"status": "totals_do_not_reconcile"}
    if total <= 0 or any(amount < 0 for amount in amounts.values()):
        return {"status": "nonpositive_or_negative_obligations"}
    ordered = sorted(amounts, key=lambda key: (-amounts[key], key))
    shares = [amounts[key] / total for key in ordered]
    return {
        "status": "complete",
        "vendor_count": len(ordered),
        "top_vendor_share_pct": float(sum(shares[:1]) * 100),
        "top_three_share_pct": float(sum(shares[:3]) * 100),
        "top_five_share_pct": float(sum(shares[:5]) * 100),
        "hhi": float(sum(share * share for share in shares) * 10000),
        "identity_basis": "reported_uei_or_recipient_id_not_corporate_parent",
        "top_vendors": [
            {
                "vendor_id": key,
                "name": names[key],
                "obligations_usd": float(amounts[key]),
                "share_pct": float(amounts[key] / total * 100),
            }
            for key in ordered[:5]
        ],
    }


async def acquisition_metrics(
    api: SpendingAPI,
    filters: dict[str, Any],
    years: list[int],
    total: Decimal,
    field: str,
    groups: dict[str, list[str]],
) -> dict[str, Any]:
    """Group annual obligations by acquisition code and calculate valid shares."""
    distributions = await asyncio.gather(
        *(api.annual({**filters, field: codes}, years) for codes in groups.values()),
        return_exceptions=True,
    )
    amounts = {}
    for name, annual in zip(groups, distributions):
        if isinstance(annual, BaseException):
            raise annual
        amounts[name] = sum(annual.values(), Decimal(0))
    definitions = CODE_DEFINITIONS_BY_FIELD.get(field, {})
    amounts["other_or_unreported"] = total - sum(amounts.values(), Decimal(0))
    valid = total > 0 and all(amount >= 0 for amount in amounts.values())
    return {
        "status": "complete" if valid else "nonpositive_or_negative_obligations",
        "filter_field": field,
        "codes": groups,
        "code_definitions": {
            name: {code: definitions[code] for code in codes if code in definitions}
            for name, codes in groups.items()
        },
        "denominator_obligations_usd": float(total),
        "buckets": {
            name: {
                "obligations_usd": float(amount),
                "share_pct": float(amount / total * 100) if valid else None,
            }
            for name, amount in amounts.items()
        },
    }


async def category_summary(
    api: SpendingAPI, request: SpendingComparisonRequest, category: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reconcile national/selected agency aggregates before zero-filling absent rows."""
    filters = filters_for(request, category)
    years = request.fiscal_years
    statuses = {}
    observed: dict[str, dict[int, Decimal]] = {}
    try:
        totals = await api.annual(filters, years)
    except SpendingUnavailable as exc:
        totals = {}
        statuses["aggregate"] = str(exc)
    path = AGENCIES[request.agency_basis]
    for year in years:
        year_filters = {
            **filters,
            "time_period": [
                {
                    "start_date": f"{year - 1}-10-01",
                    "end_date": f"{year}-09-30",
                }
            ],
        }
        try:
            rows, status = await api.pages(path, year_filters)
            seen = set()
            sum_observed = Decimal(0)
            for row in rows:
                name = row.get("name")
                if not isinstance(name, str) or not name.strip() or name in seen:
                    raise SpendingUnavailable("missing_or_duplicate_agency_identity")
                seen.add(name)
                amount = money(row["amount"])
                observed.setdefault(name, {})[year] = amount
                sum_observed += amount
            if status == "complete" and year in totals:
                if abs(sum_observed - totals[year]) > CENT:
                    status = "totals_do_not_reconcile"
            elif year not in totals:
                status = "missing_aggregate_fiscal_year"
            statuses[str(year)] = status
        except SpendingUnavailable as exc:
            statuses[str(year)] = str(exc)
        except (ValueError, KeyError, TypeError, InvalidOperation):
            statuses[str(year)] = "invalid_upstream_response"
    names = request.agencies if request.agencies is not None else sorted(observed)
    cells = []
    for name in names:
        amounts = {
            year: observed.get(name, {}).get(year, Decimal(0))
            if statuses.get(str(year)) == "complete" and name in observed
            else None
            for year in years
        }
        cells.append(
            {
                "agency": name,
                "category": category.name,
                "agency_basis": request.agency_basis,
                **annual_metrics(amounts),
                "source_references": [
                    {
                        "url": ORIGIN + path,
                        "spending_level": "transactions",
                        "filters": filters,
                        "grouping": "agency_per_fiscal_year",
                        "fiscal_years": years,
                    },
                    {
                        "url": ORIGIN + TOTALS,
                        "filters": filters,
                        "group": "fiscal_year",
                    },
                ],
                "vendors": {"status": "not_requested"},
                "competition": {"status": "not_requested"},
                "pricing": {"status": "not_requested"},
                "unassessed_metrics": UNSUPPORTED.copy(),
            }
        )
    return cells, {
        "category": category.name,
        "fiscal_year_statuses": statuses,
        "observed_agencies": sorted(observed),
        "scope": "selected_agencies"
        if request.agencies is not None
        else "national_classified_contracts",
        "status": "complete"
        if all(statuses.get(str(year)) == "complete" for year in years)
        else "partial",
    }


def share(cell: dict[str, Any], metric: str, bucket: str) -> float | None:
    """Return a bucket share only when that metric's distribution is complete."""
    data = cell.get(metric, {})
    if data.get("status") != "complete":
        return None
    return data["buckets"][bucket]["share_pct"]


def rank_findings(
    cells: list[dict[str, Any]], request: SpendingComparisonRequest
) -> list[dict[str, Any]]:
    """Deterministic materiality triage, not mission-adjusted efficiency estimates."""
    thresholds = request.thresholds
    findings = []
    for cell in cells:
        peers = [
            peer
            for peer in cells
            if peer["category"] == cell["category"]
            and peer["agency"] != cell["agency"]
            and peer["status"] == "complete"
        ]
        growth = cell["latest_yoy_growth_pct"]
        peer_growth = [
            peer["latest_yoy_growth_pct"]
            for peer in peers
            if peer["latest_yoy_growth_pct"] is not None
        ]
        cell["peer_comparison"] = {
            "basis": "same_declared_category_and_fiscal_years_not_mission_normalized",
            "valid_growth_peer_count": len(peer_growth),
            "median_growth_pct": median(peer_growth) if peer_growth else None,
            "mean_growth_pct": mean(peer_growth) if peer_growth else None,
        }
        flags = []
        cell["risk_indicators"] = flags
        cell["priority"] = "Low"
        latest = cell["annual_spending"][-1]["obligations_usd"]
        if cell["status"] != "complete":
            cell["screening_status"] = "insufficient_data"
            cell["priority"] = None
            continue
        if latest is None or latest <= 0 or latest < thresholds.materiality_usd:
            cell["screening_status"] = "below_materiality"
            continue
        cell["screening_status"] = "valid_available_metrics_only"
        if (
            len(peer_growth) >= thresholds.minimum_peers
            and growth is not None
            and growth >= thresholds.growth_pct
            and growth - median(peer_growth) >= thresholds.growth_peer_gap_pp
        ):
            flags.append(
                {
                    "signal": "growth_outlier",
                    "evidence": {
                        "latest_yoy_growth_pct": growth,
                        "peer_median_pct": median(peer_growth),
                        "peer_mean_pct": mean(peer_growth),
                        "peer_count": len(peer_growth),
                        "gap_pp": growth - median(peer_growth),
                        "growth_threshold_pct": thresholds.growth_pct,
                        "gap_threshold_pp": thresholds.growth_peer_gap_pp,
                    },
                }
            )
        vendors = cell["vendors"]
        if vendors.get("status") == "complete" and (
            vendors["top_vendor_share_pct"] >= thresholds.top_vendor_share_pct
            or vendors["hhi"] >= thresholds.hhi
        ):
            flags.append(
                {
                    "signal": "vendor_concentration",
                    "evidence": {
                        "top_vendor_share_pct": vendors["top_vendor_share_pct"],
                        "hhi": vendors["hhi"],
                        "top_vendor_threshold_pct": thresholds.top_vendor_share_pct,
                        "hhi_threshold": thresholds.hhi,
                    },
                }
            )
        for metric, bucket in (
            ("competition", "noncompetitive"),
            ("pricing", "cost_reimbursement"),
        ):
            current = share(cell, metric, bucket)
            peer_shares = [
                value
                for peer in peers
                if (value := share(peer, metric, bucket)) is not None
            ]
            if (
                current is not None
                and len(peer_shares) >= thresholds.minimum_peers
                and current - median(peer_shares) >= thresholds.acquisition_peer_gap_pp
            ):
                flags.append(
                    {
                        "signal": bucket + "_peer_difference",
                        "evidence": {
                            "share_pct": current,
                            "peer_median_share_pct": median(peer_shares),
                            "peer_count": len(peer_shares),
                            "gap_threshold_pp": thresholds.acquisition_peer_gap_pp,
                        },
                    }
                )
        noncompetitive = share(cell, "competition", "noncompetitive")
        if (
            vendors.get("status") == "complete"
            and vendors["top_vendor_share_pct"] >= thresholds.top_vendor_share_pct
            and noncompetitive is not None
            and noncompetitive >= 50
        ):
            flags.append(
                {
                    "signal": "concentration_with_limited_competition",
                    "evidence": {
                        "top_vendor_share_pct": vendors["top_vendor_share_pct"],
                        "noncompetitive_share_pct": noncompetitive,
                        "note": (
                            "Portfolio association, not proof the top vendor's "
                            "awards were sole-source."
                        ),
                    },
                }
            )
        corroborating = {flag["signal"] for flag in flags} - {
            "concentration_with_limited_competition"
        }
        cell["priority"] = (
            "High"
            if len(corroborating) >= 2
            or any(
                flag["signal"] == "concentration_with_limited_competition"
                for flag in flags
            )
            else "Medium"
            if flags
            else "Low"
        )
        if not flags:
            continue
        metric_limits = [
            f"{name}: {cell[name]['status']}"
            for name in ("vendors", "competition", "pricing")
            if cell[name]["status"] != "complete"
        ]
        findings.append(
            {
                "agency": cell["agency"],
                "category": cell["category"],
                "priority": cell["priority"],
                "signals": flags,
                "total_obligations_usd": cell["total_obligations_usd"],
                "latest_fiscal_year_obligations_usd": latest,
                "materiality_threshold_usd": thresholds.materiality_usd,
                "potential_causes": [
                    "Mission or security requirements, changes in scope or volume, "
                    "renewals, vendor switching costs, or reporting revisions."
                ],
                "possible_business_impact": (
                    "Potential budget pressure or supplier dependency; "
                    "no waste or savings established."
                ),
                "recommended_investigation": (
                    "Validate quantities, service scope, security constraints, "
                    "award histories, justifications and market alternatives "
                    "with acquisition staff."
                ),
                "cost_optimization_approach": (
                    "Assess competition, strategic sourcing, renewal pricing and "
                    "shared-service feasibility without reducing required capability."
                ),
                "confidence": "Low" if metric_limits else "Medium",
                "data_limitations": metric_limits
                + ["Peers are not mission/size normalized.", *UNSUPPORTED.values()],
                "source_references": cell["source_references"],
            }
        )
    return sorted(
        findings,
        key=lambda finding: (
            0 if finding["priority"] == "High" else 1,
            -len(finding["signals"]),
            -finding["latest_fiscal_year_obligations_usd"],
            finding["agency"],
            finding["category"],
        ),
    )


async def compare_agency_it_spending(
    request: SpendingComparisonRequest,
) -> dict[str, Any]:
    """Retrieve live aggregates first, then bounded details; persist nothing."""
    started = datetime.now(timezone.utc).isoformat()
    cells = []
    category_coverage = []
    # Fetch aggregate comparisons first so optional detail failures do not erase them.
    async with httpx.AsyncClient(
        timeout=30, follow_redirects=False, trust_env=False
    ) as client:
        api = SpendingAPI(client)
        for category in request.categories:
            summary, coverage = await category_summary(api, request, category)
            cells.extend(summary)
            category_coverage.append(coverage)

        # Enrich complete comparisons with vendor, competition, and pricing evidence.
        async def enrich_cell(cell: dict[str, Any]) -> None:
            category = next(
                category
                for category in request.categories
                if category.name == cell["category"]
            )
            filters = filters_for(request, category, cell["agency"])
            for metric in ("vendors", "competition", "pricing"):
                if cell["status"] != "complete":
                    cell[metric] = {"status": "incomplete_fiscal_years"}
                    continue
                total = money(cell["total_obligations_usd"])
                try:
                    if metric == "vendors":
                        rows, status = await api.pages(RECIPIENTS, filters)
                        cell[metric] = (
                            vendor_metrics(rows, total)
                            if status == "complete"
                            else {"status": status, "retrieved_rows": len(rows)}
                        )
                    else:
                        field, groups = (
                            ("extent_competed_type_codes", COMPETITION)
                            if metric == "competition"
                            else ("contract_pricing_type_codes", PRICING)
                        )
                        cell[metric] = await acquisition_metrics(
                            api, filters, request.fiscal_years, total, field, groups
                        )
                    cell["source_references"].append(
                        {
                            "url": ORIGIN
                            + (RECIPIENTS if metric == "vendors" else TOTALS),
                            "metric": metric,
                            "filters": filters,
                            "spending_level": "transactions",
                            "code_filters": None if metric == "vendors" else groups,
                        }
                    )
                except SpendingUnavailable as exc:
                    cell[metric] = {"status": str(exc)}
                except (ValueError, KeyError, TypeError, InvalidOperation):
                    cell[metric] = {"status": "invalid_upstream_response"}

        if request.include_details:
            await asyncio.gather(*(enrich_cell(cell) for cell in cells))
    # Flag overlapping declared scopes so consumers avoid summing duplicate categories.
    overlap = any(
        set(first.psc_codes) & set(second.psc_codes)
        and (
            not first.naics_codes
            or not second.naics_codes
            or set(first.naics_codes) & set(second.naics_codes)
        )
        for index, first in enumerate(request.categories)
        for second in request.categories[index + 1 :]
    )
    findings = rank_findings(cells, request)
    # Surface unavailable detail metrics separately from aggregate coverage.
    missing = [
        {
            "agency": cell["agency"],
            "category": cell["category"],
            "metric": metric,
            "status": cell[metric]["status"],
        }
        for cell in cells
        for metric in ("vendors", "competition", "pricing")
        if cell[metric]["status"] != "complete"
    ]
    complete_count = sum(cell["status"] == "complete" for cell in cells)
    return {
        "source": ORIGIN,
        "retrieval_started_at_utc": started,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "request": request.model_dump(mode="json"),
        "coverage": {
            "status": "complete_for_declared_aggregate_filters"
            if all(category["status"] == "complete" for category in category_coverage)
            else "partial",
            "categories": category_coverage,
            "comparisons_returned": len(cells),
            "complete_spending_comparisons": complete_count,
            "requests_made": api.requests,
            "response_bytes_received": api.bytes_received,
            "limits": {
                "requests": MAX_REQUESTS,
                "pages_per_distribution": MAX_PAGES,
                "seconds": DEADLINE_SECONDS,
                "total_response_bytes": MAX_TOTAL_BYTES,
            },
            "unassessed_detail_metrics": missing,
            "overlapping_category_codes": bool(overlap),
            "source_refresh_date": None,
            "storage": "none_request_memory_only",
        },
        "comparisons": cells,
        "executive_summary": (
            f"{complete_count} of {len(cells)} agency/category spending comparisons "
            "have reconciled fiscal-year aggregates. "
            f"{len(findings)} material screening opportunities; "
            f"{min(5, len(findings))} prioritized. Findings are investigation "
            "leads, not waste or savings estimates."
        ),
        "key_findings": findings,
        "recommended_investigations": findings[:5],
        "limitations": [
            "Net prime-contract obligations, not outlays, ceilings, budgets, "
            "unit costs or all agency IT expenditure.",
            "Categories are caller-declared exact PSC codes with optional NAICS "
            "refinement (AND); no approved six-category taxonomy is implied.",
            "Do not sum overlapping categories or combine awarding and funding views.",
            "Fiscal years are closed at least 90 days; this is not "
            "source reporting certification.",
            "Empty or missing aggregate years are unavailable, not assumed zero.",
            "Live API data can change between requests and during retrieval; "
            "no stored snapshots or historic replay.",
            "Retrieval dates are not source refresh dates. Shares require "
            "complete reconciled nonnegative distributions.",
            "Competition rate excludes follow-on codes from its numerator and "
            "includes unknown obligations in its denominator.",
            "Vendor/competition/pricing shares cover the whole analysis period; "
            "growth flags use the latest fiscal-year change.",
            "No inflation, agency-size, mission, volume or security adjustment; "
            "category peers are descriptive only.",
            "Heuristic materiality signals are not statistical significance, "
            "proof of waste, sole-source findings or estimated savings.",
            "Rate, time, memory or pagination limits may prevent complete "
            "government-wide details; narrow filters or retry later.",
            "SAM.gov and IT Dashboard are not queried by this first-release "
            "tool; FPDS-derived fields come through USAspending.",
            *UNSUPPORTED.values(),
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Deterministic report rendering from the same calculated JSON evidence."""

    def text(value: Any) -> str:
        """Escape table delimiters and flatten line breaks in displayed values."""
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    def number(value: Any, dollars: bool = False) -> str:
        """Format available values as currency or percentages for the report."""
        if value is None:
            return "Unavailable"
        return f"${value:,.2f}" if dollars else f"{value:,.2f}%"

    lines = [
        "# Executive Summary",
        report["executive_summary"],
        f"Retrieved: {report['retrieved_at_utc']}. "
        f"Agency basis: {report['request']['agency_basis']}.",
        "\n## Comparative Analysis",
        "Vendor and competition shares cover the whole analysis period. "
        "Growth is latest YoY and whole-period CAGR.",
        "| Agency | IT category | Total obligations | YoY / CAGR | Top vendor "
        "| Top 3 | Competition | Risk indicators | Priority |",
        "|---|---|---:|---|---:|---:|---:|---|---|",
    ]
    for cell in report["comparisons"]:
        vendor = cell["vendors"]
        lines.append(
            "| "
            + " | ".join(
                [
                    text(cell["agency"]),
                    text(cell["category"]),
                    number(cell["total_obligations_usd"], True),
                    number(cell["latest_yoy_growth_pct"])
                    + " / "
                    + number(cell["cagr_pct"]),
                    number(vendor.get("top_vendor_share_pct")),
                    number(vendor.get("top_three_share_pct")),
                    number(share(cell, "competition", "competitive")),
                    ", ".join(flag["signal"] for flag in cell["risk_indicators"])
                    or cell["screening_status"],
                    cell["priority"] or "Unassessed",
                ]
            )
            + " |"
        )
    lines.append("\n## Key Findings")
    for finding in report["key_findings"]:
        lines.append(
            f"- {text(finding['agency'])}, {text(finding['category'])}: "
            + ", ".join(flag["signal"] for flag in finding["signals"])
        )
    if not report["key_findings"]:
        lines.append(
            "No supported material flags; unavailable metrics are not "
            "evidence of efficiency."
        )
    lines.append("\n## Recommended Investigations")
    for index, finding in enumerate(report["recommended_investigations"], 1):
        lines.extend(
            [
                f"### {index}. {text(finding['agency'])}: {text(finding['category'])}",
                "Evidence: " + text(json.dumps(finding["signals"], sort_keys=True)),
                "Possible causes: " + text(" ".join(finding["potential_causes"])),
                "Possible impact: " + finding["possible_business_impact"],
                "Investigation: " + finding["recommended_investigation"],
                "Optimization approach: " + finding["cost_optimization_approach"],
                "Confidence: "
                + finding["confidence"]
                + ". "
                + " ".join(finding["data_limitations"]),
                "Source query references: "
                + text(json.dumps(finding["source_references"], sort_keys=True)),
            ]
        )
    lines.extend(
        [
            "\n## Coverage and Limitations",
            text(json.dumps(report["coverage"], sort_keys=True)),
        ]
    )
    lines.extend("- " + limitation for limitation in report["limitations"])
    return "\n".join(lines)
