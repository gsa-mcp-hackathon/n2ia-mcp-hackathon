"""Live spending analysis with synthetic responses and no stored source data."""

import json
from datetime import date
from decimal import Decimal

import httpx
import pytest
from fastmcp import Client
from pydantic import ValidationError

from mcp_server_n2ia.models import (
    SpendingComparisonRequest,
    latest_eligible_fiscal_year,
)
from mcp_server_n2ia.services import it_spending as service

PARAMS = {
    "agencies": ["General Services Administration", "Department of the Treasury"],
    "categories": [{"name": "Explicit IT scope", "psc_codes": ["D310"]}],
    "end_fiscal_year": 2025,
}


def test_fiscal_year_selection():
    assert latest_eligible_fiscal_year(date(2026, 11, 8)) == 2025
    assert latest_eligible_fiscal_year(date(2026, 12, 28)) == 2025
    assert latest_eligible_fiscal_year(date(2026, 12, 29)) == 2026
    request = SpendingComparisonRequest(**PARAMS)
    assert request.fiscal_years == [2023, 2024, 2025]
    assert request.agency_basis == "awarding"
    assert SpendingComparisonRequest(**PARAMS, fiscal_year_count=5).fiscal_years == [
        2021,
        2022,
        2023,
        2024,
        2025,
    ]


@pytest.mark.parametrize(
    "updates",
    [
        {"fiscal_year_count": 2},
        {"end_fiscal_year": 2999},
        {"end_fiscal_year": 2010, "fiscal_year_count": 5},
        {"agencies": ["Treasury", " treasury "]},
        {"agencies": []},
        {"agency_basis": "both"},
        {"categories": [{"name": "IT", "psc_codes": ["https://localhost"]}]},
        {"categories": [{"name": "IT", "psc_codes": ["D310", "d310"]}]},
        {"categories": [{"name": "IT", "psc_codes": ["D310"], "naics_codes": ["54"]}]},
        {"thresholds": {"materiality_usd": float("nan")}},
    ],
)
def test_invalid_request(updates):
    with pytest.raises(ValidationError):
        SpendingComparisonRequest(**{**PARAMS, **updates})


def test_money_growth_and_cagr():
    metrics = service.annual_metrics(
        {2023: Decimal(100), 2024: Decimal(120), 2025: Decimal(144)}
    )
    assert metrics["total_obligations_usd"] == 364
    assert metrics["latest_yoy_growth_pct"] == 20
    assert metrics["cagr_pct"] == pytest.approx(20)
    assert service.percent_change(0, 1) is None
    assert service.percent_change(-1, 1) is None
    assert service.percent_change(10, -5) == -150
    missing = service.annual_metrics(
        {2023: Decimal(100), 2024: None, 2025: Decimal(144)}
    )
    assert missing["total_obligations_usd"] is None
    assert missing["cagr_pct"] is None
    assert (
        service.annual_metrics({2023: Decimal(0), 2024: Decimal(1), 2025: Decimal(2)})[
            "cagr_pct"
        ]
        is None
    )
    with pytest.raises(ValueError):
        service.money(float("nan"))


def test_vendor_concentration_and_invalid_distributions():
    rows = [
        {"uei": str(index), "name": f"Vendor {index}", "amount": amount}
        for index, amount in enumerate([50, 20, 15, 10, 5])
    ]
    metrics = service.vendor_metrics(rows, Decimal(100))
    assert metrics["top_vendor_share_pct"] == 50
    assert metrics["top_three_share_pct"] == 85
    assert metrics["top_five_share_pct"] == 100
    assert metrics["hhi"] == 3250
    assert (
        service.vendor_metrics(rows[:2], Decimal(100))["status"]
        == "totals_do_not_reconcile"
    )
    assert (
        service.vendor_metrics([{"amount": 1}], Decimal(1))["status"]
        == "missing_vendor_identity"
    )
    assert (
        service.vendor_metrics([rows[0], rows[0]], Decimal(100))["status"]
        == "duplicate_vendor_identity"
    )
    assert (
        service.vendor_metrics([{"uei": "A", "amount": -1}], Decimal(-1))["status"]
        == "nonpositive_or_negative_obligations"
    )


def mock_api(monkeypatch, handler):
    original = httpx.AsyncClient

    def factory(**kwargs):
        assert kwargs["follow_redirects"] is False
        assert kwargs["trust_env"] is False
        return original(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)


def response_handler(request):
    assert request.url.host == "api.usaspending.gov"
    assert request.url.scheme == "https"
    body = json.loads(request.content)
    assert body["spending_level"] == "transactions"
    filters = body["filters"]
    assert filters["award_type_codes"] == ["A", "B", "C", "D"]
    assert filters["psc_codes"] == ["D310"]
    names = [agency["name"] for agency in filters.get("agencies", [])] or PARAMS[
        "agencies"
    ]
    annual = {2023: 100_000_000, 2024: 120_000_000, 2025: 144_000_000}
    if request.url.path == service.TOTALS:
        multiplier = 1
        if "extent_competed_type_codes" in filters:
            multiplier = {"A": 0.6, "E Civ": 0.1, "B": 0.3}[
                filters["extent_competed_type_codes"][0]
            ]
        if "contract_pricing_type_codes" in filters:
            multiplier = {"J": 0.6, "S": 0.25, "Y": 0.1}[
                filters["contract_pricing_type_codes"][0]
            ]
        return httpx.Response(
            200,
            json={
                "spending_level": "transactions",
                "results": [
                    {
                        "time_period": {"fiscal_year": str(year)},
                        "aggregated_amount": amount * len(names) * multiplier,
                    }
                    for year, amount in annual.items()
                ],
            },
        )
    if request.url.path in service.AGENCIES.values():
        year = int(filters["time_period"][0]["end_date"][:4])
        rows = [
            {"id": index, "name": name, "amount": annual[year]}
            for index, name in enumerate(names)
        ]
    else:
        assert request.url.path == service.RECIPIENTS
        rows = [
            {"uei": "FIRST", "name": "First", "amount": 364_000_000 * 0.8},
            {"recipient_id": "SECOND-C", "name": "Second", "amount": 364_000_000 * 0.2},
        ]
    return httpx.Response(
        200,
        json={
            "spending_level": "transactions",
            "results": rows,
            "page_metadata": {"hasNext": False},
        },
    )


async def test_live_shape_metrics_and_provenance(monkeypatch):
    mock_api(monkeypatch, response_handler)
    report = await service.compare_agency_it_spending(
        SpendingComparisonRequest(**PARAMS)
    )
    cell = report["comparisons"][0]
    assert cell["total_obligations_usd"] == 364_000_000
    assert cell["latest_yoy_growth_pct"] == 20
    assert cell["cagr_pct"] == pytest.approx(20)
    assert cell["vendors"]["top_three_share_pct"] == 100
    assert cell["vendors"]["hhi"] == 6800
    assert service.share(cell, "competition", "competitive") == 60
    assert service.share(cell, "competition", "follow_on") == 10
    assert service.share(cell, "pricing", "other_or_unreported") == 5
    assert cell["competition"]["code_definitions"]["competitive"]["A"] == (
        "Full and Open Competition"
    )
    assert cell["pricing"]["code_definitions"]["cost_reimbursement"]["S"] == (
        "Cost No Fee"
    )
    assert "code_reference" not in cell["competition"]
    assert cell["priority"] == "Medium"
    assert len(report["recommended_investigations"]) == 2
    assert report["coverage"]["status"] == "complete_for_declared_aggregate_filters"
    assert report["coverage"]["storage"] == "none_request_memory_only"
    assert report["retrieval_started_at_utc"] <= report["retrieved_at_utc"]
    assert cell["unassessed_metrics"]["modifications"]
    markdown = service.render_markdown(report)
    assert "$364,000,000.00" in markdown
    assert "## Recommended Investigations" in markdown
    assert "|---" in markdown and "\n\n|" not in markdown


@pytest.mark.parametrize("basis", ["awarding", "funding"])
async def test_national_scope_and_agency_attribution(monkeypatch, basis):
    def handler(request):
        body = json.loads(request.content)
        for agency in body["filters"].get("agencies", []):
            assert agency["type"] == basis
        if request.url.path in service.AGENCIES.values():
            assert request.url.path == service.AGENCIES[basis]
        return response_handler(request)

    mock_api(monkeypatch, handler)
    report = await service.compare_agency_it_spending(
        SpendingComparisonRequest(**{**PARAMS, "agencies": None, "agency_basis": basis})
    )
    assert len(report["comparisons"]) == 2
    assert (
        report["coverage"]["categories"][0]["scope"] == "national_classified_contracts"
    )
    assert {cell["agency_basis"] for cell in report["comparisons"]} == {basis}


async def test_naics_refines_psc_and_aggregates_precede_details(monkeypatch):
    paths = []

    def handler(request):
        filters = json.loads(request.content)["filters"]
        assert filters["naics_codes"] == {"require": ["541512"]}
        paths.append(request.url.path)
        return response_handler(request)

    mock_api(monkeypatch, handler)
    await service.compare_agency_it_spending(
        SpendingComparisonRequest(
            **{
                **PARAMS,
                "categories": [
                    {"name": "Scoped", "psc_codes": ["D310"], "naics_codes": ["541512"]}
                ],
            }
        )
    )
    assert paths[:4] == [
        service.TOTALS,
        service.AGENCIES["awarding"],
        service.AGENCIES["awarding"],
        service.AGENCIES["awarding"],
    ]


@pytest.mark.parametrize(
    "failure", ["missing_year", "mismatch", "duplicate_agency", "duplicate_year"]
)
async def test_incomplete_aggregate_never_zero_filled(monkeypatch, failure):
    def handler(request):
        payload = response_handler(request).json()
        if request.url.path == service.TOTALS:
            if failure == "missing_year":
                payload["results"] = [
                    row
                    for row in payload["results"]
                    if row["time_period"]["fiscal_year"] != "2024"
                ]
            if failure == "duplicate_year":
                payload["results"].append(payload["results"][0])
        elif request.url.path in service.AGENCIES.values():
            if failure == "mismatch":
                payload["results"][0]["amount"] += 1
            if failure == "duplicate_agency":
                payload["results"].append(payload["results"][0])
        return httpx.Response(200, json=payload)

    mock_api(monkeypatch, handler)
    report = await service.compare_agency_it_spending(
        SpendingComparisonRequest(**PARAMS)
    )
    assert report["comparisons"][0]["total_obligations_usd"] is None
    assert report["comparisons"][0]["cagr_pct"] is None
    assert report["comparisons"][0]["priority"] is None
    assert report["recommended_investigations"] == []
    assert report["coverage"]["status"] == "partial"


async def test_detail_page_limit_suppresses_concentration(monkeypatch):
    monkeypatch.setattr(service, "MAX_PAGES", 1)

    def handler(request):
        payload = response_handler(request).json()
        if request.url.path == service.RECIPIENTS:
            payload["page_metadata"]["hasNext"] = True
        return httpx.Response(200, json=payload)

    mock_api(monkeypatch, handler)
    report = await service.compare_agency_it_spending(
        SpendingComparisonRequest(**PARAMS)
    )
    cell = report["comparisons"][0]
    assert cell["total_obligations_usd"] == 364_000_000
    assert cell["vendors"]["status"] == "page_limit"
    assert "top_vendor_share_pct" not in cell["vendors"]
    assert report["recommended_investigations"] == []


@pytest.mark.parametrize(
    "failure", [302, 429, 503, "timeout", "json", "shape", "oversized"]
)
async def test_errors_budgets_and_redaction(monkeypatch, caplog, failure):
    secret = "seeded-token-123-45-6789"

    def handler(request):
        if isinstance(failure, int):
            return httpx.Response(
                failure, text=secret, headers={"Location": "http://127.0.0.1/"}
            )
        if failure == "timeout":
            raise httpx.ReadTimeout(secret, request=request)
        if failure == "json":
            return httpx.Response(200, text=secret)
        if failure == "oversized":
            return httpx.Response(200, content=b"x" * 101)
        return httpx.Response(200, json={"results": None})

    if failure == "oversized":
        monkeypatch.setattr(service, "MAX_RESPONSE_BYTES", 100)
    mock_api(monkeypatch, handler)
    report = await service.compare_agency_it_spending(
        SpendingComparisonRequest(**PARAMS)
    )
    assert report["comparisons"][0]["total_obligations_usd"] is None
    assert report["recommended_investigations"] == []
    assert secret not in json.dumps(report) + caplog.text


async def test_budget_exhaustion_and_no_details(monkeypatch):
    mock_api(monkeypatch, response_handler)
    report = await service.compare_agency_it_spending(
        SpendingComparisonRequest(**PARAMS, include_details=False)
    )
    assert report["coverage"]["requests_made"] == 4
    assert report["comparisons"][0]["vendors"]["status"] == "not_requested"
    monkeypatch.setattr(service, "MAX_REQUESTS", 1)
    report = await service.compare_agency_it_spending(
        SpendingComparisonRequest(**PARAMS)
    )
    assert report["coverage"]["requests_made"] == 1
    assert report["recommended_investigations"] == []
    monkeypatch.setattr(service, "DEADLINE_SECONDS", 0)
    report = await service.compare_agency_it_spending(
        SpendingComparisonRequest(**PARAMS)
    )
    assert report["coverage"]["requests_made"] == 0


async def test_mcp_registration_formats_and_invalid_input(monkeypatch):
    from fastmcp.exceptions import ToolError

    from mcp_server_n2ia.app import mcp

    mock_api(monkeypatch, response_handler)
    async with Client(mcp) as client:
        tools = await client.list_tools()
        tool = next(tool for tool in tools if tool.name == "compare_agency_it_spending")
        assert "categories" in tool.input_schema["properties"]
        assert "request" not in tool.input_schema["properties"]
        assert "comparisons" in tool.output_schema["properties"]
        response = await client.call_tool(
            tool.name, {**PARAMS, "response_format": "markdown"}
        )
        assert (
            response.structured_content["comparisons"][0]["total_obligations_usd"]
            == 364_000_000
        )
        assert "# Executive Summary" in response.structured_content["markdown"]

        def unexpected(request):
            pytest.fail("Invalid scope must not reach the source API")

        mock_api(monkeypatch, unexpected)
        with pytest.raises(ToolError):
            await client.call_tool(tool.name, {**PARAMS, "end_fiscal_year": 2999})


@pytest.mark.parametrize("failure", [503, 429, "timeout"])
async def test_transient_retry_is_bounded(monkeypatch, failure):
    attempts = []

    def handler(request):
        attempts.append(request.url.path)
        if len(attempts) == 1:
            if failure == "timeout":
                raise httpx.ReadTimeout("sensitive", request=request)
            return httpx.Response(failure, headers={"Retry-After": "0"})
        return response_handler(request)

    mock_api(monkeypatch, handler)
    report = await service.compare_agency_it_spending(
        SpendingComparisonRequest(**PARAMS, include_details=False)
    )
    assert report["coverage"]["requests_made"] == 5
    assert report["comparisons"][0]["total_obligations_usd"] == 364_000_000


async def test_long_retry_after_does_not_block(monkeypatch):
    mock_api(
        monkeypatch, lambda request: httpx.Response(429, headers={"Retry-After": "999"})
    )
    report = await service.compare_agency_it_spending(
        SpendingComparisonRequest(**PARAMS)
    )
    assert report["coverage"]["requests_made"] == 4
    assert report["recommended_investigations"] == []


def test_peer_flags_and_priority_are_material_not_waste_claims():
    cells = []
    for index, growth in enumerate([80, 0, 2, 5]):
        cell = {
            "agency": f"Agency {index}",
            "category": "IT",
            **service.annual_metrics(
                {
                    2023: Decimal(100_000_000),
                    2024: Decimal(100_000_000),
                    2025: Decimal(100_000_000) * (1 + Decimal(growth) / 100),
                }
            ),
            "vendors": {"status": "not_requested"},
            "pricing": {"status": "not_requested"},
            "competition": {
                "status": "complete",
                "buckets": {"noncompetitive": {"share_pct": 80 if index == 0 else 10}},
            },
            "source_references": [],
        }
        cells.append(cell)
    request = SpendingComparisonRequest(**PARAMS)
    findings = service.rank_findings(cells, request)
    assert len(findings) == 1
    assert findings[0]["priority"] == "High"
    assert {signal["signal"] for signal in findings[0]["signals"]} == {
        "growth_outlier",
        "noncompetitive_peer_difference",
    }
    assert cells[0]["peer_comparison"]["median_growth_pct"] == 2
    assert cells[0]["peer_comparison"]["mean_growth_pct"] == pytest.approx(7 / 3)
    assert findings[0]["confidence"] == "Low"
    assert "no waste or savings" in findings[0]["possible_business_impact"]
    assert service.rank_findings(cells[:3], request) == []
    assert (
        service.rank_findings(
            cells,
            SpendingComparisonRequest(
                **PARAMS, thresholds={"materiality_usd": 1_000_000_000}
            ),
        )
        == []
    )


async def test_overlap_warning_and_response_budget(monkeypatch):
    mock_api(monkeypatch, response_handler)
    report = await service.compare_agency_it_spending(
        SpendingComparisonRequest(
            **{
                **PARAMS,
                "categories": [
                    *PARAMS["categories"],
                    {"name": "Overlap", "psc_codes": ["D310"]},
                ],
            },
            include_details=False,
        )
    )
    assert report["coverage"]["overlapping_category_codes"] is True
    monkeypatch.setattr(service, "MAX_TOTAL_BYTES", 1)
    report = await service.compare_agency_it_spending(
        SpendingComparisonRequest(**PARAMS)
    )
    assert report["coverage"]["requests_made"] == 1
    assert report["comparisons"][0]["total_obligations_usd"] is None


async def test_no_file_writes_or_cross_request_cache(monkeypatch):
    from pathlib import Path

    calls = []

    def handler(request):
        calls.append(request.url.path)
        return response_handler(request)

    def no_files(*args, **kwargs):
        pytest.fail("Spending workflow must not open files")

    mock_api(monkeypatch, handler)
    monkeypatch.setattr("builtins.open", no_files)
    monkeypatch.setattr(Path, "open", no_files)
    for _ in range(2):
        report = await service.compare_agency_it_spending(
            SpendingComparisonRequest(**PARAMS, include_details=False)
        )
        assert report["coverage"]["requests_made"] == 4
    assert len(calls) == 8
