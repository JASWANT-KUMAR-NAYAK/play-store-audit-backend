"""
Integration/smoke tests for the report-generation pipeline
(chart_service + report_service + templates.report_templates).

These deliberately avoid shelling out to external tools like
pdftoppm/pdfinfo (used for manual visual QA during development, see
project history) since a committed test suite should not depend on
system binaries the person running `pytest` may not have installed.
Validity is checked via the PDF magic-byte header and a sane minimum
file size instead -- portable everywhere Python runs.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest

from models.analysis import LLMInsights
from models.app import AppMetadata
from models.review import Review
from services import analysis_service, chart_service, report_service
from services.exceptions import ReportGenerationError


def _target_app() -> AppMetadata:
    return AppMetadata(
        package_id="com.example.target", title="TaskFlow Pro", score=3.6,
        rating_count=12000, installs="1,000,000+", genre="Productivity",
        updated=date(2026, 3, 1), version="2.0.0", developer="Acme Inc",
    )


def _competitor() -> AppMetadata:
    return AppMetadata(package_id="com.rival.one", title="Rival Tasks", score=4.4, updated=date(2026, 8, 1))


def _reviews(count: int = 20) -> list[Review]:
    return [
        Review(review_id=f"r{i}", rating=(i % 5) + 1, content=f"Review text number {i}", review_date=datetime(2026, 7, 1))
        for i in range(count)
    ]


def _assert_looks_like_a_real_pdf(path: Path) -> None:
    assert path.exists()
    data = path.read_bytes()
    assert data.startswith(b"%PDF-"), "output does not start with the PDF magic bytes"
    assert len(data) > 5000, f"output is suspiciously small ({len(data)} bytes) for a multi-section report"


# --- End-to-end generation across the realistic scenarios --------------------


def test_full_pipeline_generates_a_valid_pdf_without_llm(tmp_path):
    result = analysis_service.analyze(_target_app(), [_competitor()], _reviews())
    chart_paths = chart_service.generate_all_charts(result, tmp_path / "charts")
    output_path = report_service.generate_report(
        analysis_result=result, chart_paths=chart_paths,
        output_path=tmp_path / "report.pdf", llm_insights=None,
    )
    _assert_looks_like_a_real_pdf(output_path)


def test_full_pipeline_generates_a_valid_pdf_with_llm_insights(tmp_path):
    result = analysis_service.analyze(_target_app(), [_competitor()], _reviews())
    chart_paths = chart_service.generate_all_charts(result, tmp_path / "charts")
    llm_insights = LLMInsights(
        available=True,
        complaint_insights=["Stability is the dominant concern."],
        praise_insights=["The interface is well liked."],
        recommendations=["Ship a stability patch first."],
    )
    output_path = report_service.generate_report(
        analysis_result=result, chart_paths=chart_paths,
        output_path=tmp_path / "report.pdf", llm_insights=llm_insights,
    )
    _assert_looks_like_a_real_pdf(output_path)


def test_pipeline_handles_zero_reviews_without_crashing(tmp_path):
    result = analysis_service.analyze(_target_app(), [_competitor()], [])
    chart_paths = chart_service.generate_all_charts(result, tmp_path / "charts")
    output_path = report_service.generate_report(
        analysis_result=result, chart_paths=chart_paths,
        output_path=tmp_path / "report.pdf", llm_insights=None,
    )
    _assert_looks_like_a_real_pdf(output_path)


def test_pipeline_handles_zero_competitors_without_crashing(tmp_path):
    result = analysis_service.analyze(_target_app(), [], _reviews())
    chart_paths = chart_service.generate_all_charts(result, tmp_path / "charts")
    output_path = report_service.generate_report(
        analysis_result=result, chart_paths=chart_paths,
        output_path=tmp_path / "report.pdf", llm_insights=None,
    )
    _assert_looks_like_a_real_pdf(output_path)


def test_pipeline_handles_competitor_with_no_score(tmp_path):
    """Regression: an app with score=None must render as a 'No data' bar/cell, not crash the chart or table."""
    scoreless_competitor = AppMetadata(package_id="com.rival.two", title="No Score App", score=None)
    result = analysis_service.analyze(_target_app(), [scoreless_competitor], _reviews())
    chart_paths = chart_service.generate_all_charts(result, tmp_path / "charts")
    output_path = report_service.generate_report(
        analysis_result=result, chart_paths=chart_paths,
        output_path=tmp_path / "report.pdf", llm_insights=None,
    )
    _assert_looks_like_a_real_pdf(output_path)


def test_pipeline_handles_xml_special_characters_in_app_names(tmp_path):
    """Regression: '&', '<', '>' in a scraped app title/review must not break reportlab's Paragraph markup."""
    target = AppMetadata(package_id="com.example.target", title="Cut & Paste Pro <Beta>", score=4.0)
    competitor = AppMetadata(package_id="com.rival.one", title="Simple To-Do & More", score=4.2)
    reviews = [
        Review(review_id="r1", rating=1, content="Bug: fails when x < y & the sync breaks too"),
        Review(review_id="r2", rating=1, content="Same crash < still > happens & annoys me"),
    ]
    result = analysis_service.analyze(target, [competitor], reviews)
    chart_paths = chart_service.generate_all_charts(result, tmp_path / "charts")
    output_path = report_service.generate_report(
        analysis_result=result, chart_paths=chart_paths,
        output_path=tmp_path / "report.pdf", llm_insights=None,
    )
    _assert_looks_like_a_real_pdf(output_path)


# --- Failure handling -----------------------------------------------------------


def test_generate_report_wraps_failures_in_report_generation_error(tmp_path):
    result = analysis_service.analyze(_target_app(), [], _reviews())
    chart_paths = chart_service.generate_all_charts(result, tmp_path / "charts")
    # Point at a chart path that doesn't exist -- reportlab's Image flowable will fail to load it.
    broken_chart_paths = chart_paths.__class__(
        rating_distribution=tmp_path / "does-not-exist.png",
        competitor_comparison=chart_paths.competitor_comparison,
    )
    with pytest.raises(ReportGenerationError):
        report_service.generate_report(
            analysis_result=result, chart_paths=broken_chart_paths,
            output_path=tmp_path / "report.pdf", llm_insights=None,
        )
