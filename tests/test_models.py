"""Tests for the pydantic models' validation contracts."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from models.analysis import AnalysisResult, ComparisonRow, ComparisonTable, RatingDistribution, ReviewSampleSummary
from models.app import AppMetadata
from models.review import Review


# --- AppMetadata --------------------------------------------------------------


def test_app_metadata_accepts_valid_package_id():
    app = AppMetadata(package_id="com.example.app", title="Example")
    assert app.package_id == "com.example.app"


@pytest.mark.parametrize(
    "bad_id",
    [
        "not-a-package-id",
        "com.rival.0",  # segment starting with a digit
        "com",  # single segment, no dot
        "",
        "com..app",  # empty segment
    ],
)
def test_app_metadata_rejects_invalid_package_id(bad_id):
    with pytest.raises(ValidationError):
        AppMetadata(package_id=bad_id, title="X")


def test_app_metadata_score_must_be_within_zero_to_five():
    with pytest.raises(ValidationError):
        AppMetadata(package_id="com.example.app", title="X", score=5.5)
    with pytest.raises(ValidationError):
        AppMetadata(package_id="com.example.app", title="X", score=-0.1)
    AppMetadata(package_id="com.example.app", title="X", score=5.0)  # boundary is inclusive


def test_app_metadata_optional_fields_default_to_none():
    app = AppMetadata(package_id="com.example.app", title="X")
    assert app.developer is None
    assert app.score is None
    assert app.rating_count is None
    assert app.installs is None
    assert app.updated is None


def test_app_metadata_rating_count_cannot_be_negative():
    with pytest.raises(ValidationError):
        AppMetadata(package_id="com.example.app", title="X", rating_count=-5)


# --- Review ---------------------------------------------------------------------


def test_review_accepts_blank_content():
    """A star-only review with no written text is a valid, common case -- not an error."""
    review = Review(review_id="r1", rating=5, content="")
    assert review.content == ""


@pytest.mark.parametrize("rating", [0, 6, -1])
def test_review_rejects_rating_out_of_bounds(rating):
    with pytest.raises(ValidationError):
        Review(review_id="r1", rating=rating, content="ok")


def test_review_rejects_blank_review_id():
    with pytest.raises(ValidationError):
        Review(review_id="   ", rating=3, content="ok")


def test_review_is_negative_and_is_positive_properties():
    assert Review(review_id="r1", rating=1, content="").is_negative is True
    assert Review(review_id="r2", rating=2, content="").is_negative is True
    assert Review(review_id="r3", rating=3, content="").is_negative is False
    assert Review(review_id="r4", rating=3, content="").is_positive is False
    assert Review(review_id="r5", rating=4, content="").is_positive is True
    assert Review(review_id="r6", rating=5, content="").is_positive is True


def test_review_date_is_optional():
    review = Review(review_id="r1", rating=3, content="ok")
    assert review.review_date is None
    dated = Review(review_id="r2", rating=3, content="ok", review_date=datetime(2026, 1, 1))
    assert dated.review_date == datetime(2026, 1, 1)


# --- AnalysisResult cross-field validation --------------------------------------


def _valid_analysis_result_kwargs(total_reviews: int) -> dict:
    target = AppMetadata(package_id="com.example.app", title="X")
    return dict(
        target_app=target,
        competitors=[],
        rating_distribution=RatingDistribution(one_star=total_reviews),
        sample_summary=ReviewSampleSummary(total_reviews_analyzed=total_reviews, average_rating=1.0),
        comparison_table=ComparisonTable(rows=[]),
    )


def test_analysis_result_accepts_matching_totals():
    kwargs = _valid_analysis_result_kwargs(total_reviews=10)
    result = AnalysisResult(**kwargs)
    assert result.rating_distribution.total == result.sample_summary.total_reviews_analyzed == 10


def test_analysis_result_rejects_mismatched_totals():
    """Regression guard: rating_distribution and sample_summary must always agree, or
    the PDF's 'total reviews analyzed' figure would silently disagree with its own chart."""
    kwargs = _valid_analysis_result_kwargs(total_reviews=10)
    kwargs["sample_summary"] = ReviewSampleSummary(total_reviews_analyzed=999, average_rating=1.0)
    with pytest.raises(ValidationError, match="does not match"):
        AnalysisResult(**kwargs)


def test_comparison_table_caps_at_four_rows():
    rows = [ComparisonRow(label=f"App {i}", package_id="com.example.app") for i in range(5)]
    with pytest.raises(ValidationError):
        ComparisonTable(rows=rows)
