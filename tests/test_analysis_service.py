"""Tests for services/analysis_service.py."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from models.analysis import ComparisonTable, ReviewSampleSummary, ThemeItem
from models.app import AppMetadata
from models.review import Review
from services import analysis_service


def _app(package_id: str = "com.example.app", **overrides) -> AppMetadata:
    defaults = dict(package_id=package_id, title="Some App")
    defaults.update(overrides)
    return AppMetadata(**defaults)


def _review(review_id: str, rating: int, content: str = "") -> Review:
    return Review(review_id=review_id, rating=rating, content=content, review_date=datetime(2026, 7, 1))


# --- _rating_distribution / _sample_summary (via analyze()) ------------------


def test_rating_distribution_counts_each_star_correctly():
    reviews = [_review("r1", 1), _review("r2", 1), _review("r3", 3), _review("r4", 5), _review("r5", 5), _review("r6", 5)]
    result = analysis_service.analyze(_app(), [], reviews)
    dist = result.rating_distribution
    assert (dist.one_star, dist.two_star, dist.three_star, dist.four_star, dist.five_star) == (2, 0, 1, 0, 3)
    assert dist.total == 6


def test_sample_summary_average_rating_is_correct():
    reviews = [_review("r1", 1), _review("r2", 5)]
    result = analysis_service.analyze(_app(), [], reviews)
    assert result.sample_summary.total_reviews_analyzed == 2
    assert result.sample_summary.average_rating == 3.0


def test_empty_reviews_produce_none_average_not_a_crash():
    result = analysis_service.analyze(_app(), [], [])
    assert result.sample_summary.total_reviews_analyzed == 0
    assert result.sample_summary.average_rating is None
    assert result.rating_distribution.total == 0
    assert result.complaint_themes == []
    assert result.praise_themes == []


def test_analyze_splits_reviews_into_negative_and_positive_correctly():
    """1-2 star -> complaint pool, 4-5 star -> praise pool, 3 star -> neither."""
    reviews = (
        [_review(f"neg{i}", 1, "terrible crashing bug here") for i in range(3)]
        + [_review(f"pos{i}", 5, "wonderful clean design here") for i in range(3)]
        + [_review("neutral", 3, "crashing wonderful mixed review")]
    )
    result = analysis_service.analyze(_app(), [], reviews)
    complaint_phrases = {t.phrase for t in result.complaint_themes}
    praise_phrases = {t.phrase for t in result.praise_themes}
    # Substring match, not exact: subsumed-unigram dedup may leave only the bigram
    # (e.g. "crashing bug" survives while bare "crashing" is correctly dropped).
    assert any("crashing" in p or "terrible" in p for p in complaint_phrases)
    assert any("wonderful" in p or "clean" in p for p in praise_phrases)
    # The lone 3-star review shouldn't be enough on its own to leak into either pool
    # (min_reviews=2 threshold in extract_common_themes handles this).


# --- _build_comparison_table (via analyze()) ----------------------------------


def test_comparison_table_always_puts_target_first():
    target = _app("com.example.target", title="Target")
    comp = _app("com.rival.one", title="Rival")
    result = analysis_service.analyze(target, [comp], [])
    assert result.comparison_table.rows[0].label == "Target"
    assert result.comparison_table.rows[1].label == "Rival"


def test_comparison_table_caps_at_max_competitors(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "MAX_COMPETITORS", 2)
    target = _app("com.example.target")
    competitors = [_app(f"com.rival.app{i}", title=f"Rival {i}") for i in range(5)]
    result = analysis_service.analyze(target, competitors, [])
    assert len(result.comparison_table.rows) == 3  # target + 2 competitors, not 5


def test_comparison_table_days_since_update_uses_reference_date():
    target = _app(updated=date(2026, 1, 1))
    table = analysis_service._build_comparison_table(target, [], reference_date=date(2026, 1, 31))
    assert table.rows[0].days_since_update == 30


def test_comparison_table_handles_missing_updated_date():
    target = _app(updated=None)
    table = analysis_service._build_comparison_table(target, [])
    assert table.rows[0].days_since_update is None


# --- _build_recommendations ----------------------------------------------------


def test_recommendations_zero_reviews_gives_single_data_quality_caveat():
    recs = analysis_service._build_recommendations(
        ReviewSampleSummary(total_reviews_analyzed=0, average_rating=None),
        [], [], ComparisonTable(rows=[]),
    )
    assert len(recs) == 1
    assert "No reviews were available" in recs[0]


def test_recommendations_leads_with_top_complaint_theme():
    summary = ReviewSampleSummary(total_reviews_analyzed=20, average_rating=2.5)
    complaint_themes = [ThemeItem(phrase="crashes constantly", count=15)]
    recs = analysis_service._build_recommendations(summary, complaint_themes, [], ComparisonTable(rows=[]))
    assert any("crashes constantly" in r for r in recs)


def test_recommendations_flags_stale_update_cadence_vs_competitors():
    target = _app(updated=date(2026, 1, 1))
    comp1 = _app("com.rival.one", updated=date(2026, 8, 1))
    comp2 = _app("com.rival.two", updated=date(2026, 8, 10))
    table = analysis_service._build_comparison_table(target, [comp1, comp2], reference_date=date(2026, 8, 20))
    summary = ReviewSampleSummary(total_reviews_analyzed=10, average_rating=4.0)
    recs = analysis_service._build_recommendations(summary, [], [], table)
    assert any("Update cadence lags" in r for r in recs)


def test_recommendations_does_not_flag_cadence_when_target_is_current():
    target = _app(updated=date(2026, 8, 15))
    comp = _app("com.rival.one", updated=date(2026, 6, 1))
    table = analysis_service._build_comparison_table(target, [comp], reference_date=date(2026, 8, 20))
    summary = ReviewSampleSummary(total_reviews_analyzed=10, average_rating=4.0)
    recs = analysis_service._build_recommendations(summary, [], [], table)
    assert not any("Update cadence lags" in r for r in recs)


def test_recommendations_includes_praise_theme_when_room_remains():
    summary = ReviewSampleSummary(total_reviews_analyzed=20, average_rating=4.5)
    praise_themes = [ThemeItem(phrase="clean interface", count=18)]
    recs = analysis_service._build_recommendations(summary, [], praise_themes, ComparisonTable(rows=[]))
    assert any("clean interface" in r for r in recs)


def test_recommendations_caps_at_three():
    summary = ReviewSampleSummary(total_reviews_analyzed=50, average_rating=3.0)
    complaint_themes = [ThemeItem(phrase="crashes constantly", count=20)]
    praise_themes = [ThemeItem(phrase="clean interface", count=20)]
    target = _app(updated=date(2026, 1, 1))
    comp = _app("com.rival.one", updated=date(2026, 8, 1))
    table = analysis_service._build_comparison_table(target, [comp], reference_date=date(2026, 8, 20))
    recs = analysis_service._build_recommendations(summary, complaint_themes, praise_themes, table)
    assert len(recs) <= 3


def test_recommendations_lead_theme_prefers_multiword_phrase():
    """Regression: the recommendation sentence must not surface a bare generic unigram when a
    more specific multi-word phrase is available at a similar rank (see text_cleaner.pick_lead_theme)."""
    summary = ReviewSampleSummary(total_reviews_analyzed=20, average_rating=3.0)
    complaint_themes = [ThemeItem(phrase="clean", count=40), ThemeItem(phrase="clean interface", count=35)]
    recs = analysis_service._build_recommendations(summary, complaint_themes, [], ComparisonTable(rows=[]))
    assert any("clean interface" in r for r in recs)
    assert not any("'clean'" in r for r in recs)  # the bare unigram should not be the featured phrase


def test_recommendations_fallback_when_no_themes_or_cadence_signal():
    summary = ReviewSampleSummary(total_reviews_analyzed=10, average_rating=3.5)
    recs = analysis_service._build_recommendations(summary, [], [], ComparisonTable(rows=[]))
    assert len(recs) == 1
    assert "mixed" in recs[0].lower() or "survey" in recs[0].lower()


# --- analyze(): AnalysisResult cross-field consistency ------------------------


def test_analyze_result_rating_distribution_matches_sample_summary_total():
    """AnalysisResult's own cross-field validator should never fire here -- analyze() must
    always produce a rating_distribution.total that matches sample_summary.total_reviews_analyzed."""
    reviews = [_review(f"r{i}", (i % 5) + 1) for i in range(37)]
    result = analysis_service.analyze(_app(), [], reviews)
    assert result.rating_distribution.total == result.sample_summary.total_reviews_analyzed == 37
