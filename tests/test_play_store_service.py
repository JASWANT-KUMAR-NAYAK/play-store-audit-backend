"""
Tests for services/play_store_service.py.

The scraper boundary (services.scraper_client) is monkeypatched
throughout -- no real network calls. This sandbox's egress allowlist
doesn't include Play Store domains anyway, so this is the only way
to exercise the real mapping/pagination/error-handling logic here;
see the module docstring history for that context.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from services import play_store_service
from services.exceptions import AppNotFoundError, ScraperNetworkError

# --- Fixtures: realistic raw scraper payloads --------------------------------


def _raw_app(package_id: str = "com.example.target", **overrides) -> dict:
    defaults = dict(
        appId=package_id, title="Example App", developer="Acme Inc",
        score=4.2, ratings=15000, installs="1,000,000+", genre="Productivity",
        updated=1750000000, version="3.2.1",
    )
    defaults.update(overrides)
    return defaults


def _raw_review(review_id: str = "rid-1", **overrides) -> dict:
    defaults = dict(
        reviewId=review_id, userName="Some User", content="Great app",
        score=4, at=datetime(2026, 6, 1), appVersion="3.2.1",
    )
    defaults.update(overrides)
    return defaults


# --- fetch_app / _to_app_metadata / _parse_updated ---------------------------


def test_fetch_app_maps_raw_dict_correctly(monkeypatch):
    monkeypatch.setattr(
        play_store_service.scraper_client, "fetch_app_details", lambda package_id, country, lang: _raw_app()
    )
    app = play_store_service.fetch_app("com.example.target", country="in", lang="en")
    assert app.package_id == "com.example.target"
    assert app.title == "Example App"
    assert app.rating_count == 15000
    assert str(app.updated) == "2025-06-15"  # Unix timestamp 1750000000 correctly converted to a date


def test_parse_updated_handles_garbage_timestamp_gracefully():
    assert play_store_service._parse_updated(None) is None
    assert play_store_service._parse_updated("not-a-number") is None
    assert play_store_service._parse_updated(99999999999999999999) is None  # OverflowError internally


# --- fetch_target_and_competitors: graceful competitor handling --------------


def test_target_not_found_propagates(monkeypatch):
    def _raise(package_id, country, lang):
        raise AppNotFoundError(f"not found: {package_id}")

    monkeypatch.setattr(play_store_service.scraper_client, "fetch_app_details", _raise)
    with pytest.raises(AppNotFoundError):
        play_store_service.fetch_target_and_competitors("com.example.target", [])


def test_competitor_not_found_is_skipped_not_fatal(monkeypatch):
    def _fetch(package_id, country, lang):
        if package_id == "com.example.target":
            return _raw_app(package_id)
        raise AppNotFoundError(f"not found: {package_id}")

    monkeypatch.setattr(play_store_service.scraper_client, "fetch_app_details", _fetch)
    target, competitors = play_store_service.fetch_target_and_competitors(
        "com.example.target", ["com.rival.missing"]
    )
    assert target.package_id == "com.example.target"
    assert competitors == []


def test_competitor_network_failure_is_skipped_not_fatal(monkeypatch):
    def _fetch(package_id, country, lang):
        if package_id == "com.example.target":
            return _raw_app(package_id)
        raise ScraperNetworkError("persistent failure")

    monkeypatch.setattr(play_store_service.scraper_client, "fetch_app_details", _fetch)
    target, competitors = play_store_service.fetch_target_and_competitors(
        "com.example.target", ["com.rival.flaky"]
    )
    assert target.package_id == "com.example.target"
    assert competitors == []


def test_mixed_good_and_bad_competitors(monkeypatch):
    def _fetch(package_id, country, lang):
        if package_id == "com.rival.missing":
            raise AppNotFoundError("gone")
        if package_id == "com.rival.flaky":
            raise ScraperNetworkError("network down")
        return _raw_app(package_id)

    monkeypatch.setattr(play_store_service.scraper_client, "fetch_app_details", _fetch)
    target, competitors = play_store_service.fetch_target_and_competitors(
        "com.example.target", ["com.rival.good", "com.rival.missing", "com.rival.flaky"]
    )
    assert [c.package_id for c in competitors] == ["com.rival.good"]


def test_competitor_ids_capped_at_max_competitors(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "MAX_COMPETITORS", 2)
    monkeypatch.setattr(
        play_store_service.scraper_client, "fetch_app_details", lambda package_id, country, lang: _raw_app(package_id)
    )
    target, competitors = play_store_service.fetch_target_and_competitors(
        "com.example.target", ["com.rival.a", "com.rival.b", "com.rival.c", "com.rival.d"]
    )
    assert len(competitors) == 2


# --- _to_review ----------------------------------------------------------------


def test_to_review_maps_valid_raw_review():
    review = play_store_service._to_review(_raw_review())
    assert review is not None
    assert review.review_id == "rid-1"
    assert review.rating == 4


def test_to_review_skips_missing_id():
    assert play_store_service._to_review(_raw_review(reviewId=None)) is None


def test_to_review_skips_missing_score():
    assert play_store_service._to_review(_raw_review(score=None)) is None


# --- fetch_reviews: pagination, dedup, resilience -----------------------------


def test_fetch_reviews_paginates_across_pages_to_reach_sample_size(monkeypatch):
    page1 = [_raw_review(f"r{i}") for i in range(100)]
    page2 = [_raw_review(f"r{i}") for i in range(100, 150)]
    calls = []

    def _fetch_batch(package_id, country, lang, count, continuation_token=None):
        calls.append(continuation_token)
        if continuation_token is None:
            return page1, "token-1"
        return page2, None

    monkeypatch.setattr(play_store_service.scraper_client, "fetch_reviews_batch", _fetch_batch)
    reviews = play_store_service.fetch_reviews("com.example.target", sample_size=150)
    assert len(reviews) == 150
    assert calls == [None, "token-1"]


def test_fetch_reviews_dedupes_across_pages(monkeypatch):
    page1 = [_raw_review("r1"), _raw_review("r2")]
    page2 = [_raw_review("r2"), _raw_review("r3")]  # r2 repeated

    def _fetch_batch(package_id, country, lang, count, continuation_token=None):
        return (page1, "tok") if continuation_token is None else (page2, None)

    monkeypatch.setattr(play_store_service.scraper_client, "fetch_reviews_batch", _fetch_batch)
    reviews = play_store_service.fetch_reviews("com.example.target", sample_size=50)
    ids = [r.review_id for r in reviews]
    assert len(ids) == len(set(ids)) == 3


def test_fetch_reviews_skips_malformed_without_losing_the_batch(monkeypatch):
    batch = [_raw_review("r1"), _raw_review(None), {"reviewId": "r3"}, _raw_review("r4")]  # missing id, missing score

    monkeypatch.setattr(
        play_store_service.scraper_client, "fetch_reviews_batch",
        lambda package_id, country, lang, count, continuation_token=None: (batch, None),
    )
    reviews = play_store_service.fetch_reviews("com.example.target", sample_size=50)
    assert [r.review_id for r in reviews] == ["r1", "r4"]


def test_fetch_reviews_empty_result_returns_empty_list_not_raise(monkeypatch):
    monkeypatch.setattr(
        play_store_service.scraper_client, "fetch_reviews_batch",
        lambda package_id, country, lang, count, continuation_token=None: ([], None),
    )
    assert play_store_service.fetch_reviews("com.example.target", sample_size=50) == []


def test_fetch_reviews_network_failure_mid_pagination_returns_partial_results(monkeypatch):
    def _fetch_batch(package_id, country, lang, count, continuation_token=None):
        if continuation_token is None:
            return [_raw_review("r1"), _raw_review("r2")], "tok"
        raise ScraperNetworkError("persistent failure")

    monkeypatch.setattr(play_store_service.scraper_client, "fetch_reviews_batch", _fetch_batch)
    reviews = play_store_service.fetch_reviews("com.example.target", sample_size=50)
    assert len(reviews) == 2  # partial results preserved, not lost to the failure


def test_fetch_reviews_app_not_found_propagates(monkeypatch):
    def _raise(package_id, country, lang, count, continuation_token=None):
        raise AppNotFoundError("gone")

    monkeypatch.setattr(play_store_service.scraper_client, "fetch_reviews_batch", _raise)
    with pytest.raises(AppNotFoundError):
        play_store_service.fetch_reviews("com.example.target", sample_size=50)
