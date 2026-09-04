"""
Fetches and normalizes app metadata and reviews for the target app
and metadata for its competitors.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from pydantic import ValidationError

from config import settings
from models.app import AppMetadata
from models.review import Review
from services import scraper_client
from services.exceptions import AppNotFoundError, ScraperNetworkError
from utils.logger import get_logger

logger = get_logger(__name__)

# Safety valve: even if the store keeps returning fresh continuation
# tokens, stop after this many pagination round-trips so a scraper
# quirk can never spin the loop forever.
_MAX_REVIEW_FETCH_LOOPS = 15

# google-play-scraper accepts at most ~200 reviews per single request
# regardless of the count passed in.
_MAX_REVIEWS_PER_REQUEST = 200


def _parse_updated(raw_timestamp: Optional[int]) -> Optional[date]:
    """google-play-scraper returns 'updated' as a Unix timestamp (or None)."""
    if raw_timestamp is None:
        return None
    try:
        return datetime.fromtimestamp(raw_timestamp).date()
    except (OSError, ValueError, OverflowError, TypeError):
        logger.debug("Could not parse 'updated' timestamp: %r", raw_timestamp)
        return None


def _to_app_metadata(raw: dict[str, Any]) -> AppMetadata:
    """Map a raw google-play-scraper app() dict onto our AppMetadata model."""
    return AppMetadata(
        package_id=raw.get("appId", ""),
        title=raw.get("title", ""),
        developer=raw.get("developer"),
        score=raw.get("score"),
        rating_count=raw.get("ratings"),
        installs=raw.get("installs"),
        genre=raw.get("genre"),
        updated=_parse_updated(raw.get("updated")),
        version=raw.get("version"),
    )


def fetch_app(package_id: str, country: Optional[str] = None, lang: Optional[str] = None) -> AppMetadata:
    """
    Fetch and normalize metadata for a single app.

    Propagates AppNotFoundError / ScraperNetworkError so callers can
    choose different handling for the target app (hard failure) vs.
    competitor apps (soft failure -- skip and continue).
    """
    country = country or settings.DEFAULT_COUNTRY
    lang = lang or settings.DEFAULT_LANG

    raw = scraper_client.fetch_app_details(package_id, country=country, lang=lang)
    metadata = _to_app_metadata(raw)
    logger.info("Fetched app metadata | %s -> '%s' (score=%s)", package_id, metadata.title, metadata.score)
    return metadata


def fetch_target_and_competitors(
    target_id: str,
    competitor_ids: list[str],
    country: Optional[str] = None,
    lang: Optional[str] = None,
) -> tuple[AppMetadata, list[AppMetadata]]:
    """
    Fetch the target app and up to MAX_COMPETITORS competitor apps.

    The target app is required -- a not-found or network error here
    propagates, since there's no report without it. Competitor
    failures are logged and skipped individually so one bad
    competitor ID doesn't sink the whole run (per spec: "Missing
    competitor IDs" must be handled gracefully).
    """
    country = country or settings.DEFAULT_COUNTRY
    lang = lang or settings.DEFAULT_LANG

    target = fetch_app(target_id, country=country, lang=lang)

    competitors: list[AppMetadata] = []
    for comp_id in competitor_ids[: settings.MAX_COMPETITORS]:
        try:
            competitors.append(fetch_app(comp_id, country=country, lang=lang))
        except AppNotFoundError:
            logger.warning("Competitor app not found, skipping | package_id=%s", comp_id)
        except ScraperNetworkError as exc:
            logger.warning("Competitor fetch failed after retries, skipping | package_id=%s error=%s", comp_id, exc)

    return target, competitors


def _to_review(raw: dict[str, Any]) -> Optional[Review]:
    """
    Map a raw google-play-scraper review dict onto our Review model.

    Returns None (rather than raising) for a malformed review -- e.g.
    a missing score -- so one bad record can't take down the whole
    batch. Per spec, "missing optional fields" must be handled
    gracefully; a missing *required* field (rating) means we simply
    can't use that review, so we skip and log instead of failing.
    """
    review_id = raw.get("reviewId")
    score = raw.get("score")
    if not review_id or score is None:
        logger.debug("Skipping malformed review (missing id/score): %r", raw.get("reviewId"))
        return None

    try:
        return Review(
            review_id=str(review_id),
            user_name=raw.get("userName"),
            rating=int(score),
            content=raw.get("content") or "",
            review_date=raw.get("at"),
            app_version=raw.get("appVersion") or raw.get("reviewCreatedVersion"),
        )
    except (ValidationError, ValueError, TypeError) as exc:
        logger.debug("Skipping malformed review %s: %s", review_id, exc)
        return None


def fetch_reviews(
    target_id: str,
    country: Optional[str] = None,
    lang: Optional[str] = None,
    sample_size: Optional[int] = None,
) -> list[Review]:
    """
    Fetch up to `sample_size` recent reviews for the target app,
    newest first, paginating via continuation tokens as needed.

    Returns an empty list (never raises) when the app genuinely has
    no reviews or every fetch attempt fails -- per spec, "empty
    review lists" is a state the rest of the pipeline must tolerate,
    not an error condition on its own. A hard AppNotFoundError is
    still propagated, since that indicates a bad package ID rather
    than an app with zero reviews.
    """
    country = country or settings.DEFAULT_COUNTRY
    lang = lang or settings.DEFAULT_LANG
    sample_size = sample_size or settings.REVIEW_SAMPLE_SIZE

    collected: list[Review] = []
    seen_ids: set[str] = set()
    token: Optional[Any] = None
    loops = 0

    while len(collected) < sample_size and loops < _MAX_REVIEW_FETCH_LOOPS:
        loops += 1
        remaining = sample_size - len(collected)
        batch_count = min(remaining, _MAX_REVIEWS_PER_REQUEST)

        try:
            batch, token = scraper_client.fetch_reviews_batch(
                target_id, country=country, lang=lang, count=batch_count, continuation_token=token
            )
        except AppNotFoundError:
            raise  # bad package ID -- let the caller treat this as fatal
        except ScraperNetworkError as exc:
            logger.warning(
                "Review fetch failed after retries on loop %d, returning %d reviews collected so far | error=%s",
                loops, len(collected), exc,
            )
            break

        if not batch:
            logger.debug("Empty review batch received, stopping pagination.")
            break

        for raw in batch:
            review = _to_review(raw)
            if review is not None and review.review_id not in seen_ids:
                seen_ids.add(review.review_id)
                collected.append(review)

        if token is None:
            break  # store has no more reviews to page through

    if not collected:
        logger.warning("No reviews collected for %s -- report will note an empty review sample.", target_id)
    else:
        logger.info("Collected %d reviews for %s (requested %d)", len(collected), target_id, sample_size)

    return collected[:sample_size]
