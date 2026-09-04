"""
Low-level wrapper around google-play-scraper.

This module is the ONLY place in the codebase that talks to
google-play-scraper directly. Everything above it (play_store_service)
works with plain dicts in, custom exceptions out -- so the scraper
library could be swapped later without touching callers.
"""

from __future__ import annotations

from typing import Any, Optional

import requests
from google_play_scraper import Sort
from google_play_scraper import app as _gps_app
from google_play_scraper import reviews as _gps_reviews
from google_play_scraper.exceptions import NotFoundError as _GPSNotFoundError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import settings
from services.exceptions import AppNotFoundError, ScraperNetworkError
from utils.logger import get_logger

logger = get_logger(__name__)

# Transient/transport-level failures worth retrying. NotFoundError is
# deliberately excluded -- a missing app is a deterministic result,
# not a transient failure, so retrying it would just waste time.
_RETRYABLE_EXCEPTIONS = (requests.exceptions.RequestException, ConnectionError, TimeoutError, OSError)

_retry_policy = retry(
    stop=stop_after_attempt(settings.REQUEST_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=settings.REQUEST_BACKOFF_SECONDS, min=1, max=20),
    retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
    reraise=True,
)


@_retry_policy
def fetch_app_details(package_id: str, country: str, lang: str) -> dict[str, Any]:
    """
    Fetch raw app details for a single package ID.

    Raises:
        AppNotFoundError: the package ID doesn't exist for this locale
            (not retried).
        ScraperNetworkError: transport failure after all retries.
    """
    logger.debug("Fetching app details | package_id=%s country=%s lang=%s", package_id, country, lang)
    try:
        return _gps_app(package_id, lang=lang, country=country)
    except _GPSNotFoundError as exc:
        raise AppNotFoundError(f"App not found: {package_id} ({country}/{lang})") from exc
    except _RETRYABLE_EXCEPTIONS:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize any other scraper failure
        raise ScraperNetworkError(f"Failed to fetch app details for {package_id}: {exc}") from exc


@_retry_policy
def fetch_reviews_batch(
    package_id: str,
    country: str,
    lang: str,
    count: int,
    continuation_token: Optional[Any] = None,
) -> tuple[list[dict[str, Any]], Optional[Any]]:
    """
    Fetch one page of raw reviews, newest first.

    Returns (batch, continuation_token). continuation_token is None
    once the store has no further reviews to page through. Used by
    play_store_service's Phase 4 pagination loop.
    """
    logger.debug("Fetching reviews batch | package_id=%s count=%s", package_id, count)
    try:
        batch, token = _gps_reviews(
            package_id,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,
            count=count,
            continuation_token=continuation_token,
        )
        return batch, token
    except _GPSNotFoundError as exc:
        raise AppNotFoundError(f"App not found: {package_id} ({country}/{lang})") from exc
    except _RETRYABLE_EXCEPTIONS:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ScraperNetworkError(f"Failed to fetch reviews for {package_id}: {exc}") from exc
