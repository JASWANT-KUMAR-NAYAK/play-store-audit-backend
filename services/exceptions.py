"""Custom exceptions for the Play Store data-fetch layer."""

from __future__ import annotations


class PlayStoreServiceError(Exception):
    """Base class for all play_store_service / scraper_client errors."""


class AppNotFoundError(PlayStoreServiceError):
    """Raised when a package ID does not exist on the store for the given locale.

    Deterministic -- never retried.
    """


class ScraperNetworkError(PlayStoreServiceError):
    """Raised when a scraper request fails after all retry attempts are exhausted."""


class ReportGenerationError(Exception):
    """Raised when PDF assembly fails for any reason (bad layout data, reportlab
    internal error, unwritable output path, etc).

    Deliberately a plain Exception, not a PlayStoreServiceError -- report
    generation is a different failure domain from data fetching.
    """


class LLMProviderError(Exception):
    """Raised by an LLMProvider when the call fails or the response can't be
    parsed into valid insights (auth failure, network error, malformed JSON,
    missing fields).

    Always caught by generate_llm_insights() and converted into an
    unavailable LLMInsights -- never allowed to propagate and crash
    the report pipeline, per spec: the app must work without an LLM key.
    """
