"""
Provider-agnostic LLM insight generation (Anthropic / OpenAI, config-selectable).

generate_llm_insights() is the single public entry point and NEVER
raises. Any failure -- no provider configured, missing key, network
error, malformed model response -- degrades to
LLMInsights(available=False, unavailable_reason=...), per spec: the
report must still generate without an LLM key, with AI-generated
insights clearly marked unavailable rather than crashing the pipeline
or silently disappearing.

Only ONE batched call is made per report (never per-review, never
per-theme) and it receives aggregated, anonymized statistics only --
no raw review text, usernames, or review IDs.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Optional

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import settings
from models.analysis import AnalysisResult, LLMInsights, ThemeItem
from services.exceptions import LLMProviderError
from utils.logger import get_logger

logger = get_logger(__name__)

_MAX_COMPLAINT_INSIGHTS = 5
_MAX_PRAISE_INSIGHTS = 5
_MAX_RECOMMENDATIONS = 3

SYSTEM_PROMPT = """You are a senior mobile app product analyst producing insights for a professional Play Store competitor & review audit report. You will be given aggregated, anonymized review statistics for one app -- rating distribution, sample size, and the most frequent complaint/praise keyword themes with mention counts.

Respond with ONLY a single JSON object and nothing else -- no markdown code fences, no preamble, no explanation before or after. The JSON object must have exactly these three keys:

{
  "complaint_insights": ["3 to 5 short plain-English sentences summarizing the most important complaint patterns"],
  "praise_insights": ["3 to 5 short plain-English sentences summarizing the most important things users like"],
  "recommendations": ["exactly 3 short, specific, actionable recommendations for the app's product team"]
}

Base every insight and recommendation strictly on the data provided -- do not invent details the data doesn't support. Synthesize the keyword/count data into genuine plain-English insight rather than repeating raw counts verbatim. Each string should read as a single clear sentence suitable for a client-facing PDF report. Do not mention that you are an AI or that this was generated from a sample."""


# --- Payload construction -------------------------------------------------


def _theme_payload(themes: list[ThemeItem]) -> list[dict[str, Any]]:
    return [{"phrase": t.phrase, "mentions": t.count} for t in themes]


def _build_payload(result: AnalysisResult) -> dict[str, Any]:
    """
    Aggregated, anonymized payload for the single batched LLM call.

    Deliberately contains no raw review text, usernames, or review
    IDs -- only app metadata, rating statistics, and the
    already-aggregated complaint/praise theme counts from Phase 5.
    """
    summary = result.sample_summary
    dist = result.rating_distribution
    return {
        "target_app": {
            "title": result.target_app.title,
            "category": result.target_app.genre,
            "current_score": result.target_app.score,
            "rating_count": result.target_app.rating_count,
            "installs": result.target_app.installs,
        },
        "review_sample": {
            "total_reviews_analyzed": summary.total_reviews_analyzed,
            "average_rating": summary.average_rating,
            "rating_distribution": {
                "1_star": dist.one_star,
                "2_star": dist.two_star,
                "3_star": dist.three_star,
                "4_star": dist.four_star,
                "5_star": dist.five_star,
            },
        },
        "complaint_themes": _theme_payload(result.complaint_themes),
        "praise_themes": _theme_payload(result.praise_themes),
        "competitors": [
            {"name": c.title, "score": c.score, "installs": c.installs} for c in result.competitors
        ],
    }


# --- Response parsing -------------------------------------------------------


def _strip_markdown_fence(text: str) -> str:
    """Defensively strip a ```json ... ``` or ``` ... ``` fence, even though the prompt asks for none."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    text = text.strip("`")
    if text.lower().startswith("json"):
        text = text[4:]
    return text.strip()


def _parse_response_text(raw_text: str) -> LLMInsights:
    """Parse + validate the model's JSON reply into an LLMInsights. Raises LLMProviderError on any problem."""
    text = _strip_markdown_fence(raw_text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMProviderError(f"Model response was not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise LLMProviderError("Model response was valid JSON but not a JSON object.")

    complaint_insights = data.get("complaint_insights")
    praise_insights = data.get("praise_insights")
    recommendations = data.get("recommendations")

    if not all(isinstance(x, list) for x in (complaint_insights, praise_insights, recommendations)):
        raise LLMProviderError("Model response was missing one or more required list fields.")

    def _clean(items: list, cap: int) -> list[str]:
        return [str(item).strip() for item in items if str(item).strip()][:cap]

    return LLMInsights(
        available=True,
        complaint_insights=_clean(complaint_insights, _MAX_COMPLAINT_INSIGHTS),
        praise_insights=_clean(praise_insights, _MAX_PRAISE_INSIGHTS),
        recommendations=_clean(recommendations, _MAX_RECOMMENDATIONS),
    )


# --- Provider interface + implementations ----------------------------------


class LLMProvider(ABC):
    """Interface every concrete provider implements, so providers can be swapped without touching callers."""

    @abstractmethod
    def generate_insights(self, payload: dict[str, Any]) -> LLMInsights:
        """Run the single batched call and return a populated LLMInsights. Raises LLMProviderError on failure."""


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str) -> None:
        import anthropic  # deferred import -- only needed when this provider is actually selected

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._retryable = (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.APITimeoutError)

    def generate_insights(self, payload: dict[str, Any]) -> LLMInsights:
        anthropic = self._anthropic

        @retry(
            stop=stop_after_attempt(settings.REQUEST_MAX_ATTEMPTS),
            wait=wait_exponential(multiplier=settings.REQUEST_BACKOFF_SECONDS, min=1, max=20),
            retry=retry_if_exception_type(self._retryable),
            reraise=True,
        )
        def _call():
            return self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": json.dumps(payload, indent=2)}],
            )

        try:
            response = _call()
        except anthropic.AuthenticationError as exc:
            raise LLMProviderError(f"Anthropic authentication failed -- check ANTHROPIC_API_KEY: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - any other SDK failure must degrade, not crash
            raise LLMProviderError(f"Anthropic API call failed: {exc}") from exc

        text_blocks = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        if not text_blocks:
            raise LLMProviderError("Anthropic response contained no text content.")

        return _parse_response_text("".join(text_blocks))


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str) -> None:
        import openai  # deferred import -- only needed when this provider is actually selected

        self._openai = openai
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model
        self._retryable = (openai.APIConnectionError, openai.RateLimitError, openai.APITimeoutError)

    def generate_insights(self, payload: dict[str, Any]) -> LLMInsights:
        openai = self._openai

        @retry(
            stop=stop_after_attempt(settings.REQUEST_MAX_ATTEMPTS),
            wait=wait_exponential(multiplier=settings.REQUEST_BACKOFF_SECONDS, min=1, max=20),
            retry=retry_if_exception_type(self._retryable),
            reraise=True,
        )
        def _call():
            return self._client.chat.completions.create(
                model=self._model,
                max_tokens=1024,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, indent=2)},
                ],
            )

        try:
            response = _call()
        except openai.AuthenticationError as exc:
            raise LLMProviderError(f"OpenAI authentication failed -- check OPENAI_API_KEY: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(f"OpenAI API call failed: {exc}") from exc

        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise LLMProviderError("OpenAI response contained no content.")

        return _parse_response_text(content)


# --- Provider resolution + public entry point -------------------------------


def _resolve_provider() -> tuple[Optional[LLMProvider], Optional[str]]:
    """
    Build the configured provider, if any.

    Returns (provider, unavailable_reason) -- exactly one is None.
    A provider is returned only when LLM_PROVIDER names a supported
    provider AND the matching API key is set; every other case
    (unset, missing key, unrecognized name, missing SDK) returns a
    human-readable reason instead, never raises.
    """
    provider_name = (settings.LLM_PROVIDER or "").strip().lower()

    if not provider_name:
        return None, "no LLM provider was configured for this run"

    if provider_name == "anthropic":
        if not settings.ANTHROPIC_API_KEY:
            return None, "LLM_PROVIDER is set to 'anthropic' but ANTHROPIC_API_KEY is not set"
        try:
            return AnthropicProvider(settings.ANTHROPIC_API_KEY, settings.ANTHROPIC_MODEL), None
        except ImportError as exc:
            return None, f"the 'anthropic' package is not installed: {exc}"

    if provider_name == "openai":
        if not settings.OPENAI_API_KEY:
            return None, "LLM_PROVIDER is set to 'openai' but OPENAI_API_KEY is not set"
        try:
            return OpenAIProvider(settings.OPENAI_API_KEY, settings.OPENAI_MODEL), None
        except ImportError as exc:
            return None, f"the 'openai' package is not installed: {exc}"

    return None, f"LLM_PROVIDER value '{provider_name}' is not recognized (expected 'anthropic' or 'openai')"


def generate_llm_insights(result: AnalysisResult) -> LLMInsights:
    """
    Run the single optional batched LLM call and return the result.

    NEVER raises. Any failure -- no provider configured, missing key,
    unrecognized provider name, network error, malformed model
    response -- degrades to LLMInsights(available=False,
    unavailable_reason=...) so report_service/report_templates render
    the deterministic-only report exactly as if no LLM had been
    requested at all. Safe to call unconditionally on every run.
    """
    provider, reason = _resolve_provider()
    if provider is None:
        logger.info("LLM insights skipped: %s", reason)
        return LLMInsights(available=False, unavailable_reason=reason)

    payload = _build_payload(result)

    try:
        insights = provider.generate_insights(payload)
    except LLMProviderError as exc:
        logger.warning("LLM insight generation failed, falling back to deterministic-only: %s", exc)
        return LLMInsights(available=False, unavailable_reason=str(exc))
    except Exception as exc:  # noqa: BLE001 - absolute last-resort safety net
        logger.warning("Unexpected error during LLM insight generation, falling back: %s", exc)
        return LLMInsights(available=False, unavailable_reason=f"Unexpected error: {exc}")

    logger.info(
        "LLM insights generated | complaint=%d praise=%d recommendations=%d",
        len(insights.complaint_insights),
        len(insights.praise_insights),
        len(insights.recommendations),
    )
    return insights
