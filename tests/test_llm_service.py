"""
Focused tests for services/llm_service.py.

Covers: successful Anthropic/OpenAI response parsing (through the
real provider classes), malformed JSON handling, provider/API
failure handling (both the raw provider layer and the
generate_llm_insights() safety net), missing-provider/missing-key
fallback, and that exactly one batched call is made per report.

No real network calls or API keys are used anywhere in this file.
The underlying SDK client methods (client.messages.create /
client.chat.completions.create) are monkeypatched to return fixture
responses shaped like the real Anthropic/OpenAI SDKs -- constructing
a provider with a fake key string is safe since neither SDK validates
the key or contacts the network until a method is actually called.

Failure-path tests deliberately raise exception types that are NOT in
each provider's retryable set (see llm_service.AnthropicProvider /
OpenAIProvider._retryable), so they fail on the first attempt instead
of triggering tenacity's real exponential backoff -- this keeps the
suite fast without needing to mock time.sleep. Retry-attempt timing
itself is out of scope for this file.
"""

from __future__ import annotations

import json
import types
from datetime import date

import pytest

from config import settings
from models.analysis import (
    AnalysisResult,
    ComparisonRow,
    ComparisonTable,
    RatingDistribution,
    ReviewSampleSummary,
    ThemeItem,
)
from models.app import AppMetadata
from services import llm_service
from services.exceptions import LLMProviderError

# --- Shared fixtures / fakes ------------------------------------------------


@pytest.fixture
def analysis_result() -> AnalysisResult:
    """A small, realistic AnalysisResult -- enough to build a valid LLM payload."""
    target = AppMetadata(
        package_id="com.example.target", title="TaskFlow Pro", score=3.6,
        rating_count=12000, installs="1,000,000+", genre="Productivity",
        updated=date(2026, 3, 1), version="2.0.0",
    )
    competitor = AppMetadata(package_id="com.rival.one", title="Rival Tasks", score=4.4)
    return AnalysisResult(
        target_app=target,
        competitors=[competitor],
        rating_distribution=RatingDistribution(one_star=15, two_star=15, three_star=10, four_star=20, five_star=20),
        sample_summary=ReviewSampleSummary(total_reviews_analyzed=80, average_rating=3.19),
        complaint_themes=[ThemeItem(phrase="crashes constantly", count=23)],
        praise_themes=[ThemeItem(phrase="clean interface", count=40)],
        comparison_table=ComparisonTable(
            rows=[
                ComparisonRow.from_app(target, "Target", days_since_update=178),
                ComparisonRow.from_app(competitor, "Rival Tasks", days_since_update=25),
            ]
        ),
        deterministic_recommendations=["Fix the crash bug."],
    )


VALID_JSON_REPLY = json.dumps(
    {
        "complaint_insights": ["Crashes spike right after updates.", "Battery drain compounds the frustration."],
        "praise_insights": ["The interface is repeatedly called clean and simple."],
        "recommendations": [
            "Ship a stability patch before new features.",
            "Audit the update pipeline.",
            "Promote the clean UI in ASO copy.",
        ],
    }
)


def _anthropic_response(text: str) -> types.SimpleNamespace:
    """Shape a fake response matching anthropic's Message: content is a list of blocks with .type/.text."""
    return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=text)])


def _openai_response(text: str) -> types.SimpleNamespace:
    """Shape a fake response matching OpenAI's ChatCompletion: choices[0].message.content."""
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))])


# --- Successful response parsing, through the real provider classes --------


def test_anthropic_provider_parses_successful_response(monkeypatch):
    provider = llm_service.AnthropicProvider(api_key="fake-key", model="claude-sonnet-5")
    monkeypatch.setattr(provider._client.messages, "create", lambda **kwargs: _anthropic_response(VALID_JSON_REPLY))

    insights = provider.generate_insights({"some": "payload"})

    assert insights.available is True
    assert insights.complaint_insights == [
        "Crashes spike right after updates.",
        "Battery drain compounds the frustration.",
    ]
    assert insights.praise_insights == ["The interface is repeatedly called clean and simple."]
    assert len(insights.recommendations) == 3


def test_openai_provider_parses_successful_response(monkeypatch):
    provider = llm_service.OpenAIProvider(api_key="fake-key", model="gpt-4o-mini")
    monkeypatch.setattr(
        provider._client.chat.completions, "create", lambda **kwargs: _openai_response(VALID_JSON_REPLY)
    )

    insights = provider.generate_insights({"some": "payload"})

    assert insights.available is True
    assert insights.praise_insights == ["The interface is repeatedly called clean and simple."]
    assert len(insights.recommendations) == 3


def test_anthropic_provider_strips_markdown_fence(monkeypatch):
    """Real models sometimes wrap JSON in ```json fences despite instructions not to -- must still parse."""
    fenced = f"```json\n{VALID_JSON_REPLY}\n```"
    provider = llm_service.AnthropicProvider(api_key="fake-key", model="claude-sonnet-5")
    monkeypatch.setattr(provider._client.messages, "create", lambda **kwargs: _anthropic_response(fenced))

    insights = provider.generate_insights({"some": "payload"})

    assert insights.available is True
    assert len(insights.recommendations) == 3


# --- Malformed JSON ----------------------------------------------------------


def test_anthropic_provider_raises_on_malformed_json(monkeypatch):
    provider = llm_service.AnthropicProvider(api_key="fake-key", model="claude-sonnet-5")
    monkeypatch.setattr(
        provider._client.messages, "create", lambda **kwargs: _anthropic_response("this isn't json {broken")
    )

    with pytest.raises(LLMProviderError, match="not valid JSON"):
        provider.generate_insights({"some": "payload"})


def test_openai_provider_raises_on_malformed_json(monkeypatch):
    provider = llm_service.OpenAIProvider(api_key="fake-key", model="gpt-4o-mini")
    monkeypatch.setattr(
        provider._client.chat.completions, "create", lambda **kwargs: _openai_response("not json either [[[")
    )

    with pytest.raises(LLMProviderError, match="not valid JSON"):
        provider.generate_insights({"some": "payload"})


def test_raises_on_json_missing_required_fields(monkeypatch):
    provider = llm_service.AnthropicProvider(api_key="fake-key", model="claude-sonnet-5")
    incomplete = json.dumps({"complaint_insights": ["a"], "praise_insights": ["b"]})  # no recommendations key
    monkeypatch.setattr(provider._client.messages, "create", lambda **kwargs: _anthropic_response(incomplete))

    with pytest.raises(LLMProviderError, match="required list field"):
        provider.generate_insights({"some": "payload"})


def test_raises_on_json_that_is_not_an_object(monkeypatch):
    provider = llm_service.AnthropicProvider(api_key="fake-key", model="claude-sonnet-5")
    monkeypatch.setattr(provider._client.messages, "create", lambda **kwargs: _anthropic_response('["a", "b"]'))

    with pytest.raises(LLMProviderError, match="not a JSON object"):
        provider.generate_insights({"some": "payload"})


# --- Provider/API failure ----------------------------------------------------


def test_anthropic_provider_wraps_api_failure(monkeypatch):
    provider = llm_service.AnthropicProvider(api_key="fake-key", model="claude-sonnet-5")

    def _raise(**kwargs):
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr(provider._client.messages, "create", _raise)

    with pytest.raises(LLMProviderError, match="Anthropic API call failed"):
        provider.generate_insights({"some": "payload"})


def test_openai_provider_wraps_api_failure(monkeypatch):
    provider = llm_service.OpenAIProvider(api_key="fake-key", model="gpt-4o-mini")

    def _raise(**kwargs):
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr(provider._client.chat.completions, "create", _raise)

    with pytest.raises(LLMProviderError, match="OpenAI API call failed"):
        provider.generate_insights({"some": "payload"})


def test_anthropic_provider_raises_when_response_has_no_text(monkeypatch):
    provider = llm_service.AnthropicProvider(api_key="fake-key", model="claude-sonnet-5")
    empty_response = types.SimpleNamespace(content=[])
    monkeypatch.setattr(provider._client.messages, "create", lambda **kwargs: empty_response)

    with pytest.raises(LLMProviderError, match="no text content"):
        provider.generate_insights({"some": "payload"})


def test_generate_llm_insights_falls_back_when_provider_raises_llm_error(monkeypatch, analysis_result):
    """The orchestrator must convert a provider failure into available=False, never let it propagate."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "fake-key")

    class _FailingProvider:
        def generate_insights(self, payload):
            raise LLMProviderError("simulated API failure")

    monkeypatch.setattr(llm_service, "AnthropicProvider", lambda api_key, model: _FailingProvider())

    insights = llm_service.generate_llm_insights(analysis_result)

    assert insights.available is False
    assert "simulated API failure" in insights.unavailable_reason


def test_generate_llm_insights_falls_back_on_unexpected_exception(monkeypatch, analysis_result):
    """Even a bug that raises something other than LLMProviderError must not crash the pipeline."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "fake-key")

    class _BuggyProvider:
        def generate_insights(self, payload):
            raise ValueError("totally unexpected bug, not even LLMProviderError")

    monkeypatch.setattr(llm_service, "AnthropicProvider", lambda api_key, model: _BuggyProvider())

    insights = llm_service.generate_llm_insights(analysis_result)

    assert insights.available is False
    assert "Unexpected error" in insights.unavailable_reason


# --- Missing provider / key fallback ----------------------------------------


def test_resolve_provider_when_unset(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", None)
    provider, reason = llm_service._resolve_provider()
    assert provider is None
    assert "no LLM provider" in reason


def test_resolve_provider_anthropic_missing_key(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", None)
    provider, reason = llm_service._resolve_provider()
    assert provider is None
    assert "ANTHROPIC_API_KEY" in reason


def test_resolve_provider_openai_missing_key(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", None)
    provider, reason = llm_service._resolve_provider()
    assert provider is None
    assert "OPENAI_API_KEY" in reason


def test_resolve_provider_unrecognized_name(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "gemini")
    provider, reason = llm_service._resolve_provider()
    assert provider is None
    assert "not recognized" in reason


def test_generate_llm_insights_never_raises_with_no_provider_configured(monkeypatch, analysis_result):
    """This is the default state of a fresh checkout (.env.example ships with LLM_PROVIDER blank)."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", None)

    insights = llm_service.generate_llm_insights(analysis_result)

    assert insights.available is False
    assert insights.complaint_insights == []
    assert insights.praise_insights == []
    assert insights.recommendations == []


# --- Exactly one batched call -------------------------------------------------


def test_generate_llm_insights_makes_exactly_one_batched_call(monkeypatch, analysis_result):
    """The spec requires ONE batched call per report -- never per-review, never per-theme."""
    call_count = {"n": 0}

    def _counting_create(**kwargs):
        call_count["n"] += 1
        return _anthropic_response(VALID_JSON_REPLY)

    monkeypatch.setattr(settings, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "fake-key")

    real_provider = llm_service.AnthropicProvider(api_key="fake-key", model="claude-sonnet-5")
    monkeypatch.setattr(real_provider._client.messages, "create", _counting_create)
    monkeypatch.setattr(llm_service, "AnthropicProvider", lambda api_key, model: real_provider)

    insights = llm_service.generate_llm_insights(analysis_result)

    assert insights.available is True
    assert call_count["n"] == 1, f"expected exactly one batched call, got {call_count['n']}"


def test_payload_stays_a_single_aggregate_regardless_of_theme_count(analysis_result):
    """
    Confirms _build_payload produces one aggregated document (not a
    per-theme or per-review structure) even as themes scale up --
    the shape that makes a single batched call possible in the first
    place.
    """
    analysis_result.complaint_themes = [ThemeItem(phrase=f"issue {i}", count=5) for i in range(50)]
    payload = llm_service._build_payload(analysis_result)

    assert isinstance(payload, dict)
    assert isinstance(payload["complaint_themes"], list)
    assert len(payload["complaint_themes"]) == 50
    assert set(payload.keys()) == {
        "target_app", "review_sample", "complaint_themes", "praise_themes", "competitors",
    }
