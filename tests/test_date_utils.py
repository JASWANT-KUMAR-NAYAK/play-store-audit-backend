"""Tests for utils/date_utils.py."""

from __future__ import annotations

from datetime import date

from utils import date_utils


def test_days_since_computes_correct_difference():
    assert date_utils.days_since(date(2026, 1, 1), reference=date(2026, 1, 31)) == 30


def test_days_since_returns_none_for_missing_date():
    """A missing update date must read as 'unknown', never silently as '0 days ago'."""
    assert date_utils.days_since(None, reference=date(2026, 1, 31)) is None


def test_days_since_defaults_reference_to_today():
    result = date_utils.days_since(date.today())
    assert result == 0


def test_days_since_handles_future_date_as_negative():
    assert date_utils.days_since(date(2026, 2, 1), reference=date(2026, 1, 1)) == -31


def test_format_date_renders_expected_format():
    assert date_utils.format_date(date(2026, 6, 1)) == "Jun 01, 2026"


def test_format_date_handles_none():
    assert date_utils.format_date(None) == "Unknown"
