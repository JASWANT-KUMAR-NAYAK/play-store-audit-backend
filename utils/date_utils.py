"""Date parsing and 'days since last update' calculations."""

from __future__ import annotations

from datetime import date
from typing import Optional


def days_since(target_date: Optional[date], reference: Optional[date] = None) -> Optional[int]:
    """
    Days elapsed between `target_date` and `reference` (default: today).

    Returns None if target_date is unknown -- callers must treat a
    missing update date as "unknown", not as "zero days ago".
    """
    if target_date is None:
        return None
    reference = reference or date.today()
    return (reference - target_date).days


def format_date(target_date: Optional[date]) -> str:
    """Human-readable date for report display, e.g. 'Jun 01, 2026'."""
    if target_date is None:
        return "Unknown"
    return target_date.strftime("%b %d, %Y")
