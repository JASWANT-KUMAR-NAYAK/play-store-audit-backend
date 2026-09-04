"""Review data model for individual Play Store reviews of the target app."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Review(BaseModel):
    """
    A single user review of the target app.

    review_id is kept as a string since google-play-scraper returns
    opaque alphanumeric IDs, not integers. content may be empty (some
    users leave a star rating with no text) -- that's valid, not an
    error, and downstream text analysis must skip it rather than fail.
    """

    review_id: str = Field(..., description="Opaque review identifier from the Play Store")
    user_name: Optional[str] = Field(default=None, description="Display name of the reviewer")
    rating: int = Field(..., ge=1, le=5, description="Star rating, 1-5")
    content: str = Field(default="", description="Review text body; may be empty")
    review_date: Optional[datetime] = Field(default=None, description="When the review was posted")
    app_version: Optional[str] = Field(
        default=None, description="App version the review was left against, if known"
    )

    @field_validator("review_id")
    @classmethod
    def _require_nonblank_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("review_id cannot be blank")
        return value

    @field_validator("content")
    @classmethod
    def _normalize_content(cls, value: Optional[str]) -> str:
        return value.strip() if value else ""

    @field_validator("user_name", "app_version")
    @classmethod
    def _strip_optional(cls, value: Optional[str]) -> Optional[str]:
        return value.strip() if isinstance(value, str) else value

    @property
    def is_negative(self) -> bool:
        """True for 1-2 star reviews -- the complaint-analysis pool."""
        return self.rating <= 2

    @property
    def is_positive(self) -> bool:
        """True for 4-5 star reviews -- the praise-analysis pool."""
        return self.rating >= 4
