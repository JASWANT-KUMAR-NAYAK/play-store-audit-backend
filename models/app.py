"""App metadata model for the target app and its competitors."""

from __future__ import annotations

import re
from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# Android package IDs are reverse-domain style: segments separated by
# dots, each segment starting with a letter and containing only
# letters, digits, and underscores.
_PACKAGE_ID_PATTERN = re.compile(
    r"^[a-zA-Z][a-zA-Z0-9_]*(\.[a-zA-Z][a-zA-Z0-9_]*)+$"
)


class AppMetadata(BaseModel):
    """
    Metadata for a single Play Store app (target or competitor).

    Fields are optional where the Play Store listing may legitimately
    omit them (e.g. a brand-new app with no rating yet) -- the model
    should never fail to construct just because one field is missing.
    """

    package_id: str = Field(..., description="Play Store package ID, e.g. com.example.app")
    title: str = Field(..., description="App display name")
    developer: Optional[str] = Field(default=None, description="Publishing developer/org name")
    score: Optional[float] = Field(
        default=None, ge=0.0, le=5.0, description="Current average rating (0-5)"
    )
    rating_count: Optional[int] = Field(
        default=None, ge=0, description="Total number of ratings"
    )
    installs: Optional[str] = Field(
        default=None, description="Install band as shown on the store, e.g. '1,000,000+'"
    )
    genre: Optional[str] = Field(default=None, description="Category/genre label")
    updated: Optional[date] = Field(default=None, description="Date the current version was released")
    version: Optional[str] = Field(default=None, description="Current app version string")

    @field_validator("package_id")
    @classmethod
    def _validate_package_id(cls, value: str) -> str:
        value = value.strip()
        if not _PACKAGE_ID_PATTERN.match(value):
            raise ValueError(
                f"'{value}' is not a valid Play Store package ID "
                "(expected reverse-domain form, e.g. com.example.app)"
            )
        return value

    @field_validator("title", "developer", "installs", "genre", "version")
    @classmethod
    def _strip_strings(cls, value: Optional[str]) -> Optional[str]:
        return value.strip() if isinstance(value, str) else value
