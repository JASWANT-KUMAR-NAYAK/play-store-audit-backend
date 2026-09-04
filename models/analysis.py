"""
Analysis output models.

These describe the shape of every deterministic and (optionally)
LLM-derived result the analysis/report services pass between each
other -- rating distribution, complaint/praise themes, the
competitor comparison table, and the final aggregated AnalysisResult
consumed by report_service in Phase 7.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from models.app import AppMetadata


class RatingDistribution(BaseModel):
    """Count of reviews per star rating within the fetched sample."""

    one_star: int = Field(default=0, ge=0)
    two_star: int = Field(default=0, ge=0)
    three_star: int = Field(default=0, ge=0)
    four_star: int = Field(default=0, ge=0)
    five_star: int = Field(default=0, ge=0)

    @property
    def total(self) -> int:
        return (
            self.one_star
            + self.two_star
            + self.three_star
            + self.four_star
            + self.five_star
        )


class ReviewSampleSummary(BaseModel):
    """Top-line stats for the fetched review sample."""

    total_reviews_analyzed: int = Field(..., ge=0)
    average_rating: Optional[float] = Field(
        default=None, ge=0.0, le=5.0, description="Mean rating across the fetched sample"
    )


class ThemeItem(BaseModel):
    """A single recurring keyword/phrase found via frequency analysis."""

    phrase: str = Field(..., description="The keyword or n-gram phrase")
    count: int = Field(..., ge=1, description="Number of reviews the phrase occurred in")

    @property
    def display(self) -> str:
        return f"{self.phrase} ({self.count})"


class ComparisonRow(BaseModel):
    """One row of the target-vs-competitors comparison table."""

    label: str = Field(..., description="'Target' or a competitor's app title")
    package_id: str
    score: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    rating_count: Optional[int] = Field(default=None, ge=0)
    installs: Optional[str] = None
    genre: Optional[str] = None
    version: Optional[str] = None
    days_since_update: Optional[int] = Field(default=None, ge=0)

    @classmethod
    def from_app(
        cls, app: AppMetadata, label: str, days_since_update: Optional[int]
    ) -> "ComparisonRow":
        """Build a comparison row directly from an AppMetadata instance."""
        return cls(
            label=label,
            package_id=app.package_id,
            score=app.score,
            rating_count=app.rating_count,
            installs=app.installs,
            genre=app.genre,
            version=app.version,
            days_since_update=days_since_update,
        )


class ComparisonTable(BaseModel):
    """Target app row plus up to 3 competitor rows."""

    rows: list[ComparisonRow] = Field(default_factory=list, max_length=4)


class LLMInsights(BaseModel):
    """Output of the optional batched LLM insight-generation call."""

    available: bool = Field(
        ..., description="False when no LLM key/provider was configured or the call failed"
    )
    complaint_insights: list[str] = Field(default_factory=list, max_length=5)
    praise_insights: list[str] = Field(default_factory=list, max_length=5)
    recommendations: list[str] = Field(default_factory=list, max_length=3)
    unavailable_reason: Optional[str] = Field(
        default=None, description="Human-readable reason shown in the PDF when unavailable"
    )


class AnalysisResult(BaseModel):
    """Aggregate result passed from analysis_service into report_service."""

    target_app: AppMetadata
    competitors: list[AppMetadata] = Field(default_factory=list, max_length=3)
    rating_distribution: RatingDistribution
    sample_summary: ReviewSampleSummary
    complaint_themes: list[ThemeItem] = Field(default_factory=list)
    praise_themes: list[ThemeItem] = Field(default_factory=list)
    comparison_table: ComparisonTable
    deterministic_recommendations: list[str] = Field(default_factory=list, max_length=3)
    llm_insights: Optional[LLMInsights] = None

    @model_validator(mode="after")
    def _check_distribution_matches_summary(self) -> "AnalysisResult":
        if self.rating_distribution.total != self.sample_summary.total_reviews_analyzed:
            raise ValueError(
                "rating_distribution total "
                f"({self.rating_distribution.total}) does not match "
                "sample_summary.total_reviews_analyzed "
                f"({self.sample_summary.total_reviews_analyzed})"
            )
        return self
