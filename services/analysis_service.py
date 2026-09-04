"""
Deterministic rating/review analysis (no ML).

Consumes the AppMetadata + Review models produced by
play_store_service and produces the AnalysisResult that
report_service (Phase 7) will render into the PDF.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
from typing import Optional

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
from models.review import Review
from utils import date_utils, text_cleaner
from utils.logger import get_logger

logger = get_logger(__name__)


def _rating_distribution(reviews: list[Review]) -> RatingDistribution:
    counts = Counter(r.rating for r in reviews)
    return RatingDistribution(
        one_star=counts.get(1, 0),
        two_star=counts.get(2, 0),
        three_star=counts.get(3, 0),
        four_star=counts.get(4, 0),
        five_star=counts.get(5, 0),
    )


def _sample_summary(reviews: list[Review]) -> ReviewSampleSummary:
    total = len(reviews)
    average = round(sum(r.rating for r in reviews) / total, 2) if total else None
    return ReviewSampleSummary(total_reviews_analyzed=total, average_rating=average)


def _build_comparison_table(
    target: AppMetadata, competitors: list[AppMetadata], reference_date: Optional[date] = None
) -> ComparisonTable:
    reference_date = reference_date or date.today()
    rows = [
        ComparisonRow.from_app(
            target, label="Target", days_since_update=date_utils.days_since(target.updated, reference_date)
        )
    ]
    for comp in competitors[: settings.MAX_COMPETITORS]:
        rows.append(
            ComparisonRow.from_app(
                comp,
                label=comp.title or comp.package_id,
                days_since_update=date_utils.days_since(comp.updated, reference_date),
            )
        )
    return ComparisonTable(rows=rows)


def _build_recommendations(
    sample_summary: ReviewSampleSummary,
    complaint_themes: list[ThemeItem],
    praise_themes: list[ThemeItem],
    comparison_table: ComparisonTable,
) -> list[str]:
    """
    Rule-based recommendations from the deterministic analysis alone.

    Always returns at least one recommendation, even with a tiny or
    empty review sample -- an audit report with a blank
    recommendations section looks unfinished, so a data-quality
    caveat is itself an actionable recommendation.
    """
    if sample_summary.total_reviews_analyzed == 0:
        return [
            "No reviews were available in the sampled window -- rerun "
            "once review volume increases, or supplement with a "
            "manual qualitative review pass."
        ]

    recommendations: list[str] = []

    lead_complaint = text_cleaner.pick_lead_theme(complaint_themes)
    if lead_complaint:
        recommendations.append(
            f"Prioritize addressing '{lead_complaint.phrase}' -- a "
            f"frequent complaint, appearing in {lead_complaint.count} of "
            "the sampled negative reviews."
        )

    target_row = comparison_table.rows[0] if comparison_table.rows else None
    competitor_days = [
        row.days_since_update for row in comparison_table.rows[1:] if row.days_since_update is not None
    ]
    if target_row and target_row.days_since_update is not None and competitor_days:
        avg_competitor_days = sum(competitor_days) / len(competitor_days)
        if target_row.days_since_update > avg_competitor_days * 1.5:
            recommendations.append(
                f"Update cadence lags competitors ({target_row.days_since_update} "
                f"days since last release vs. a {avg_competitor_days:.0f}-day "
                "competitor average) -- consider a more frequent release "
                "schedule to stay visible in store ranking signals."
            )

    lead_praise = text_cleaner.pick_lead_theme(praise_themes)
    if lead_praise and len(recommendations) < 3:
        recommendations.append(
            f"Lean into '{lead_praise.phrase}' in App Store Optimization "
            "(ASO) copy and marketing -- it's a consistently praised "
            "aspect across positive reviews."
        )

    if not recommendations:
        recommendations.append(
            "Review sentiment is mixed without one dominant theme -- "
            "consider a structured user survey to surface specific "
            "improvement areas."
        )

    return recommendations[:3]


def analyze(
    target: AppMetadata,
    competitors: list[AppMetadata],
    reviews: list[Review],
) -> AnalysisResult:
    """
    Run the full deterministic analysis pipeline and return the
    aggregate AnalysisResult consumed by report_service.

    Safe to call with an empty `reviews` list (an app with no reviews
    in the sampled window), an empty `competitors` list, or MORE than
    MAX_COMPETITORS competitors -- every downstream step degrades
    gracefully rather than raising. The competitor cap is enforced
    here (not just assumed from the caller) so this function is safe
    to call directly, not only through the CLI path that already
    caps it upstream.
    """
    competitors = competitors[: settings.MAX_COMPETITORS]

    rating_distribution = _rating_distribution(reviews)
    sample_summary = _sample_summary(reviews)

    negative_reviews = [r for r in reviews if r.is_negative]
    positive_reviews = [r for r in reviews if r.is_positive]

    complaint_themes = text_cleaner.extract_common_themes(negative_reviews)
    praise_themes = text_cleaner.extract_common_themes(positive_reviews)

    comparison_table = _build_comparison_table(target, competitors)
    recommendations = _build_recommendations(
        sample_summary, complaint_themes, praise_themes, comparison_table
    )

    logger.info(
        "Analysis complete | reviews=%d avg_rating=%s complaint_themes=%d praise_themes=%d competitors=%d",
        sample_summary.total_reviews_analyzed,
        sample_summary.average_rating,
        len(complaint_themes),
        len(praise_themes),
        len(competitors),
    )

    return AnalysisResult(
        target_app=target,
        competitors=competitors,
        rating_distribution=rating_distribution,
        sample_summary=sample_summary,
        complaint_themes=complaint_themes,
        praise_themes=praise_themes,
        comparison_table=comparison_table,
        deterministic_recommendations=recommendations,
        llm_insights=None,
    )
