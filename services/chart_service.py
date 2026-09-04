"""
Chart generation for the audit PDF report.

Charts are rendered via matplotlib's non-interactive Agg backend and
saved as PNG files for embedding into the PDF in Phase 7. Each
function takes an explicit output directory -- this module has no
opinion about temp vs. persistent storage; the caller decides.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import -- no display in this environment

from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from config import settings  # noqa: E402
from models.analysis import AnalysisResult, ComparisonTable, RatingDistribution  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

# Red (1-star) -> green (5-star): a deliberate semantic gradient,
# since "bad vs. good" is the entire point of a rating distribution.
_RATING_COLORS = ["#c0392b", "#e67e22", "#f1c40f", "#8bc34a", "#27ae60"]

_TARGET_COLOR = "#2563eb"  # highlighted -- this is "us"
_COMPETITOR_COLOR = "#94a3b8"  # neutral gray -- "them"
_NO_DATA_COLOR = "#e2e8f0"

_TITLE_COLOR = "#1e293b"
_LABEL_COLOR = "#334155"
_MUTED_COLOR = "#64748b"


@dataclass(frozen=True)
class ChartPaths:
    """Filesystem paths to every chart generated for one report run."""

    rating_distribution: Path
    competitor_comparison: Path


def _ensure_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def _empty_chart_placeholder(ax: plt.Axes, message: str) -> None:
    """Render a clean 'no data' placeholder instead of a blank/misleading chart."""
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, color=_MUTED_COLOR, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)


def generate_rating_distribution_chart(
    distribution: RatingDistribution,
    output_dir: Path,
    filename: str = "rating_distribution.png",
) -> Path:
    """
    Bar chart of 1-5 star review counts in the fetched sample.

    An empty sample (total == 0) renders a placeholder with a "no
    data" annotation rather than an all-zero chart that looks broken
    -- the PDF can always embed a valid image regardless of data
    availability.
    """
    _ensure_dir(output_dir)
    out_path = output_dir / filename
    fig, ax = plt.subplots(figsize=(6, 4))

    if distribution.total == 0:
        _empty_chart_placeholder(ax, "No review data available\nfor this sample")
    else:
        stars = [1, 2, 3, 4, 5]
        counts = [
            distribution.one_star,
            distribution.two_star,
            distribution.three_star,
            distribution.four_star,
            distribution.five_star,
        ]
        bars = ax.bar(stars, counts, color=_RATING_COLORS, edgecolor="white", linewidth=0.5)
        ax.set_xticks(stars)
        ax.set_xticklabels([f"{s}\u2605" for s in stars])
        ax.set_ylabel("Number of reviews")
        ax.set_ylim(0, max(counts) * 1.15 if max(counts) > 0 else 1)
        headroom = max(counts) * 0.02 if max(counts) > 0 else 0.02
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + headroom,
                    str(count),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=_LABEL_COLOR,
                )
        _style_axes(ax)

    ax.set_title("Rating Distribution (Sampled Reviews)", fontsize=13, fontweight="bold", color=_TITLE_COLOR)
    fig.tight_layout()
    fig.savefig(out_path, dpi=settings.CHART_DPI)
    plt.close(fig)

    logger.info("Saved rating distribution chart -> %s", out_path)
    return out_path


def generate_competitor_comparison_chart(
    comparison_table: ComparisonTable,
    output_dir: Path,
    filename: str = "competitor_comparison.png",
) -> Path:
    """
    Bar chart comparing the target app's score against up to 3
    competitors.

    The target bar is visually highlighted. An app with no published
    score (score is None) renders as a small flat "No data" bar
    rather than being silently dropped or misrepresented as an actual
    0.0 rating.
    """
    _ensure_dir(output_dir)
    out_path = output_dir / filename
    fig, ax = plt.subplots(figsize=(6, 4))

    if not comparison_table.rows:
        _empty_chart_placeholder(ax, "No comparison data available")
    else:
        labels = [row.label for row in comparison_table.rows]
        scores = [row.score for row in comparison_table.rows]
        colors = [
            _TARGET_COLOR if i == 0 else (_NO_DATA_COLOR if s is None else _COMPETITOR_COLOR)
            for i, s in enumerate(scores)
        ]
        # Give a "no data" bar a small visible stub so its label has an anchor point.
        heights = [s if s is not None else 0.15 for s in scores]

        bars = ax.bar(labels, heights, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_ylabel("Store rating (0-5)")
        ax.set_ylim(0, 5.3)
        for bar, score in zip(bars, scores):
            label = f"{score:.1f}" if score is not None else "No data"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.08,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                color=_LABEL_COLOR,
            )
        _style_axes(ax)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    ax.set_title("Rating Comparison: Target vs. Competitors", fontsize=13, fontweight="bold", color=_TITLE_COLOR)
    fig.tight_layout()
    fig.savefig(out_path, dpi=settings.CHART_DPI)
    plt.close(fig)

    logger.info("Saved competitor comparison chart -> %s", out_path)
    return out_path


def generate_all_charts(analysis_result: AnalysisResult, output_dir: Path) -> ChartPaths:
    """Generate every chart needed for one report run."""
    return ChartPaths(
        rating_distribution=generate_rating_distribution_chart(analysis_result.rating_distribution, output_dir),
        competitor_comparison=generate_competitor_comparison_chart(analysis_result.comparison_table, output_dir),
    )
