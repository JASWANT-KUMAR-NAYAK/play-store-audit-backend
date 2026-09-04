"""Assembles all analysis output into the final PDF report."""

from __future__ import annotations

from datetime import date
from functools import partial
from pathlib import Path
from typing import Optional

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate

from models.analysis import AnalysisResult, LLMInsights
from services.chart_service import ChartPaths
from services.exceptions import ReportGenerationError
from templates import report_templates
from templates.report_templates import MARGIN
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_report(
    analysis_result: AnalysisResult,
    chart_paths: ChartPaths,
    output_path: Path,
    llm_insights: Optional[LLMInsights] = None,
    generated_date: Optional[date] = None,
) -> Path:
    """
    Build the full audit PDF and write it to output_path. Returns output_path.

    Works correctly with llm_insights=None: per spec, the report must
    generate using deterministic insights alone when no LLM provider
    is configured, clearly noting the AI layer was unavailable rather
    than silently omitting it or crashing.

    Raises ReportGenerationError (never a raw reportlab/IO exception)
    on any assembly failure, so a bad chart path or a reportlab
    internal error can't take down the whole application.
    """
    generated_date = generated_date or date.today()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        story = report_templates.build_story(analysis_result, chart_paths, llm_insights, generated_date)

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            leftMargin=MARGIN,
            rightMargin=MARGIN,
            topMargin=MARGIN,
            bottomMargin=MARGIN,
            title=f"{analysis_result.target_app.title} - Play Store Audit Report",
            author="Play Store Audit Report Generator",
        )

        doc.build(
            story,
            onFirstPage=report_templates.first_page_decorations,
            onLaterPages=partial(
                report_templates.later_page_decorations,
                app_title=analysis_result.target_app.title or analysis_result.target_app.package_id,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - any assembly/IO failure must degrade gracefully, not crash
        raise ReportGenerationError(f"Failed to generate PDF report: {exc}") from exc

    logger.info("PDF report generated -> %s", output_path)
    return output_path
