"""
reportlab section/layout builders for the audit PDF.

build_story() is the single public entry point: it assembles every
section into the flowable list report_service.generate_report() hands
to a SimpleDocTemplate. Everything else here is a private per-section
helper.

All dynamic text (app titles, scraped review phrases, LLM output)
goes through P()/_esc() before reaching a Paragraph -- reportlab's
Paragraph markup treats "&", "<", ">" as XML, and an app title like
"Cut & Paste Pro" would otherwise break rendering.
"""

from __future__ import annotations

from datetime import date
from typing import Optional
from xml.sax.saxutils import escape as _xml_escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, Spacer, Table, TableStyle

from models.analysis import AnalysisResult, ComparisonTable, LLMInsights, ThemeItem
from models.app import AppMetadata
from services.chart_service import ChartPaths
from utils import date_utils, text_cleaner

# --- Layout constants ---------------------------------------------------

PAGE_WIDTH, PAGE_HEIGHT = letter
MARGIN = 0.75 * inch
USABLE_WIDTH = PAGE_WIDTH - 2 * MARGIN
CHART_WIDTH = 6 * inch
CHART_HEIGHT = 4 * inch

# --- Color palette (matches chart_service's semantic colors) ------------

PRIMARY = colors.HexColor("#2563eb")
DARK = colors.HexColor("#1e293b")
MUTED = colors.HexColor("#64748b")
LIGHT_BG = colors.HexColor("#f1f5f9")
TARGET_HIGHLIGHT_BG = colors.HexColor("#eff6ff")
BORDER = colors.HexColor("#e2e8f0")


def _esc(text: Optional[object]) -> str:
    """XML-escape any value before it reaches a Paragraph. None -> ''."""
    if text is None:
        return ""
    return _xml_escape(str(text))


def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles: dict[str, ParagraphStyle] = {
        "CoverTitle": ParagraphStyle(
            "CoverTitle", parent=base["Title"], fontSize=24, leading=30,
            textColor=DARK, alignment=TA_CENTER, spaceAfter=14,
        ),
        "CoverAppName": ParagraphStyle(
            "CoverAppName", parent=base["Title"], fontSize=19, leading=24,
            textColor=PRIMARY, alignment=TA_CENTER, spaceAfter=6,
        ),
        "CoverMeta": ParagraphStyle(
            "CoverMeta", parent=base["Normal"], fontSize=10.5,
            textColor=MUTED, alignment=TA_CENTER, spaceAfter=4,
        ),
        "SectionHeading": ParagraphStyle(
            "SectionHeading", parent=base["Heading1"], fontSize=16,
            textColor=DARK, spaceBefore=4, spaceAfter=10,
        ),
        "SubHeading": ParagraphStyle(
            "SubHeading", parent=base["Heading2"], fontSize=11.5,
            textColor=PRIMARY, spaceBefore=10, spaceAfter=6,
        ),
        "Body": ParagraphStyle(
            "Body", parent=base["Normal"], fontSize=10, leading=15,
            textColor=DARK, spaceAfter=8,
        ),
        "BulletBody": ParagraphStyle(
            "BulletBody", parent=base["Normal"], fontSize=10, leading=15,
            textColor=DARK, leftIndent=14, spaceAfter=6,
        ),
        "MutedItalic": ParagraphStyle(
            "MutedItalic", parent=base["Normal"], fontSize=9.5, leading=13,
            textColor=MUTED, fontName="Helvetica-Oblique", spaceAfter=8,
        ),
        "TableCell": ParagraphStyle(
            "TableCell", parent=base["Normal"], fontSize=9, leading=11.5, textColor=DARK,
        ),
        "TableKey": ParagraphStyle(
            "TableKey", parent=base["Normal"], fontSize=9.5, leading=12,
            textColor=DARK, fontName="Helvetica-Bold",
        ),
        "TableHeader": ParagraphStyle(
            "TableHeader", parent=base["Normal"], fontSize=9, leading=11.5,
            textColor=colors.white, fontName="Helvetica-Bold",
        ),
    }
    return styles


_STYLES = _build_styles()


def P(text: object, style_name: str) -> Paragraph:
    """Escape + wrap in one step -- the only way dynamic text should reach a Paragraph."""
    return Paragraph(_esc(text), _STYLES[style_name])


def _section_heading(text: str) -> Paragraph:
    return Paragraph(_esc(text), _STYLES["SectionHeading"])


def _divider_bar(width: float = 2 * inch, height: float = 3, color=PRIMARY) -> Table:
    bar = Table([[""]], colWidths=[width], rowHeights=[height])
    bar.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), color)]))
    bar.hAlign = "CENTER"
    return bar


# --- Section 1: Cover page ----------------------------------------------


def _cover_page(target: AppMetadata, generated_date: date) -> list:
    elems: list = [Spacer(1, 1.6 * inch)]
    elems.append(P("Play Store Competitor & Review Audit Report", "CoverTitle"))
    elems.append(Spacer(1, 0.25 * inch))
    elems.append(_divider_bar())
    elems.append(Spacer(1, 0.35 * inch))
    elems.append(P(target.title or target.package_id, "CoverAppName"))
    elems.append(P(f"Package ID: {target.package_id}", "CoverMeta"))
    if target.developer:
        elems.append(P(f"Developer: {target.developer}", "CoverMeta"))
    elems.append(Spacer(1, 0.5 * inch))
    elems.append(P(f"Generated on {generated_date.strftime('%B %d, %Y')}", "CoverMeta"))
    return elems


# --- Section 2: Executive summary ----------------------------------------


def _health_band(average_rating: Optional[float]) -> str:
    if average_rating is None:
        return "undetermined (insufficient review data)"
    if average_rating >= 4.5:
        return "excellent"
    if average_rating >= 4.0:
        return "good"
    if average_rating >= 3.5:
        return "mixed"
    if average_rating >= 3.0:
        return "concerning"
    return "poor"


def _key_findings(result: AnalysisResult) -> list[str]:
    findings: list[str] = []
    summary = result.sample_summary

    if summary.total_reviews_analyzed:
        findings.append(
            f"{summary.total_reviews_analyzed} recent reviews sampled, "
            f"averaging {summary.average_rating:.2f} / 5."
        )
    else:
        findings.append("No reviews were available in the sampled window.")

    if result.target_app.score is not None:
        count_str = f"{result.target_app.rating_count:,}" if result.target_app.rating_count is not None else "an unknown number of"
        findings.append(f"Current store rating: {result.target_app.score:.1f} / 5 ({count_str} total ratings).")

    competitor_scores = [c.score for c in result.competitors if c.score is not None]
    if competitor_scores and result.target_app.score is not None:
        lower = sum(1 for cs in competitor_scores if cs < result.target_app.score)
        findings.append(
            f"Rated higher than {lower} of {len(competitor_scores)} tracked "
            "competitor(s) with published scores."
        )

    lead_complaint = text_cleaner.pick_lead_theme(result.complaint_themes)
    if lead_complaint:
        findings.append(f"Top complaint theme: '{lead_complaint.phrase}' ({lead_complaint.count} mentions).")
    lead_praise = text_cleaner.pick_lead_theme(result.praise_themes)
    if lead_praise:
        findings.append(f"Top praise theme: '{lead_praise.phrase}' ({lead_praise.count} mentions).")

    return findings[:5]


def _executive_summary(result: AnalysisResult) -> list:
    summary = result.sample_summary
    elems: list = [_section_heading("Executive Summary")]

    if summary.total_reviews_analyzed == 0:
        narrative = (
            f"No reviews were available for {result.target_app.title or result.target_app.package_id} "
            "in the sampled window, so this report's review-based findings are limited. "
            "The sections below still include store metadata and competitor comparison, "
            "which do not depend on review volume."
        )
    else:
        band = _health_band(summary.average_rating)
        narrative = (
            f"Based on a sample of {summary.total_reviews_analyzed} recent reviews "
            f"(averaging {summary.average_rating:.2f} out of 5), "
            f"{result.target_app.title or result.target_app.package_id}'s user sentiment "
            f"is currently {band}."
        )
        lead_complaint = text_cleaner.pick_lead_theme(result.complaint_themes)
        if lead_complaint:
            narrative += (
                f" The most frequent complaint theme is '{lead_complaint.phrase}', "
                f"mentioned in {lead_complaint.count} of the sampled negative reviews."
            )
        lead_praise = text_cleaner.pick_lead_theme(result.praise_themes)
        if lead_praise:
            narrative += f" On the positive side, '{lead_praise.phrase}' is the most consistently praised aspect."

    elems.append(P(narrative, "Body"))

    for finding in _key_findings(result):
        elems.append(P(f"\u2022 {finding}", "BulletBody"))

    return elems


# --- Section 3: Target app overview --------------------------------------


def _target_overview(target: AppMetadata) -> list:
    elems: list = [_section_heading("Target App Overview")]

    updated_str = date_utils.format_date(target.updated)
    days = date_utils.days_since(target.updated)
    if days is not None:
        updated_str += f" ({days} days ago)"

    rows = [
        ("App Name", target.title or "Unknown"),
        ("Package ID", target.package_id),
        ("Developer", target.developer or "Not available"),
        ("Category", target.genre or "Not available"),
        ("Current Rating", f"{target.score:.1f} / 5" if target.score is not None else "Not available"),
        ("Rating Count", f"{target.rating_count:,}" if target.rating_count is not None else "Not available"),
        ("Install Band", target.installs or "Not available"),
        ("Current Version", target.version or "Not available"),
        ("Last Updated", updated_str),
    ]
    table_data = [[P(k, "TableKey"), P(v, "TableCell")] for k, v in rows]
    tbl = Table(table_data, colWidths=[1.8 * inch, USABLE_WIDTH - 1.8 * inch])
    tbl.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("BACKGROUND", (0, 0), (0, -1), LIGHT_BG),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    elems.append(tbl)
    return elems


# --- Section 4: Rating & review analysis ---------------------------------


def _rating_review_analysis(result: AnalysisResult, chart_paths: ChartPaths) -> list:
    elems: list = [_section_heading("Rating & Review Analysis")]
    chart_block: list = [Image(str(chart_paths.rating_distribution), width=CHART_WIDTH, height=CHART_HEIGHT)]
    chart_block.append(Spacer(1, 8))

    summary = result.sample_summary
    if summary.total_reviews_analyzed:
        chart_block.append(P(f"Total reviews analyzed: {summary.total_reviews_analyzed}", "Body"))
        chart_block.append(P(f"Average rating of sampled reviews: {summary.average_rating:.2f} / 5", "Body"))
    else:
        chart_block.append(P("No reviews were available to analyze in the sampled window.", "Body"))

    elems.append(KeepTogether(chart_block))
    return elems


# --- AI availability note (shared by sections 5, 6, 8) -------------------


def _ai_availability_note(llm_insights: Optional[LLMInsights]) -> list:
    """
    A single, one-time note when the AI insight layer is unavailable --
    rather than repeating a placeholder in each of the three sections
    that could have carried AI content. Returns [] when insights ARE
    available, since each section then shows its own AI content block.
    """
    if llm_insights is not None and llm_insights.available:
        return []
    reason = (llm_insights.unavailable_reason if llm_insights else None) or "no LLM provider was configured for this run"
    note = (
        f"Note: AI-generated insight summaries were not available for this report ({reason}). "
        "All findings below are derived from deterministic keyword and frequency analysis of the sampled reviews."
    )
    return [P(note, "MutedItalic"), Spacer(1, 6)]


# --- Sections 5 & 6: complaint / praise themes ---------------------------


def _theme_table(themes: list[ThemeItem]) -> Table:
    header = [P("Phrase", "TableHeader"), P("Mentions", "TableHeader")]
    data = [header] + [[P(t.phrase, "TableCell"), P(str(t.count), "TableCell")] for t in themes]
    tbl = Table(data, colWidths=[USABLE_WIDTH * 0.75, USABLE_WIDTH * 0.25], repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
                ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return tbl


def _themed_section(
    heading: str,
    themes: list[ThemeItem],
    empty_message: str,
    intro: str,
    ai_insights: list[str],
) -> list:
    elems: list = [_section_heading(heading)]
    if themes:
        elems.append(P(intro, "Body"))
        elems.append(_theme_table(themes))
    else:
        elems.append(P(empty_message, "Body"))

    if ai_insights:
        elems.append(Spacer(1, 10))
        elems.append(P("AI-Generated Insights", "SubHeading"))
        for insight in ai_insights:
            elems.append(P(f"\u2022 {insight}", "BulletBody"))

    elems.append(Spacer(1, 12))
    return elems


def _complaints_section(result: AnalysisResult, llm_insights: Optional[LLMInsights]) -> list:
    ai_insights = llm_insights.complaint_insights if llm_insights and llm_insights.available else []
    return _themed_section(
        heading="What Users Complain About",
        themes=result.complaint_themes,
        empty_message="No recurring complaint themes were identified in the sampled negative reviews.",
        intro="Recurring keyword and phrase patterns identified in 1- and 2-star reviews:",
        ai_insights=ai_insights,
    )


def _praise_section(result: AnalysisResult, llm_insights: Optional[LLMInsights]) -> list:
    ai_insights = llm_insights.praise_insights if llm_insights and llm_insights.available else []
    return _themed_section(
        heading="What Users Like",
        themes=result.praise_themes,
        empty_message="No recurring praise themes were identified in the sampled positive reviews.",
        intro="Recurring keyword and phrase patterns identified in 4- and 5-star reviews:",
        ai_insights=ai_insights,
    )


# --- Section 7: competitor comparison ------------------------------------


_COMPARISON_METRIC_ROWS: list[tuple[str, "callable"]] = [
    ("Rating", lambda row: f"{row.score:.1f} / 5" if row.score is not None else "N/A"),
    ("Rating Count", lambda row: f"{row.rating_count:,}" if row.rating_count is not None else "N/A"),
    ("Install Band", lambda row: row.installs or "N/A"),
    ("Category", lambda row: row.genre or "N/A"),
    ("Version", lambda row: row.version or "N/A"),
    ("Days Since Update", lambda row: str(row.days_since_update) if row.days_since_update is not None else "N/A"),
]


def _comparison_table_flowable(comparison_table: ComparisonTable) -> Table:
    header = [P("Metric", "TableHeader")] + [P(row.label, "TableHeader") for row in comparison_table.rows]
    data = [header]
    for label, getter in _COMPARISON_METRIC_ROWS:
        data.append([P(label, "TableKey")] + [P(getter(row), "TableCell") for row in comparison_table.rows])

    n_apps = len(comparison_table.rows)
    metric_col_width = 1.5 * inch
    app_col_width = (USABLE_WIDTH - metric_col_width) / max(n_apps, 1)
    col_widths = [metric_col_width] + [app_col_width] * n_apps

    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("BACKGROUND", (0, 1), (0, -1), LIGHT_BG),
    ]
    if n_apps >= 1:
        # Subtly highlight the Target column (always column index 1).
        style_cmds.append(("BACKGROUND", (1, 1), (1, -1), TARGET_HIGHLIGHT_BG))
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


def _competitor_comparison(result: AnalysisResult, chart_paths: ChartPaths) -> list:
    elems: list = [_section_heading("Competitor Comparison")]
    elems.append(Image(str(chart_paths.competitor_comparison), width=CHART_WIDTH, height=CHART_HEIGHT))
    elems.append(Spacer(1, 10))
    if len(result.comparison_table.rows) > 1:
        elems.append(_comparison_table_flowable(result.comparison_table))
    else:
        elems.append(P("No competitor apps were available for comparison in this run.", "Body"))
    return elems


# --- Section 8: recommendations -------------------------------------------


def _recommendations_section(result: AnalysisResult, llm_insights: Optional[LLMInsights]) -> list:
    elems: list = [_section_heading("Recommendations")]
    elems.append(P("Data-Driven Recommendations", "SubHeading"))
    for i, rec in enumerate(result.deterministic_recommendations, start=1):
        elems.append(P(f"{i}. {rec}", "BulletBody"))

    if llm_insights and llm_insights.available and llm_insights.recommendations:
        elems.append(Spacer(1, 10))
        elems.append(P("AI-Enhanced Recommendations", "SubHeading"))
        for i, rec in enumerate(llm_insights.recommendations, start=1):
            elems.append(P(f"{i}. {rec}", "BulletBody"))

    return elems


# --- Section 9: methodology / disclaimer ----------------------------------


def _methodology_section(result: AnalysisResult, llm_insights: Optional[LLMInsights]) -> list:
    elems: list = [Spacer(1, 16), _section_heading("Methodology & Disclaimer")]
    text = (
        "This report is based on publicly available information from the Google Play Store, "
        f"including app metadata and a sample of {result.sample_summary.total_reviews_analyzed} "
        "recent user reviews for the target app, retrieved at the time this report was generated. "
        "Competitor data reflects each app's public store listing at the same time. "
        "This report is an independent analysis and is not affiliated with, endorsed by, or "
        "sponsored by Google LLC or Google Play. All trademarks and app names are the property "
        "of their respective owners. Review sampling reflects a snapshot in time and may not "
        "capture every review or fully represent overall user sentiment."
    )
    elems.append(P(text, "Body"))

    if llm_insights and llm_insights.available:
        ai_note = (
            "Sections marked 'AI-Generated Insights' or 'AI-Enhanced Recommendations' were "
            "produced by a large language model based on aggregated review data and should be "
            "reviewed for accuracy before acting on them."
        )
        elems.append(P(ai_note, "Body"))

    return elems


# --- Page decorations (footer / page numbers) -----------------------------


def first_page_decorations(canvas, doc) -> None:  # noqa: ARG001 -- signature required by reportlab
    """No footer on the cover page -- keep it clean."""


def later_page_decorations(canvas, doc, app_title: str) -> None:
    canvas.saveState()
    canvas.setStrokeColor(BORDER)
    canvas.line(MARGIN, 0.62 * inch, PAGE_WIDTH - MARGIN, 0.62 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    # NOTE: canvas.drawString is raw low-level text drawing, not
    # Paragraph's XML-markup parser -- no _esc() needed here.
    canvas.drawString(MARGIN, 0.48 * inch, f"{app_title} - Play Store Audit Report")
    canvas.drawRightString(PAGE_WIDTH - MARGIN, 0.48 * inch, f"Page {doc.page}")
    canvas.restoreState()


# --- Public entry point ---------------------------------------------------


def build_story(
    result: AnalysisResult,
    chart_paths: ChartPaths,
    llm_insights: Optional[LLMInsights],
    generated_date: date,
) -> list:
    """Assemble the complete flowable list for the audit PDF, in section order."""
    story: list = []

    story += _cover_page(result.target_app, generated_date)
    story.append(PageBreak())

    story += _executive_summary(result)
    story += _target_overview(result.target_app)
    story.append(PageBreak())

    story += _rating_review_analysis(result, chart_paths)
    story.append(PageBreak())

    story += _ai_availability_note(llm_insights)
    story += _complaints_section(result, llm_insights)
    story += _praise_section(result, llm_insights)
    story.append(PageBreak())

    story += _competitor_comparison(result, chart_paths)
    story.append(PageBreak())

    story += _recommendations_section(result, llm_insights)
    story += _methodology_section(result, llm_insights)

    return story
