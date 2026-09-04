"""
Play Store Competitor & Review Audit Report Generator.

CLI entrypoint. Usage:
    python main.py --target com.example.app --competitor com.rival.one \
        --competitor com.rival.two --country in --lang en

Pipeline wiring (fetch -> analyze -> report) is added incrementally in
Phases 3-7; this entrypoint currently handles argument parsing and
input validation only.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from slugify import slugify

from config import settings
from services import chart_service, llm_service, play_store_service, report_service
from services.analysis_service import analyze
from services.exceptions import AppNotFoundError, ReportGenerationError, ScraperNetworkError
from utils.logger import get_logger, set_global_level

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Play Store competitor & review audit PDF report."
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target app's Play Store package ID, e.g. com.example.app",
    )
    parser.add_argument(
        "--competitor",
        action="append",
        default=[],
        dest="competitors",
        help=(
            "Competitor package ID. Repeat up to "
            f"{settings.MAX_COMPETITORS} times."
        ),
    )
    parser.add_argument(
        "--country",
        default=settings.DEFAULT_COUNTRY,
        help=f"Play Store country code (default: {settings.DEFAULT_COUNTRY})",
    )
    parser.add_argument(
        "--lang",
        default=settings.DEFAULT_LANG,
        help=f"Play Store language code (default: {settings.DEFAULT_LANG})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging (scraper request attempts, retry backoff, etc).",
    )
    args = parser.parse_args(argv)

    if len(args.competitors) > settings.MAX_COMPETITORS:
        parser.error(
            f"At most {settings.MAX_COMPETITORS} --competitor values are "
            f"supported, got {len(args.competitors)}."
        )

    return args


def main(argv: list[str] | None = None) -> int:
    start_time = time.perf_counter()

    def _finish(exit_code: int) -> int:
        elapsed = time.perf_counter() - start_time
        logger.info("Run finished in %.1fs (exit code %d)", elapsed, exit_code)
        return exit_code

    args = parse_args(argv)
    if args.verbose:
        set_global_level(logging.DEBUG)

    logger.info(
        "Starting audit run | target=%s competitors=%s country=%s lang=%s",
        args.target,
        args.competitors,
        args.country,
        args.lang,
    )

    try:
        try:
            target, competitors = play_store_service.fetch_target_and_competitors(
                target_id=args.target,
                competitor_ids=args.competitors,
                country=args.country,
                lang=args.lang,
            )
        except AppNotFoundError as exc:
            logger.error("Target app could not be found: %s", exc)
            return _finish(1)
        except ScraperNetworkError as exc:
            logger.error("Network failure while fetching target app: %s", exc)
            return _finish(1)

        logger.info("Target: %s", target.model_dump_json(indent=2))
        for comp in competitors:
            logger.info("Competitor: %s", comp.model_dump_json(indent=2))

        try:
            reviews = play_store_service.fetch_reviews(
                target_id=args.target, country=args.country, lang=args.lang
            )
        except AppNotFoundError as exc:
            logger.error("Target app could not be found while fetching reviews: %s", exc)
            return _finish(1)

        if reviews:
            avg_rating = sum(r.rating for r in reviews) / len(reviews)
            logger.info("Fetched %d reviews | sample average rating: %.2f", len(reviews), avg_rating)
        else:
            logger.warning("No reviews were available for this app -- proceeding with an empty sample.")

        result = analyze(target=target, competitors=competitors, reviews=reviews)
        logger.info(
            "Analysis complete | complaint_themes=%d praise_themes=%d recommendations=%d",
            len(result.complaint_themes),
            len(result.praise_themes),
            len(result.deterministic_recommendations),
        )

        chart_output_dir = settings.OUTPUT_DIR / "charts"
        chart_paths = chart_service.generate_all_charts(result, chart_output_dir)
        logger.info("Rating distribution chart -> %s", chart_paths.rating_distribution)
        logger.info("Competitor comparison chart -> %s", chart_paths.competitor_comparison)

        report_filename = f"{slugify(target.title or args.target)}-audit-report.pdf"
        report_path = settings.OUTPUT_DIR / report_filename

        llm_insights = llm_service.generate_llm_insights(result)

        try:
            report_service.generate_report(
                analysis_result=result,
                chart_paths=chart_paths,
                output_path=report_path,
                llm_insights=llm_insights,
            )
        except ReportGenerationError as exc:
            logger.error("PDF generation failed: %s", exc)
            return _finish(1)

        logger.info("Report generated successfully -> %s", report_path)
        return _finish(0)

    except Exception as exc:  # noqa: BLE001 - absolute last resort; never show the end user a raw traceback
        logger.error("Unexpected error, run aborted: %s", exc)
        if args.verbose:
            logger.exception("Full traceback:")
        return _finish(1)


if __name__ == "__main__":
    sys.exit(main())
