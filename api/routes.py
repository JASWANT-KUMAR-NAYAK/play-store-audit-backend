from services.supabase_service import get_supabase_client

from uuid import uuid4
from pathlib import Path
from fastapi.responses import FileResponse

from fastapi import APIRouter, HTTPException

from api.schemas import AuditRequest, AuditResponse
from config import settings
from services import chart_service, llm_service, play_store_service, report_service
from services.analysis_service import analyze
from services.exceptions import (
    AppNotFoundError,
    ReportGenerationError,
    ScraperNetworkError,
)
from slugify import slugify


router = APIRouter(prefix="/api")

audits: dict[str, dict] = {}


@router.post("/audits", response_model=AuditResponse)
def create_audit(request: AuditRequest):
    audit_id = str(uuid4())

    try:
        target, competitors = play_store_service.fetch_target_and_competitors(
            target_id=request.target,
            competitor_ids=request.competitors,
            country=request.country,
            lang=request.language,
        )

        reviews = play_store_service.fetch_reviews(
            target_id=request.target,
            country=request.country,
            lang=request.language,
        )

        result = analyze(
            target=target,
            competitors=competitors,
            reviews=reviews,
        )

        chart_output_dir = settings.OUTPUT_DIR / "charts"
        chart_paths = chart_service.generate_all_charts(
            result,
            chart_output_dir,
        )

        llm_insights = llm_service.generate_llm_insights(result)

        report_filename = f"{slugify(target.title or request.target)}-audit-report.pdf"
        report_path = settings.OUTPUT_DIR / report_filename

        try:
            report_service.generate_report(
                analysis_result=result,
                chart_paths=chart_paths,
                output_path=report_path,
                llm_insights=llm_insights,
            )
        except ReportGenerationError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"PDF generation failed: {exc}",
            ) from exc

        audits[audit_id] = {
            "audit_id": audit_id,
            "target": target.model_dump(),
            "competitors": [c.model_dump() for c in competitors],
            "reviews_count": len(reviews),
            "analysis": result.model_dump(),
            "charts": {
                "rating_distribution": str(chart_paths.rating_distribution),
                "competitor_comparison": str(chart_paths.competitor_comparison),
            },
            "report_path": str(report_path),
            "llm_insights": llm_insights.model_dump(),
            "status": "completed",
        }
        supabase = get_supabase_client()

        supabase.table("audits").insert(
            {
                "audit_id": audit_id,
                "target_package": request.target,
                "target_name": target.title,
                "status": "completed",
            }
        ).execute()

        return AuditResponse(
            audit_id=audit_id,
            status="completed",
        )

    except AppNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ScraperNetworkError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc


@router.get("/audits/{audit_id}")
def get_audit(audit_id: str):
    audit = audits.get(audit_id)

    if audit is None:
        raise HTTPException(
            status_code=404,
            detail="Audit not found",
        )

    return audit
@router.get("/audits/{audit_id}/report")
def download_report(audit_id: str):
    audit = audits.get(audit_id)

    if audit is None:
        raise HTTPException(status_code=404, detail="Audit not found")

    report_path = audit.get("report_path")

    if not report_path:
        raise HTTPException(status_code=404, detail="Report not available")

    path = Path(report_path)

    if not path.is_file():
        raise HTTPException(status_code=404, detail="Report file not found")

    return FileResponse(
        path=path,
        media_type="application/pdf",
        filename=path.name,
    )