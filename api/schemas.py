from pydantic import BaseModel, Field


class AuditRequest(BaseModel):

    target: str = Field(..., min_length=1)

    competitors: list[str] = Field(default_factory=list)

    country: str = "in"

    language: str = "en"

    review_sample_size: int = Field(default=180, ge=50, le=200)


class AuditResponse(BaseModel):

    audit_id: str

    status: str