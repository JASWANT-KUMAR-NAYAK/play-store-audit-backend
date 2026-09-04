"""
Central configuration for the Play Store Audit Report Generator.

All tunable constants and environment-derived secrets live here so the
rest of the application never reads os.environ directly.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
SUPABASE_URL: str | None = os.getenv("SUPABASE_URL") or None
SUPABASE_KEY: str | None = os.getenv("SUPABASE_KEY") or None
# --- Paths -------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUT_DIR: Path = PROJECT_ROOT / "output"

# --- Scraper defaults ----------------------------------------------------

DEFAULT_COUNTRY: str = os.getenv("DEFAULT_COUNTRY", "in")
DEFAULT_LANG: str = os.getenv("DEFAULT_LANG", "en")

# Target review sample size. google-play-scraper paginates internally;
# the service layer requests batches until this count (or the app's
# total review count) is reached.
REVIEW_SAMPLE_SIZE: int = int(os.getenv("REVIEW_SAMPLE_SIZE", "180"))

# Max competitor apps supported per report, per locked product scope.
MAX_COMPETITORS: int = 3

# --- Network / retry behavior --------------------------------------------

REQUEST_MAX_ATTEMPTS: int = 4
REQUEST_BACKOFF_SECONDS: float = 1.5  # base for exponential backoff

# --- LLM configuration -----------------------------------------------------

# "anthropic", "openai", or "" / None to disable AI-generated insights.
LLM_PROVIDER: str | None = os.getenv("LLM_PROVIDER") or None

ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY") or None
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY") or None

ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL") or "claude-sonnet-5"
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

# --- Analysis tuning ---------------------------------------------------

# Theme extraction n-gram sizes (unigrams + bigrams by default).
THEME_NGRAM_SIZES: tuple[int, ...] = (1, 2)
THEME_TOP_N: int = 10

# --- Chart output --------------------------------------------------------

CHART_DPI: int = 150
