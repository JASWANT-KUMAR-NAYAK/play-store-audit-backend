# Play Store Competitor & Review Audit Report Generator

A command-line tool that generates a professional PDF audit report for a
Google Play Store app: rating and review analysis, recurring complaint/praise
themes, a competitor comparison, and actionable recommendations. Built to be
a realistic freelance deliverable — one command in, one polished PDF out.

Optionally enriches the deterministic analysis with AI-generated insights
via Anthropic or OpenAI. Works fully without an LLM key: the report still
generates, with AI-specific sections clearly marked unavailable.

## What it does

Given a target app's Play Store package ID (and up to 3 optional
competitors), it:

1. Fetches app metadata for the target and competitors
2. Fetches ~150–200 recent reviews for the target app
3. Runs deterministic analysis (no ML) — rating distribution, recurring
   complaint/praise keyword themes, competitor comparison
4. Optionally sends the aggregated, anonymized results through a single
   batched LLM call for plain-English insights and recommendations
5. Renders everything into a 9-section PDF report

## Requirements

- Python 3.10+
- Internet access to `play.google.com` (and, optionally, your chosen LLM
  provider's API)

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill in what you need:

```bash
cp .env.example .env
```

| Variable | Required? | Notes |
|---|---|---|
| `LLM_PROVIDER` | No | `anthropic`, `openai`, or leave blank to disable AI insights |
| `ANTHROPIC_API_KEY` | Only if `LLM_PROVIDER=anthropic` | |
| `OPENAI_API_KEY` | Only if `LLM_PROVIDER=openai` | |
| `ANTHROPIC_MODEL` | No | Defaults to `claude-sonnet-5` |
| `OPENAI_MODEL` | No | Defaults to `gpt-4o-mini` |
| `DEFAULT_COUNTRY` | No | Defaults to `in` |
| `DEFAULT_LANG` | No | Defaults to `en` |
| `REVIEW_SAMPLE_SIZE` | No | Defaults to `180` |

Leaving `LLM_PROVIDER` blank is a fully supported, first-class mode — the
report generates from deterministic analysis alone.

## Usage

```bash
python main.py --target com.example.app \
    --competitor com.rival.one \
    --competitor com.rival.two \
    --country in \
    --lang en
```

| Flag | Required? | Description |
|---|---|---|
| `--target` | Yes | Target app's Play Store package ID |
| `--competitor` | No | Competitor package ID. Repeatable up to 3 times |
| `--country` | No | Play Store country code (default from `.env`, else `in`) |
| `--lang` | No | Play Store language code (default from `.env`, else `en`) |
| `--verbose` / `-v` | No | Enable debug-level logging (scraper request attempts, retry backoff) |

The report is written to `output/<app-name>-audit-report.pdf`. Chart PNGs
used in the report are written to `output/charts/`.

### Example

```bash
python main.py --target com.spotify.music --competitor com.google.android.apps.youtube.music -v
```

## Output

A PDF with 9 sections: cover page, executive summary, target app overview,
rating & review analysis (with a rating-distribution chart), what users
complain about, what users like, competitor comparison (with a chart and
table), recommendations, and a methodology/disclaimer page. The report does
not claim any official affiliation with Google.

## Project structure

```
play_store_audit/
├── main.py                    # CLI entrypoint
├── config/settings.py         # env loading + tunable constants
├── models/                    # pydantic data models (App, Review, Analysis)
├── services/
│   ├── scraper_client.py      # low-level google-play-scraper wrapper (retry/backoff)
│   ├── play_store_service.py  # fetch + normalize app metadata and reviews
│   ├── analysis_service.py    # deterministic rating/theme/recommendation analysis
│   ├── chart_service.py       # matplotlib chart generation
│   ├── llm_service.py         # optional Anthropic/OpenAI insight generation
│   ├── report_service.py      # PDF assembly orchestration
│   └── exceptions.py          # custom exception hierarchy
├── templates/report_templates.py  # reportlab section/layout builders
├── utils/                     # text cleaning, date helpers, logging
└── tests/                     # pytest suite (see below)
```

## Running tests

```bash
pytest tests/ -v
```

The suite is self-contained — every external boundary (the Play Store
scraper, Anthropic, OpenAI) is mocked, so it runs without network access or
API keys.

## Notes & limitations

- Theme extraction is deterministic (regex tokenization + frequency
  counting), not ML-based, per the locked V1 scope.
- The LLM layer, when enabled, makes exactly one batched call per report
  using aggregated, anonymized statistics — never raw review text.
- This is an independent, unofficial tool. It is not affiliated with,
  endorsed by, or sponsored by Google LLC or Google Play.
