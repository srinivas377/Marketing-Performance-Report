# AI Marketing Performance Analyzer

A Python application that analyzes SEO and paid-advertising CSV datasets, identifies performance bottlenecks and growth opportunities with deterministic Python rules, calls an LLM API to generate grounded, prioritized business recommendations, and outputs clean **HTML and JSON reports** (`output/report.html` and `output/report.json`). Whenever the source CSV files change, the analyzer recalculates all metrics, tracks run-over-run trends, and regenerates updated recommendations automatically.

---

## Key Highlights & Results on the Supplied Dataset

Running the analyzer on `data/seo_data.csv` (12 product pages) and `data/ads_data.csv` (10 campaigns) surfaces:

- **Paid Advertising Summary:** **$6,680** total spend generating **$32,230** revenue (**4.82×** blended ROAS, **286** purchases, **$23** average CPA).
- **SEO Summary (28 days):** **254,900** impressions and **4,605** clicks (**1.81%** overall CTR, **10.8** weighted average position, **58.5%** of impressions stuck on page 2+).
- **Top P1 Issues & Opportunities Detected (7 issues, 6 opportunities):**
  - **Paid Ads (`ADS-01`):** *Meta - Prospecting* consumes 14% of ad spend ($950) at a **2.20× ROAS** (below the 3.0× target) and 1.00% conversion rate (~$1,815 modelled revenue gap vs. non-brand average).
  - **SEO Ranking Upside (`SEO-01`):** 4 high-impression pages sit on page 2 (positions 8–20), led by *Black Mermaid Prom Dress (`P-101`)* with 52,000 impressions at position 11.8 (moving toward page 1 adds ~2,012 modelled clicks / 28d).
  - **SEO CTR Gap (`SEO-02`):** 3 top-8 pages underperform their rank benchmark CTR, led by *Red Satin Evening Gown (`P-102`)* at position 4.3 with 3.00% CTR vs. ~7.6% benchmark.
  - **Indexing & Content Hygiene (`SEO-03`, `SEO-04`, `SEO-05`):** 2 pages flagged as not indexed (`P-104`, `P-110` accounting for 30,700 impressions), 2 pages missing product descriptions (`P-103`, `P-110`), and 4 slow-loading pages (`load_vs_position_corr = 0.89`).
  - **Cross-Channel (`X-01`, `X-02`) & Reallocation (`ADS-03`):** Paid search campaigns (*Google Search - Prom*, *Google Search - Evening*) overlap with organic page-2/unindexed gaps, and shifting 20% (~$590) of budget from weak campaigns to proven winners (*Meta - Retargeting* 8.0×, *Google Search - Prom* 6.0×, *Meta - Sale Creative* 5.5×) adds ~$796 modelled revenue on the same spend.

---

## Architecture & Design Decisions

```text
seo_data.csv ─┐                                         ┌─> output/report.html
              ├─> clean & validate ─> metrics ─> ranked ┼─> LLM API ─> output/report.json
ads_data.csv ─┘     (loader.py)      (seo.py,   findings│   (llm.py)
                                      ads.py,           │
                                      cross.py)         │
                                         ▲
                 previous run snapshot (history.py) ──> trend & decline detection
```

1. **Zero Hallucinated Metrics (Python Computes, LLM Synthesizes):** Python calculates every KPI, correlation, and dollar estimate first. The LLM receives only the structured analysis JSON, is instructed not to invent numbers, and must link every recommendation to specific finding IDs (`ADS-01`, `SEO-02`, etc.). Unknown IDs are stripped during validation.
2. **Multi-Provider Support + Graceful Resilience:** Supports Anthropic (Claude), OpenAI, Google Gemini, and xAI Grok behind one interface, with automatic transient-error retry (e.g. temporary 503/429 spikes), malformed-JSON retry, and a clearly labeled rule-based fallback if an API key is missing or an endpoint is unreachable (never disguising fallback output as AI-generated).
3. **Real Dynamic Updates:** Supports both one-click re-runs and a built-in `--watch` mode using SHA-256 content hashing and debouncing so identical saves don't trigger redundant API calls, while real data edits snapshot the change and generate run-over-run trend diagnostics.

---

## Project Structure

```text
main.py                      CLI entry point (run once, --watch, --no-llm, --provider, --version)
pyproject.toml               Pytest configuration (pythonpath = ["."])
requirements.txt             Runtime dependencies
requirements-dev.txt         Test dependencies (pytest)
.env.example                 Template for LLM provider and API key
analyzer/
  config.py                  Centralized thresholds, CTR curve, and modelling assumptions
  loader.py                  CSV ingestion, validation, type coercion, and data-quality reporting
  seo.py                     SEO metrics, category/page summaries, and issue/opportunity detectors
  ads.py                     Ad metrics (ROAS, CPA, CVR, CTR, CPC, frequency) and detectors
  cross.py                   Cross-channel detection (paid search vs. organic ranking gaps)
  history.py                 Run-to-run snapshots and period-change / decline detection
  findings.py                Finding dataclass, priority ranking (P1/P2/P3), and stable IDs
  llm.py                     Prompting, multi-provider API calls, JSON validation, and fallback
  report.py                  Report builder, strict JSON writer, and Jinja2 HTML renderer
  templates/report.html.j2   Executive-friendly responsive & printable HTML report template
  watcher.py                 Polling + debounced SHA-256 file-change watcher
  pipeline.py                End-to-end orchestration
data/                        Supplied datasets (seo_data.csv, ads_data.csv)
output/                      Generated reports (report.html, report.json)
tests/                       Automated pytest suite (22 tests)
```

---

## Installation & Setup

Requires **Python 3.10+**.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Configuring the LLM API

1. Copy the template file:
   - macOS/Linux: `cp .env.example .env`
   - Windows: `copy .env.example .env`
2. Open `.env` and add **one** API key for your chosen provider:

| `LLM_PROVIDER` | Environment Variable | Default Model | Notes |
|---|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5` | Claude SDK (default if unset) |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` | OpenAI Chat Completions |
| `gemini` | `GEMINI_API_KEY` | `gemini-flash-lite-latest` | Gemini OpenAI-compatible endpoint |
| `grok` | `XAI_API_KEY` | `grok-4` | xAI OpenAI-compatible endpoint |

- **Auto-detection:** If `LLM_PROVIDER` is left blank in `.env`, the analyzer automatically selects whichever provider key is populated.
- **Model overrides:** Set `LLM_MODEL=<model_id>` in `.env` or pass `--model <model_id>` on the command line at any time.
- **Security:** `.env` is listed in `.gitignore`. No API keys or credentials are stored in source code.

---

## How to Run

```bash
# Standard run: analyze both CSVs, call the LLM API, write output/report.html + output/report.json
python main.py

# Watch mode: run immediately, then automatically regenerate whenever either CSV changes
python main.py --watch

# Offline / rule-based mode: skip the LLM API call (visibly labeled in the report)
python main.py --no-llm

# Override provider, model, or input/output paths
python main.py --provider gemini --model gemini-flash-lite-latest
python main.py --seo path/to/seo.csv --ads path/to/ads.csv --out path/to/output
```

Open `output/report.html` in any browser (Windows: `start output\report.html`, macOS: `open output/report.html`).

---

## How the CSV Analysis Works

### 1. Ingestion & Data Quality (`analyzer/loader.py`)
- Validates required columns and UTF-8 encoding.
- Coerces numeric columns (`errors="coerce"`), normalizes boolean `indexed` flags, and fills missing optional labels (`category` → `Uncategorized`, `platform`/`campaign_type` → `Unknown`).
- Excludes corrupted rows (missing IDs, duplicates, negative metrics, `clicks > impressions`) without crashing, and logs every exclusion or anomaly (e.g., empty descriptions, duplicate `url_slug`, zero spend, `purchases > clicks`) into the **Data quality** section of the report.
- Uses `safe_div` everywhere so zero-spend or zero-impression rows yield `"n/a"` (`None` in strict JSON) instead of crashing.

### 2. Metrics Calculated
- **SEO (`analyzer/seo.py`):** Per-page and per-category impressions, clicks, CTR, expected CTR by position, impression-weighted average position, page-2 impression share, indexing status, missing descriptions, and load-time correlation with rank.
- **Advertising (`analyzer/ads.py`):** Portfolio, platform, campaign-type, and per-campaign spend, revenue, purchases, ROAS (`revenue ÷ spend`), CPA (`spend ÷ purchases`), conversion rate (`purchases ÷ clicks`), CTR, CPC, AOV, reach frequency (`impressions ÷ reach`), and spend share. Non-brand averages serve as the baseline so branded search doesn't skew benchmarks.

### 3. Detection Rules & Prioritization (`analyzer/config.py`, `analyzer/findings.py`)

| Area | Detector | Trigger Logic |
|---|---|---|
| Ads | Weak campaign | ROAS < 3.0× target, or < 70% of non-brand average with ≥ 10% spend share, or high spend with CVR < 75% of non-brand average |
| Ads | Budget reallocation | Shifts 20% of flagged weak spend to proven scale candidates (ROAS ≥ 1.25× non-brand average & ≥ 20 purchases) at 70% marginal efficiency |
| Ads | Intra-type spread | Best-vs-worst ROAS ratio ≥ 2.0× within the same campaign type |
| Ads | Portfolio ROAS | Overall blended ROAS below 3.0× target |
| SEO | Not indexed | `indexed == False` (flags inconsistency if impressions > 0) |
| SEO | Low CTR for rank | Top-8 page with ≥ 5,000 impressions and CTR < 60% of position benchmark |
| SEO | Page-2 upside | Positions 8–20 with ≥ 5,000 impressions; models 50% gap closure toward position 7 |
| SEO | Content & speed | Empty descriptions and pages > 1.25× median load time |
| SEO | Keyword overlap | Same-category pages with ≥ 0.35 Jaccard token similarity |
| Cross | Paid vs. organic gap | Non-brand Search campaigns matching categories that have page-2 or unindexed pages |
| Trend | Period declines | Run-over-run organic clicks/impressions down ≥ 15%, ROAS down ≥ 15%, or spend up ≥ 5% while purchases drop ≥ 5% |

All findings are ranked by **severity (`P1` high → `P2` medium → `P3` low)** and then by **modelled revenue at stake**.  
*Note on assumptions:* Because the supplied CSVs are single 28-day snapshots without date columns, trend comparison activates from the second run onward once data changes, and the benchmark CTR curve in `config.py` is an explicit assumption documented in the report's *Methodology & assumptions* section.

---

## Regenerating Reports on Data Changes

- **Automatic (`python main.py --watch`):** Polls `seo_data.csv` and `ads_data.csv` every 2 seconds (configurable via `--interval`), waits for editor writes to settle, verifies an actual content change via SHA-256, and rebuilds both reports with fresh LLM recommendations. If a CSV is mid-edit or temporarily broken, it logs an error and keeps watching.
- **Manual:** Edit either CSV and re-run `python main.py`.

Each run on new data stores a compact snapshot in `output/history/`. On subsequent changes, `history.py` adds a **"What changed since the previous data"** comparison table and flags any emerging declines. Re-running on unchanged CSVs does not create duplicate snapshots.

**Quick Live-Demo Recipe:**
1. Run `python main.py --watch`.
2. In `data/ads_data.csv`, change `Meta - Retargeting` (`spend_usd` `410` → `1500`, `purchases` `41` → `10`, `revenue_usd` `3280` → `800`) and save.
3. Within a few seconds, the updated report shows lower blended ROAS, a **"What changed"** comparison table, a trend-decline alert, and updated AI recommendations.

---

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

All **22 tests** cover CSV validation/cleaning, manual-math verification of SEO/Ads metrics, keyword cannibalization, zero-spend edge cases, run-to-run trend detection, strict JSON compliance, LLM provider auto-detection/JSON parsing/retry/fallback (using a mock caller with no API key required), and the file watcher.
