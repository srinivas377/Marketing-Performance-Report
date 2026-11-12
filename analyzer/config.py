"""Thresholds and assumptions used by the detection rules.

Everything that is a judgement call lives here so it can be explained in one
place and tuned without touching the logic. The values are also written into
the report ("Methodology & assumptions") so a reader can see what was used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Thresholds:
    # ---------------- Advertising ----------------
    target_roas: float = 3.0           # campaigns below this are "below target"
    par_gap_ratio: float = 0.70        # ROAS < 70% of non-brand average = below par
    high_spend_share: float = 0.10     # "high spend" = >= 10% of total ad spend
    low_cvr_ratio: float = 0.75        # conversion rate < 75% of non-brand average
    scale_roas_ratio: float = 1.25     # scale candidate: ROAS >= 1.25x non-brand average
    scale_min_purchases: int = 20      # ...and enough purchases to trust the ROAS
    reallocation_share: float = 0.20   # move 20% of flagged campaigns' budget
    scale_marginal_efficiency: float = 0.70  # extra $ earns 70% of a winner's average ROAS
    spread_ratio: float = 2.0          # best/worst ROAS inside one campaign type
    brand_keywords: tuple[str, ...] = ("brand",)  # branded search harvests existing demand

    # ---------------- SEO ----------------
    min_impressions: int = 5000        # ignore tiny pages for CTR / ranking checks
    ctr_underperform_ratio: float = 0.60   # actual CTR < 60% of the benchmark for its position
    striking_min_pos: float = 8.0      # "striking distance" = positions 8..20
    striking_max_pos: float = 20.0
    striking_target_pos: float = 7.0   # modelled target when a page is pushed up
    gap_closure: float = 0.50          # assume we only close half of any modelled gap
    slow_page_ratio: float = 1.25      # slow = > 1.25x the median load time
    cannibalization_similarity: float = 0.35  # Jaccard similarity of name+description words
    organic_value_factor: float = 0.50  # an organic click is worth 50% of a paid click

    # ---------------- Trends (run-to-run) ----------------
    trend_change_pct: float = 15.0     # flag changes of >= 15%
    trend_min_clicks: int = 20         # ignore pages with a tiny click base


# Benchmark organic CTR by average position (position, CTR). An *assumption*
# based on commonly published CTR-by-rank studies; linear interpolation is used
# between points. Replace with your own Search Console averages if you have them.
CTR_CURVE: tuple[tuple[float, float], ...] = (
    (1, 0.28), (2, 0.15), (3, 0.11), (4, 0.08), (5, 0.065), (6, 0.05),
    (7, 0.04), (8, 0.032), (9, 0.027), (10, 0.022), (15, 0.010),
    (20, 0.006), (30, 0.003),
)


@dataclass
class AppConfig:
    seo_path: Path = PROJECT_ROOT / "data" / "seo_data.csv"
    ads_path: Path = PROJECT_ROOT / "data" / "ads_data.csv"
    output_dir: Path = PROJECT_ROOT / "output"
    use_llm: bool = True
    provider: str | None = None   # anthropic | openai | gemini | grok (env LLM_PROVIDER)
    model: str | None = None      # overrides the provider default (env LLM_MODEL)
    thresholds: Thresholds = field(default_factory=Thresholds)

    @property
    def history_dir(self) -> Path:
        return self.output_dir / "history"
