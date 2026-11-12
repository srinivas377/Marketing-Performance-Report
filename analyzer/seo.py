"""SEO analysis: metrics, summary and issue / opportunity detection."""
from __future__ import annotations

import itertools
import re

import numpy as np
import pandas as pd

from .config import CTR_CURVE, Thresholds
from .findings import Finding
from .utils import safe_div

_XS = np.array([p for p, _ in CTR_CURVE], dtype=float)
_YS = np.array([c for _, c in CTR_CURVE], dtype=float)
_STOP = {"a", "an", "and", "for", "with", "the", "of", "in", "on", "to", "s"}


def expected_ctr(position: float) -> float:
    """Benchmark organic CTR for an average position (linear interpolation)."""
    return float(np.interp(position, _XS, _YS))


# --------------------------------------------------------------------------
# Enrichment + summary
# --------------------------------------------------------------------------
def enrich(df: pd.DataFrame, th: Thresholds) -> pd.DataFrame:
    d = df.copy()
    d["ctr"] = (d["clicks_28d"] / d["impressions_28d"].where(d["impressions_28d"] > 0)).fillna(0.0)
    d["expected_ctr"] = d["avg_position"].map(lambda p: expected_ctr(p) if pd.notna(p) else np.nan)
    d["ctr_ratio"] = d["ctr"] / d["expected_ctr"]
    median_load = d["page_load_ms"].median()
    d["slow"] = (d["page_load_ms"] > median_load * th.slow_page_ratio).fillna(False) if pd.notna(median_load) else False
    d["not_indexed"] = (d["indexed"] == False).fillna(False).astype(bool)  # noqa: E712
    d["missing_desc"] = d["description"].eq("")
    d["page2"] = (d["avg_position"] > 10).fillna(False)
    d["striking"] = (
        d["avg_position"].between(th.striking_min_pos, th.striking_max_pos)
        & (d["impressions_28d"] >= th.min_impressions)
        & ~d["not_indexed"]
    ).fillna(False)
    d["low_ctr"] = (
        (d["impressions_28d"] >= th.min_impressions)
        & (d["ctr_ratio"] < th.ctr_underperform_ratio)
        & (d["avg_position"] < th.striking_min_pos)
        & ~d["not_indexed"]
    ).fillna(False)
    return d


def _flags(row: pd.Series) -> list[str]:
    out = []
    if row["not_indexed"]:
        out.append("Not indexed")
    if row["missing_desc"]:
        out.append("No description")
    if row["low_ctr"]:
        out.append("Low CTR for rank")
    if row["striking"]:
        out.append("Page 2 – ranking upside")
    if row["slow"]:
        out.append("Slow page")
    return out


def _weighted_position(d: pd.DataFrame):
    v = d.dropna(subset=["avg_position"])
    if v.empty or v["impressions_28d"].sum() <= 0:
        return None
    return float(np.average(v["avg_position"], weights=v["impressions_28d"]))


def summarize_seo(df: pd.DataFrame, th: Thresholds) -> tuple[pd.DataFrame, dict]:
    d = enrich(df, th)
    impressions = float(d["impressions_28d"].sum())
    clicks = float(d["clicks_28d"].sum())

    valid = d.dropna(subset=["avg_position", "page_load_ms"])
    corr = None
    if len(valid) >= 6 and valid["page_load_ms"].nunique() > 1 and valid["avg_position"].nunique() > 1:
        corr = float(valid["page_load_ms"].corr(valid["avg_position"]))

    by_category = []
    for cat, g in d.groupby("category", sort=False):
        by_category.append({
            "category": cat,
            "pages": int(len(g)),
            "impressions": float(g["impressions_28d"].sum()),
            "clicks": float(g["clicks_28d"].sum()),
            "ctr": safe_div(g["clicks_28d"].sum(), g["impressions_28d"].sum(), 0.0),
            "avg_position": _weighted_position(g),
        })
    by_category.sort(key=lambda r: -r["impressions"])

    pages = []
    for _, r in d.sort_values("impressions_28d", ascending=False).iterrows():
        pages.append({
            "product_id": r["product_id"], "product_name": r["product_name"],
            "category": r["category"], "url_slug": r["url_slug"],
            "impressions": float(r["impressions_28d"]), "clicks": float(r["clicks_28d"]),
            "ctr": float(r["ctr"]), "expected_ctr": r["expected_ctr"],
            "avg_position": r["avg_position"], "indexed": None if pd.isna(r["indexed"]) else bool(r["indexed"]),
            "page_load_ms": r["page_load_ms"], "flags": _flags(r),
        })

    summary = {
        "totals": {
            "pages": int(len(d)),
            "impressions": impressions,
            "clicks": clicks,
            "ctr": safe_div(clicks, impressions, 0.0),
            "avg_position": _weighted_position(d),
            "page2_impression_share": safe_div(d.loc[d["page2"], "impressions_28d"].sum(), impressions, 0.0),
            "indexed_pages": int((d["indexed"] == True).sum()),  # noqa: E712
            "not_indexed_pages": int(d["not_indexed"].sum()),
            "missing_descriptions": int(d["missing_desc"].sum()),
            "avg_load_ms": None if d["page_load_ms"].isna().all() else float(d["page_load_ms"].mean()),
            "load_vs_position_corr": corr,
        },
        "by_category": by_category,
        "pages": pages,
    }
    return d, summary


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
def _sev_from_click_share(gain_clicks: float, total_clicks: float) -> str:
    share = safe_div(gain_clicks, total_clicks, 0.0)
    return "high" if share >= 0.10 else "medium" if share >= 0.03 else "low"


def _tokens(row: pd.Series) -> set[str]:
    text = f"{row['product_name']} {row['description']}".lower()
    return {t for t in re.findall(r"[a-z0-9]+", text) if t not in _STOP}


def detect_seo(d: pd.DataFrame, summary: dict, th: Thresholds, value_per_click: float) -> list[Finding]:
    total_clicks = summary["totals"]["clicks"]
    total_impr = summary["totals"]["impressions"]
    findings: list[Finding] = []

    # 1) Pages flagged as not indexed ------------------------------------------------
    ni = d[d["not_indexed"]]
    if not ni.empty:
        impr = float(ni["impressions_28d"].sum())
        clicks = float(ni["clicks_28d"].sum())
        names = ", ".join(f"{r.product_name} ({r.product_id})" for r in ni.itertuples())
        inconsistent = (
            f" They still show {impr:,.0f} impressions, which is inconsistent with being unindexed, so the flag may be stale or the pages "
            "may have recently dropped out of the index."
            if impr > 0 else ""
        )
        findings.append(Finding(
            area="SEO", kind="issue", severity="high",
            title=f"{len(ni)} product page(s) flagged as not indexed",
            detail=(
                f"{names} {'is' if len(ni) == 1 else 'are'} marked as not indexed. Together they account for "
                f"{safe_div(impr, total_impr, 0) * 100:.0f}% of impressions ({impr:,.0f}) and {clicks:,.0f} clicks.{inconsistent}"
            ),
            suggested_action="Check each URL in Google Search Console (URL Inspection) for noindex tags, canonical conflicts or robots.txt blocks, fix the cause and request re-indexing.",
            value_usd=clicks * value_per_click,
            entities=[{"product_id": r.product_id, "name": r.product_name, "impressions": r.impressions_28d,
                       "clicks": r.clicks_28d, "position": r.avg_position} for r in ni.itertuples()],
        ))

    # 2) High impressions, CTR far below benchmark for the page's rank ----------------
    low = d[d["low_ctr"]].copy()
    if not low.empty:
        low["gain"] = (low["impressions_28d"] * (low["expected_ctr"] - low["ctr"])).clip(lower=0) * th.gap_closure
        low = low.sort_values("gain", ascending=False)
        gain = float(low["gain"].sum())
        top = low.iloc[0]
        findings.append(Finding(
            area="SEO", kind="issue", severity=_sev_from_click_share(gain, total_clicks),
            title=f"{len(low)} well-ranked page(s) get far fewer clicks than their position should earn",
            detail=(
                f"These pages rank in the top {th.striking_min_pos:.0f} but their click-through rate is below "
                f"{th.ctr_underperform_ratio * 100:.0f}% of the benchmark for their position. The biggest gap is {top['product_name']} "
                f"({top['product_id']}): {top['impressions_28d']:,.0f} impressions at position {top['avg_position']:.1f} but only "
                f"{top['ctr'] * 100:.1f}% CTR (benchmark ≈ {top['expected_ctr'] * 100:.1f}%). Closing half of the gap on all of them is worth roughly "
                f"{gain:,.0f} extra clicks per 28 days."
            ),
            suggested_action="Rewrite the title tags and meta descriptions to lead with the product's key attribute, price/offer and a clear call to action; add product schema (price, rating, availability) for rich results; A/B test and re-check CTR in 4 weeks.",
            value_usd=gain * value_per_click,
            entities=[{"product_id": r.product_id, "name": r.product_name, "impressions": r.impressions_28d,
                       "position": r.avg_position, "ctr": r.ctr, "expected_ctr": r.expected_ctr,
                       "est_extra_clicks": r.gain} for r in low.itertuples()],
        ))

    # 3) Striking distance: high impressions but stuck on page 2 -----------------------
    st = d[d["striking"]].copy()
    if not st.empty:
        target = expected_ctr(th.striking_target_pos)
        st["gain"] = (st["impressions_28d"] * (target - st["ctr"])).clip(lower=0) * th.gap_closure
        st = st.sort_values("gain", ascending=False)
        gain = float(st["gain"].sum())
        top = st.iloc[0]
        findings.append(Finding(
            area="SEO", kind="opportunity", severity=_sev_from_click_share(gain, total_clicks),
            title=f"{len(st)} high-impression page(s) are stuck on page 2 of Google",
            detail=(
                f"These pages are shown often but rank between position {th.striking_min_pos:.0f} and {th.striking_max_pos:.0f}, where few people click. "
                f"{top['product_name']} ({top['product_id']}) has {top['impressions_28d']:,.0f} impressions at position {top['avg_position']:.1f} and a "
                f"{top['ctr'] * 100:.2f}% CTR. Moving these pages towards position {th.striking_target_pos:.0f} would add roughly {gain:,.0f} clicks per 28 days "
                f"(assuming only half of the modelled gain is achieved)."
            ),
            suggested_action="For each page: expand the product description with unique copy, strengthen internal links from category and blog pages, improve the title for the main keyword, and earn a few relevant backlinks. Start with the page with the largest impression count.",
            value_usd=gain * value_per_click,
            entities=[{"product_id": r.product_id, "name": r.product_name, "impressions": r.impressions_28d,
                       "position": r.avg_position, "ctr": r.ctr, "est_extra_clicks": r.gain} for r in st.itertuples()],
        ))

    # 4) Missing descriptions ----------------------------------------------------------
    md = d[d["missing_desc"]]
    if not md.empty:
        impr = float(md["impressions_28d"].sum())
        big = bool((md["impressions_28d"] >= th.min_impressions).any())
        names = ", ".join(f"{r.product_name} ({r.product_id})" for r in md.itertuples())
        findings.append(Finding(
            area="SEO", kind="issue", severity="medium" if big else "low",
            title=f"{len(md)} product page(s) have no description",
            detail=f"{names} {'has' if len(md) == 1 else 'have'} an empty description while receiving {impr:,.0f} impressions. Thin pages rank and convert worse.",
            suggested_action="Write unique 100–150 word descriptions covering fabric, fit, occasion, sizing and care, and include the main search term naturally.",
            value_usd=None,
            entities=[{"product_id": r.product_id, "name": r.product_name, "impressions": r.impressions_28d,
                       "position": r.avg_position} for r in md.itertuples()],
        ))

    # 5) Slow pages ---------------------------------------------------------------------
    sl = d[d["slow"]]
    if not sl.empty:
        corr = summary["totals"]["load_vs_position_corr"]
        corr_txt = (
            f" Across all pages, slower load time goes with a worse ranking position (correlation {corr:.2f}); this is a correlation, not proof of cause."
            if corr is not None and corr >= 0.3 else ""
        )
        names = ", ".join(f"{r.product_name} ({r.page_load_ms:.0f})" for r in sl.itertuples())
        findings.append(Finding(
            area="SEO", kind="issue", severity="medium" if (corr or 0) >= 0.5 else "low",
            title=f"{len(sl)} page(s) load noticeably slower than the rest",
            detail=f"Load time is more than {th.slow_page_ratio:.2f}× the median ({d['page_load_ms'].median():.0f}) for: {names}.{corr_txt}",
            suggested_action="Compress and lazy-load product images, serve them in WebP/AVIF, remove unused scripts and re-test with PageSpeed Insights.",
            value_usd=None,
            entities=[{"product_id": r.product_id, "name": r.product_name, "page_load_ms": r.page_load_ms,
                       "position": r.avg_position} for r in sl.itertuples()],
        ))

    # 6) Possible keyword cannibalisation -----------------------------------------------
    tok = {r.product_id: _tokens(pd.Series({"product_name": r.product_name, "description": r.description})) for r in d.itertuples()}
    for cat, g in d.groupby("category"):
        for a, b in itertools.combinations(g.itertuples(), 2):
            ta, tb = tok[a.product_id], tok[b.product_id]
            sim = safe_div(len(ta & tb), len(ta | tb), 0.0)
            if sim >= th.cannibalization_similarity:
                both_p2 = bool(a.page2 and b.page2)
                shared = ", ".join(sorted(ta & tb))
                pos_a = "n/a" if pd.isna(a.avg_position) else f"{a.avg_position:.1f}"
                pos_b = "n/a" if pd.isna(b.avg_position) else f"{b.avg_position:.1f}"
                findings.append(Finding(
                    area="SEO", kind="opportunity", severity="medium" if both_p2 else "low",
                    title=f"Possible keyword overlap: {a.product_name} vs {b.product_name}",
                    detail=(
                        f"Both pages are in '{cat}' and share much of the same wording ({shared}), so they may compete for the same searches. "
                        f"Positions are {pos_a} and {pos_b} with {a.impressions_28d + b.impressions_28d:,.0f} combined impressions. "
                        "This is a hypothesis based on text similarity only."
                    ),
                    suggested_action="Give each page a distinct primary keyword and angle (e.g. silhouette, neckline or occasion), cross-link them, or consolidate them if they are near-duplicates.",
                    value_usd=None,
                    entities=[{"product_id": x.product_id, "name": x.product_name, "impressions": x.impressions_28d,
                               "position": x.avg_position} for x in (a, b)],
                ))
    return findings
