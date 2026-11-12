"""Cross-channel insight: paid search that overlaps with organic ranking gaps."""
from __future__ import annotations

import re

import pandas as pd

from .findings import Finding
from .utils import safe_div


def detect_cross_channel(seo_d: pd.DataFrame, ads_summary: dict) -> list[Finding]:
    """Match Google Search campaigns to SEO categories by keyword in the campaign name.

    Example: "Google Search - Prom" matches the category "Prom Dresses" because
    the first word of the category ("prom") appears in the campaign name.
    """
    findings: list[Finding] = []
    keywords = {
        str(cat).split()[0].lower(): cat
        for cat in seo_d["category"].dropna().unique()
        if str(cat).split()
    }
    for c in ads_summary["campaigns"]:
        if str(c["campaign_type"]).lower() != "search" or c["is_brand"]:
            continue
        words = set(re.findall(r"[a-z0-9]+", str(c["campaign"]).lower()))
        for kw, cat in keywords.items():
            if kw not in words:
                continue
            pages = seo_d[seo_d["category"].str.lower().str.startswith(kw)]
            gaps = pages[pages["page2"] | pages["not_indexed"]]
            if gaps.empty:
                continue
            impr = float(pages["impressions_28d"].sum())
            gap_impr = float(gaps["impressions_28d"].sum())
            names = ", ".join(
                f"{r.product_name} ({'not indexed' if r.not_indexed else f'pos {r.avg_position:.1f}'})" for r in gaps.itertuples()
            )
            roas_txt = f"{c['roas']:.2f}×" if c["roas"] is not None else "n/a"
            findings.append(Finding(
                area="Cross-channel", kind="opportunity",
                severity="medium" if safe_div(gap_impr, impr, 0) >= 0.5 else "low",
                title=f"Paid search '{c['campaign']}' overlaps with organic gaps in {cat}",
                detail=(
                    f"You spend ${c['spend']:,.0f} (ROAS {roas_txt}) on paid search for this theme while "
                    f"{len(gaps)} of {len(pages)} organic pages in '{cat}' are on page 2 or not indexed: {names}. "
                    f"These pages hold {safe_div(gap_impr, impr, 0) * 100:.0f}% of the category's {impr:,.0f} organic impressions."
                ),
                suggested_action="Keep the paid campaign running while it is profitable, but treat these organic pages as a priority so free traffic gradually replaces paid clicks.",
                value_usd=None,
                entities=[{"product_id": r.product_id, "name": r.product_name, "position": r.avg_position,
                           "impressions": r.impressions_28d, "not_indexed": bool(r.not_indexed)} for r in gaps.itertuples()],
            ))
    return findings
