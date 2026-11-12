"""Run-to-run history: the basis for trend detection.

The supplied CSVs are a single snapshot (no dates), so a "declining clicks"
check is only possible by comparing this run with the previous one. Each time
the data changes a small snapshot is saved; the next run compares against it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Thresholds
from .findings import Finding
from .utils import clean, safe_div


def build_snapshot(data_hash: str, seo_d: pd.DataFrame, seo_summary: dict, ads_summary: dict) -> dict[str, Any]:
    return clean({
        "data_hash": data_hash,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seo": {
            "totals": {k: seo_summary["totals"][k] for k in ("impressions", "clicks", "ctr")},
            "pages": {r.product_id: {"name": r.product_name, "clicks": r.clicks_28d, "impressions": r.impressions_28d}
                      for r in seo_d.itertuples()},
        },
        "ads": {
            "totals": {k: ads_summary["totals"][k] for k in ("spend", "revenue", "purchases", "roas")},
            "campaigns": {c["campaign"]: {"spend": c["spend"], "revenue": c["revenue"],
                                          "purchases": c["purchases"], "roas": c["roas"]}
                          for c in ads_summary["campaigns"]},
        },
    })


def load_history(history_dir: Path) -> list[dict]:
    snaps = []
    for p in sorted(Path(history_dir).glob("*.json")):
        try:
            snaps.append(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue  # ignore a corrupted snapshot rather than failing the run
    snaps.sort(key=lambda s: s.get("generated_at", ""))
    return snaps


def pick_previous(history: list[dict], current_hash: str) -> dict | None:
    """Most recent snapshot whose data differs from the current data."""
    for snap in reversed(history):
        if snap.get("data_hash") != current_hash:
            return snap
    return None


def save_snapshot(history_dir: Path, snapshot: dict, history: list[dict]) -> bool:
    """Save only when the data is new (re-running on identical data adds nothing)."""
    if history and history[-1].get("data_hash") == snapshot["data_hash"]:
        return False
    history_dir = Path(history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (history_dir / f"{stamp}_{snapshot['data_hash'][:8]}.json").write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8")
    return True


def _change(label: str, prev, cur, good_when: str) -> dict:
    pct = None if (prev is None or cur is None or prev == 0) else (cur - prev) / abs(prev) * 100
    direction = None
    if pct is not None and abs(pct) >= 0.05 and good_when != "neutral":
        direction = "good" if (pct > 0) == (good_when == "up") else "bad"
    return {"metric": label, "previous": prev, "current": cur, "change_pct": pct, "direction": direction}


def compare(prev: dict, cur: dict, th: Thresholds, value_per_click: float) -> tuple[dict, list[Finding]]:
    ps, cs = prev["seo"]["totals"], cur["seo"]["totals"]
    pa, ca = prev["ads"]["totals"], cur["ads"]["totals"]
    rows = [
        _change("Organic impressions", ps["impressions"], cs["impressions"], "up"),
        _change("Organic clicks", ps["clicks"], cs["clicks"], "up"),
        _change("Organic CTR", ps["ctr"], cs["ctr"], "up"),
        _change("Ad spend", pa["spend"], ca["spend"], "neutral"),
        _change("Ad revenue", pa["revenue"], ca["revenue"], "up"),
        _change("Ad purchases", pa["purchases"], ca["purchases"], "up"),
        _change("Blended ROAS", pa["roas"], ca["roas"], "up"),
    ]
    changes = {"previous_generated_at": prev.get("generated_at"), "rows": rows}
    by = {r["metric"]: r for r in rows}
    t = th.trend_change_pct
    findings: list[Finding] = []

    # --- SEO decline -------------------------------------------------------------------
    clicks, impr = by["Organic clicks"], by["Organic impressions"]
    seo_drops = [(r["metric"], r["change_pct"]) for r in (clicks, impr) if r["change_pct"] is not None and r["change_pct"] <= -t]
    page_drops = []
    for pid, now in cur["seo"]["pages"].items():
        before = prev["seo"]["pages"].get(pid)
        if before and before["clicks"] >= th.trend_min_clicks and now["clicks"] is not None:
            pct = (now["clicks"] - before["clicks"]) / before["clicks"] * 100
            if pct <= -t:
                page_drops.append({"product_id": pid, "name": now["name"], "clicks_before": before["clicks"],
                                   "clicks_now": now["clicks"], "change_pct": pct})
    if seo_drops or page_drops:
        lost = max(0.0, ps["clicks"] - cs["clicks"])
        worst = min([p for _, p in seo_drops], default=0)
        parts = [f"{m.lower()} {p:+.1f}%" for m, p in seo_drops]
        if page_drops:
            parts.append(f"{len(page_drops)} page(s) lost {t:.0f}%+ of their clicks")
        findings.append(Finding(
            area="Trend", kind="issue", severity="high" if worst <= -2 * t else "medium",
            title="Organic performance declined since the previous data",
            detail="Compared with the previous run: " + "; ".join(parts) + f". Organic clicks went from {ps['clicks']:,.0f} to {cs['clicks']:,.0f}.",
            suggested_action="Check Search Console for ranking drops, indexing errors or recent site changes on the affected pages, and review seasonality before reacting.",
            value_usd=lost * value_per_click,
            entities=sorted(page_drops, key=lambda x: x["change_pct"]),
        ))

    # --- Ads decline ---------------------------------------------------------------------
    roas, spend, purch = by["Blended ROAS"], by["Ad spend"], by["Ad purchases"]
    reasons = []
    if roas["change_pct"] is not None and roas["change_pct"] <= -t:
        prev_roas = "n/a" if pa["roas"] is None else f"{pa['roas']:.2f}×"
        cur_roas = "n/a" if ca["roas"] is None else f"{ca['roas']:.2f}×"
        reasons.append(f"blended ROAS fell {abs(roas['change_pct']):.1f}% ({prev_roas} → {cur_roas})")
    if (spend["change_pct"] or 0) >= 5 and (purch["change_pct"] or 0) <= -5:
        reasons.append(f"spend rose {spend['change_pct']:.1f}% while purchases fell {abs(purch['change_pct']):.1f}%")
    camp_drops = []
    for name, now in cur["ads"]["campaigns"].items():
        before = prev["ads"]["campaigns"].get(name)
        if before and before["roas"] and now["roas"] is not None:
            pct = (now["roas"] - before["roas"]) / before["roas"] * 100
            if pct <= -t:
                camp_drops.append({"campaign": name, "roas_before": before["roas"], "roas_now": now["roas"],
                                   "spend_before": before["spend"], "spend_now": now["spend"],
                                   "purchases_before": before["purchases"], "purchases_now": now["purchases"],
                                   "change_pct": pct})
    if camp_drops:
        reasons.append(f"{len(camp_drops)} campaign(s) lost {t:.0f}%+ of their ROAS")
    if reasons:
        lost_rev = max(0.0, pa["revenue"] - ca["revenue"])
        high = (spend["change_pct"] or 0) >= 5 and (purch["change_pct"] or 0) <= -5 or (roas["change_pct"] or 0) <= -2 * t
        findings.append(Finding(
            area="Trend", kind="issue", severity="high" if high else "medium",
            title="Advertising efficiency declined since the previous data",
            detail="Compared with the previous run: " + "; ".join(reasons) + f". Ad revenue went from ${pa['revenue']:,.0f} to ${ca['revenue']:,.0f} on spend of ${pa['spend']:,.0f} → ${ca['spend']:,.0f}.",
            suggested_action="Identify the campaigns driving the drop (listed below), pause or cap the worst ones, and check for tracking problems, creative fatigue or landing-page changes.",
            value_usd=lost_rev,
            entities=sorted(camp_drops, key=lambda x: x["change_pct"]),
        ))
    return changes, findings
