"""Advertising analysis: metrics, summary and issue / opportunity detection."""
from __future__ import annotations

import pandas as pd

from .config import Thresholds
from .findings import Finding
from .utils import safe_div

_ACTION_BY_TYPE = {
    "prospecting": "Refresh the creative and narrow or rebuild audiences; pause the weakest ad sets and test one new angle at a time before spending more.",
    "shopping": "Audit the product feed (titles, prices, availability, images), exclude products that get clicks but never sell, and split high/low performers into separate bids.",
    "performance max": "Review asset groups and listing groups, add brand/negative exclusions, and feed in stronger audience signals; cap budget until ROAS recovers.",
    "search": "Review the search-terms report, add negative keywords, tighten match types and check ad-to-landing-page relevance.",
    "retargeting": "Check audience size and frequency caps and exclude recent purchasers.",
}
_DEFAULT_ACTION = "Review targeting, creative and landing page, and reduce budget until efficiency improves."


def _is_brand(name: str, th: Thresholds) -> bool:
    n = str(name).lower()
    return any(k in n for k in th.brand_keywords)


def _metrics(g: dict) -> dict:
    return {
        "roas": safe_div(g["revenue"], g["spend"]),
        "cpa": safe_div(g["spend"], g["purchases"]),
        "cvr": safe_div(g["purchases"], g["clicks"]),
        "ctr": safe_div(g["clicks"], g["impressions"]),
        "cpc": safe_div(g["spend"], g["clicks"]),
        "aov": safe_div(g["revenue"], g["purchases"]),
    }


def _group(df: pd.DataFrame, key: str) -> list[dict]:
    out = []
    for k, g in df.groupby(key, sort=False):
        row = {
            key: k, "campaigns": int(len(g)),
            "spend": float(g["spend_usd"].sum()), "revenue": float(g["revenue_usd"].sum()),
            "purchases": float(g["purchases"].sum()), "clicks": float(g["clicks"].sum()),
            "impressions": float(g["impressions"].sum()),
        }
        row.update(_metrics(row))
        out.append(row)
    out.sort(key=lambda r: -r["spend"])
    return out


def summarize_ads(df: pd.DataFrame, th: Thresholds) -> dict:
    total = {
        "spend": float(df["spend_usd"].sum()), "revenue": float(df["revenue_usd"].sum()),
        "purchases": float(df["purchases"].sum()), "clicks": float(df["clicks"].sum()),
        "impressions": float(df["impressions"].sum()),
    }
    total.update(_metrics(total))
    total["campaigns"] = int(len(df))
    total["revenue_per_click"] = safe_div(total["revenue"], total["clicks"], 0.0)

    brand_mask = df["campaign"].map(lambda n: _is_brand(n, th))
    nb = df[~brand_mask]
    if nb.empty or nb["spend_usd"].sum() <= 0:
        nb = df
    nb_tot = {"spend": float(nb["spend_usd"].sum()), "revenue": float(nb["revenue_usd"].sum()),
              "purchases": float(nb["purchases"].sum()), "clicks": float(nb["clicks"].sum()),
              "impressions": float(nb["impressions"].sum())}
    nb_m = _metrics(nb_tot)
    par_roas, par_cvr = nb_m["roas"], nb_m["cvr"]

    campaigns = []
    for _, r in df.iterrows():
        row = {
            "campaign": r["campaign"], "platform": r["platform"], "campaign_type": r["campaign_type"],
            "spend": float(r["spend_usd"]), "revenue": float(r["revenue_usd"]),
            "purchases": float(r["purchases"]), "clicks": float(r["clicks"]),
            "impressions": float(r["impressions"]),
            "reach": None if pd.isna(r["reach"]) else float(r["reach"]),
            "is_brand": _is_brand(r["campaign"], th),
        }
        row.update(_metrics(row))
        row["frequency"] = safe_div(row["impressions"], row["reach"])
        row["spend_share"] = safe_div(row["spend"], total["spend"], 0.0)
        row["flags"] = _campaign_flags(row, par_roas, par_cvr, th)
        campaigns.append(row)
    campaigns.sort(key=lambda r: -r["spend"])

    return {
        "totals": total,
        "benchmarks": {
            "target_roas": th.target_roas, "non_brand_roas": par_roas, "non_brand_cvr": par_cvr,
        },
        "by_platform": _group(df, "platform"),
        "by_campaign_type": _group(df, "campaign_type"),
        "campaigns": campaigns,
    }


def _campaign_flags(c: dict, par_roas, par_cvr, th: Thresholds) -> list[str]:
    flags = []
    if c["roas"] is not None and c["roas"] < th.target_roas:
        flags.append("Below target ROAS")
    if not c["is_brand"] and par_roas and c["roas"] is not None and c["roas"] < th.par_gap_ratio * par_roas:
        flags.append("Well below average ROAS")
    if (not c["is_brand"] and par_cvr and c["cvr"] is not None and c["spend_share"] >= th.high_spend_share
            and c["cvr"] < th.low_cvr_ratio * par_cvr):
        flags.append("High spend, weak conversion")
    if (not c["is_brand"] and par_roas and c["roas"] is not None and c["roas"] >= th.scale_roas_ratio * par_roas
            and c["purchases"] >= th.scale_min_purchases):
        flags.append("Scale candidate")
    if c["is_brand"]:
        flags.append("Branded (defends existing demand)")
    return flags


def detect_ads(summary: dict, th: Thresholds) -> list[Finding]:
    findings: list[Finding] = []
    tot = summary["totals"]
    par_roas = summary["benchmarks"]["non_brand_roas"]
    par_cvr = summary["benchmarks"]["non_brand_cvr"]
    campaigns = summary["campaigns"]

    # 1) Inefficient campaigns (one finding per campaign) -------------------------------
    donors: list[dict] = []
    for c in campaigns:
        flags = set(c["flags"])
        below_target = "Below target ROAS" in flags
        below_par = "Well below average ROAS" in flags
        weak_conv = "High spend, weak conversion" in flags
        if c["is_brand"] or not (below_target or weak_conv or (below_par and c["spend_share"] >= th.high_spend_share)):
            continue
        donors.append(c)
        reasons = []
        if below_target:
            reasons.append(f"ROAS is {c['roas']:.2f}× against a target of {th.target_roas:.1f}×")
        elif par_roas:
            reasons.append(f"ROAS is {c['roas']:.2f}× versus a non-brand average of {par_roas:.2f}×")
        if weak_conv:
            reasons.append(f"only {c['cvr'] * 100:.2f}% of clicks convert versus {par_cvr * 100:.2f}% on average")
        gap = max(0.0, (c["spend"] * par_roas - c["revenue"])) if par_roas else 0.0
        severity = "high" if (below_target or (below_par and c["spend_share"] >= th.high_spend_share)) else "medium"
        cpa_txt = f" Cost per purchase is ${c['cpa']:.0f}." if c["cpa"] else ""
        findings.append(Finding(
            area="Ads", kind="issue", severity=severity,
            title=f"{c['campaign']}: {c['spend_share'] * 100:.0f}% of budget with weak returns",
            detail=(
                f"It spent ${c['spend']:,.0f} and earned ${c['revenue']:,.0f} from {c['purchases']:.0f} purchases. "
                + (lambda s: s[0].upper() + s[1:])("; ".join(reasons)) + f".{cpa_txt} If it matched the non-brand average ROAS it would have earned about ${gap:,.0f} more."
            ),
            suggested_action=_ACTION_BY_TYPE.get(str(c["campaign_type"]).lower(), _DEFAULT_ACTION),
            value_usd=gap,
            entities=[{"campaign": c["campaign"], "spend": c["spend"], "revenue": c["revenue"], "roas": c["roas"],
                       "cvr": c["cvr"], "cpa": c["cpa"], "purchases": c["purchases"]}],
        ))

    # 2) Budget reallocation opportunity -------------------------------------------------
    scalers = [c for c in campaigns if "Scale candidate" in c["flags"]]
    if donors and scalers:
        moved = th.reallocation_share * sum(c["spend"] for c in donors)
        d_roas = safe_div(sum(c["revenue"] for c in donors), sum(c["spend"] for c in donors), 0.0)
        s_roas = safe_div(sum(c["revenue"] for c in scalers), sum(c["spend"] for c in scalers), 0.0)
        net = max(0.0, moved * (s_roas * th.scale_marginal_efficiency - d_roas))
        s_names = ", ".join(f"{c['campaign']} ({c['roas']:.1f}×)" for c in sorted(scalers, key=lambda x: -x["roas"]))
        d_names = ", ".join(c["campaign"] for c in donors)
        findings.append(Finding(
            area="Ads", kind="opportunity", severity="medium",
            title=f"Shift budget from {len(donors)} weak campaign(s) to proven performers",
            detail=(
                f"Campaigns with the best returns: {s_names}. Weak campaigns: {d_names} (combined ROAS {d_roas:.2f}×). "
                f"Moving {th.reallocation_share * 100:.0f}% of the weak campaigns' budget (about ${moved:,.0f}) to the winners, assuming extra spend earns only "
                f"{th.scale_marginal_efficiency * 100:.0f}% of their average ROAS, would add roughly ${net:,.0f} of revenue for the same total spend. "
                "Branded search is excluded because it mostly captures demand that already exists; retargeting can only scale as far as its audience allows."
            ),
            suggested_action="Move budget in steps of about 10–20% every week, watch ROAS and CPA after each step, and stop increasing a campaign when its ROAS falls toward the target.",
            value_usd=net,
            entities=[{"campaign": c["campaign"], "role": "scale", "roas": c["roas"], "spend": c["spend"]} for c in scalers]
                     + [{"campaign": c["campaign"], "role": "reduce", "roas": c["roas"], "spend": c["spend"]} for c in donors],
        ))

    # 3) Large ROAS spread inside the same campaign type -----------------------------------
    by_type: dict[str, list[dict]] = {}
    for c in campaigns:
        if not c["is_brand"] and c["roas"]:
            by_type.setdefault(str(c["campaign_type"]), []).append(c)
    for ctype, cs in by_type.items():
        if len(cs) < 2:
            continue
        best, worst = max(cs, key=lambda x: x["roas"]), min(cs, key=lambda x: x["roas"])
        if worst["roas"] > 0 and best["roas"] / worst["roas"] >= th.spread_ratio:
            findings.append(Finding(
                area="Ads", kind="opportunity", severity="low",
                title=f"{ctype} campaigns vary widely: {worst['roas']:.1f}× to {best['roas']:.1f}× ROAS",
                detail=(
                    f"Within {ctype}, {best['campaign']} returns {best['roas']:.2f}× while {worst['campaign']} returns {worst['roas']:.2f}× "
                    f"on ${worst['spend']:,.0f} of spend. The difference is probably creative, offer or audience rather than the channel itself."
                ),
                suggested_action=f"Identify what {best['campaign']} does differently (offer, creative, audience) and test the same approach in {worst['campaign']}.",
                value_usd=None,
                entities=[{"campaign": x["campaign"], "roas": x["roas"], "spend": x["spend"]} for x in sorted(cs, key=lambda y: -y["roas"])],
            ))

    # 4) Portfolio-level check: is overall ROAS below target? -------------------------------
    if tot["roas"] is not None and tot["roas"] < th.target_roas:
        findings.append(Finding(
            area="Ads", kind="issue", severity="high",
            title=f"Overall ad ROAS ({tot['roas']:.2f}×) is below the {th.target_roas:.1f}× target",
            detail=f"${tot['spend']:,.0f} of spend produced ${tot['revenue']:,.0f} of revenue across {tot['campaigns']} campaigns.",
            suggested_action="Pause or cut the lowest-ROAS campaigns first and re-test creative and landing pages before increasing budget.",
            value_usd=max(0.0, tot["spend"] * th.target_roas - tot["revenue"]),
        ))
    return findings
