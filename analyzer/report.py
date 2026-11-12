"""Assemble the report dictionary and write it as JSON and HTML."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import __version__
from .config import CTR_CURVE, Thresholds
from .findings import Finding
from .utils import clean

TEMPLATE_DIR = Path(__file__).parent / "templates"


_ENTITY_FMT = [
    ("role", lambda v: f"[{v}]"),
    ("impressions", lambda v: f"{v:,.0f} impressions"),
    ("clicks", lambda v: f"{v:,.0f} clicks"),
    ("position", lambda v: f"position {v:.1f}" if v is not None else "position n/a"),
    ("ctr", lambda v: f"CTR {v * 100:.2f}%"),
    ("expected_ctr", lambda v: f"benchmark {v * 100:.1f}%"),
    ("est_extra_clicks", lambda v: f"≈ +{v:,.0f} clicks"),
    ("page_load_ms", lambda v: f"load {v:.0f}"),
    ("spend", lambda v: f"${v:,.0f} spend"),
    ("revenue", lambda v: f"${v:,.0f} revenue"),
    ("roas", lambda v: f"ROAS {v:.2f}×" if v is not None else "ROAS n/a"),
    ("cvr", lambda v: f"conv. rate {v * 100:.2f}%" if v is not None else "conv. rate n/a"),
    ("cpa", lambda v: f"CPA ${v:,.0f}" if v is not None else "CPA n/a"),
    ("purchases", lambda v: f"{v:,.0f} purchases"),
    ("clicks_before", lambda v: f"clicks {v:,.0f}"),
    ("clicks_now", lambda v: f"→ {v:,.0f}"),
    ("roas_before", lambda v: f"ROAS {v:.2f}×"),
    ("roas_now", lambda v: f"→ {v:.2f}×"),
    ("change_pct", lambda v: f"({v:+.1f}%)"),
]


def _entity_text(e: dict) -> str:
    label = e.get("name") or e.get("campaign") or ""
    if e.get("product_id"):
        label = f"{label} ({e['product_id']})"
    parts = []
    for key, fmt in _ENTITY_FMT:
        if key in e and (e[key] is not None or key in ("position", "roas", "cvr", "cpa")):
            try:
                parts.append(fmt(e[key]))
            except (TypeError, ValueError):
                pass
    if e.get("not_indexed"):
        parts.append("not indexed")
    return f"{label}: " + " · ".join(parts) if parts else label


def build_report(*, data_hash: str, cfg_paths: dict, seo_summary: dict, ads_summary: dict,
                 findings: list[Finding], ai: dict, changes: dict | None,
                 quality: dict[str, list[str]], th: Thresholds, value_per_click: float) -> dict:
    fdicts = [f.to_dict() for f in findings]
    for fd in fdicts:
        fd["entities_text"] = [_entity_text(e) for e in fd["entities"]]
    return clean({
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_version": data_hash[:12],
            "files": cfg_paths,
            "tool_version": __version__,
        },
        "executive_summary": ai.get("executive_summary") or "",
        "ai": {k: ai.get(k) for k in ("source", "provider", "model", "error")},
        "seo_summary": seo_summary,
        "ads_summary": ads_summary,
        "issues": [f for f in fdicts if f["kind"] == "issue"],
        "opportunities": [f for f in fdicts if f["kind"] == "opportunity"],
        "recommendations": ai["recommendations"],
        "changes_since_previous": changes,
        "data_quality": quality,
        "assumptions": {
            "thresholds": asdict(th),
            "benchmark_ctr_by_position": [{"position": p, "ctr": c} for p, c in CTR_CURVE],
            "organic_click_value_usd": value_per_click,
            "note": "Revenue figures for findings are modelled estimates, overlap between findings is possible, and they should not be added together.",
        },
    })


def write_json(report: dict, path: Path) -> None:
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def _usd(v):
    return "—" if v is None else f"${v:,.0f}"


def _num(v, digits=0):
    return "—" if v is None else f"{v:,.{digits}f}"


def _pct(v, digits=1):
    return "—" if v is None else f"{v * 100:.{digits}f}%"


def _x(v):
    return "—" if v is None else f"{v:.2f}×"


def _signed(v):
    return "—" if v is None else f"{v:+.1f}%"


def render_html(report: dict) -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=select_autoescape(["html", "j2"]))
    env.filters.update(usd=_usd, num=_num, pct=_pct, x=_x, signed=_signed)
    return env.get_template("report.html.j2").render(r=report)


def write_html(report: dict, path: Path) -> None:
    path.write_text(render_html(report), encoding="utf-8")
