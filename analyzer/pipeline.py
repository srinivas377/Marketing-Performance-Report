"""The end-to-end run: CSV -> analysis -> findings -> LLM -> report files."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from . import ads, cross, history, llm, report, seo
from .config import AppConfig
from .findings import rank_and_number
from .loader import load_ads, load_seo
from .utils import DataError
from .watcher import content_hash


def run_once(cfg: AppConfig, llm_caller: Callable | None = None, log: Callable[[str], None] = print) -> dict:
    th = cfg.thresholds

    # 1) Read + clean ---------------------------------------------------------------
    data_hash = content_hash([cfg.seo_path, cfg.ads_path])
    if data_hash is None:
        raise DataError("Could not read the CSV files (missing or locked).")
    seo_res, ads_res = load_seo(cfg.seo_path), load_ads(cfg.ads_path)
    log(f"Loaded {seo_res.rows_used}/{seo_res.rows_total} SEO rows and {ads_res.rows_used}/{ads_res.rows_total} ad campaigns.")

    # 2) Metrics + summaries ---------------------------------------------------------
    ads_summary = ads.summarize_ads(ads_res.df, th)
    value_per_click = ads_summary["totals"]["revenue_per_click"] * th.organic_value_factor
    seo_d, seo_summary = seo.summarize_seo(seo_res.df, th)

    # 3) Issues + opportunities ------------------------------------------------------
    findings = (
        seo.detect_seo(seo_d, seo_summary, th, value_per_click)
        + ads.detect_ads(ads_summary, th)
        + cross.detect_cross_channel(seo_d, ads_summary)
    )

    # 4) Trends against the previous (different) data version -------------------------
    hist = history.load_history(cfg.history_dir)
    snapshot = history.build_snapshot(data_hash, seo_d, seo_summary, ads_summary)
    previous = history.pick_previous(hist, data_hash)
    changes = None
    if previous:
        changes, trend_findings = history.compare(previous, snapshot, th, value_per_click)
        findings += trend_findings
    findings = rank_and_number(findings)
    log(f"Found {sum(f.kind == 'issue' for f in findings)} issue(s) and {sum(f.kind == 'opportunity' for f in findings)} opportunity(ies).")

    quality = {"seo": seo_res.notes, "ads": ads_res.notes}

    # 5) LLM recommendations (falls back, visibly, if unavailable) ---------------------
    ai = None
    if cfg.use_llm:
        try:
            llm_cfg = llm.load_llm_config(cfg.provider, cfg.model)
            log(f"Calling {llm_cfg.provider} ({llm_cfg.model}) for recommendations ...")
            payload = llm.build_payload(seo_summary, ads_summary, findings, changes, quality, th)
            ai = llm.generate_ai_recommendations(payload, findings, llm_cfg, **({"caller": llm_caller} if llm_caller else {}))
        except llm.LLMError as exc:
            log(f"LLM unavailable: {exc}")
            ai = llm.rule_based_recommendations(findings, seo_summary, ads_summary, str(exc))
    else:
        ai = llm.rule_based_recommendations(findings, seo_summary, ads_summary, "AI step skipped (--no-llm).")

    if not ai.get("executive_summary"):
        ai["executive_summary"] = llm.rule_based_recommendations(findings, seo_summary, ads_summary, None)["executive_summary"]

    # 6) Report files -----------------------------------------------------------------
    rep = report.build_report(
        data_hash=data_hash,
        cfg_paths={"seo": Path(cfg.seo_path).name, "ads": Path(cfg.ads_path).name},
        seo_summary=seo_summary, ads_summary=ads_summary, findings=findings, ai=ai,
        changes=changes, quality=quality, th=th, value_per_click=value_per_click,
    )
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report.write_json(rep, out / "report.json")
    report.write_html(rep, out / "report.html")
    history.save_snapshot(cfg.history_dir, snapshot, hist)
    at, st = ads_summary["totals"], seo_summary["totals"]
    roas_s = f"{at['roas']:.2f}x" if at["roas"] is not None else "n/a"
    trend_s = " | Trend comparison included" if changes else ""
    log(f"Summary: Ads ${at['spend']:,.0f} spend -> ${at['revenue']:,.0f} rev ({roas_s} ROAS) | SEO {st['clicks']:,.0f} clicks ({st['ctr']*100:.2f}% CTR){trend_s}")
    log(f"Report written: {out / 'report.html'} and {out / 'report.json'}")
    return rep
