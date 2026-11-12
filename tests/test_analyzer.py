"""Run with:  pytest -q"""
from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

from analyzer import ads, history, llm, seo
from analyzer.config import AppConfig, Thresholds
from analyzer.loader import load_ads, load_seo
from analyzer.pipeline import run_once
from analyzer.utils import DataError
from analyzer.watcher import watch

DATA = Path(__file__).resolve().parent.parent / "data"
TH = Thresholds()


@pytest.fixture
def workdir(tmp_path):
    shutil.copy(DATA / "seo_data.csv", tmp_path / "seo_data.csv")
    shutil.copy(DATA / "ads_data.csv", tmp_path / "ads_data.csv")
    return tmp_path


def cfg_for(w: Path, **kw) -> AppConfig:
    return AppConfig(seo_path=w / "seo_data.csv", ads_path=w / "ads_data.csv", output_dir=w / "out", **kw)


# ----------------------------- loading / cleaning -----------------------------
def test_loads_supplied_data_cleanly():
    s, a = load_seo(DATA / "seo_data.csv"), load_ads(DATA / "ads_data.csv")
    assert (s.rows_used, a.rows_used) == (12, 10)


def test_bad_rows_are_excluded_and_reported(tmp_path):
    (tmp_path / "ads.csv").write_text(
        "campaign,platform,campaign_type,spend_usd,clicks,purchases,revenue_usd,impressions,reach\n"
        "A,Google,Search,100,50,5,400,1000,100\n"
        "B,Meta,Prospecting,abc,50,5,400,1000,100\n"          # non-numeric spend
        "C,Meta,Prospecting,100,5000,5,400,1000,100\n"        # clicks > impressions
        "D,Meta,Prospecting,-5,50,5,400,1000,100\n"           # negative spend
        "A,Google,Search,100,50,5,400,1000,100\n")            # duplicate
    r = load_ads(tmp_path / "ads.csv")
    assert list(r.df["campaign"]) == ["A"]
    assert len(r.notes) >= 4


def test_missing_column_and_missing_file_raise_clear_errors(tmp_path):
    (tmp_path / "bad.csv").write_text("campaign,spend_usd\nA,1\n")
    with pytest.raises(DataError, match="missing required column"):
        load_ads(tmp_path / "bad.csv")
    with pytest.raises(DataError, match="not found"):
        load_seo(tmp_path / "nope.csv")


def test_zero_spend_does_not_crash(tmp_path):
    (tmp_path / "ads.csv").write_text(
        "campaign,platform,campaign_type,spend_usd,clicks,purchases,revenue_usd,impressions,reach\n"
        "Z,Google,Search,0,0,0,0,0,0\nA,Meta,Prospecting,100,10,1,50,500,50\n")
    s = ads.summarize_ads(load_ads(tmp_path / "ads.csv").df, TH)
    z = next(c for c in s["campaigns"] if c["campaign"] == "Z")
    assert z["roas"] is None and z["cpa"] is None


# ----------------------------- detection -----------------------------
@pytest.fixture
def analysed():
    sdf, adf = load_seo(DATA / "seo_data.csv").df, load_ads(DATA / "ads_data.csv").df
    a_sum = ads.summarize_ads(adf, TH)
    d, s_sum = seo.summarize_seo(sdf, TH)
    vpc = a_sum["totals"]["revenue_per_click"] * TH.organic_value_factor
    return d, s_sum, a_sum, seo.detect_seo(d, s_sum, TH, vpc), ads.detect_ads(a_sum, TH)


def test_ads_totals_match_manual_calculation(analysed):
    _, _, a_sum, _, _ = analysed
    assert a_sum["totals"]["spend"] == 6680 and a_sum["totals"]["revenue"] == 32230
    assert a_sum["totals"]["roas"] == pytest.approx(32230 / 6680)


def test_worst_campaign_is_meta_prospecting(analysed):
    from analyzer.findings import rank_and_number
    *_, ads_f = analysed
    top = next(f for f in rank_and_number(ads_f) if f.kind == "issue")   # ranking = severity, then $ at stake
    assert "Meta - Prospecting" in top.title and top.severity == "high" and top.id == "ADS-01"


def test_branded_campaign_is_never_flagged_or_scaled(analysed):
    _, _, a_sum, _, ads_f = analysed
    brand = next(c for c in a_sum["campaigns"] if c["is_brand"])
    assert "Scale candidate" not in brand["flags"]
    assert not any("Brand" in f.title for f in ads_f if f.kind == "issue")


def test_seo_flags_unindexed_and_page2_pages(analysed):
    d, _, _, seo_f, _ = analysed
    assert set(d.loc[d["not_indexed"], "product_id"]) == {"P-104", "P-110"}
    striking = next(f for f in seo_f if "page 2" in f.title)
    assert {e["product_id"] for e in striking.entities} >= {"P-101", "P-107", "P-103"}


def test_possible_cannibalisation_detected(analysed):
    seo_f = analysed[3]
    pair = next(f for f in seo_f if "overlap" in f.title)
    assert {e["product_id"] for e in pair.entities} == {"P-101", "P-107"}


def test_expected_ctr_is_monotonic():
    assert seo.expected_ctr(1) > seo.expected_ctr(5) > seo.expected_ctr(15) > seo.expected_ctr(40)


# ----------------------------- trends / dynamic updates -----------------------------
def _edit_ads(path: Path, campaign: str, **changes):
    df = pd.read_csv(path)
    for k, v in changes.items():
        df.loc[df["campaign"] == campaign, k] = v
    df.to_csv(path, index=False)


def test_changed_data_changes_report_and_flags_decline(workdir):
    cfg = cfg_for(workdir, use_llm=False)
    first = run_once(cfg, log=lambda m: None)
    assert first["changes_since_previous"] is None

    _edit_ads(workdir / "ads_data.csv", "Meta - Retargeting", spend_usd=1500, purchases=10, revenue_usd=800)
    second = run_once(cfg, log=lambda m: None)

    assert second["ads_summary"]["totals"]["spend"] == first["ads_summary"]["totals"]["spend"] + 1090
    assert second["meta"]["data_version"] != first["meta"]["data_version"]
    rows = {r["metric"]: r for r in second["changes_since_previous"]["rows"]}
    assert rows["Ad spend"]["change_pct"] > 0 and rows["Blended ROAS"]["direction"] == "bad"
    titles = [f["title"] for f in second["issues"]]
    assert any("Advertising efficiency declined" in t for t in titles)
    assert any("Meta - Retargeting" in t for t in titles)   # now a weak campaign


def test_rerun_on_identical_data_keeps_one_snapshot_and_same_comparison(workdir):
    cfg = cfg_for(workdir, use_llm=False)
    run_once(cfg, log=lambda m: None)
    run_once(cfg, log=lambda m: None)
    assert len(list(cfg.history_dir.glob("*.json"))) == 1


def test_seo_click_drop_is_flagged(workdir):
    cfg = cfg_for(workdir, use_llm=False)
    run_once(cfg, log=lambda m: None)
    df = pd.read_csv(workdir / "seo_data.csv")
    df["clicks_28d"] = (df["clicks_28d"] * 0.5).astype(int)
    df.to_csv(workdir / "seo_data.csv", index=False)
    rep = run_once(cfg, log=lambda m: None)
    assert any("Organic performance declined" in f["title"] for f in rep["issues"])


def test_report_json_is_strict_json(workdir):
    run_once(cfg_for(workdir, use_llm=False), log=lambda m: None)
    text = (workdir / "out" / "report.json").read_text()
    json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))  # rejects NaN/Infinity
    assert (workdir / "out" / "report.html").read_text().startswith("<!doctype html>")


# ----------------------------- LLM -----------------------------
GOOD = json.dumps({
    "executive_summary": "Summary.",
    "recommendations": [
        {"priority": "high", "title": "Cut Meta - Prospecting", "action": "Reduce budget 30%.", "why": "ROAS 2.2x",
         "expected_impact": "Save ~$250", "effort": "low", "related_findings": ["ADS-01", "NOPE-99"]},
        {"priority": "P3", "title": "t", "action": "a"},
    ]})


def test_llm_path_uses_programmatic_call_and_validates(workdir, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    calls = []

    def fake(cfg, system, user):
        calls.append((cfg.provider, system, user))
        return "```json\n" + GOOD + "\n```"          # fenced reply must still parse

    rep = run_once(cfg_for(workdir, provider="anthropic"), llm_caller=fake, log=lambda m: None)
    assert len(calls) == 1 and "ADS-01" in calls[0][2]          # findings really are sent to the model
    assert rep["ai"]["source"] == "llm"
    first = rep["recommendations"][0]
    assert first["priority"] == "P1" and first["related_findings"] == ["ADS-01"]   # unknown id dropped
    assert "Summary." == rep["executive_summary"]


def test_llm_retries_once_then_falls_back_visibly(workdir, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    n = {"c": 0}

    def bad(cfg, s, u):
        n["c"] += 1
        return "sorry, no json"

    rep = run_once(cfg_for(workdir), llm_caller=bad, log=lambda m: None)
    assert n["c"] == 2
    assert rep["ai"]["source"] == "rules" and "JSON" in rep["ai"]["error"]
    assert "AI recommendations were not generated" in (workdir / "out" / "report.html").read_text()


def test_missing_api_key_falls_back(workdir, monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    rep = run_once(cfg_for(workdir), log=lambda m: None)
    assert rep["ai"]["source"] == "rules" and "ANTHROPIC_API_KEY" in rep["ai"]["error"]


def test_unknown_provider_rejected():
    with pytest.raises(llm.LLMError):
        llm.load_llm_config("bogus")


def test_llm_auto_detects_provider_when_unset_and_fast_fails_on_api_error(workdir, monkeypatch):
    for k in ("LLM_PROVIDER", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    cfg = llm.load_llm_config()
    assert cfg.provider == "gemini"

    calls = {"n": 0}

    def api_fail(c, s, u):
        calls["n"] += 1
        raise llm.LLMError("gemini API call failed (AuthenticationError): invalid key")

    rep = run_once(cfg_for(workdir), llm_caller=api_fail, log=lambda m: None)
    assert calls["n"] == 1
    assert rep["ai"]["source"] == "rules" and "AuthenticationError" in rep["ai"]["error"]


def test_missing_optional_fields_and_nan_position_handled_cleanly(tmp_path):
    (tmp_path / "seo.csv").write_text(
        "product_id,product_name,category,url_slug,description,impressions_28d,clicks_28d,avg_position,indexed,page_load_ms\n"
        "P-1,,Prom Dresses,slug-1,Black mermaid prom dress,10000,100,bad,True,-5\n"
        "P-2,Black Mermaid Prom Gown,,slug-1,Black mermaid prom dress gown,9000,90,12.0,True,70\n")
    res = load_seo(tmp_path / "seo.csv")
    assert res.df.loc[0, "product_name"] == "P-1"
    assert res.df.loc[1, "category"] == "Uncategorized"
    assert any("page_load_ms" in n for n in res.notes)
    assert any("duplicate url_slug" in n for n in res.notes)
    # Also test cannibalization with NaN position in the same category
    res.df.loc[1, "category"] = "Prom Dresses"
    d, s_sum = seo.summarize_seo(res.df, TH)
    findings = seo.detect_seo(d, s_sum, TH, 1.0)
    overlap = next(f for f in findings if "overlap" in f.title)
    assert "nan" not in overlap.detail.lower() and "n/a" in overlap.detail


def test_history_compare_handles_zero_spend_transition():
    prev = {
        "seo": {"totals": {"impressions": 1000.0, "clicks": 50.0, "ctr": 0.05}, "pages": {}},
        "ads": {"totals": {"spend": 100.0, "revenue": 400.0, "purchases": 4.0, "roas": 4.0}, "campaigns": {}},
    }
    cur = {
        "seo": {"totals": {"impressions": 1000.0, "clicks": 50.0, "ctr": 0.05}, "pages": {}},
        "ads": {"totals": {"spend": 0.0, "revenue": 0.0, "purchases": 0.0, "roas": None}, "campaigns": {}},
    }
    changes, findings = history.compare(prev, cur, TH, 1.0)
    assert len(changes["rows"]) == 7 and isinstance(findings, list)


# ----------------------------- watcher -----------------------------
def test_watcher_triggers_only_on_real_content_change(workdir):
    paths = [workdir / "seo_data.csv", workdir / "ads_data.csv"]
    hits, stop = [], {"v": False}
    t = threading.Thread(target=watch, args=(paths, lambda: hits.append(1)),
                         kwargs=dict(interval=0.1, debounce=0.1, stop=lambda: stop["v"]))
    t.start()
    time.sleep(0.4)
    paths[1].write_bytes(paths[1].read_bytes())            # touched, identical content
    time.sleep(0.6)
    assert hits == []
    _edit_ads(paths[1], "Meta - Prospecting", spend_usd=2000)   # real change
    time.sleep(0.8)
    stop["v"] = True
    t.join(timeout=3)
    assert hits == [1]
