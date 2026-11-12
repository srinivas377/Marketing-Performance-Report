"""LLM integration: turn the computed findings into prioritised recommendations.

Design:
  * The Python code finds the problems and calculates every number.
  * The LLM only receives those results (a JSON payload) and writes the
    recommendations. It is told to use nothing else and to cite finding IDs.
  * The reply is requested as JSON, validated, and unknown finding IDs are
    dropped. If the call fails, the report falls back to rule-based actions and
    says so clearly (it never pretends the AI wrote them).

Providers: Anthropic (Claude SDK) and anything that speaks the OpenAI API
(OpenAI, Gemini's OpenAI-compatible endpoint, xAI Grok).
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from .config import Thresholds
from .findings import Finding, PRIORITY_LABEL
from .utils import clean


class LLMError(Exception):
    """Any problem getting usable recommendations from the model."""


class LLMParseError(LLMError):
    """Raised when the model reply is missing, malformed JSON, or unusable."""


# provider -> (env var for the key, default model, base_url or None)
# Model names change often: override with LLM_MODEL in .env if a default is retired.
PROVIDERS: dict[str, tuple[str, str, str | None]] = {
    "anthropic": ("ANTHROPIC_API_KEY", "claude-sonnet-5-5", None),
    "openai": ("OPENAI_API_KEY", "gpt-4o-mini", None),
    "gemini": ("GEMINI_API_KEY", "gemini-flash-lite-latest", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    "grok": ("XAI_API_KEY", "grok-4", "https://api.x.ai/v1"),
}


@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: str
    base_url: str | None = None
    max_tokens: int = 3000
    timeout: float = 90.0


def load_llm_config(provider: str | None = None, model: str | None = None) -> LLMConfig:
    env_provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if provider:
        chosen = provider.strip().lower()
    elif env_provider:
        chosen = env_provider
    else:
        chosen = next((p for p, (k, _, _) in PROVIDERS.items() if (os.getenv(k) or "").strip()), "anthropic")
    if chosen not in PROVIDERS:
        raise LLMError(f"Unknown LLM provider '{chosen}'. Choose one of: {', '.join(PROVIDERS)}.")
    key_env, default_model, base_url = PROVIDERS[chosen]
    api_key = (os.getenv(key_env) or "").strip()
    if not api_key:
        other = next((p for p, (k, _, _) in PROVIDERS.items() if (os.getenv(k) or "").strip()), None)
        hint = (
            f" Found {PROVIDERS[other][0]} set — run with --provider {other} or set LLM_PROVIDER={other}."
            if other else ""
        )
        raise LLMError(f"{key_env} is not set. Copy .env.example to .env and add your key (or run with --no-llm).{hint}")
    return LLMConfig(chosen, model or os.getenv("LLM_MODEL") or default_model, api_key, base_url)


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a senior e-commerce performance-marketing analyst writing for a non-technical business owner.

You receive a JSON object produced by a Python analysis of the store's SEO and paid-advertising data. It contains summaries, a ranked list of findings (each with an id such as ADS-01 or SEO-02), and data-quality notes.

Rules:
1. Use ONLY facts and numbers present in the JSON. Never invent metrics, benchmarks, dates, competitors or tools' results. If something is not in the data, say it needs checking instead of guessing.
2. Every recommendation must be tied to one or more finding ids in "related_findings" and must quote the specific numbers (campaign / page names, spend, ROAS, CTR, position, impressions) that justify it.
3. Be concrete and actionable: say exactly what to change, on which campaign or page, and what to measure afterwards. No generic advice such as "improve SEO".
4. Priority: P1 = do this week (high severity or largest revenue at stake), P2 = do this month, P3 = nice to have. Order recommendations from P1 downwards.
5. Estimated values in the findings are modelled assumptions: describe impact as "roughly" / "up to", never as a guarantee.
6. Write in plain English. Explain any unavoidable jargon (ROAS, CTR) in a few words the first time.
7. If "changes" shows a decline versus the previous data, address it in the top recommendations.

Return ONLY valid JSON (no markdown fences, no commentary) in exactly this shape:
{
  "executive_summary": "3-4 sentences: overall health, the single biggest problem, the single biggest opportunity.",
  "recommendations": [
    {
      "priority": "P1",
      "title": "short action title",
      "action": "specific steps to take",
      "why": "evidence from the data, with numbers",
      "expected_impact": "what should improve and roughly by how much, or what to measure",
      "effort": "low | medium | high",
      "related_findings": ["ADS-01"]
    }
  ]
}
Give between 5 and 8 recommendations."""


def _pct(v):
    return None if v is None else round(v * 100, 2)


def build_payload(seo_summary: dict, ads_summary: dict, findings: list[Finding],
                  changes: dict | None, quality: dict[str, list[str]], th: Thresholds) -> dict[str, Any]:
    st, at = seo_summary["totals"], ads_summary["totals"]
    payload = {
        "context": "Online dress retailer; 28-day SEO data per product page and paid-ad data per campaign.",
        "assumptions": {
            "target_roas": th.target_roas,
            "estimated_values_are_modelled": True,
            "organic_click_value_factor": th.organic_value_factor,
        },
        "seo_summary": {
            "pages": st["pages"], "impressions": st["impressions"], "clicks": st["clicks"],
            "ctr_pct": _pct(st["ctr"]), "avg_position": st["avg_position"],
            "share_of_impressions_on_page2_pct": _pct(st["page2_impression_share"]),
            "not_indexed_pages": st["not_indexed_pages"], "missing_descriptions": st["missing_descriptions"],
            "by_category": [{**c, "ctr_pct": _pct(c["ctr"])} for c in seo_summary["by_category"]],
            "pages": [{"id": p["product_id"], "name": p["product_name"], "impressions": p["impressions"],
                       "clicks": p["clicks"], "ctr_pct": _pct(p["ctr"]), "position": p["avg_position"],
                       "indexed": p["indexed"], "flags": p["flags"]} for p in seo_summary["pages"]],
        },
        "ads_summary": {
            "spend": at["spend"], "revenue": at["revenue"], "purchases": at["purchases"], "roas": at["roas"],
            "cpa": at["cpa"], "conversion_rate_pct": _pct(at["cvr"]),
            "non_brand_roas": ads_summary["benchmarks"]["non_brand_roas"],
            "by_platform": ads_summary["by_platform"],
            "campaigns": [{"name": c["campaign"], "platform": c["platform"], "type": c["campaign_type"],
                           "spend": c["spend"], "revenue": c["revenue"], "purchases": c["purchases"],
                           "roas": c["roas"], "cpa": c["cpa"], "conversion_rate_pct": _pct(c["cvr"]),
                           "share_of_spend_pct": _pct(c["spend_share"]), "flags": c["flags"]}
                          for c in ads_summary["campaigns"]],
        },
        "findings": [{**f.to_dict(), "entities": f.entities[:8]} for f in findings],
        "changes_vs_previous_data": changes,
        "data_quality_notes": quality,
    }
    return clean(payload)


# --------------------------------------------------------------------------
# Provider calls
# --------------------------------------------------------------------------
def _is_transient(exc: Exception) -> bool:
    name = type(exc).__name__
    msg = str(exc).lower()
    if "credit_balance_exhausted" in msg or "insufficient_quota" in msg:
        return False
    return name in {"InternalServerError", "APITimeoutError", "RateLimitError"} or any(
        code in msg for code in ("503", "502", "500", "unavailable", "timed out")
    )


def call_llm(cfg: LLMConfig, system: str, user: str) -> str:
    """One programmatic API call (with 1 brief retry on transient server errors)."""
    for attempt in range(2):
        try:
            if cfg.provider == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=cfg.api_key, timeout=cfg.timeout)
                resp = client.messages.create(
                    model=cfg.model, max_tokens=cfg.max_tokens, system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            import openai
            client = openai.OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.timeout)
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            return resp.choices[0].message.content or ""
        except ImportError as exc:
            raise LLMError(f"The '{cfg.provider}' SDK is not installed: {exc}. Run: pip install -r requirements.txt") from exc
        except Exception as exc:  # network, auth, rate limit, bad model name ...
            if attempt == 0 and _is_transient(exc):
                time.sleep(2.0)
                continue
            raise LLMError(f"{cfg.provider} API call failed ({type(exc).__name__}): {exc}") from exc
    raise LLMError("Unknown LLM failure")


# --------------------------------------------------------------------------
# Parsing + validation
# --------------------------------------------------------------------------
def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise LLMParseError("The model reply did not contain a JSON object.")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise LLMParseError(f"The model reply was not valid JSON: {exc}") from exc


def _norm_priority(p: Any) -> str:
    s = str(p).strip().upper()
    if s in {"P1", "P2", "P3"}:
        return s
    return {"HIGH": "P1", "MEDIUM": "P2", "LOW": "P3", "1": "P1", "2": "P2", "3": "P3"}.get(s, "P2")


def validate(obj: dict, valid_ids: set[str]) -> dict:
    recs_in = obj.get("recommendations")
    if not isinstance(recs_in, list) or not recs_in:
        raise LLMParseError("The model reply had no 'recommendations' list.")
    recs = []
    for r in recs_in:
        if not isinstance(r, dict) or not r.get("title") or not r.get("action"):
            continue
        related = r.get("related_findings") or []
        if isinstance(related, str):
            related = [related]
        effort = str(r.get("effort", "medium")).strip().lower()
        recs.append({
            "priority": _norm_priority(r.get("priority")),
            "title": str(r["title"]).strip(),
            "action": str(r["action"]).strip(),
            "why": str(r.get("why", "")).strip(),
            "expected_impact": str(r.get("expected_impact", "")).strip(),
            "effort": effort if effort in {"low", "medium", "high"} else "medium",
            "related_findings": [i for i in (str(x).strip() for x in related) if i in valid_ids],
        })
    if not recs:
        raise LLMParseError("None of the recommendations in the model reply were usable.")
    recs.sort(key=lambda r: r["priority"])
    summary = str(obj.get("executive_summary", "")).strip()
    return {"executive_summary": summary, "recommendations": recs[:10]}


def generate_ai_recommendations(payload: dict, findings: list[Finding], cfg: LLMConfig,
                                caller: Callable[[LLMConfig, str, str], str] = call_llm) -> dict:
    """Call the model (one retry on malformed JSON/unusable output) and return validated recommendations."""
    user = ("Here is the analysis JSON. Produce the recommendations as instructed.\n\n"
            + json.dumps(payload, ensure_ascii=False))
    valid_ids = {f.id for f in findings}
    last: LLMParseError | None = None
    for attempt in range(2):
        prompt = user if attempt == 0 else user + "\n\nYour previous reply could not be parsed. Reply with ONLY the JSON object."
        try:
            result = validate(_extract_json(caller(cfg, SYSTEM_PROMPT, prompt)), valid_ids)
            result.update(source="llm", provider=cfg.provider, model=cfg.model, error=None)
            return result
        except LLMParseError as exc:
            last = exc
    raise last or LLMError("Unknown LLM failure")


# --------------------------------------------------------------------------
# Fallback (clearly labelled in the report)
# --------------------------------------------------------------------------
def rule_based_recommendations(findings: list[Finding], seo_summary: dict, ads_summary: dict, error: str | None) -> dict:
    recs = []
    for f in findings[:6]:
        recs.append({
            "priority": f.priority, "title": f.title, "action": f.suggested_action,
            "why": f.detail, "expected_impact": (f"Roughly ${f.value_usd:,.0f} of revenue at stake (modelled)." if f.value_usd else "See finding detail."),
            "effort": "medium", "related_findings": [f.id],
        })
    high = sum(1 for f in findings if f.severity == "high")
    a, s = ads_summary["totals"], seo_summary["totals"]
    summary = (
        f"Paid ads returned {a['roas']:.2f}× on ${a['spend']:,.0f} of spend; organic search earned {s['clicks']:,.0f} clicks "
        f"({s['ctr'] * 100:.1f}% CTR). {high} high-priority finding(s) need attention first."
    )
    return {"executive_summary": summary, "recommendations": recs, "source": "rules",
            "provider": None, "model": None, "error": error}
