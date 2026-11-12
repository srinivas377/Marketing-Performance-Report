"""Read and validate the two CSV files.

Bad data never crashes the run and is never silently ignored: invalid rows are
excluded from the maths and every exclusion is reported as a data-quality note.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import DataError

SEO_REQUIRED = [
    "product_id", "product_name", "category", "url_slug", "description",
    "impressions_28d", "clicks_28d", "avg_position", "indexed", "page_load_ms",
]
ADS_REQUIRED = [
    "campaign", "platform", "campaign_type", "spend_usd", "clicks",
    "purchases", "revenue_usd", "impressions", "reach",
]

_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f"}


@dataclass
class LoadResult:
    df: pd.DataFrame
    notes: list[str] = field(default_factory=list)   # data-quality notes
    rows_total: int = 0
    rows_used: int = 0


def _read(path: Path, required: list[str]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataError(f"File not found: {path}")
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        raise DataError(f"{path.name} is empty.") from None
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataError(f"{path.name} could not be parsed as CSV: {exc}") from exc
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataError(f"{path.name} is missing required column(s): {', '.join(missing)}")
    if df.empty:
        raise DataError(f"{path.name} has a header but no data rows.")
    return df


def _parse_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or pd.isna(value):
        return pd.NA
    s = str(value).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    return pd.NA


def _numeric(df: pd.DataFrame, cols: list[str], notes: list[str], label: str) -> None:
    """Coerce columns to numbers; non-numeric text becomes NaN (and is reported)."""
    for col in cols:
        original = df[col]
        df[col] = pd.to_numeric(original, errors="coerce")
        bad = original.notna() & df[col].isna()
        if bad.any():
            notes.append(f"{label}: {int(bad.sum())} non-numeric value(s) in '{col}' were treated as missing.")


def _drop(df: pd.DataFrame, mask: pd.Series, id_col: str, reason: str, notes: list[str]) -> pd.DataFrame:
    if mask.any():
        ids = ", ".join(str(v) for v in df.loc[mask, id_col].tolist())
        notes.append(f"Excluded {int(mask.sum())} row(s) – {reason} ({ids}).")
    return df.loc[~mask].copy()


def load_seo(path: Path) -> LoadResult:
    notes: list[str] = []
    raw = _read(path, SEO_REQUIRED)
    total = len(raw)
    df = raw[SEO_REQUIRED].copy()

    for col in ("product_id", "product_name", "category", "url_slug"):
        df[col] = df[col].astype("string").str.strip()
    df = _drop(df, df["product_id"].isna() | df["product_id"].eq(""), "product_name", "missing product_id", notes)

    dup = df["product_id"].duplicated(keep="first")
    df = _drop(df, dup, "product_id", "duplicate product_id (first occurrence kept)", notes)

    missing_name = df["product_name"].isna() | df["product_name"].eq("")
    if missing_name.any():
        notes.append(f"SEO: {int(missing_name.sum())} page(s) had no product_name and defaulted to product_id.")
        df.loc[missing_name, "product_name"] = df.loc[missing_name, "product_id"]

    missing_cat = df["category"].isna() | df["category"].eq("")
    if missing_cat.any():
        notes.append(f"SEO: {int(missing_cat.sum())} page(s) had no category and were grouped under 'Uncategorized'.")
        df.loc[missing_cat, "category"] = "Uncategorized"

    df["url_slug"] = df["url_slug"].fillna("")
    dup_slug = df["url_slug"].ne("") & df["url_slug"].duplicated(keep=False)
    if dup_slug.any():
        ids = ", ".join(df.loc[dup_slug, "product_id"].tolist())
        notes.append(f"SEO: {int(dup_slug.sum())} page(s) share a duplicate url_slug ({ids}).")

    _numeric(df, ["impressions_28d", "clicks_28d", "avg_position", "page_load_ms"], notes, "SEO")

    bad_core = df["impressions_28d"].isna() | df["clicks_28d"].isna() | (df["impressions_28d"] < 0) | (df["clicks_28d"] < 0)
    df = _drop(df, bad_core, "product_id", "missing or negative impressions/clicks", notes)

    bad_ctr = df["clicks_28d"] > df["impressions_28d"]
    df = _drop(df, bad_ctr, "product_id", "clicks exceed impressions (impossible CTR > 100%)", notes)

    # Position / load time: keep the row, but exclude the bad value from those calculations.
    bad_pos = df["avg_position"].notna() & (df["avg_position"] < 1)
    if bad_pos.any():
        notes.append(f"SEO: {int(bad_pos.sum())} avg_position value(s) below 1 were ignored.")
        df.loc[bad_pos, "avg_position"] = np.nan
    if df["avg_position"].isna().any():
        notes.append(f"SEO: {int(df['avg_position'].isna().sum())} page(s) have no valid avg_position and are skipped by ranking checks.")
    bad_load = df["page_load_ms"].notna() & (df["page_load_ms"] <= 0)
    if bad_load.any():
        notes.append(f"SEO: {int(bad_load.sum())} page_load_ms value(s) <= 0 were ignored.")
        df.loc[bad_load, "page_load_ms"] = np.nan

    parsed = [_parse_bool(v) for v in df["indexed"]]
    df["indexed"] = pd.array(parsed, dtype="boolean")
    if df["indexed"].isna().any():
        notes.append(f"SEO: {int(df['indexed'].isna().sum())} page(s) have an unreadable 'indexed' value and are treated as unknown.")

    desc = df["description"].fillna("").astype(str).str.strip()
    df["description"] = desc
    missing_desc = desc.eq("")
    if missing_desc.any():
        ids = ", ".join(df.loc[missing_desc, "product_id"].tolist())
        notes.append(f"SEO: {int(missing_desc.sum())} page(s) have an empty description ({ids}) – kept and flagged as a content issue.")

    median_load = df["page_load_ms"].median()
    if pd.notna(median_load) and median_load < 150:
        notes.append(
            f"SEO: page_load_ms values are very small (median {median_load:.0f}) for milliseconds. "
            "Load times are compared with each other only; please confirm the unit of this column."
        )

    if df.empty:
        raise DataError("seo_data.csv has no valid rows after cleaning.")
    return LoadResult(df.reset_index(drop=True), notes, total, len(df))


def load_ads(path: Path) -> LoadResult:
    notes: list[str] = []
    raw = _read(path, ADS_REQUIRED)
    total = len(raw)
    df = raw[ADS_REQUIRED].copy()

    for col in ("campaign", "platform", "campaign_type"):
        df[col] = df[col].astype("string").str.strip()
    df = _drop(df, df["campaign"].isna() | df["campaign"].eq(""), "platform", "missing campaign name", notes)

    for col in ("platform", "campaign_type"):
        missing_val = df[col].isna() | df[col].eq("")
        if missing_val.any():
            notes.append(f"Ads: {int(missing_val.sum())} campaign(s) had no '{col}' and were grouped under 'Unknown'.")
            df.loc[missing_val, col] = "Unknown"

    dup = df["campaign"].duplicated(keep="first")
    df = _drop(df, dup, "campaign", "duplicate campaign name (first occurrence kept)", notes)

    num_cols = ["spend_usd", "clicks", "purchases", "revenue_usd", "impressions", "reach"]
    _numeric(df, num_cols, notes, "Ads")

    core = ["spend_usd", "clicks", "purchases", "revenue_usd", "impressions"]
    bad_core = df[core].isna().any(axis=1) | (df[core] < 0).any(axis=1)
    df = _drop(df, bad_core, "campaign", "missing or negative spend/clicks/purchases/revenue/impressions", notes)

    bad_ctr = df["clicks"] > df["impressions"]
    df = _drop(df, bad_ctr, "campaign", "clicks exceed impressions", notes)

    if (df["purchases"] > df["clicks"]).any():
        names = ", ".join(df.loc[df["purchases"] > df["clicks"], "campaign"].tolist())
        notes.append(f"Ads: purchases exceed clicks for {names} (possible view-through conversions) – kept as is.")
    if (df["spend_usd"] == 0).any():
        names = ", ".join(df.loc[df["spend_usd"] == 0, "campaign"].tolist())
        notes.append(f"Ads: zero spend for {names} – ROAS cannot be calculated for these and they are skipped in ROAS checks.")
    bad_reach = df["reach"].notna() & (df["reach"] < 0)
    if bad_reach.any():
        notes.append(f"Ads: {int(bad_reach.sum())} negative reach value(s) were ignored.")
        df.loc[bad_reach, "reach"] = np.nan
    if df["reach"].isna().any():
        notes.append(f"Ads: {int(df['reach'].isna().sum())} campaign(s) have no valid reach; frequency is not shown for them.")

    if df.empty:
        raise DataError("ads_data.csv has no valid rows after cleaning.")
    return LoadResult(df.reset_index(drop=True), notes, total, len(df))
