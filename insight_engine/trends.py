"""Part B - Trend detection.

pct_change = (current - previous) / previous * 100, computed per
(district, indicator) between CONSECUTIVE calendar months only (a gap is never
bridged, so we never describe a 2-month change as "vs previous month").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

TREND_COLUMNS = ["district", "indicator", "month", "prev_month", "value", "prev_value",
                 "pct_change", "undefined_baseline", "is_significant", "ratio"]


def compute_trends(df: pd.DataFrame, indicators: list[str], cfg: Config) -> pd.DataFrame:
    long = df.melt(id_vars=["district", "month", "month_idx"], value_vars=indicators,
                   var_name="indicator", value_name="value").dropna(subset=["value"])
    if long.empty:
        return pd.DataFrame(columns=TREND_COLUMNS)
    long = long.sort_values(["indicator", "district", "month_idx"], kind="mergesort")
    g = long.groupby(["indicator", "district"], sort=False)
    long["prev_value"] = g["value"].shift(1)
    long["prev_idx"] = g["month_idx"].shift(1)
    long["prev_month"] = g["month"].shift(1)

    t = long[(long["month_idx"] - long["prev_idx"]) == 1].copy()   # consecutive months only
    if t.empty:
        return pd.DataFrame(columns=TREND_COLUMNS)

    prev, cur = t["prev_value"], t["value"]
    nonzero = prev != 0
    t["pct_change"] = np.where(nonzero, (cur - prev) / prev.where(nonzero, 1.0) * 100.0, np.nan)
    # 0 -> x (x != 0): percentage is undefined; treated as unbounded change (see README).
    t["undefined_baseline"] = (~nonzero) & (cur != 0)
    mag = t["pct_change"].abs().round(9)                           # kills float noise at the boundary
    thr = cfg.trend_threshold_pct
    t["is_significant"] = ((mag >= thr) | t["undefined_baseline"]).astype(bool)
    t["ratio"] = np.where(t["undefined_baseline"], np.inf, (mag / thr).fillna(0.0))
    return t[TREND_COLUMNS].reset_index(drop=True)
