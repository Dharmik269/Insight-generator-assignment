"""Part D - Pearson correlation matrix + flagged pairs, with honest fragility checks."""
from __future__ import annotations

import warnings as _w
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from .config import Config

PAIR_COLUMNS = ["a", "b", "r", "p_value", "n_obs", "n_districts", "first_month", "last_month",
                "loo_driver", "loo_r", "fragile", "fragile_reason"]


def correlation_matrix(df: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    """Standard pandas.DataFrame.corr() (Pearson, pairwise-complete)."""
    with _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        return df[indicators].corr(method="pearson")


def _safe_r(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(x.corr(y))


def detect_correlations(df: pd.DataFrame, indicators: list[str], cfg: Config):
    warns: list[str] = []
    corr = correlation_matrix(df, indicators)
    rows = []
    for a, b in combinations(indicators, 2):
        r = corr.loc[a, b]
        if not np.isfinite(r):
            warns.append(f"Correlation {a} vs {b} undefined (constant column or <3 paired values).")
            continue
        if round(abs(float(r)), 9) < cfg.corr_threshold:
            continue
        sub = df[["district", "month", "month_idx", a, b]].dropna(subset=[a, b])
        n = len(sub)
        p = float(stats.pearsonr(sub[a], sub[b]).pvalue) if n >= 3 else float("nan")
        n_d = int(sub["district"].nunique())

        # Leave-one-district-out: does a single district create the relationship?
        loo_driver, loo_r = None, float("nan")
        worst = None
        for d in sorted(sub["district"].unique()):
            rest = sub[sub["district"] != d]
            rr = _safe_r(rest[a], rest[b])
            score = abs(rr) if np.isfinite(rr) else -1.0     # undefined after removal == fully fragile
            if worst is None or score < worst[0]:
                worst, loo_driver, loo_r = (score, d), d, rr
        reasons = []
        if n_d < cfg.min_districts_for_corr:
            reasons.append(f"only {n_d} districts (<{cfg.min_districts_for_corr} recommended)")
        if loo_driver is not None and (not np.isfinite(loo_r) or abs(loo_r) < cfg.corr_threshold):
            reasons.append(f"relationship disappears without {loo_driver}")
        rows.append(dict(a=a, b=b, r=float(r), p_value=p, n_obs=n, n_districts=n_d,
                         first_month=sub["month"].min(), last_month=sub["month"].max(),
                         loo_driver=loo_driver, loo_r=float(loo_r), fragile=bool(reasons),
                         fragile_reason="; ".join(reasons)))
    return pd.DataFrame(rows, columns=PAIR_COLUMNS), corr, warns
