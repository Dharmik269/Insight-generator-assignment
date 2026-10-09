"""Part C - Outlier detection (IQR rule or Z-score), per indicator, pooled over
all district-months in the dataset.

Why pooled and not per-month: with 6 districts a per-month sample has n=6, where a
Z-score can never exceed (n-1)/sqrt(n) = 2.04, so a threshold of 3 would be
unreachable.  Pooling gives n = districts x months.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

OUTLIER_COLUMNS = ["district", "indicator", "month", "value", "direction", "method", "z",
                   "mean", "median", "lower_fence", "upper_fence", "ratio", "pool_n"]
_EPS = 1e-9


def detect_outliers(df: pd.DataFrame, indicators: list[str], cfg: Config):
    rows: list[dict] = []
    warnings: list[str] = []
    for ind in indicators:
        s = df[["district", "month", ind]].dropna(subset=[ind])
        n = len(s)
        need = 4 if cfg.outlier_method == "iqr" else 3
        if n < need:
            warnings.append(f"Outliers skipped for '{ind}': need >= {need} values, have {n}.")
            continue
        v = s[ind].to_numpy(dtype=float)
        mean, median = float(v.mean()), float(np.median(v))
        std = float(v.std(ddof=1))
        q1, q3 = (float(x) for x in np.percentile(v, [25, 75]))   # linear interpolation (= pandas default)
        iqr = q3 - q1

        if cfg.outlier_method == "iqr":
            if iqr == 0:
                warnings.append(f"IQR outliers skipped for '{ind}': IQR is 0 (values nearly constant).")
                continue
            k = cfg.iqr_multiplier
            lo, hi = q1 - k * iqr, q3 + k * iqr
        else:
            if std == 0:
                warnings.append(f"Z-score outliers skipped for '{ind}': standard deviation is 0.")
                continue
            lo = hi = np.nan

        z_all = (v - mean) / std if std > 0 else np.full_like(v, np.nan)
        for i, (district, month) in enumerate(zip(s["district"], s["month"])):
            x = float(v[i])
            if cfg.outlier_method == "iqr":
                if lo - x > _EPS:
                    direction, dist = "below", (lo - x) / iqr
                elif x - hi > _EPS:
                    direction, dist = "above", (x - hi) / iqr
                else:
                    continue
                ratio = (cfg.iqr_multiplier + dist) / cfg.iqr_multiplier
            else:
                z = float(z_all[i])
                if round(abs(z), 9) < cfg.z_threshold:
                    continue
                direction = "below" if z < 0 else "above"
                ratio = abs(z) / cfg.z_threshold
            rows.append(dict(district=district, indicator=ind, month=month, value=x,
                             direction=direction, method=cfg.outlier_method,
                             z=float(z_all[i]), mean=mean, median=median,
                             lower_fence=float(lo), upper_fence=float(hi),
                             ratio=float(ratio), pool_n=n))
    out = pd.DataFrame(rows, columns=OUTLIER_COLUMNS)
    return out, warnings
