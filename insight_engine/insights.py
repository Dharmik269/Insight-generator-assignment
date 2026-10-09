"""Part E - Automated insight generation.

Every number in every sentence is read from the data/configuration at run time.
Text is a template; there are no district- or value-specific strings.
Severity is derived from  observed_magnitude / configured_threshold  (never a magic
absolute number), then bucketed with the configurable cut-offs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import (ACRONYMS, INSIGHT_TYPES, PERCENT_COLUMNS, SEVERITY_RANK, Config)
from .correlations import detect_correlations
from .outliers import detect_outliers
from .trends import compute_trends

INSIGHT_COLUMNS = ["insight_id", "type", "indicator", "entity", "period", "value",
                   "prev_value", "change_pct", "severity", "explanation"]
ALL_DISTRICTS = "All districts"
LIMITATIONS = (
    "Correlation is computed on pooled district-months. With few districts/months the "
    "coefficient is statistically fragile (two points always give r = +/-1). Treat flagged "
    "pairs as hypotheses to investigate, not conclusions. Correlation is not causation."
)


@dataclass
class AnalysisResult:
    insights: pd.DataFrame
    trends: pd.DataFrame
    outliers: pd.DataFrame
    corr_matrix: pd.DataFrame
    corr_pairs: pd.DataFrame
    targets: dict
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- helpers
def label(ind: str) -> str:
    return " ".join(w.upper() if w in ACRONYMS else w.capitalize() for w in ind.split("_"))


def fmt(x, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    s = f"{float(x):.{nd}f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and not math.isfinite(p)):
        return "n/a"
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def severity_from_ratio(ratio: float, cfg: Config) -> str:
    if ratio is None or (isinstance(ratio, float) and math.isnan(ratio)):
        raise ValueError("Severity ratio is NaN - refusing to guess a severity.")
    if ratio >= cfg.sev_high_ratio:
        return "High"
    if ratio >= cfg.sev_medium_ratio:
        return "Medium"
    return "Low"


def severity_from_corr(r: float, fragile: bool, cfg: Config) -> str:
    headroom = (abs(r) - cfg.corr_threshold) / (1.0 - cfg.corr_threshold)
    if headroom >= cfg.corr_high_frac:
        sev = "High"
    elif headroom >= cfg.corr_medium_frac:
        sev = "Medium"
    else:
        sev = "Low"
    # Fragile evidence is never allowed to be "High".
    return "Medium" if (fragile and sev == "High") else sev


def default_targets(df: pd.DataFrame, indicators: list[str], lower_is_better) -> dict:
    """Data-derived default targets. Real programme targets should be supplied by the user."""
    out = {}
    for ind in indicators:
        s = df[ind].dropna()
        if s.empty:
            continue
        if ind in lower_is_better:
            out[ind] = float(math.ceil(s.quantile(0.75)))         # ceiling = upper quartile
        elif ind in PERCENT_COLUMNS:
            out[ind] = 75.0                                       # illustrative coverage floor
        else:
            out[ind] = float(math.floor(s.quantile(0.25)))        # floor = lower quartile
    return out


def _sentinel(x: float) -> float:
    return 1e18 if x == float("inf") else x


# ---------------------------------------------------------------- builders
def _trend_rows(trends: pd.DataFrame, cfg: Config) -> list[dict]:
    rows = []
    for t in trends[trends["is_significant"]].itertuples(index=False):
        lab, thr = label(t.indicator), fmt(cfg.trend_threshold_pct, 1)
        rose = t.value > t.prev_value
        # Polarity: for "lower is better" indicators a rise is bad; otherwise a drop is bad.
        verdict = "a deterioration" if (rose == (t.indicator in cfg.lower_is_better)) else "an improvement"
        verb = "rose" if rose else "dropped"
        if t.undefined_baseline:
            body = (f"{lab} in {t.district} {verb} from {fmt(t.prev_value)} to {fmt(t.value)} "
                    f"compared to the previous month (percentage change undefined from a zero baseline)")
            change = float("nan")
        else:
            body = (f"{lab} in {t.district} {verb} by {fmt(abs(t.pct_change), 1)}% compared to the previous "
                    f"month ({fmt(t.prev_value)} -> {fmt(t.value)}), exceeding the {thr}% significant-change threshold")
            change = round(float(t.pct_change), 1)
        rows.append(dict(type="trend", indicator=t.indicator, entity=t.district, period=t.month,
                         value=round(float(t.value), 2), prev_value=round(float(t.prev_value), 2),
                         change_pct=change, severity=severity_from_ratio(float(t.ratio), cfg),
                         explanation=f"{body}; assessed as {verdict}.", _score=_sentinel(float(t.ratio))))
    return rows


def _outlier_rows(out: pd.DataFrame, cfg: Config) -> list[dict]:
    rows = []
    for o in out.itertuples(index=False):
        lab = label(o.indicator)
        dev = (o.value - o.mean) / o.mean * 100.0 if o.mean != 0 else float("nan")
        side = "lower" if o.direction == "below" else "upper"
        if o.method == "iqr":
            fence = o.lower_fence if o.direction == "below" else o.upper_fence
            why = (f"{o.direction} the {side} IQR fence of {fmt(fence)} "
                   f"(multiplier {fmt(cfg.iqr_multiplier)}; pooled median {fmt(o.median)})")
        else:
            why = f"beyond the Z-score threshold of {fmt(cfg.z_threshold)}"
        z_part = f"; {fmt(abs(o.z), 1)}\u03c3" if np.isfinite(o.z) else ""
        if np.isfinite(dev):
            pct_txt = f"{fmt(abs(dev), 1)}% {o.direction} the pooled mean ({fmt(o.mean)}{z_part})"
        else:
            pct_txt = f"{o.direction} the pooled mean ({fmt(o.mean)}{z_part})"
        rows.append(dict(type="outlier", indicator=o.indicator, entity=o.district, period=o.month,
                         value=round(float(o.value), 2), prev_value=round(float(o.mean), 2),
                         change_pct=round(float(dev), 1) if np.isfinite(dev) else float("nan"),
                         severity=severity_from_ratio(float(o.ratio), cfg),
                         explanation=(f"{o.district}'s {lab} of {fmt(o.value)} in {o.month} is {pct_txt} and lies "
                                      f"{why}; n = {int(o.pool_n)} district-months. Flagged for review."),
                         _score=float(o.ratio)))
    return rows


def _corr_rows(pairs: pd.DataFrame, cfg: Config) -> list[dict]:
    rows = []
    for c in pairs.itertuples(index=False):
        strength = "strongly " if abs(c.r) >= 0.9 else ""
        sign = "negatively" if c.r < 0 else "positively"
        period = c.first_month if c.first_month == c.last_month else f"{c.first_month}..{c.last_month}"
        txt = (f"{label(c.a)} and {label(c.b)} are {strength}{sign} correlated (Pearson r = {fmt(c.r)}, "
               f"n = {c.n_obs} district-months across {c.n_districts} districts, "
               f"p {fmt_p(c.p_value)}), above the |r| >= {fmt(cfg.corr_threshold)} threshold.")
        if c.fragile:
            txt += f" LOW CONFIDENCE: {c.fragile_reason}."
            if c.loo_driver is not None:
                txt += f" Dropping the most influential district ({c.loo_driver}) gives r = {fmt(c.loo_r)}."
        txt += " Correlation does not imply causation."
        rows.append(dict(type="correlation", indicator=f"{c.a}:{c.b}", entity=ALL_DISTRICTS, period=period,
                         value=round(float(c.r), 2), prev_value=float("nan"), change_pct=float("nan"),
                         severity=severity_from_corr(float(c.r), bool(c.fragile), cfg),
                         explanation=txt, _score=abs(float(c.r))))
    return rows


def _breach_rows(df: pd.DataFrame, targets: dict, cfg: Config) -> list[dict]:
    rows = []
    for ind, limit in targets.items():
        if ind not in df.columns:
            continue
        lower_better = ind in cfg.lower_is_better
        for r in df[["district", "month", ind]].dropna(subset=[ind]).itertuples(index=False):
            x = float(r[2])
            gap = (x - limit) if lower_better else (limit - x)      # >0 means breach
            if round(gap, 9) <= 0:
                continue
            dev = gap / abs(limit) * 100.0 if limit != 0 else float("inf")
            ratio = dev / cfg.breach_scale_pct
            rel = "above the maximum" if lower_better else "below the minimum"
            dev_txt = f"{fmt(dev, 1)}% {'over' if lower_better else 'short'}" if np.isfinite(dev) else "target is zero"
            signed = (x - limit) / abs(limit) * 100.0 if limit != 0 else float("nan")
            rows.append(dict(type="threshold_breach", indicator=ind, entity=r.district, period=r.month,
                             value=round(x, 2), prev_value=round(float(limit), 2),
                             change_pct=round(signed, 1) if np.isfinite(signed) else float("nan"),
                             severity=severity_from_ratio(ratio, cfg),
                             explanation=(f"{r.district}'s {label(ind)} of {fmt(x)} in {r.month} is {rel} target "
                                          f"of {fmt(limit)} ({dev_txt})."),
                             _score=_sentinel(ratio)))
    return rows


# ---------------------------------------------------------------- orchestration
def run_analysis(df: pd.DataFrame, indicators: list[str], cfg: Config) -> AnalysisResult:
    warns: list[str] = []
    trends = compute_trends(df, indicators, cfg)
    outliers, w = detect_outliers(df, indicators, cfg)
    warns += w
    pairs, corr, w = detect_correlations(df, indicators, cfg)
    warns += w
    if trends.empty:
        warns.append("No consecutive-month pairs found: no trends computed.")

    targets = cfg.target_map() or default_targets(df, indicators, cfg.lower_is_better)

    rows = _trend_rows(trends, cfg) + _outlier_rows(outliers, cfg) + _corr_rows(pairs, cfg) \
        + _breach_rows(df, targets, cfg)
    ins = pd.DataFrame(rows, columns=INSIGHT_COLUMNS[1:] + ["_score"])
    if not ins.empty:
        ins["_t"] = ins["type"].map({t: i for i, t in enumerate(INSIGHT_TYPES)})
        ins["_s"] = ins["severity"].map(SEVERITY_RANK)
        # Total, deterministic ordering => identical input+config always yields identical IDs.
        ins = ins.sort_values(["_t", "_s", "_score", "entity", "indicator", "period"],
                              ascending=[True, False, False, True, True, True], kind="mergesort")
        ins = ins.drop(columns=["_t", "_s"]).reset_index(drop=True)
    ins.insert(0, "insight_id", [f"INS-{i + 1:04d}" for i in range(len(ins))])
    ins = ins.drop(columns="_score")[INSIGHT_COLUMNS]
    return AnalysisResult(ins, trends, outliers, corr, pairs, targets, warns)


def filter_insights(ins: pd.DataFrame, districts=None, months=None, indicators=None) -> pd.DataFrame:
    """Live filters. Correlation insights are dataset-wide (entity = 'All districts'),
    so only the indicator filter (both members of the pair) applies to them."""
    if ins.empty:
        return ins
    is_corr = ins["type"] == "correlation"
    keep = pd.Series(True, index=ins.index)
    if districts is not None:
        keep &= is_corr | ins["entity"].isin(districts)
    if months is not None:
        keep &= is_corr | ins["period"].isin(months)
    if indicators is not None:
        sel = set(indicators)
        parts = ins["indicator"].str.split(":")
        keep &= parts.map(lambda p: all(x in sel for x in p))
    return ins[keep].reset_index(drop=True)
