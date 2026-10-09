"""Safe export. Text cells starting with = + - @ TAB CR are prefixed with a quote so
Excel/Sheets can never execute them as formulas (CSV/formula injection)."""
from __future__ import annotations

import json
import math

import pandas as pd

from .config import Config
from .insights import AnalysisResult, INSIGHT_COLUMNS, LIMITATIONS

_DANGEROUS = ("=", "+", "-", "@", "\t", "\r")
TEXT_COLUMNS = ["insight_id", "type", "indicator", "entity", "period", "severity", "explanation"]


def sanitize_cell(x):
    if isinstance(x, str) and x.startswith(_DANGEROUS):
        return "'" + x
    return x


def _safe_frame(ins: pd.DataFrame) -> pd.DataFrame:
    out = ins[INSIGHT_COLUMNS].copy()
    for c in TEXT_COLUMNS:
        out[c] = out[c].map(sanitize_cell)
    return out


def insights_csv(ins: pd.DataFrame) -> str:
    return _safe_frame(ins).to_csv(index=False, na_rep="", lineterminator="\n")


def corr_csv(corr: pd.DataFrame) -> str:
    return corr.round(4).to_csv(na_rep="", lineterminator="\n")


def _clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if hasattr(v, "item"):
        v = v.item()
        return _clean(v)
    return v


def insights_json(res: AnalysisResult, cfg: Config, ins: pd.DataFrame | None = None) -> str:
    ins = res.insights if ins is None else ins
    records = [{k: _clean(v) for k, v in r.items()} for r in _safe_frame(ins).to_dict("records")]
    payload = {"config": cfg.as_dict(), "targets_used": res.targets, "warnings": res.warnings,
               "limitations": LIMITATIONS, "insight_count": len(records), "insights": records}
    return json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
