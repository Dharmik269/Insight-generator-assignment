"""Part A - Data loading & validation.

Design rule for clinical data: FAIL CLOSED. Anything malformed (bad numbers,
out-of-range percentages, duplicate keys, odd characters) raises ValidationError
instead of being silently coerced or imputed. Genuinely empty cells are kept as
NaN (never imputed) and reported.
"""
from __future__ import annotations

import csv
import datetime as _dt
import io
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import PERCENT_COLUMNS

MAX_BYTES = 5 * 1024 * 1024
MAX_ROWS = 100_000
MAX_INDICATORS = 50
COLUMN_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
# Must start with a letter/digit (blocks "=", "+", "-", "@" CSV-formula injection),
# then letters/digits/space and a few harmless punctuation chars. Unicode letters OK.
DISTRICT_RE = re.compile(r"^[^\W_][\w .'&()\-]{0,63}$")
MONTH_RE = re.compile(r"^(\d{4})-(\d{2})(?:-(\d{2}))?$")
MAX_ABS_VALUE = 1e12


class ValidationError(ValueError):
    """Raised when the input file does not satisfy the schema/safety rules."""


@dataclass
class LoadResult:
    df: pd.DataFrame
    indicators: list[str]
    head: str
    info: str
    missing: pd.Series
    warnings: list[str] = field(default_factory=list)


def _read_bytes(source, max_bytes: int) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    elif hasattr(source, "read"):
        raw = source.read(max_bytes + 1)
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
    else:
        path = os.fspath(source)
        if not os.path.isfile(path):
            raise ValidationError(f"File not found: {path}")
        if os.path.getsize(path) > max_bytes:
            raise ValidationError(f"File larger than {max_bytes // (1024 * 1024)} MB limit.")
        with open(path, "rb") as fh:
            raw = fh.read()
    if len(raw) > max_bytes:
        raise ValidationError(f"File larger than {max_bytes // (1024 * 1024)} MB limit.")
    return raw


def _month_index(s: str) -> int | None:
    """'2026-07' or '2026-07-01' -> absolute month number; None if invalid."""
    m = MONTH_RE.match(s)
    if not m:
        return None
    y, mo = int(m.group(1)), int(m.group(2))
    if not (1900 <= y <= 2100 and 1 <= mo <= 12):
        return None
    if m.group(3) is not None:
        try:
            _dt.date(y, mo, int(m.group(3)))
        except ValueError:
            return None
    return y * 12 + (mo - 1)


def _month_str(idx: int) -> str:
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def _rows(mask: pd.Series, limit: int = 5) -> str:
    lines = [int(i) + 2 for i in mask[mask].index[:limit]]  # +2 => spreadsheet line no.
    extra = int(mask.sum()) - len(lines)
    return ", ".join(map(str, lines)) + (f" (+{extra} more)" if extra > 0 else "")


def load_csv(source, *, max_bytes: int = MAX_BYTES) -> LoadResult:
    raw = _read_bytes(source, max_bytes)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValidationError("File is not valid UTF-8 text.") from exc
    if "\x00" in text:
        raise ValidationError("File contains NUL bytes; not a plain CSV.")
    if not text.strip():
        raise ValidationError("File is empty.")

    header = next(csv.reader(io.StringIO(text)), [])
    cols = [h.strip().lower() for h in header]
    if len(set(cols)) != len(cols):
        raise ValidationError("Duplicate column names in header.")
    for c in cols:
        if not COLUMN_RE.match(c):
            raise ValidationError(f"Invalid column name {c!r}: use letters, digits, underscore.")
    missing_req = [c for c in ("month", "district") if c not in cols]
    if missing_req:
        raise ValidationError(f"Missing required column(s): {', '.join(missing_req)}")
    indicators = [c for c in cols if c not in ("month", "district")]
    if not indicators:
        raise ValidationError("No indicator columns found besides month/district.")
    if len(indicators) > MAX_INDICATORS:
        raise ValidationError(f"Too many indicator columns (max {MAX_INDICATORS}).")

    try:
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False,
                         na_values=[""], index_col=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValidationError(f"CSV could not be parsed: {exc}") from exc
    if len(df.columns) != len(cols):
        raise ValidationError("Row/column count mismatch in CSV.")
    df.columns = cols
    if df.empty:
        raise ValidationError("CSV has a header but no data rows.")
    if len(df) > MAX_ROWS:
        raise ValidationError(f"Too many rows (max {MAX_ROWS}).")

    for c in df.columns:
        s = df[c].str.strip()
        df[c] = s.where(s != "", np.nan)

    # --- keys -------------------------------------------------------------
    bad = df["district"].isna()
    if bad.any():
        raise ValidationError(f"Blank district at line(s): {_rows(bad)}")
    bad = df["month"].isna()
    if bad.any():
        raise ValidationError(f"Blank month at line(s): {_rows(bad)}")

    df["district"] = df["district"].str.replace(r"\s+", " ", regex=True)
    bad = ~df["district"].str.match(DISTRICT_RE)
    if bad.any():
        raise ValidationError(f"Invalid district name at line(s): {_rows(bad)} "
                              "(allowed: letters, digits, space and . ' & ( ) -)")
    folded = df.groupby(df["district"].str.casefold())["district"].nunique()
    clash = folded[folded > 1].index.tolist()
    if clash:
        raise ValidationError(f"Inconsistent district spelling/case for: {', '.join(clash)}")

    idx = df["month"].map(_month_index)
    bad = idx.isna()
    if bad.any():
        raise ValidationError(f"Invalid month at line(s): {_rows(bad)} (use YYYY-MM or YYYY-MM-DD)")
    df["month_idx"] = idx.astype("int64")
    df["month"] = df["month_idx"].map(_month_str)

    dup = df.duplicated(["district", "month_idx"], keep=False)
    if dup.any():
        raise ValidationError(f"Duplicate (district, month) rows at line(s): {_rows(dup)}")

    # --- indicators -------------------------------------------------------
    for c in indicators:
        num = pd.to_numeric(df[c], errors="coerce")
        bad = df[c].notna() & num.isna()                 # text such as "abc" or "nan"
        if bad.any():
            raise ValidationError(f"Non-numeric value in '{c}' at line(s): {_rows(bad)}")
        bad = num.notna() & (~np.isfinite(num) | (num.abs() > MAX_ABS_VALUE))
        if bad.any():
            raise ValidationError(f"Infinite/absurd value in '{c}' at line(s): {_rows(bad)}")
        bad = num < 0
        if bad.any():
            raise ValidationError(f"Negative value in '{c}' at line(s): {_rows(bad)}")
        if c in PERCENT_COLUMNS:
            bad = num > 100
            if bad.any():
                raise ValidationError(f"Percentage above 100 in '{c}' at line(s): {_rows(bad)}")
        df[c] = num.astype("float64")

    df = df[["month", "month_idx", "district", *indicators]]
    df = df.sort_values(["district", "month_idx"], kind="mergesort").reset_index(drop=True)

    show = df.drop(columns="month_idx")
    buf = io.StringIO()
    show.info(buf=buf)
    missing = show.isna().sum()

    warnings: list[str] = []
    n_d = df["district"].nunique()
    if n_d < 10:
        warnings.append(f"Only {n_d} district(s): correlations need >=10 districts to be stable.")
    per = df.groupby("district")["month_idx"].nunique()
    if (per < 3).any():
        warnings.append(f"{int((per < 3).sum())} district(s) have <3 months: trend detection is limited "
                        f"(longest history: {int(per.max())} months).")
    gaps = [d for d, g in df.groupby("district")["month_idx"] if g.diff().dropna().gt(1).any()]
    if gaps:
        warnings.append("Months are not contiguous for: " + ", ".join(gaps) +
                        ". Trends are only computed between consecutive months.")
    if int(missing.sum()) > 0:
        warnings.append(f"{int(missing.sum())} missing value(s); they are NOT imputed and are skipped.")

    return LoadResult(df=df, indicators=indicators, head=show.head().to_string(),
                      info=buf.getvalue(), missing=missing, warnings=warnings)


def format_report(lr: LoadResult) -> str:
    out = ["=== head() ===", lr.head, "", "=== info() ===", lr.info.rstrip(), "",
           "=== missing values per column ===", lr.missing.to_string()]
    if lr.warnings:
        out += ["", "=== warnings ==="] + [f"- {w}" for w in lr.warnings]
    return "\n".join(out)
