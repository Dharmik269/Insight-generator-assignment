# Assignment 4 — Automated Insight Generation (Auto-Analytics Engine)

A general-purpose engine that ingests a district-level healthcare CSV and automatically produces
**trends, outliers, correlations and threshold breaches** as structured, human-readable insights,
with a Streamlit UI (live filters, sliders, charts).  Backend and UI are pure Python; no separate API.

> **Decision-support only.** Flags must be reviewed by a qualified person before any clinical or
> operational action. The sample data is aggregated district data; do **not** load patient-level
> identifiable data into a tool that has not been through your organisation's privacy review.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py          # interactive UI
python main.py                # CLI: prints head()/info()/missing report, writes outputs/
python -m pytest -q           # 49 tests
```

CLI options: `python main.py --csv file.csv --trend 15 --method zscore --z 2.5 --corr 0.8`

## Project layout

| Path | Purpose |
|---|---|
| `data/district_health.csv` | The exact dataset from the assignment |
| `insight_engine/config.py` | All thresholds in one validated, immutable `Config` |
| `insight_engine/loader.py` | **Part A** — load + strict validation + head/info/missing report |
| `insight_engine/trends.py` | **Part B** — month-over-month % change |
| `insight_engine/outliers.py` | **Part C** — IQR rule or Z-score |
| `insight_engine/correlations.py` | **Part D** — Pearson matrix, flagged pairs, fragility checks |
| `insight_engine/insights.py` | **Part E** — insight rows, severity, explanations, live filters |
| `insight_engine/export.py` | CSV/JSON export (formula-injection safe) |
| `app.py` | **Part F** — Streamlit UI |
| `main.py` | CLI runner |
| `outputs/` | `insights.csv`, `insights.json`, `correlation_matrix.csv` generated from the sample data |
| `tests/test_engine.py` | 49 tests |

## How it works

**Trend (B).** `pct_change = (current − previous) / previous × 100` per `(district, indicator)`,
only between *consecutive* calendar months (a missing month is never bridged). `is_significant` when
`|pct_change| ≥ threshold` (default 10 %, slider). Previous = 0 → percentage undefined; reported as
an unbounded change (severity High) instead of dividing by zero.

**Outlier (C).** Per indicator, pooled over all district-months. IQR: outlier if `v > Q3+k·IQR` or
`v < Q1−k·IQR` (k = 1.5, slider). Alternative Z-score (threshold 3, slider). Outliers and trends are
separate insight types and are never merged.

**Correlation (D).** `DataFrame.corr()` (Pearson) over all indicators; pairs with `|r| ≥ 0.70`
(slider) are flagged. Each flagged pair is stress-tested with a leave-one-district-out check, and
marked **LOW CONFIDENCE** when there are fewer than 10 districts or the relationship vanishes
without a single district.

**Threshold breach (E).** Value beyond a target (floor for "higher is better", ceiling for "lower is
better"). Targets are editable in the UI; defaults are derived from the data (quartiles) except a
75 % floor for the percentage indicators, which is an *illustrative* default — set your programme's real targets.

**Severity (never a magic number).** `ratio = observed magnitude ÷ configured threshold`
(trend: `|%Δ| / threshold`; outlier: Z → `|z| / threshold`, IQR → `(k + distance beyond fence in IQRs) / k`;
breach: `% beyond target / breach scale`). `ratio ≥ 1.75 → High`, `≥ 1.25 → Medium`, else `Low` (both cut-offs are
sliders). Correlation uses the share of headroom between the threshold and |r| = 1 (≥⅔ High, ≥⅓ Medium);
fragile correlations are capped at Medium.

**Insight fields:** `insight_id, type, indicator, entity, period, value, prev_value, change_pct, severity, explanation`.
Explanations are templates; every number comes from the data. For outliers `prev_value` is the pooled mean
baseline and `change_pct` the % deviation from it; for breaches `prev_value` is the target; for correlations
`value` is *r*.  IDs are assigned after a total, deterministic sort, so the same input + settings always gives
identical output.

## Sample output (default settings, `outputs/insights.csv`)

| ID | Type | Entity | Indicator | Period | Value | Prev/Base | Δ% | Severity |
|---|---|---|---|---|---|---|---|---|
| INS-0004 | trend | Ahmedabad | anc_coverage | 2026-08 | 69 | 85 | −18.8 | High |
| INS-0007 | outlier | Mehsana | anc_coverage | 2026-08 | 42 | 78.33 (pooled mean) | −46.4 | High |
| INS-0001 | trend | Mehsana | high_risk_cases | 2026-08 | 28 | 11 | +154.5 | High |
| INS-0012 | correlation | All districts | anc_coverage:high_risk_cases | 2026-07..2026-08 | r = −0.93 | – | – | Medium (low confidence) |

## Limitations 

* **2 months × 6 districts = 12 rows.** Pearson correlations are statistically fragile (the assignment
  recommends ≥ 3 months and ≥ 10 districts). The engine says so in the UI, the insight text and the JSON.
* With only 2 months each district has a single month-over-month comparison; no longer-run trend exists.
* **Z-score on 12 points is weak:** the extreme value inflates the standard deviation (masking). Here Mehsana's
  ANC of 42 is 2.8σ from the pooled mean, below the default threshold of 3, so the Z-score method does not flag it
  while the IQR rule does. The sample text's "3.1σ below state mean 76" uses a different baseline than the pooled mean.
* The IQR rule also flags Ahmedabad's 69 % ANC (just under the lower fence 69.88) — shown as **Low** severity.
* Small counts (e.g. 7 → 6 high-risk cases) produce large percentage changes; read the absolute numbers too.
* Missing values are skipped, never imputed.

## Security & safety design

* **Fail-closed validation:** non-numeric, NaN/inf, negative, >100 % percentages, invalid months, duplicate
  `(district, month)`, inconsistent district spelling, duplicate/odd column names, non-UTF-8, NUL bytes → clear error, no analysis.
* **Limits:** 5 MB file, 100 000 rows, 50 indicators.
* **Injection:** district names must match a strict allow-list (so `=cmd…`, `@SUM`, `<script>` are rejected); exports additionally
  prefix any text cell starting with `= + - @ TAB CR` with `'`; UI never uses `unsafe_allow_html`.
* **No code execution paths:** no `eval/exec/pickle/subprocess`; checked with `bandit` (clean) and `pip-audit` (no known vulnerabilities).
* **Privacy:** runs locally, no network calls; telemetry off, XSRF protection on, stack traces hidden (`.streamlit/config.toml`).
* **Numerical safety:** boundary comparisons use rounding to avoid float artefacts (exactly −10 % counts as significant);
  division by zero and constant columns handled; `Config` rejects NaN/out-of-range settings; severity refuses to guess on NaN.
* **Deterministic, strict JSON** (`allow_nan=False`, no timestamps).
