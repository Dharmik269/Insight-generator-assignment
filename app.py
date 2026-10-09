"""Part F - Streamlit UI.  Run:  streamlit run app.py

Security notes
* Everything runs locally; no network calls, no telemetry (see .streamlit/config.toml).
* Uploaded files are size-limited and validated (fail closed) before any analysis.
* User/data-derived text is displayed with st.dataframe / st.text (never HTML, never
  unsafe_allow_html), so a malicious cell value cannot inject markup or script.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from insight_engine import Config, ValidationError, filter_insights, load_csv, run_analysis
from insight_engine.config import SEVERITY_LEVELS
from insight_engine.export import corr_csv, insights_csv, insights_json
from insight_engine.insights import LIMITATIONS, default_targets, label
from insight_engine.correlations import correlation_matrix
from insight_engine.loader import MAX_BYTES

DEFAULT_CSV = Path(__file__).parent / "data" / "district_health.csv"
SEV_COLORS = {"Low": "#2e9e5b", "Medium": "#e8a317", "High": "#d62f2f"}

st.set_page_config(page_title="Auto-Analytics Engine", layout="wide")
st.title("Auto-Analytics Engine \u2014 District Health Insights")
st.caption("Automated trend, outlier, correlation and threshold-breach detection. "
           "Decision-support only: every flag must be verified by a qualified person.")

# ------------------------------------------------------------------ data
with st.sidebar:
    st.header("1 \u00b7 Data")
    up = st.file_uploader("Upload CSV (optional)", type=["csv"],
                          help=f"Max {MAX_BYTES // (1024 * 1024)} MB. Must contain month, district + numeric indicators.")


@st.cache_data(show_spinner=False)
def _load(raw: bytes | None):
    return load_csv(DEFAULT_CSV if raw is None else raw)


try:
    lr = _load(up.getvalue() if up is not None else None)
except ValidationError as exc:
    st.error(f"Data validation failed: {exc}")
    st.stop()
except Exception:  # never leak internals / stack traces for unexpected input
    st.error("The file could not be processed. Please check it is a valid CSV.")
    st.stop()

df, indicators = lr.df, lr.indicators

# ------------------------------------------------------------------ sidebar controls
with st.sidebar:
    st.header("2 \u00b7 Thresholds")
    trend_thr = st.slider("Trend: significant |% change| \u2265", 1.0, 100.0, 10.0, 0.5)
    method = st.radio("Outlier method", ["iqr", "zscore"], horizontal=True,
                      format_func=lambda m: "IQR rule" if m == "iqr" else "Z-score")
    iqr_mult = st.slider("IQR multiplier", 0.5, 5.0, 1.5, 0.1, disabled=method != "iqr")
    z_thr = st.slider("Z-score threshold", 1.0, 6.0, 3.0, 0.1, disabled=method != "zscore")
    corr_thr = st.slider("Correlation: |r| \u2265", 0.30, 0.99, 0.70, 0.01)
    with st.expander("Severity & breach settings"):
        sev_med = st.slider("Medium from (\u00d7 threshold)", 1.0, 3.0, 1.25, 0.05)
        sev_high = st.slider("High from (\u00d7 threshold)", 1.0, 5.0, 1.75, 0.05)
        corr_med = st.slider("Correlation Medium from (share of headroom)", 0.05, 0.9, 0.33, 0.01)
        corr_high = st.slider("Correlation High from (share of headroom)", 0.1, 0.95, 0.67, 0.01)
        breach_scale = st.slider("Breach: % beyond target = 1 severity unit", 1.0, 50.0, 10.0, 0.5)
        lower_better = st.multiselect("Indicators where LOWER is better", indicators,
                                      default=[i for i in indicators if i == "high_risk_cases"])
        st.caption("Targets (default = data-derived; set your programme targets):")
        defaults = default_targets(df, indicators, frozenset(lower_better))
        targets = tuple((i, st.number_input(f"{label(i)} target", value=float(defaults[i]),
                                            key=f"tgt_{i}", format="%.2f"))
                        for i in indicators if i in defaults)

    st.header("3 \u00b7 Filters")
    f_districts = st.multiselect("District", sorted(df["district"].unique()),
                                 default=sorted(df["district"].unique()))
    f_months = st.multiselect("Month", sorted(df["month"].unique()), default=sorted(df["month"].unique()))
    f_inds = st.multiselect("Indicator", indicators, default=indicators, format_func=label)

try:
    cfg = Config(trend_threshold_pct=trend_thr, outlier_method=method, iqr_multiplier=iqr_mult,
                 z_threshold=z_thr, corr_threshold=corr_thr, sev_medium_ratio=sev_med,
                 sev_high_ratio=sev_high, corr_medium_frac=corr_med, corr_high_frac=corr_high,
                 breach_scale_pct=breach_scale, lower_is_better=frozenset(lower_better), targets=targets)
except ValueError as exc:
    st.sidebar.error(f"Invalid settings: {exc}")
    st.stop()

res = run_analysis(df, indicators, cfg)
view = filter_insights(res.insights, f_districts, f_months, f_inds)

# ------------------------------------------------------------------ Part A report
with st.expander("Data validation report (head / info / missing values)"):
    st.text(lr.head)
    st.text(lr.info)
    st.dataframe(lr.missing.rename("missing values"))
for w in lr.warnings + res.warnings:
    st.warning(w)

# ------------------------------------------------------------------ summary + insights
c1, c2, c3, c4 = st.columns(4)
counts = view["severity"].value_counts().reindex(SEVERITY_LEVELS, fill_value=0)
c1.metric("Insights shown", len(view))
c2.metric("High", int(counts["High"]))
c3.metric("Medium", int(counts["Medium"]))
c4.metric("Low", int(counts["Low"]))

tab_ins, tab_charts, tab_corr, tab_dl = st.tabs(["Insights", "Charts", "Correlation", "Download"])

with tab_ins:
    if view.empty:
        st.info("No insights match the current filters/thresholds.")
    else:
        sev_f = st.multiselect("Severity", list(SEVERITY_LEVELS), default=list(SEVERITY_LEVELS))
        type_f = st.multiselect("Type", sorted(view["type"].unique()), default=sorted(view["type"].unique()))
        shown = view[view["severity"].isin(sev_f) & view["type"].isin(type_f)]
        st.dataframe(shown, hide_index=True, column_config={
            "explanation": st.column_config.TextColumn("explanation", width="large"),
            "change_pct": st.column_config.NumberColumn("change_pct", format="%.1f"),
        })

with tab_charts:
    left, right = st.columns(2)
    with left:
        st.subheader("Severity counts")
        if view.empty:
            st.info("Nothing to plot.")
        else:
            bar = view.groupby(["type", "severity"]).size().reset_index(name="count")
            fig = px.bar(bar, x="type", y="count", color="severity", barmode="stack",
                         category_orders={"severity": list(SEVERITY_LEVELS)}, color_discrete_map=SEV_COLORS)
            st.plotly_chart(fig)
    with right:
        st.subheader("Per-district trend")
        pick = st.selectbox("Indicator", f_inds or indicators, format_func=label)
        sub = df[df["district"].isin(f_districts) & df["month"].isin(f_months)]
        if sub.empty or sub[pick].dropna().empty:
            st.info("No data for the selected filters.")
        else:
            fig = px.line(sub.sort_values("month_idx"), x="month", y=pick, color="district", markers=True,
                          labels={pick: label(pick)})
            fig.update_xaxes(type="category")
            st.plotly_chart(fig)

with tab_corr:
    st.warning(LIMITATIONS)
    sub = df[df["district"].isin(f_districts) & df["month"].isin(f_months)]
    sel = [i for i in f_inds]
    if len(sel) < 2 or len(sub) < 3:
        st.info("Select at least 2 indicators and enough rows (\u22653) to see a correlation matrix.")
    else:
        if len(sub) < 10:
            st.warning(f"Only {len(sub)} rows in the current selection \u2014 treat these correlations as unstable.")
        cm = correlation_matrix(sub, sel)
        fig = px.imshow(cm.rename(index=label, columns=label), zmin=-1, zmax=1, text_auto=".2f",
                        color_continuous_scale="RdBu_r", aspect="auto")
        st.plotly_chart(fig)
        st.caption("Heatmap follows the district/month filters; correlation *insights* always use the full dataset.")
        flagged = res.corr_pairs.assign(a=lambda d: d["a"].map(label), b=lambda d: d["b"].map(label))
        st.subheader(f"Pairs with |r| \u2265 {corr_thr:.2f} (full dataset)")
        st.dataframe(flagged, hide_index=True)

with tab_dl:
    st.caption("Exports apply the current filters. Text cells are formula-injection safe.")
    st.download_button("Insights CSV", insights_csv(view), "insights.csv", "text/csv")
    st.download_button("Insights JSON", insights_json(res, cfg, view), "insights.json", "application/json")
    st.download_button("Correlation matrix CSV", corr_csv(res.corr_matrix), "correlation_matrix.csv", "text/csv")
