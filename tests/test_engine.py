import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from insight_engine import Config, ValidationError, filter_insights, load_csv, run_analysis
from insight_engine.export import insights_csv, insights_json, sanitize_cell

CSV = Path(__file__).parent.parent / "data" / "district_health.csv"
HEADER = "month,district,anc_coverage,institutional_delivery,immunization,high_risk_cases\n"


def run(cfg=None, source=CSV):
    lr = load_csv(source)
    cfg = cfg or Config()
    return lr, cfg, run_analysis(lr.df, lr.indicators, cfg)


def load_text(body: str):
    return load_csv((HEADER + body).encode())


# ------------------------------------------------------------ assignment patterns
def test_ahmedabad_anc_trend_is_high():
    _, _, res = run()
    r = res.insights.query("type=='trend' and entity=='Ahmedabad' and indicator=='anc_coverage'").iloc[0]
    assert (r.value, r.prev_value, r.change_pct, r.severity) == (69.0, 85.0, -18.8, "High")
    assert "18.8%" in r.explanation and "10%" in r.explanation


def test_mehsana_anc_is_outlier_high_and_not_conflated_with_trend():
    _, _, res = run()
    o = res.insights.query("type=='outlier' and entity=='Mehsana' and indicator=='anc_coverage'")
    assert len(o) == 1 and o.iloc[0].severity == "High" and o.iloc[0].value == 42.0
    assert {"trend", "outlier"} <= set(res.insights["type"])


def test_mehsana_high_risk_cases_flagged():
    _, _, res = run()
    t = res.insights.query("type=='trend' and entity=='Mehsana' and indicator=='high_risk_cases'").iloc[0]
    assert t.change_pct == 154.5 and t.severity == "High"


def test_all_four_insight_types_and_required_fields():
    _, _, res = run()
    assert set(res.insights["type"]) == {"trend", "outlier", "correlation", "threshold_breach"}
    assert list(res.insights.columns) == ["insight_id", "type", "indicator", "entity", "period", "value",
                                          "prev_value", "change_pct", "severity", "explanation"]
    assert set(res.insights["severity"]) <= {"Low", "Medium", "High"}
    assert res.insights["insight_id"].is_unique
    assert res.insights["insight_id"].iloc[0] == "INS-0001"


def test_correlation_matrix_matches_pandas_corr():
    lr, _, res = run()
    pd.testing.assert_frame_equal(res.corr_matrix, lr.df[lr.indicators].corr())


def test_correlation_flags_are_marked_low_confidence_with_6_districts():
    _, _, res = run()
    c = res.insights.query("type=='correlation'")
    assert len(c) >= 1 and c["explanation"].str.contains("LOW CONFIDENCE").all()
    assert (c["severity"] != "High").all()


# ------------------------------------------------------------ data-driven, not hardcoded
def test_insights_follow_the_data_not_hardcoded_text():
    body = ("2026-01,Alpha,50,80,90,10\n2026-02,Alpha,75,80,90,10\n2026-03,Alpha,60,80,90,10\n"
            "2026-01,Beta,60,80,90,5\n2026-02,Beta,61,80,90,5\n2026-03,Beta,60,80,90,5\n")
    lr = load_text(body)
    res = run_analysis(lr.df, lr.indicators, Config())
    t = res.insights.query("type=='trend' and indicator=='anc_coverage' and entity=='Alpha'")
    assert sorted(t["change_pct"]) == [-20.0, 50.0]          # computed, 2 consecutive pairs
    assert not res.insights["explanation"].str.contains("Ahmedabad|Mehsana").any()


def test_threshold_is_configurable_and_boundary_inclusive():
    body = "2026-01,A,100,80,90,10\n2026-02,A,90,80,90,10\n"      # exactly -10 %
    lr = load_text(body)
    assert len(run_analysis(lr.df, lr.indicators, Config(trend_threshold_pct=10)).trends.query("is_significant")) == 1
    assert len(run_analysis(lr.df, lr.indicators, Config(trend_threshold_pct=10.5)).trends.query("is_significant")) == 0


def test_severity_scales_with_threshold_not_magic_number():
    lr = load_csv(CSV)
    lo = run_analysis(lr.df, lr.indicators, Config(trend_threshold_pct=10)).insights
    hi = run_analysis(lr.df, lr.indicators, Config(trend_threshold_pct=30)).insights
    a = lambda d: d.query("type=='trend' and entity=='Ahmedabad' and indicator=='anc_coverage'")
    assert a(lo).iloc[0].severity == "High" and a(hi).empty       # 18.8 % no longer significant at 30 %


def test_zscore_method_is_configurable():
    _, _, res = run(Config(outlier_method="zscore", z_threshold=2.5))
    assert (res.outliers["method"] == "zscore").all()
    assert ((res.outliers["z"].abs()) >= 2.5).all()
    assert not run(Config(outlier_method="zscore", z_threshold=3.0))[2].outliers.query("indicator=='anc_coverage'").size


# ------------------------------------------------------------ numeric edge cases
def test_zero_baseline_does_not_divide_by_zero():
    lr = load_text("2026-01,A,50,80,90,0\n2026-02,A,50,80,90,5\n2026-01,B,50,80,90,0\n2026-02,B,50,80,90,0\n")
    res = run_analysis(lr.df, lr.indicators, Config())
    t = res.insights.query("type=='trend' and indicator=='high_risk_cases'")
    assert len(t) == 1 and t.iloc[0].entity == "A" and t.iloc[0].severity == "High"
    assert math.isnan(t.iloc[0].change_pct) and "undefined" in t.iloc[0].explanation
    json.loads(insights_json(res, Config()))                      # still valid JSON (no NaN tokens)


def test_gap_months_are_not_compared():
    lr = load_text("2026-01,A,90,80,90,10\n2026-03,A,50,80,90,10\n")
    assert run_analysis(lr.df, lr.indicators, Config()).trends.empty
    assert any("contiguous" in w for w in lr.warnings)


def test_missing_values_are_reported_not_imputed():
    lr = load_text("2026-01,A,,80,90,10\n2026-02,A,60,80,90,10\n")
    assert int(lr.missing["anc_coverage"]) == 1 and np.isnan(lr.df["anc_coverage"].iloc[0])
    run_analysis(lr.df, lr.indicators, Config())                  # must not crash


def test_constant_column_does_not_crash_or_flag():
    lr = load_text("2026-01,A,50,80,90,10\n2026-02,A,50,80,90,10\n2026-01,B,50,81,90,12\n2026-02,B,50,82,90,11\n")
    res = run_analysis(lr.df, lr.indicators, Config())
    assert res.corr_pairs.query("a=='anc_coverage' or b=='anc_coverage'").empty


def test_deterministic_output():
    a = insights_csv(run()[2].insights)
    b = insights_csv(run()[2].insights)
    assert a == b


# ------------------------------------------------------------ filters
def test_filters():
    _, _, res = run()
    ins = res.insights
    f = filter_insights(ins, districts=["Mehsana"], months=None, indicators=["anc_coverage"])
    assert set(f.query("type!='correlation'")["entity"]) == {"Mehsana"}
    assert (f["indicator"].str.split(":").map(lambda p: set(p) <= {"anc_coverage"})).all()
    assert filter_insights(ins, districts=[], months=None, indicators=None).query("type!='correlation'").empty


# ------------------------------------------------------------ validation / security
@pytest.mark.parametrize("body,msg", [
    ("2026-13,A,50,80,90,10\n", "Invalid month"),
    ("2026-01,A,abc,80,90,10\n", "Non-numeric"),
    ("2026-01,A,nan,80,90,10\n", "Non-numeric"),
    ("2026-01,A,inf,80,90,10\n", "Infinite"),
    ("2026-01,A,150,80,90,10\n", "above 100"),
    ("2026-01,A,-5,80,90,10\n", "Negative"),
    ("2026-01,A,50,80,90,-1\n", "Negative"),
    ("2026-01,A,50,80,90,10\n2026-01,A,51,80,90,10\n", "Duplicate"),
    ("2026-01,A,50,80,90,10\n2026-02,a,51,80,90,10\n", "Inconsistent"),
    ("2026-01,,50,80,90,10\n", "Blank district"),
    ("2026-01,=cmd|' /C calc'!A0,50,80,90,10\n", "Invalid district"),
    ("2026-01,@SUM(1),50,80,90,10\n", "Invalid district"),
    ("2026-01,<script>alert(1)</script>,50,80,90,10\n", "Invalid district"),
])
def test_bad_input_is_rejected(body, msg):
    with pytest.raises(ValidationError, match=msg):
        load_text(body)


@pytest.mark.parametrize("raw", [b"", b"   \n", b"month,district\n2026-01,A\n", b"a,b\n1,2\n",
                                 b"month,district,x,x\n2026-01,A,1,2\n", b"\xff\xfe\x00bad",
                                 HEADER.encode()])
def test_malformed_files_rejected(raw):
    with pytest.raises(ValidationError):
        load_csv(raw)


def test_oversize_file_rejected():
    with pytest.raises(ValidationError, match="larger"):
        load_csv(b"x" * 100, max_bytes=10)


def test_missing_file_rejected():
    with pytest.raises(ValidationError):
        load_csv("does/not/exist.csv")


def test_iso_date_month_format_accepted():
    lr = load_text("2026-07-01,A,50,80,90,10\n2026-08-01,A,40,80,90,10\n")
    assert list(lr.df["month"]) == ["2026-07", "2026-08"]


def test_export_neutralises_formula_injection():
    assert sanitize_cell("=1+1") == "'=1+1" and sanitize_cell("@x") == "'@x"
    assert sanitize_cell("Ahmedabad") == "Ahmedabad" and sanitize_cell(-5.0) == -5.0
    df = pd.DataFrame([dict(insight_id="INS-0001", type="trend", indicator="x", entity="=evil",
                            period="2026-01", value=-1.0, prev_value=1.0, change_pct=-200.0,
                            severity="High", explanation="+cmd")])
    out = insights_csv(df)
    assert "'=evil" in out and "'+cmd" in out and "-1.0" in out  # numbers keep their minus sign


@pytest.mark.parametrize("kw", [dict(trend_threshold_pct=0), dict(trend_threshold_pct=float("nan")),
                                dict(corr_threshold=1.0), dict(corr_threshold=-0.5),
                                dict(outlier_method="eval"), dict(sev_medium_ratio=2, sev_high_ratio=1.5),
                                dict(iqr_multiplier=float("inf")), dict(breach_scale_pct=0)])
def test_invalid_config_rejected(kw):
    with pytest.raises(ValueError):
        Config(**kw)


def test_json_output_is_strict_and_valid():
    lr, cfg, res = run()
    payload = json.loads(insights_json(res, cfg))
    assert payload["insight_count"] == len(res.insights) and "limitations" in payload
