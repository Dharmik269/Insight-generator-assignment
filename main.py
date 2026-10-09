"""Command-line runner: validates the CSV, prints the Part A report, writes outputs.

    python main.py                                  # uses data/district_health.csv
    python main.py --csv other.csv --trend 15 --method zscore
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from insight_engine import Config, ValidationError, load_csv, run_analysis
from insight_engine.export import corr_csv, insights_csv, insights_json
from insight_engine.loader import format_report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--csv", default="data/district_health.csv")
    p.add_argument("--out", default="outputs")
    p.add_argument("--trend", type=float, default=10.0, help="trend threshold %% (default 10)")
    p.add_argument("--method", choices=["iqr", "zscore"], default="iqr")
    p.add_argument("--iqr-mult", type=float, default=1.5)
    p.add_argument("--z", type=float, default=3.0)
    p.add_argument("--corr", type=float, default=0.70)
    a = p.parse_args(argv)
    try:
        cfg = Config(trend_threshold_pct=a.trend, outlier_method=a.method,
                     iqr_multiplier=a.iqr_mult, z_threshold=a.z, corr_threshold=a.corr)
        lr = load_csv(a.csv)
    except (ValidationError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(format_report(lr))
    res = run_analysis(lr.df, lr.indicators, cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "insights.csv").write_text(insights_csv(res.insights), encoding="utf-8")
    (out / "insights.json").write_text(insights_json(res, cfg), encoding="utf-8")
    (out / "correlation_matrix.csv").write_text(corr_csv(res.corr_matrix), encoding="utf-8")
    print(f"\n{len(res.insights)} insights written to {out}/")
    for w in res.warnings:
        print(f"warning: {w}")
    for r in res.insights.itertuples(index=False):
        print(f"{r.insight_id} [{r.severity:<6}] {r.type:<16} {r.explanation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
