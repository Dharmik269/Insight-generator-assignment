"""All tunable parameters live here. Config validates itself on creation so that
a bad slider value / CLI flag can never silently produce misleading insights."""
from __future__ import annotations

import math
from dataclasses import dataclass

INSIGHT_TYPES = ("trend", "outlier", "correlation", "threshold_breach")
SEVERITY_LEVELS = ("Low", "Medium", "High")
SEVERITY_RANK = {"Low": 0, "Medium": 1, "High": 2}

# Schema metadata (NOT narratives): which columns are percentages, which
# indicators are "bad when they go up", and acronyms for display labels.
PERCENT_COLUMNS = frozenset({"anc_coverage", "institutional_delivery", "immunization"})
LOWER_IS_BETTER_DEFAULT = frozenset({"high_risk_cases"})
ACRONYMS = frozenset({"anc", "hiv", "tb", "opd", "ipd", "phc", "chc"})


def _finite(name: str, x: float) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"{name} must be a finite number, got {x!r}")
    return float(x)


@dataclass(frozen=True)
class Config:
    # Part B
    trend_threshold_pct: float = 10.0
    # Part C
    outlier_method: str = "iqr"          # "iqr" | "zscore"
    iqr_multiplier: float = 1.5
    z_threshold: float = 3.0
    # Part D
    corr_threshold: float = 0.70
    min_districts_for_corr: int = 10     # assignment: >=10 districts for stability
    # Severity (all data-derived: ratio = observed magnitude / configured threshold)
    sev_medium_ratio: float = 1.25
    sev_high_ratio: float = 1.75
    corr_medium_frac: float = 1 / 3      # share of headroom between threshold and |r|=1
    corr_high_frac: float = 2 / 3
    # Threshold breaches
    breach_scale_pct: float = 10.0       # % beyond target that equals 1 severity unit
    lower_is_better: frozenset = LOWER_IS_BETTER_DEFAULT
    targets: tuple = ()                  # ((indicator, limit), ...) ; empty -> data-derived defaults

    def __post_init__(self) -> None:
        t = _finite("trend_threshold_pct", self.trend_threshold_pct)
        if not 0 < t <= 1000:
            raise ValueError("trend_threshold_pct must be in (0, 1000]")
        if self.outlier_method not in ("iqr", "zscore"):
            raise ValueError("outlier_method must be 'iqr' or 'zscore'")
        if not 0 < _finite("iqr_multiplier", self.iqr_multiplier) <= 10:
            raise ValueError("iqr_multiplier must be in (0, 10]")
        if not 0 < _finite("z_threshold", self.z_threshold) <= 10:
            raise ValueError("z_threshold must be in (0, 10]")
        if not 0 < _finite("corr_threshold", self.corr_threshold) < 1:
            raise ValueError("corr_threshold must be in (0, 1)")
        if not isinstance(self.min_districts_for_corr, int) or self.min_districts_for_corr < 2:
            raise ValueError("min_districts_for_corr must be an integer >= 2")
        m, h = _finite("sev_medium_ratio", self.sev_medium_ratio), _finite("sev_high_ratio", self.sev_high_ratio)
        if not 1 <= m < h:
            raise ValueError("Need 1 <= sev_medium_ratio < sev_high_ratio")
        cm, ch = _finite("corr_medium_frac", self.corr_medium_frac), _finite("corr_high_frac", self.corr_high_frac)
        if not 0 < cm < ch < 1:
            raise ValueError("Need 0 < corr_medium_frac < corr_high_frac < 1")
        if _finite("breach_scale_pct", self.breach_scale_pct) <= 0:
            raise ValueError("breach_scale_pct must be > 0")
        object.__setattr__(self, "lower_is_better", frozenset(self.lower_is_better))
        object.__setattr__(
            self, "targets",
            tuple((str(k), _finite(f"target[{k}]", v)) for k, v in self.targets),
        )

    def target_map(self) -> dict[str, float]:
        return dict(self.targets)

    def as_dict(self) -> dict:
        return {
            "trend_threshold_pct": self.trend_threshold_pct,
            "outlier_method": self.outlier_method,
            "iqr_multiplier": self.iqr_multiplier,
            "z_threshold": self.z_threshold,
            "corr_threshold": self.corr_threshold,
            "min_districts_for_corr": self.min_districts_for_corr,
            "sev_medium_ratio": self.sev_medium_ratio,
            "sev_high_ratio": self.sev_high_ratio,
            "corr_medium_frac": round(self.corr_medium_frac, 4),
            "corr_high_frac": round(self.corr_high_frac, 4),
            "breach_scale_pct": self.breach_scale_pct,
            "lower_is_better": sorted(self.lower_is_better),
            "targets": {k: v for k, v in self.targets},
        }
