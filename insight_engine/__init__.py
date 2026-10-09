"""Auto-Analytics Engine: trends, outliers, correlations and threshold breaches
for district-level healthcare indicators, turned into structured insights."""
from .config import Config
from .loader import ValidationError, load_csv
from .insights import run_analysis, filter_insights

__all__ = ["Config", "ValidationError", "load_csv", "run_analysis", "filter_insights"]
