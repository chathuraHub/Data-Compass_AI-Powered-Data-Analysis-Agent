"""Dataset profiling: feature-type inference, data-quality checks and summaries.

Pure pandas / numpy (no Streamlit), so every function is easy to unit-test.
Pandas dtype and *feature type* are deliberately separate concepts: a booking ID
stored as text has dtype ``object``/``str`` but feature type ``Identifier``.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NUMERICAL = "Numerical"
CATEGORICAL = "Categorical"
DATETIME = "Date/time"
BOOLEAN = "Boolean"
IDENTIFIER = "Identifier"
TEXT = "Text"
OTHER = "Other"

_ID_TOKENS = {"id", "uuid", "guid", "key", "idx"}
_CAMEL_ID = re.compile(r"[a-z0-9](ID|Id)$")
_BOOL_WORDS = {"true", "false", "yes", "no", "y", "n", "t", "f", "0", "1"}
_DATE_RE = re.compile(
    r"^\s*("
    r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
    r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
    r"|\d{1,2}\s+[A-Za-z]{3,9}\.?,?\s+\d{2,4}"
    r"|[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}"
    r")([ T]\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?)?\s*(Z|[+-]\d{2}:?\d{2})?\s*$"
)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def is_text_dtype(s: pd.Series) -> bool:
    """True for object / string / category columns (works on pandas 2 and 3)."""
    return (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    )


def safe_to_datetime(s: pd.Series) -> pd.Series:
    """Parse dates leniently without noisy warnings. Unparseable values -> NaT."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(s, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            return pd.to_datetime(s, errors="coerce")


def as_datetime(s: pd.Series) -> pd.Series:
    return s if pd.api.types.is_datetime64_any_dtype(s) else safe_to_datetime(s)


def as_numeric(s: pd.Series) -> pd.Series:
    """Numeric view of a column (handles numbers stored as text, e.g. '1,200')."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    cleaned = s.astype("string").str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def safe_duplicated(df: pd.DataFrame, keep="first") -> pd.Series:
    try:
        return df.duplicated(keep=keep)
    except TypeError:  # unhashable cells such as lists / dicts
        return df.astype(str).duplicated(keep=keep)


def _sample(s: pd.Series, n: int = 5000) -> pd.Series:
    return s if len(s) <= n else s.sample(n, random_state=0)


def looks_like_dates(s: pd.Series) -> bool:
    """True when a text column mostly contains parseable date strings."""
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return False
    vals = _sample(s.dropna()).astype(str)
    if vals.empty:
        return False
    if vals.map(lambda v: bool(_DATE_RE.match(v))).mean() < 0.9:
        return False
    return safe_to_datetime(vals).notna().mean() >= 0.9


def numeric_text_ratio(s: pd.Series) -> float:
    vals = _sample(s.dropna()).astype(str).str.strip()
    if vals.empty:
        return 0.0
    cleaned = vals.str.replace(",", "", regex=False)
    return float(pd.to_numeric(cleaned, errors="coerce").notna().mean())


def _name_is_id(name) -> bool:
    name = str(name)
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", name.lower()) if t]
    return bool(_ID_TOKENS.intersection(tokens)) or bool(_CAMEL_ID.search(name))


# --------------------------------------------------------------------------- #
# Feature type inference
# --------------------------------------------------------------------------- #
def infer_feature_type(name, s: pd.Series) -> str:
    try:
        nn = s.dropna()
        n = len(nn)
        if n == 0:
            return OTHER
        if pd.api.types.is_bool_dtype(s):
            return BOOLEAN
        if pd.api.types.is_datetime64_any_dtype(s):
            return DATETIME
        if pd.api.types.is_timedelta64_dtype(s) or pd.api.types.is_complex_dtype(s):
            return OTHER

        nunique = nn.nunique()
        uniq_ratio = nunique / n

        if (not pd.api.types.is_float_dtype(s)) and _name_is_id(name) and uniq_ratio >= 0.5:
            return IDENTIFIER

        if pd.api.types.is_numeric_dtype(s):
            if nunique <= 2 and set(nn.unique()) <= {0, 1}:
                return BOOLEAN
            if (
                pd.api.types.is_integer_dtype(s)
                and uniq_ratio == 1
                and n >= 20
                and (nn.is_monotonic_increasing or nn.is_monotonic_decreasing)
            ):
                return IDENTIFIER
            return NUMERICAL

        # text-like columns from here on
        text = nn.astype(object).astype(str)
        if looks_like_dates(nn):
            return DATETIME
        if nunique <= 2 and {v.strip().lower() for v in text.unique()} <= _BOOL_WORDS:
            return BOOLEAN
        if numeric_text_ratio(nn) >= 0.98:
            return NUMERICAL
        samp = _sample(text)
        avg_len = samp.str.len().mean()
        space_share = samp.str.contains(r"\s", regex=True).mean()
        if uniq_ratio >= 0.98 and n >= 20 and avg_len <= 24 and space_share < 0.2:
            return IDENTIFIER
        if avg_len > 30 and uniq_ratio > 0.5:
            return TEXT
        return CATEGORICAL
    except (TypeError, ValueError):
        return OTHER


# --------------------------------------------------------------------------- #
# Outliers (IQR rule)
# --------------------------------------------------------------------------- #
def iqr_stats(s: pd.Series) -> dict:
    x = as_numeric(s).dropna()
    empty = {"count": 0, "pct": 0.0, "lower": np.nan, "upper": np.nan, "q1": np.nan, "q3": np.nan}
    if len(x) < 4:
        return empty
    q1, q3 = float(x.quantile(0.25)), float(x.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (x < lower) | (x > upper)
    return {
        "count": int(mask.sum()),
        "pct": round(100 * float(mask.mean()), 2),
        "lower": lower,
        "upper": upper,
        "q1": q1,
        "q3": q3,
    }


def outlier_mask(s: pd.Series) -> pd.Series:
    """Boolean mask (aligned to s) of values outside the IQR fences."""
    x = as_numeric(s)
    st = iqr_stats(x)
    if np.isnan(st["lower"]):
        return pd.Series(False, index=s.index)
    return ((x < st["lower"]) | (x > st["upper"])).fillna(False)


# --------------------------------------------------------------------------- #
# Profile object
# --------------------------------------------------------------------------- #
@dataclass
class Profile:
    overview: dict
    columns: pd.DataFrame
    feature_types: dict
    outliers: pd.DataFrame
    missing: pd.DataFrame
    issues: list = field(default_factory=list)
    date_text_cols: list = field(default_factory=list)
    inconsistent: dict = field(default_factory=dict)

    def cols_of(self, *types: str) -> list:
        return [c for c, t in self.feature_types.items() if t in types]


def _category_variants(s: pd.Series) -> tuple[int, int]:
    """(distinct raw values, distinct after trimming/lower-casing)."""
    vals = s.dropna().astype(object).astype(str).unique()
    norm = {re.sub(r"\s+", " ", v.strip().lower()) for v in vals}
    return len(vals), len(norm)


def profile_dataset(df: pd.DataFrame) -> Profile:
    n_rows, n_cols = df.shape
    miss = df.isna().sum()
    feature_types: dict = {}
    rows: list = []
    outlier_rows: list = []
    type_concerns: dict = {}
    date_text: list = []
    inconsistent: dict = {}
    constant: list = []
    high_missing: list = []

    for col in df.columns:
        s = df[col]
        nn = s.dropna()
        ftype = infer_feature_type(col, s)
        feature_types[col] = ftype
        n_nonnull = len(nn)
        try:
            distinct = int(nn.nunique())
        except TypeError:
            distinct = int(nn.astype(str).nunique())
        uniq_pct = round(100 * distinct / n_nonnull, 1) if n_nonnull else 0.0
        miss_n = int(miss[col])
        miss_pct = round(100 * miss_n / n_rows, 2) if n_rows else 0.0

        flags: list = []
        text_like = is_text_dtype(s)

        if miss_n:
            if miss_pct > 30:
                flags.append(f"High missingness ({miss_pct}%)")
                high_missing.append(col)
            else:
                flags.append(f"Missing {miss_pct}%")
        if n_nonnull and distinct <= 1:
            flags.append("Constant value")
            constant.append(col)

        if text_like and n_nonnull:
            if ftype == NUMERICAL:
                flags.append("Numbers stored as text")
                type_concerns.setdefault(col, []).append("numbers stored as text")
            elif ftype == DATETIME:
                flags.append("Dates stored as text")
                date_text.append(col)
                type_concerns.setdefault(col, []).append("dates stored as text")
            else:
                ratio = numeric_text_ratio(nn)
                if 0.5 <= ratio < 0.98:
                    flags.append("Mixed numeric/text values")
                    type_concerns.setdefault(col, []).append("mixed numeric and text values")
            if pd.api.types.is_object_dtype(s):
                n_types = _sample(nn).map(type).nunique()
                if n_types > 1:
                    flags.append("Mixed Python types")
                    type_concerns.setdefault(col, []).append("mixed Python types")
            if ftype == CATEGORICAL:
                raw, norm = _category_variants(nn)
                if norm < raw:
                    flags.append(f"Inconsistent spelling/casing ({raw - norm} extra variants)")
                    inconsistent[col] = raw - norm

        if ftype == NUMERICAL:
            o = iqr_stats(s)
            x = as_numeric(s).dropna()
            outlier_rows.append(
                {
                    "Column": col,
                    "Outliers": o["count"],
                    "Outlier %": o["pct"],
                    "Lower bound": o["lower"],
                    "Upper bound": o["upper"],
                    "Q1": o["q1"],
                    "Q3": o["q3"],
                    "Min": float(x.min()) if len(x) else np.nan,
                    "Max": float(x.max()) if len(x) else np.nan,
                }
            )
            if o["count"]:
                flags.append(f"Possible outliers ({o['count']})")

        examples = ", ".join(str(v)[:18] for v in nn.astype(object).unique()[:3]) if n_nonnull else ""
        rows.append(
            {
                "Column": col,
                "Pandas dtype": str(s.dtype),
                "Feature type": ftype,
                "Missing": miss_n,
                "Missing %": miss_pct,
                "Distinct": distinct,
                "Unique %": uniq_pct,
                "Examples": examples,
                "Quality flags": " · ".join(flags),
            }
        )

    columns = pd.DataFrame(rows)
    outliers = pd.DataFrame(outlier_rows)
    missing_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Missing": [int(miss[c]) for c in df.columns],
            "Missing %": [round(100 * int(miss[c]) / n_rows, 2) if n_rows else 0.0 for c in df.columns],
        }
    )

    total_cells = n_rows * n_cols
    missing_cells = int(miss.sum())
    dup_rows = int(safe_duplicated(df).sum()) if n_rows else 0
    counts = pd.Series(feature_types).value_counts()
    other_n = sum(int(counts.get(t, 0)) for t in (IDENTIFIER, TEXT, OTHER))
    overview = {
        "rows": n_rows,
        "columns": n_cols,
        "missing_cells": missing_cells,
        "missing_pct": round(100 * missing_cells / total_cells, 2) if total_cells else 0.0,
        "duplicate_rows": dup_rows,
        "duplicate_pct": round(100 * dup_rows / n_rows, 2) if n_rows else 0.0,
        "n_numerical": int(counts.get(NUMERICAL, 0)),
        "n_categorical": int(counts.get(CATEGORICAL, 0)),
        "n_datetime": int(counts.get(DATETIME, 0)),
        "n_boolean": int(counts.get(BOOLEAN, 0)),
        "n_other": other_n,
        "n_identifier": int(counts.get(IDENTIFIER, 0)),
        "n_text": int(counts.get(TEXT, 0)),
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1024**2, 3),
    }

    issues: list = []
    cols_with_missing = [c for c in df.columns if miss[c] > 0]
    if cols_with_missing:
        issues.append(
            {
                "key": "missing",
                "severity": "warning",
                "title": "Missing values",
                "detail": f"{overview['missing_pct']}% of cells missing in {len(cols_with_missing)} column(s).",
                "columns": cols_with_missing,
                "count": missing_cells,
            }
        )
    if dup_rows:
        issues.append(
            {
                "key": "duplicates",
                "severity": "warning",
                "title": "Duplicate rows",
                "detail": f"{dup_rows} fully duplicated row(s) ({overview['duplicate_pct']}% of rows).",
                "columns": [],
                "count": dup_rows,
            }
        )
    if type_concerns:
        issues.append(
            {
                "key": "types",
                "severity": "warning",
                "title": "Potentially inconsistent data types",
                "detail": "; ".join(f"{c}: {', '.join(v)}" for c, v in type_concerns.items()),
                "columns": list(type_concerns),
                "count": len(type_concerns),
            }
        )
    out_cols = outliers.loc[outliers["Outliers"] > 0, "Column"].tolist() if len(outliers) else []
    if out_cols:
        issues.append(
            {
                "key": "outliers",
                "severity": "info",
                "title": "Numerical columns with potential outliers",
                "detail": "IQR rule flags values for investigation; they are not necessarily errors.",
                "columns": out_cols,
                "count": int(outliers["Outliers"].sum()),
            }
        )
    if inconsistent:
        issues.append(
            {
                "key": "categories",
                "severity": "info",
                "title": "Inconsistent categories",
                "detail": "Same category written with different casing or spacing.",
                "columns": list(inconsistent),
                "count": int(sum(inconsistent.values())),
            }
        )
    if constant:
        issues.append(
            {
                "key": "constant",
                "severity": "info",
                "title": "Constant columns",
                "detail": "These columns hold a single value and carry no information.",
                "columns": constant,
                "count": len(constant),
            }
        )

    return Profile(
        overview=overview,
        columns=columns,
        feature_types=feature_types,
        outliers=outliers,
        missing=missing_df,
        issues=issues,
        date_text_cols=date_text,
        inconsistent=inconsistent,
    )


# --------------------------------------------------------------------------- #
# Summaries used by the Overview page
# --------------------------------------------------------------------------- #
def numeric_summary(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    rows = []
    for c in cols:
        x = as_numeric(df[c]).dropna()
        if x.empty:
            continue
        rows.append(
            {
                "Column": c,
                "Count": int(x.count()),
                "Mean": x.mean(),
                "Median": x.median(),
                "Std dev": x.std(),
                "Min": x.min(),
                "Q1": x.quantile(0.25),
                "Q3": x.quantile(0.75),
                "Max": x.max(),
                "Skewness": x.skew(),
            }
        )
    return pd.DataFrame(rows)


def categorical_frequency(s: pd.Series, top: int = 25) -> tuple[pd.DataFrame, int]:
    vals = s.dropna().astype(object).astype(str)
    vc = vals.value_counts()
    total = int(vc.sum())
    table = pd.DataFrame(
        {
            "Category": vc.index,
            "Count": vc.values,
            "Percent": (vc.values / total * 100).round(2) if total else [],
        }
    )
    return table.head(top), int(len(vc))


def datetime_summary(s: pd.Series) -> dict:
    dt = as_datetime(s).dropna()
    if dt.empty:
        return {}
    return {
        "earliest": dt.min(),
        "latest": dt.max(),
        "range_days": int((dt.max() - dt.min()).days),
        "valid": int(len(dt)),
    }


def records_per_period(s: pd.Series, grain: str = "M") -> pd.Series:
    dt = as_datetime(s).dropna()
    if dt.empty:
        return pd.Series(dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        per = dt.dt.to_period(grain).value_counts().sort_index()
        full = pd.period_range(per.index.min(), per.index.max(), freq=grain)
    if len(full) <= 5000:
        per = per.reindex(full, fill_value=0)
    per.index = per.index.to_timestamp()
    return per


def correlation_matrix(df: pd.DataFrame, cols: list, method: str = "pearson") -> pd.DataFrame:
    num = pd.DataFrame({c: as_numeric(df[c]) for c in cols})
    num = num.loc[:, num.nunique(dropna=True) > 1]
    return num.corr(method=method)
