"""Validated pandas query engine.

The LLM never writes Python. It fills in a ``QuerySpec`` (filters, group-by,
metrics, ...). This module validates every column, operator and aggregation
against the real DataFrame and then runs *fixed* pandas code. Used by both the
AI Data Analyst and the Chart Builder.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

from .profiler import as_datetime, looks_like_dates


class QueryError(ValueError):
    """A problem with the request that the caller (or the LLM) can fix."""


FilterOp = Literal["==", "!=", ">", ">=", "<", "<=", "in", "not_in", "contains", "between", "is_null", "not_null"]
AggName = Literal["count", "sum", "mean", "median", "min", "max", "std", "nunique"]
Grain = Literal["day", "week", "month", "quarter", "year"]


class Filter(BaseModel):
    column: str = Field(description="Exact column name.")
    op: FilterOp = Field(description="Comparison operator. 'between' is inclusive and needs 2 values.")
    value: Optional[str] = Field(default=None, description="Single value, written as text (e.g. '30', 'Sedan', '2025-03-01').")
    values: Optional[list[str]] = Field(default=None, description="List of values for 'in', 'not_in' and 'between'.")


class Metric(BaseModel):
    column: str = Field(default="*", description="Column to aggregate, or '*' to count rows.")
    agg: AggName = Field(description="Aggregation. sum/mean/median/std need a numeric column.")
    alias: Optional[str] = Field(default=None, description="Optional name for the result column.")


class QuerySpec(BaseModel):
    filters: list[Filter] = Field(default_factory=list, description="Row filters, combined with AND.")
    group_by: list[str] = Field(default_factory=list, description="Columns to group by.")
    time_column: Optional[str] = Field(default=None, description="Date column to group by period (use with time_grain).")
    time_grain: Optional[Grain] = Field(default=None, description="Period for time_column. Defaults to month.")
    metrics: list[Metric] = Field(default_factory=list, description="Aggregations to compute. Empty + group_by => row counts.")
    select_columns: list[str] = Field(default_factory=list, description="Only for raw-row lookups (no metrics/group_by).")
    sort_by: Optional[str] = Field(default=None, description="Result column to sort by.")
    ascending: bool = Field(default=False, description="Sort direction (default descending).")
    limit: int = Field(default=20, description="Max rows to return (1-200).")


@dataclass
class QueryResult:
    table: pd.DataFrame
    explanation: str
    total_rows: int
    group_cols: list
    metric_cols: list
    time_col: Optional[str]


# --------------------------------------------------------------------------- #
# Column helpers
# --------------------------------------------------------------------------- #
def resolve_column(df: pd.DataFrame, name: str) -> str:
    if name in df.columns:
        return name
    lower = {str(c).lower(): c for c in df.columns}
    key = str(name).strip().lower()
    if key in lower:
        return lower[key]
    close = difflib.get_close_matches(str(name), [str(c) for c in df.columns], n=3, cutoff=0.6)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise QueryError(f"Unknown column '{name}'.{hint} Available columns: {list(map(str, df.columns))}")


def col_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if looks_like_dates(s):
        return "datetime"
    return "text"


def truncate_dates(dt: pd.Series, grain: str) -> pd.Series:
    if grain == "day":
        return dt.dt.normalize()
    freq = {"week": "W", "month": "M", "quarter": "Q", "year": "Y"}[grain]
    return dt.dt.to_period(freq).dt.start_time


def auto_grain(dt: pd.Series) -> str:
    d = dt.dropna()
    if d.empty:
        return "month"
    days = (d.max() - d.min()).days
    return "month" if days > 730 else "week" if days > 120 else "day"


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
def _num(v, col):
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        raise QueryError(f"Value '{v}' is not a number, but '{col}' is a numeric column.")


def _ts(v, col):
    try:
        ts = pd.to_datetime(v)
    except (TypeError, ValueError):
        raise QueryError(f"Value '{v}' is not a valid date for column '{col}'.")
    if pd.isna(ts):
        raise QueryError(f"Value '{v}' is not a valid date for column '{col}'.")
    return ts


_DAY = pd.Timedelta(days=1)


def _build_mask(df: pd.DataFrame, f: Filter):
    col = resolve_column(df, f.column)
    s = df[col]
    kind = col_kind(s)
    op = f.op
    if op == "is_null":
        return s.isna(), f"{col} is missing"
    if op == "not_null":
        return s.notna(), f"{col} is not missing"
    vals = list(f.values) if f.values else ([f.value] if f.value is not None else [])
    if not vals:
        raise QueryError(f"Filter on '{col}' with operator '{op}' needs a value.")

    if op in ("in", "not_in"):
        if kind == "numeric":
            m = s.isin([_num(v, col) for v in vals])
        else:
            m = s.astype(object).astype(str).str.strip().str.lower().isin([str(v).strip().lower() for v in vals])
        return (~m if op == "not_in" else m), f"{col} {op.replace('_', ' ')} {vals}"

    if op == "contains":
        m = s.astype(object).astype(str).str.contains(str(vals[0]), case=False, regex=False, na=False)
        return m, f"{col} contains '{vals[0]}'"

    if op == "between":
        if len(vals) != 2:
            raise QueryError(f"'between' on '{col}' needs exactly two values (low, high).")
        if kind == "numeric":
            lo, hi = _num(vals[0], col), _num(vals[1], col)
            return s.between(lo, hi), f"{col} between {lo:g} and {hi:g}"
        if kind == "datetime":
            dt = as_datetime(s)
            lo, hi = _ts(vals[0], col), _ts(vals[1], col)
            upper = (dt < hi + _DAY) if hi == hi.normalize() else (dt <= hi)
            return (dt >= lo) & upper, f"{col} between {lo.date()} and {hi.date()}"
        raise QueryError(f"'between' only works on numeric or date columns; '{col}' is {kind}.")

    v = vals[0]
    if kind == "numeric":
        x = _num(v, col)
        ops = {"==": s == x, "!=": (s != x) & s.notna(), ">": s > x, ">=": s >= x, "<": s < x, "<=": s <= x}
        return ops[op], f"{col} {op} {x:g}"
    if kind == "datetime":
        dt = as_datetime(s)
        ts = _ts(v, col)
        date_only = ts == ts.normalize()
        eq = (dt.dt.normalize() == ts.normalize()) if date_only else (dt == ts)
        ops = {
            "==": eq,
            "!=": ~eq & dt.notna(),
            ">": (dt >= ts + _DAY) if date_only else (dt > ts),
            ">=": dt >= ts,
            "<": dt < ts,
            "<=": (dt < ts + _DAY) if date_only else (dt <= ts),
        }
        return ops[op], f"{col} {op} {ts.date() if date_only else ts}"
    if kind == "bool":
        if op not in ("==", "!="):
            raise QueryError(f"Operator '{op}' is not valid for boolean column '{col}'.")
        truth = str(v).strip().lower() in {"true", "1", "yes", "y", "t"}
        eq = s == truth
        return (eq if op == "==" else (~eq & s.notna())), f"{col} {op} {truth}"
    if op not in ("==", "!="):
        raise QueryError(f"Operator '{op}' is not valid for text column '{col}'. Use ==, !=, in or contains.")
    norm = s.astype(object).astype(str).str.strip().str.lower()
    eq = norm == str(v).strip().lower()
    return (eq & s.notna() if op == "==" else (~eq & s.notna())), f"{col} {op} '{v}'"


def apply_filters(df: pd.DataFrame, filters: list[Filter]):
    notes = []
    out = df
    for f in filters:
        f = f if isinstance(f, Filter) else Filter.model_validate(f)
        mask, note = _build_mask(out, f)
        out = out.loc[mask.fillna(False)]
        notes.append(note)
    return out, notes


# --------------------------------------------------------------------------- #
# Query execution
# --------------------------------------------------------------------------- #
_NUMERIC_AGGS = {"sum", "mean", "median", "std"}


def _scalar_agg(s: pd.Series, agg: str):
    if agg == "count":
        return int(s.count())
    if agg == "nunique":
        return int(s.nunique())
    return getattr(s, agg)()


def execute_query(df: pd.DataFrame, spec: QuerySpec, max_rows: int = 200) -> QueryResult:
    spec = spec if isinstance(spec, QuerySpec) else QuerySpec.model_validate(spec)
    base = df.copy(deep=False)
    n_before = len(base)

    # ---- group columns / time grain
    group_cols = [resolve_column(base, c) for c in spec.group_by]
    period_col = None
    grain = None
    if spec.time_column:
        tc = resolve_column(base, spec.time_column)
        if col_kind(base[tc]) != "datetime":
            raise QueryError(f"'{tc}' is not a date/time column, so it cannot be grouped by period.")
        dt = as_datetime(base[tc])
        grain = spec.time_grain or "month"
        period_col = f"{tc} ({grain})"
        base[period_col] = truncate_dates(dt, grain)
        group_cols = [period_col] + [c for c in group_cols if c != tc]

    # ---- metrics (validated before anything runs)
    metric_defs = []
    for m in spec.metrics:
        if m.column == "*":
            if m.agg != "count":
                raise QueryError("Column '*' can only be used with agg 'count' (row count).")
            col = "*"
        else:
            col = resolve_column(base, m.column)
            kind = col_kind(base[col])
            if m.agg in _NUMERIC_AGGS and kind != "numeric":
                raise QueryError(
                    f"Cannot compute {m.agg} of '{col}': it is a {kind} column, not numeric. "
                    "Use count or nunique, or pick a numeric column."
                )
            if m.agg in ("min", "max") and kind not in ("numeric", "datetime"):
                raise QueryError(f"min/max need a numeric or date column; '{col}' is {kind}.")
            if kind == "datetime" and not pd.api.types.is_datetime64_any_dtype(base[col]):
                base[col] = as_datetime(base[col])
        alias = m.alias or ("row_count" if col == "*" else f"{m.agg}_{col}")
        while alias in group_cols:
            alias += "_value"
        metric_defs.append((alias, col, m.agg))

    if group_cols and not metric_defs:
        metric_defs = [("row_count", "*", "count")]

    work, filter_notes = apply_filters(base, spec.filters)
    n_after = len(work)
    limit = max(1, min(int(spec.limit or 20), max_rows))
    lines = [f"Rows analysed: {n_after:,} of {n_before:,}" + (" (after filters)" if filter_notes else "")]
    lines.append("Filters: " + "; ".join(filter_notes) if filter_notes else "Filters: none")

    # ---- run
    if group_cols:
        work = work.assign(__n__=1)
        named = {}
        for alias, col, agg in metric_defs:
            named[alias] = ("__n__", "sum") if col == "*" else (col, agg)
        table = work.groupby(group_cols, dropna=False, observed=True).agg(**named).reset_index()
        for c in group_cols:
            if c != period_col and table[c].isna().any():
                table[c] = table[c].astype(object).where(table[c].notna(), "(missing)")
        gtxt = ", ".join(group_cols)
        lines.append(f"Grouped by: {gtxt}")
    elif metric_defs:
        row = {}
        for alias, col, agg in metric_defs:
            row[alias] = int(len(work)) if col == "*" else _scalar_agg(work[col], agg)
        table = pd.DataFrame([row])
    else:
        cols = [resolve_column(work, c) for c in spec.select_columns] or list(work.columns)
        table = work[cols].copy()
        lines.append("Returned raw rows")

    if metric_defs:
        lines.append(
            "Metrics: "
            + "; ".join(f"{'row count' if c == '*' else f'{a} of {c}'} → {al}" for al, c, a in metric_defs)
        )

    # ---- sort
    if spec.sort_by:
        sb = resolve_column(table, spec.sort_by)
        table = table.sort_values(sb, ascending=spec.ascending, kind="stable")
        lines.append(f"Sorted by {sb} ({'ascending' if spec.ascending else 'descending'})")
    elif period_col:
        table = table.sort_values(group_cols, ascending=True, kind="stable")
    elif group_cols and metric_defs:
        table = table.sort_values(metric_defs[0][0], ascending=spec.ascending, kind="stable")
        lines.append(f"Sorted by {metric_defs[0][0]} ({'ascending' if spec.ascending else 'descending'})")

    total = len(table)
    table = table.head(limit).reset_index(drop=True)
    if total > limit:
        lines.append(f"Showing {limit} of {total:,} rows")

    return QueryResult(
        table=table,
        explanation="\n".join(f"- {l}" for l in lines),
        total_rows=total,
        group_cols=group_cols,
        metric_cols=[a for a, _, _ in metric_defs],
        time_col=period_col,
    )
