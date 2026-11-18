"""Chart Builder engine: spec -> validated data -> Plotly figure."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import pandas as pd
import plotly.express as px
from pydantic import BaseModel, Field

from .profiler import as_datetime
from .query_engine import (
    Filter,
    Metric,
    QueryError,
    QueryResult,
    QuerySpec,
    apply_filters,
    auto_grain,
    col_kind,
    execute_query,
    resolve_column,
)
from .ui import PALETTE

ChartType = Literal["bar", "line", "area", "scatter", "histogram", "box", "pie"]
AGG_CHARTS = {"bar", "line", "area", "pie"}


class ChartError(ValueError):
    pass


class ChartSpec(BaseModel):
    chart_type: ChartType = Field(description="bar, line, area, pie (aggregated) or scatter, histogram, box (raw rows).")
    x: Optional[str] = Field(default=None, description="X-axis / category column (exact name).")
    y: Optional[str] = Field(default=None, description="Value column. Omit to count rows.")
    color: Optional[str] = Field(default=None, description="Optional column that splits the data into series.")
    aggregation: Literal["none", "count", "sum", "mean", "median", "min", "max", "nunique"] = Field(
        default="none", description="Use 'none' only for scatter/histogram/box."
    )
    time_grain: Optional[Literal["day", "week", "month", "quarter", "year"]] = Field(
        default=None, description="Only when x is a date column."
    )
    top_n: Optional[int] = Field(default=None, description="Keep only the top N categories.")
    title: Optional[str] = None


@dataclass
class ChartData:
    table: pd.DataFrame
    spec: ChartSpec
    x_col: Optional[str]
    y_col: Optional[str]
    color_col: Optional[str]
    grain: Optional[str] = None
    notes: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_spec(df: pd.DataFrame, spec: ChartSpec) -> list[str]:
    errs: list[str] = []
    for label, c in (("X", spec.x), ("Y", spec.y), ("Color", spec.color)):
        if c and c not in df.columns:
            errs.append(f"{label} column '{c}' does not exist in the dataset.")
    if errs:
        return errs
    ct, agg = spec.chart_type, spec.aggregation
    kx = col_kind(df[spec.x]) if spec.x else None
    ky = col_kind(df[spec.y]) if spec.y else None

    if ct in AGG_CHARTS:
        if not spec.x:
            errs.append("Choose an X column.")
        if agg == "none":
            errs.append("Choose an aggregation (count, sum, mean, …) — this chart type summarises rows.")
        if agg in ("sum", "mean", "median"):
            if not spec.y:
                errs.append(f"'{agg}' needs a numeric Y column.")
            elif ky != "numeric":
                errs.append(f"Cannot take the {agg} of '{spec.y}': it is not numeric. Pick a numeric column or use count.")
        if agg == "nunique" and not spec.y:
            errs.append("'nunique' needs a Y column to count distinct values of.")
        if agg in ("min", "max") and (not spec.y or ky not in ("numeric", "datetime")):
            errs.append(f"'{agg}' needs a numeric or date Y column.")
        if ct == "pie":
            if kx == "datetime":
                errs.append("Pie charts need a categorical X column, not a date.")
            if agg not in ("sum", "count", "nunique", "none"):
                errs.append("Pie charts only make sense with sum, count or nunique.")
    elif ct == "scatter":
        if not spec.x or not spec.y:
            errs.append("Scatter plots need both X and Y.")
        elif kx != "numeric" or ky != "numeric":
            errs.append("Scatter plots need numeric X and Y columns (convert text numbers in the Cleaning Studio).")
    elif ct == "histogram":
        if not spec.x:
            errs.append("Histograms need an X column.")
        elif kx != "numeric":
            errs.append("Histograms need a numeric X column.")
    elif ct == "box":
        if not spec.y:
            errs.append("Box plots need a numeric Y column.")
        elif ky != "numeric":
            errs.append("Box plots need a numeric Y column.")

    if spec.color and col_kind(df[spec.color]) == "text" and df[spec.color].nunique() > 30:
        errs.append(f"Color column '{spec.color}' has {df[spec.color].nunique()} distinct values; pick one with ≤ 30.")
    return errs


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def prepare_chart_data(df: pd.DataFrame, spec: ChartSpec, filters: list[Filter] | None = None) -> ChartData:
    errs = validate_spec(df, spec)
    if errs:
        raise ChartError("\n".join(errs))
    filters = filters or []
    notes: list = []
    ct = spec.chart_type

    if ct in AGG_CHARTS:
        x = spec.x
        x_is_date = col_kind(df[x]) == "datetime"
        grain = None
        group_by: list = []
        time_col = None
        if x_is_date:
            time_col = x
            grain = spec.time_grain
            if not grain:
                grain = auto_grain(as_datetime(df[x]))
                notes.append(f"Dates grouped by {grain} automatically.")
        else:
            group_by.append(x)
        if spec.color and spec.color != x:
            group_by.append(spec.color)
        agg = spec.aggregation
        alias = "Count" if (agg == "count" and not spec.y) else f"{agg.capitalize()} of {spec.y}"
        metric = Metric(column=spec.y or "*", agg=agg, alias=alias)
        qs = QuerySpec(filters=filters, group_by=group_by, time_column=time_col, time_grain=grain,
                       metrics=[metric], limit=5000)
        try:
            res: QueryResult = execute_query(df, qs, max_rows=5000)
        except QueryError as e:
            raise ChartError(str(e))
        table = res.table
        x_col = res.time_col or x
        y_col = res.metric_cols[0]
        if ct in ("line", "area") and not x_is_date:
            table = table.sort_values(x_col)
        if ct in ("bar", "pie") and not x_is_date:
            n_cats = table[x_col].nunique()
            top_n = spec.top_n or (25 if n_cats > 40 else None)
            if top_n and n_cats > top_n:
                totals = table.groupby(x_col)[y_col].sum().sort_values(ascending=False)
                table = table[table[x_col].isin(totals.head(top_n).index)]
                notes.append(f"Showing the top {top_n} of {n_cats} values of {x}.")
        if table.empty:
            raise ChartError("No rows match the current filters.")
        return ChartData(table.reset_index(drop=True), spec, x_col, y_col, spec.color if spec.color != x else None, grain, notes)

    # ---- raw-row charts
    try:
        work, _ = apply_filters(df, filters)
    except QueryError as e:
        raise ChartError(str(e))
    cols = [c for c in (spec.x, spec.y, spec.color) if c]
    cols = list(dict.fromkeys(cols))
    t = work[cols].copy()
    required = {"histogram": [spec.x], "scatter": [spec.x, spec.y], "box": [spec.y]}[ct]
    t = t.dropna(subset=required)
    if t.empty:
        raise ChartError("No rows match the current filters.")
    cap = 5000 if ct == "scatter" else 50000
    if len(t) > cap:
        t = t.sample(cap, random_state=0)
        notes.append(f"Showing a random sample of {cap:,} rows.")
    if ct == "box" and spec.x and col_kind(t[spec.x]) == "text" and t[spec.x].nunique() > 25:
        top = t[spec.x].value_counts().head(25).index
        t = t[t[spec.x].isin(top)]
        notes.append(f"Showing the 25 most frequent values of {spec.x}.")
    return ChartData(t.reset_index(drop=True), spec, spec.x, spec.y, spec.color, None, notes)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def auto_title(spec: ChartSpec) -> str:
    if spec.title:
        return spec.title
    ct = spec.chart_type
    if ct == "histogram":
        return f"Distribution of {spec.x}"
    if ct == "scatter":
        return f"{spec.y} vs {spec.x}"
    if ct == "box":
        return f"{spec.y}" + (f" by {spec.x}" if spec.x else "")
    what = "Count" if (spec.aggregation == "count" and not spec.y) else f"{spec.aggregation.capitalize()} of {spec.y}"
    return f"{what} by {spec.x}" + (f" and {spec.color}" if spec.color else "")


def _style(fig, title: str):
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=55, b=10),
        template="plotly_white",
        legend_title_text="",
    )
    return fig


def build_figure(cd: ChartData):
    t, s = cd.table, cd.spec
    common = dict(color_discrete_sequence=PALETTE)
    ct = s.chart_type
    if ct == "bar":
        fig = px.bar(t, x=cd.x_col, y=cd.y_col, color=cd.color_col, barmode="group",
                     text_auto=None if cd.color_col else ".3s", **common)
    elif ct == "line":
        fig = px.line(t, x=cd.x_col, y=cd.y_col, color=cd.color_col, markers=True, **common)
    elif ct == "area":
        fig = px.area(t, x=cd.x_col, y=cd.y_col, color=cd.color_col, **common)
    elif ct == "pie":
        fig = px.pie(t, names=cd.x_col, values=cd.y_col, hole=0.35, **common)
    elif ct == "scatter":
        fig = px.scatter(t, x=cd.x_col, y=cd.y_col, color=cd.color_col, opacity=0.7, **common)
    elif ct == "histogram":
        fig = px.histogram(t, x=cd.x_col, color=cd.color_col, nbins=30, barmode="overlay", opacity=0.8, **common)
    else:
        fig = px.box(t, x=cd.x_col, y=cd.y_col, color=cd.color_col, **common)
    return _style(fig, auto_title(s))


def auto_chart(res: QueryResult, title: str = ""):
    """Quick chart for an AI Analyst result table (or None when not useful)."""
    t = res.table
    if len(t) < 2 or not res.metric_cols or not res.group_cols:
        return None
    y = res.metric_cols[0]
    if res.time_col:
        color = next((g for g in res.group_cols if g != res.time_col), None)
        fig = px.line(t, x=res.time_col, y=y, color=color, markers=True, color_discrete_sequence=PALETTE)
    else:
        x = res.group_cols[0]
        color = res.group_cols[1] if len(res.group_cols) > 1 else None
        fig = px.bar(t.head(40), x=x, y=y, color=color, barmode="group", color_discrete_sequence=PALETTE)
    return _style(fig, title or y)


# --------------------------------------------------------------------------- #
# Insights
# --------------------------------------------------------------------------- #
def _fmt_period(ts, grain: str | None) -> str:
    try:
        if grain == "year":
            return f"{ts.year}"
        if grain == "quarter":
            return f"Q{(ts.month - 1) // 3 + 1} {ts.year}"
        if grain == "month":
            return ts.strftime("%b %Y")
        return ts.strftime("%d %b %Y")
    except AttributeError:
        return str(ts)


def simple_insights(cd: ChartData) -> list[str]:
    s, t = cd.spec, cd.table
    if s.chart_type not in ("bar", "line", "area", "pie") or cd.color_col or not cd.y_col or len(t) < 2:
        return []
    out = []
    hi, lo = t.loc[t[cd.y_col].idxmax()], t.loc[t[cd.y_col].idxmin()]
    is_time = pd.api.types.is_datetime64_any_dtype(t[cd.x_col])
    lab = (lambda v: _fmt_period(v, cd.grain)) if is_time else str
    out.append(f"Highest: **{lab(hi[cd.x_col])}** ({hi[cd.y_col]:,.2f}).")
    out.append(f"Lowest: **{lab(lo[cd.x_col])}** ({lo[cd.y_col]:,.2f}).")
    if is_time:
        ts = t.sort_values(cd.x_col)
        last, prev = ts[cd.y_col].iloc[-1], ts[cd.y_col].iloc[-2]
        if prev:
            out.append(f"Latest period vs previous: {100 * (last - prev) / abs(prev):+.1f}%.")
    elif lo[cd.y_col] > 0 and len(t) > 2:
        out.append(f"The top value is {hi[cd.y_col] / lo[cd.y_col]:.1f}× the lowest.")
    return out


# --------------------------------------------------------------------------- #
# Natural language -> ChartSpec
# --------------------------------------------------------------------------- #
def snap_spec(spec: ChartSpec, df: pd.DataFrame) -> ChartSpec:
    """Fix casing/typos in column names and fill sensible aggregation defaults."""
    data = spec.model_dump()
    for k in ("x", "y", "color"):
        if data.get(k):
            try:
                data[k] = resolve_column(df, data[k])
            except QueryError:
                pass
    if data["chart_type"] in AGG_CHARTS and data["aggregation"] == "none":
        data["aggregation"] = "count" if not data.get("y") else "sum"
    if data["chart_type"] in ("scatter", "histogram", "box"):
        data["aggregation"] = "none"
    return ChartSpec(**data)


def nl_to_spec(prompt: str, df: pd.DataFrame, feature_types: dict, llm) -> ChartSpec:
    schema = "\n".join(f"- {c} ({feature_types.get(c, 'Other')})" for c in list(df.columns)[:80])
    system = (
        "Convert the user's chart request into a ChartSpec for the dataset below. "
        "Use ONLY the exact column names listed. Rules: bar/line/area/pie need an aggregation "
        "(sum for money/quantities, mean for averages/rates, count for 'how many'); put the date column on x "
        "for trends and set time_grain from words like daily/weekly/monthly; use color for 'by <category>' splits "
        "on trends; scatter/histogram/box use aggregation 'none'.\n\nColumns:\n" + schema
    )
    structured = llm.with_structured_output(ChartSpec)
    out = structured.invoke([("system", system), ("human", prompt)])
    if isinstance(out, dict):
        out = ChartSpec(**out)
    if out is None:
        raise ChartError("The model could not turn that into a chart. Try naming the columns explicitly.")
    return snap_spec(out, df)
