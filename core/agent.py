"""AI Data Analyst: LangChain tools + LangGraph agent + Groq model.

Tools are created per request (``build_tools``) with the DataFrame captured in a
closure, so there is no global state. Every tool runs fixed, validated pandas
code — the model never executes arbitrary Python. Each tool also pushes its
computed table / explanation / chart into ``sink`` so the UI can show real
result tables next to the model's written answer.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq

from .chart_engine import auto_chart
from .profiler import NUMERICAL, as_numeric, categorical_frequency, datetime_summary
from .query_engine import QueryError, QuerySpec, execute_query, resolve_column

DEFAULT_MODEL = "openai/gpt-oss-120b"


# --------------------------------------------------------------------------- #
# Model helpers
# --------------------------------------------------------------------------- #
def groq_ready() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def get_llm(temperature: float = 0) -> ChatGroq:
    return ChatGroq(model=os.getenv("GROQ_MODEL", DEFAULT_MODEL), temperature=temperature)


def friendly_error(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()
    if "api key" in low or "401" in low or "invalid_api_key" in low:
        return "The Groq API key was rejected. Check GROQ_API_KEY in your .env file."
    if ("rate" in low and "limit" in low) or "429" in low:
        return "Groq's rate limit was reached. Wait a moment and try again."
    if "recursion" in low:
        return "The analysis took too many steps. Try a simpler or more specific question."
    return f"The analyst hit an error: {msg[:300]}"


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@dataclass
class ToolResult:
    title: str
    table: pd.DataFrame
    explanation: str
    chart: object = None


def _as_text(table: pd.DataFrame, max_rows: int = 40) -> str:
    t = table.head(max_rows).copy()
    for c in t.columns:
        if pd.api.types.is_float_dtype(t[c]):
            t[c] = t[c].round(4)
    return t.to_string(index=False)


def build_tools(df: pd.DataFrame, feature_types: dict, sink: list):
    @tool
    def dataset_info() -> str:
        """Show the dataset shape and every column with its pandas dtype, feature type and missing count."""
        rows = [
            {"column": c, "dtype": str(df[c].dtype), "feature_type": feature_types.get(c, "Other"),
             "missing": int(df[c].isna().sum()), "distinct": int(df[c].nunique())}
            for c in df.columns
        ]
        t = pd.DataFrame(rows)
        sink.append(ToolResult("Dataset columns", t, f"- Shape: {df.shape[0]:,} rows × {df.shape[1]} columns"))
        return f"Shape: {df.shape}\n{_as_text(t, 100)}"

    @tool
    def run_query(spec: QuerySpec) -> str:
        """Filter, group, aggregate, rank or build time series from the dataset.

        Examples:
        - Average value per category: group_by=['category'], metrics=[{column:'value', agg:'mean'}]
        - Rows per category: group_by=['category'] (no metrics => row counts)
        - Monthly revenue: time_column='date', time_grain='month', metrics=[{column:'revenue', agg:'sum'}]
        - Filtered lookup: filters=[{column:'status', op:'==', value:'Completed'}]
        Text matching is case-insensitive. Always use this tool for numbers.
        """
        try:
            q = spec if isinstance(spec, QuerySpec) else QuerySpec.model_validate(spec)
            res = execute_query(df, q)
        except (QueryError, ValueError) as e:
            return f"ERROR: {e}"
        title = "Query result" if not res.group_cols else "Result by " + ", ".join(res.group_cols)
        sink.append(ToolResult(title, res.table, res.explanation, auto_chart(res, title)))
        return f"{res.explanation}\n\nRESULT ({res.total_rows} rows):\n{_as_text(res.table)}"

    @tool
    def column_summary(column: str) -> str:
        """Summarise ONE column: statistics for numeric, top values for categorical, range for dates."""
        try:
            col = resolve_column(df, column)
        except QueryError as e:
            return f"ERROR: {e}"
        s = df[col]
        ftype = feature_types.get(col, "Other")
        if ftype == NUMERICAL:
            x = as_numeric(s).dropna()
            t = x.describe().round(4).rename("value").reset_index().rename(columns={"index": "statistic"})
            expl = f"- Descriptive statistics of {col} ({int(s.isna().sum())} missing values excluded)"
        elif ftype == "Date/time":
            info = datetime_summary(s)
            t = pd.DataFrame({"statistic": ["earliest", "latest", "range_days"],
                              "value": [str(info.get("earliest")), str(info.get("latest")), info.get("range_days")]})
            expl = f"- Date range of {col}"
        else:
            t, n = categorical_frequency(s, top=20)
            expl = f"- Top values of {col} ({n} distinct values)"
        sink.append(ToolResult(f"Summary of {col}", t, expl))
        return f"{expl}\n{_as_text(t)}"

    @tool
    def correlation(column_a: str, column_b: str, method: Literal["pearson", "spearman"] = "pearson") -> str:
        """Correlation between two numeric columns (pearson or spearman). Correlation is not causation."""
        try:
            a, b = resolve_column(df, column_a), resolve_column(df, column_b)
        except QueryError as e:
            return f"ERROR: {e}"
        if feature_types.get(a) != NUMERICAL or feature_types.get(b) != NUMERICAL:
            return "ERROR: both columns must be numeric."
        pair = pd.DataFrame({a: as_numeric(df[a]), b: as_numeric(df[b])}).dropna()
        if len(pair) < 3:
            return "ERROR: not enough rows with both values."
        r = float(pair[a].corr(pair[b], method=method))
        t = pd.DataFrame({"column_a": [a], "column_b": [b], "method": [method], "correlation": [round(r, 4)],
                          "rows_used": [len(pair)]})
        sink.append(ToolResult(f"Correlation: {a} vs {b}", t, f"- {method.title()} correlation on {len(pair):,} complete rows"))
        return _as_text(t)

    return [dataset_info, run_query, column_summary, correlation]


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
def schema_text(df: pd.DataFrame, feature_types: dict, max_cols: int = 60) -> str:
    lines = []
    for c in list(df.columns)[:max_cols]:
        nn = df[c].dropna()
        ex = ", ".join(str(v)[:20] for v in nn.astype(object).unique()[:3])
        lines.append(f"- {c} | dtype={df[c].dtype} | type={feature_types.get(c, 'Other')} | e.g. {ex}")
    if df.shape[1] > max_cols:
        lines.append(f"... and {df.shape[1] - max_cols} more columns (call dataset_info)")
    return f"{len(df):,} rows × {df.shape[1]} columns\n" + "\n".join(lines)


SYSTEM_PROMPT = """You are Data Compass's data analyst. You answer questions about ONE uploaded dataset.

DATASET
{schema}

RULES
1. Every number you report must come from a tool result. Never calculate, estimate or guess numbers yourself.
2. Use run_query for filtering, grouping, ranking, aggregation and time series. Use column_summary for one column,
   correlation for two numeric columns, and dataset_info only if the schema above is not enough.
3. Use the exact column names above. For date columns use time_column + time_grain to get daily/weekly/monthly results.
4. If a filter depends on a category value you are unsure about, call column_summary on that column first.
5. If a tool returns ERROR, read it, fix your call and retry (at most twice), then explain the problem plainly.
6. Answer concisely: lead with the direct answer, then 1-3 sentences of context. State the filters you applied and any
   caveats (missing values, small groups). Result tables and charts are shown to the user automatically, so do not
   re-print whole tables.
7. Correlation shows association, not causation. Say so when you report one.
8. If the question cannot be answered from this data, say so instead of guessing.
"""


@dataclass
class AnalystResponse:
    answer: str
    results: list = field(default_factory=list)
    error: str | None = None


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def run_analyst(question: str, df: pd.DataFrame, feature_types: dict, memory, thread_id: str) -> AnalystResponse:
    if not groq_ready():
        return AnalystResponse("", error="GROQ_API_KEY is missing. Add it to your .env file and restart the app.")
    sink: list[ToolResult] = []
    try:
        agent = create_agent(
            model=get_llm(),
            tools=build_tools(df, feature_types, sink),
            system_prompt=SYSTEM_PROMPT.format(schema=schema_text(df, feature_types)),
            checkpointer=memory,
        )
        out = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            {"configurable": {"thread_id": thread_id}, "recursion_limit": 16},
        )
        return AnalystResponse(_content_to_text(out["messages"][-1].content), results=sink)
    except Exception as e:  # keep the chat alive whatever the provider returns
        return AnalystResponse("", results=sink, error=friendly_error(e))
