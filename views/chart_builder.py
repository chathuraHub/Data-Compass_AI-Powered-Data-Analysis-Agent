"""Page 4 — Chart Builder: natural-language prompt or manual controls -> validated Plotly chart."""
import streamlit as st

from core import ui
from core.state import cached_profile
from core.agent import get_llm, groq_ready, friendly_error
from core.chart_engine import (
    ChartError, ChartSpec, build_figure, nl_to_spec, prepare_chart_data, simple_insights, validate_spec,
)
from core.profiler import BOOLEAN, CATEGORICAL, DATETIME, NUMERICAL, as_datetime, as_numeric
from core.query_engine import Filter

df = ui.require_data()
profile = cached_profile(df)
ftypes = profile.feature_types
ss = st.session_state
W = ui.stretch()

ui.page_header("Chart Builder", "Describe a chart in words, then fine-tune columns, aggregation and filters. Every chart is validated before it is drawn.", icon_name="chart")
ui.source_badge()
st.write("")

NONE = "(none)"
CHART_TYPES = {"Bar": "bar", "Line": "line", "Area": "area", "Pie": "pie", "Scatter": "scatter",
               "Histogram": "histogram", "Box": "box"}
AGGS = ["count", "sum", "mean", "median", "min", "max", "nunique", "none"]
GRAINS = ["auto", "day", "week", "month", "quarter", "year"]
col_opts = [NONE] + list(map(str, df.columns))

# ---- keep widget state valid when the dataset (original vs cleaned) changes
def _valid(key: str, options: list) -> None:
    if key in ss and ss[key] not in options:
        del ss[key]


for k, opts in (("cb_type", list(CHART_TYPES)), ("cb_x", col_opts), ("cb_y", col_opts), ("cb_color", col_opts),
                ("cb_agg", AGGS), ("cb_grain", GRAINS)):
    _valid(k, opts)
first_cat = next(iter(profile.cols_of(CATEGORICAL, BOOLEAN)), None)
ss.setdefault("cb_type", "Bar")
ss.setdefault("cb_x", first_cat or col_opts[1])
ss.setdefault("cb_y", NONE)
ss.setdefault("cb_color", NONE)
ss.setdefault("cb_agg", "count")
ss.setdefault("cb_grain", "auto")
ss.setdefault("cb_topn", 0)

# ------------------------------------------------------------ natural-language prompt
with st.container(border=True):
    pc1, pc2 = st.columns([5, 1])
    prompt = pc1.text_input("Describe the chart", placeholder="e.g. Plot monthly revenue by vehicle type",
                            key="cb_prompt", label_visibility="collapsed")
    go = pc2.button("Generate chart", type="primary")
    if go:
        if not prompt.strip():
            st.warning("Type a chart request first.")
        elif not groq_ready():
            st.error("GROQ_API_KEY not found. Add it to `.env` — or use the manual controls below.")
        else:
            with st.spinner("Interpreting your request…"):
                try:
                    spec = nl_to_spec(prompt, df, ftypes, get_llm())
                    problems = validate_spec(df, spec)
                    if problems:
                        st.error("I understood the request but the chart isn't valid:\n\n- " + "\n- ".join(problems))
                    else:
                        ss.cb_type = next(k for k, v in CHART_TYPES.items() if v == spec.chart_type)
                        ss.cb_x = spec.x or NONE
                        ss.cb_y = spec.y or NONE
                        ss.cb_color = spec.color or NONE
                        ss.cb_agg = spec.aggregation
                        ss.cb_grain = spec.time_grain or "auto"
                        ss.cb_topn = int(spec.top_n or 0)
                        st.toast("Chart configured — adjust below if needed")
                except ChartError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(friendly_error(e))

# ------------------------------------------------------------ manual configuration
st.subheader("Chart configuration")
c = st.columns(6)
ctype_label = c[0].selectbox("Chart type", list(CHART_TYPES), key="cb_type")
x = c[1].selectbox("X-axis", col_opts, key="cb_x")
y = c[2].selectbox("Y-axis", col_opts, key="cb_y")
color = c[3].selectbox("Color / series", col_opts, key="cb_color")
agg = c[4].selectbox("Aggregation", AGGS, key="cb_agg")
grain = c[5].selectbox("Date grouping", GRAINS, key="cb_grain", help="Applies when X is a date column.")
top_n = st.number_input("Show only top N categories (0 = all)", min_value=0, max_value=200, key="cb_topn")

# ------------------------------------------------------------ filters
filters: list = []
with st.expander("Filters (dates, categories, numeric ranges)"):
    f1, f2, f3 = st.columns(3)
    date_cols = profile.cols_of(DATETIME)
    cat_cols = profile.cols_of(CATEGORICAL, BOOLEAN)
    num_cols = profile.cols_of(NUMERICAL)
    with f1:
        dcol = st.selectbox("Date column", [NONE] + date_cols, key="cb_f_date")
        if dcol != NONE:
            dts = as_datetime(df[dcol]).dropna()
            if not dts.empty:
                lo, hi = dts.min().date(), dts.max().date()
                rng = st.date_input("Date range", value=(lo, hi), min_value=lo, max_value=hi, key=f"cb_f_range_{dcol}")
                if isinstance(rng, (tuple, list)) and len(rng) == 2 and (rng[0], rng[1]) != (lo, hi):
                    filters.append(Filter(column=dcol, op="between", values=[str(rng[0]), str(rng[1])]))
    with f2:
        ccol = st.selectbox("Category column", [NONE] + cat_cols, key="cb_f_cat")
        if ccol != NONE:
            vals = df[ccol].dropna().astype(str).value_counts().head(100).index.tolist()
            chosen = st.multiselect("Keep only these values", vals, key=f"cb_f_vals_{ccol}")
            if chosen:
                filters.append(Filter(column=ccol, op="in", values=chosen))
    with f3:
        ncol = st.selectbox("Numeric column", [NONE] + num_cols, key="cb_f_num")
        if ncol != NONE:
            xs = as_numeric(df[ncol]).dropna()
            if not xs.empty and xs.min() < xs.max():
                lo, hi = float(xs.min()), float(xs.max())
                sel = st.slider("Range", lo, hi, (lo, hi), key=f"cb_f_nr_{ncol}")
                if sel != (lo, hi):
                    filters.append(Filter(column=ncol, op="between", values=[str(sel[0]), str(sel[1])]))

# ------------------------------------------------------------ build + render
spec = ChartSpec(
    chart_type=CHART_TYPES[ctype_label],
    x=None if x == NONE else x,
    y=None if y == NONE else y,
    color=None if color == NONE else color,
    aggregation=agg,
    time_grain=None if grain == "auto" else grain,
    top_n=int(top_n) or None,
)

try:
    cd = prepare_chart_data(df, spec, filters)
except ChartError as e:
    st.error(str(e))
    st.stop()

fig = build_figure(cd)
st.plotly_chart(fig, key="cb_fig", config={"displaylogo": False}, **W)
for n in cd.notes:
    st.caption("ℹ️ " + n)

dl1, dl2, _ = st.columns([1, 1, 3])
dl1.download_button("⬇ Chart (HTML)", fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
                    file_name="chart.html", mime="text/html")
dl2.download_button("⬇ Data (CSV)", cd.table.to_csv(index=False).encode("utf-8"),
                    file_name="chart_data.csv", mime="text/csv")
st.caption("Tip: use the camera icon in the chart toolbar to save a PNG.")

left, right = st.columns([3, 2])
with left:
    st.subheader("Result data")
    st.dataframe(cd.table, hide_index=True, **W)
with right:
    tips = simple_insights(cd)
    if tips:
        st.subheader("Insights")
        for t in tips:
            st.markdown(f"- {t}")
