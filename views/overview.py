"""Page 1 — Overview & EDA. Runs automatically on the active dataset."""
import pandas as pd
import plotly.express as px
import streamlit as st

from core import ui
from core.state import cached_profile
from core.profiler import (
    CATEGORICAL, DATETIME, NUMERICAL, as_numeric, categorical_frequency, correlation_matrix,
    datetime_summary, numeric_summary, outlier_mask, records_per_period, safe_duplicated,
)

df = ui.require_data()
p = cached_profile(df)
ov = p.overview
W = ui.stretch()

ui.page_header("Overview & EDA", "Your dataset, profiled automatically: quality, distributions, correlations and outliers.", icon_name="dashboard")
ui.source_badge()
st.write("")

# ------------------------------------------------------------------ KPI cards
ui.kpi_grid([
    ("Total rows", f"{ov['rows']:,}", "", "table"),
    ("Total columns", f"{ov['columns']:,}", "", "columns"),
    ("Missing cells", f"{ov['missing_pct']}%", f"{ov['missing_cells']:,} of {ov['rows'] * ov['columns']:,} cells", "alert"),
    ("Duplicate rows", f"{ov['duplicate_pct']}%", f"{ov['duplicate_rows']:,} fully duplicated rows", "copy"),
    ("Memory", f"{ov['memory_mb']:.1f} MB", "", "database"),
    ("Numerical", ov["n_numerical"], "", "sigma"),
    ("Categorical", ov["n_categorical"], "", "tag"),
    ("Date/time", ov["n_datetime"], "", "calendar"),
    ("Boolean", ov["n_boolean"], "", "toggle"),
    ("Other", ov["n_other"], f"{ov['n_identifier']} identifier · {ov['n_text']} text", "dots"),
])
st.write("")

# ------------------------------------------------------------ quality summary
warnings_ = [i for i in p.issues if i["severity"] == "warning"]
if warnings_:
    st.warning("**Data quality warning:** " + " ".join(f"{i['title']} — {i['detail']}" for i in warnings_[:3]))
else:
    st.success("No major data quality problems detected.")
with st.expander("Data quality summary", expanded=bool(p.issues)):
    if not p.issues:
        st.write("Nothing to flag.")
    for i in p.issues:
        cols = f" (`{'`, `'.join(map(str, i['columns'][:8]))}`)" if i["columns"] else ""
        st.markdown(f"**{i['title']}** — {i['detail']}{cols}")
    st.caption("Issues are flagged for investigation. Nothing is changed automatically — use the Data Cleaning Studio to fix them.")

# ------------------------------------------------------------ column profiler
st.subheader("Column profiler")
c1, c2 = st.columns([2, 3])
q = c1.text_input("Search columns", placeholder="Type a column name…", label_visibility="collapsed")
types = c2.multiselect("Feature type", sorted(p.columns["Feature type"].unique()), placeholder="Filter by feature type",
                       label_visibility="collapsed")
prof = p.columns
if q:
    prof = prof[prof["Column"].astype(str).str.contains(q, case=False, regex=False)]
if types:
    prof = prof[prof["Feature type"].isin(types)]
st.dataframe(
    prof, hide_index=True, **W,
    column_config={
        "Missing %": st.column_config.NumberColumn(format="%.2f%%"),
        "Unique %": st.column_config.NumberColumn(format="%.1f%%"),
    },
)
st.caption("Pandas dtype is how the column is stored; feature type is what it represents (e.g. a booking ID stored as text is an Identifier).")

# ------------------------------------------------------- missing & duplicates
st.subheader("Missing values & duplicates")
left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.markdown(f"**Missing values** — {ov['missing_cells']:,} cells ({ov['missing_pct']}%)")
        st.dataframe(
            p.missing.sort_values("Missing", ascending=False), hide_index=True, **W,
            column_config={
                "Missing %": st.column_config.ProgressColumn(format="%.2f%%", min_value=0, max_value=100),
            },
        )
        with_missing = p.missing.loc[p.missing["Missing"] > 0, "Column"].tolist()
        if with_missing:
            pick = st.selectbox("Show rows with missing values in", ["Any column"] + with_missing)
            rows = df[df.isna().any(axis=1)] if pick == "Any column" else df[df[pick].isna()]
            st.caption(f"{len(rows):,} matching rows (first 200 shown)")
            st.dataframe(rows.head(200), **W)
with right:
    with st.container(border=True):
        st.markdown(f"**Duplicate rows** — {ov['duplicate_rows']:,} ({ov['duplicate_pct']}%)")
        st.caption(
            "A repeated value in one column (e.g. the same vehicle type) is normal. "
            "A fully duplicated record may need investigation. Nothing is deleted automatically — "
            "remove duplicates in the Data Cleaning Studio."
        )
        if ov["duplicate_rows"]:
            dup = df[safe_duplicated(df, keep=False)]
            try:
                dup = dup.assign(duplicate_group=dup.groupby(list(df.columns), dropna=False).ngroup()).sort_values("duplicate_group")
            except (TypeError, ValueError):
                pass
            st.caption(f"{len(dup):,} rows belong to a duplicated group (first 100 shown)")
            st.dataframe(dup.head(100), **W)
        else:
            st.success("No fully duplicated rows.")

# ------------------------------------------------ distributions by feature type
st.subheader("Distributions")
t_num, t_cat, t_dt = st.tabs(["Numerical", "Categorical", "Date/time"])
with t_num:
    num_cols = p.cols_of(NUMERICAL)
    if not num_cols:
        st.info("No numerical columns.")
    else:
        st.dataframe(numeric_summary(df, num_cols).round(3), hide_index=True, **W)
        col = st.selectbox("Column", num_cols, key="ov_num")
        x = as_numeric(df[col]).dropna()
        a, b = st.columns(2)
        xf = x.to_frame(name=col)
        a.plotly_chart(px.histogram(xf, x=col, nbins=30, color_discrete_sequence=[ui.PURPLE], title=f"Histogram — {col}").update_layout(
            showlegend=False, xaxis_title=col, margin=dict(t=50, b=10)), key="ov_hist", **W)
        b.plotly_chart(px.box(xf, y=col, points="outliers", color_discrete_sequence=[ui.PURPLE], title=f"Box plot — {col}").update_layout(
            showlegend=False, yaxis_title=col, margin=dict(t=50, b=10)), key="ov_box", **W)
with t_cat:
    cat_cols = p.cols_of(CATEGORICAL)
    if not cat_cols:
        st.info("No categorical columns.")
    else:
        col = st.selectbox("Column", cat_cols, key="ov_cat")
        freq, n_distinct = categorical_frequency(df[col])
        top = freq.iloc[0]
        st.markdown(f"**{n_distinct}** distinct categories · most frequent: **{top['Category']}** ({top['Percent']}%)")
        a, b = st.columns([3, 2])
        a.plotly_chart(px.bar(freq, x="Category", y="Count", color_discrete_sequence=[ui.PURPLE], title=f"Frequency — {col}"
                              ).update_layout(margin=dict(t=50, b=10)), key="ov_catbar", **W)
        b.dataframe(freq, hide_index=True, **W)
with t_dt:
    dt_cols = p.cols_of(DATETIME)
    if not dt_cols:
        st.info("No date/time columns.")
    else:
        col = st.selectbox("Column", dt_cols, key="ov_dt")
        info = datetime_summary(df[col])
        if info:
            k = st.columns(3)
            k[0].metric("Earliest", str(info["earliest"].date()))
            k[1].metric("Latest", str(info["latest"].date()))
            k[2].metric("Range", f"{info['range_days']:,} days")
            grain_label = st.radio("Records per", ["Day", "Week", "Month"], index=2, horizontal=True, key="ov_grain")
            per = records_per_period(df[col], {"Day": "D", "Week": "W", "Month": "M"}[grain_label])
            per_df = per.rename_axis("period").reset_index(name="Records")
            st.plotly_chart(px.line(per_df, x="period", y="Records", markers=len(per_df) < 60, color_discrete_sequence=[ui.PURPLE],
                                    title=f"Records per {grain_label.lower()} — {col}").update_layout(
                showlegend=False, xaxis_title="", yaxis_title="Records", margin=dict(t=50, b=10)), key="ov_dtline", **W)

# ---------------------------------------------- correlation & outliers
st.subheader("Correlation & outliers")
cl, ol = st.columns(2)
num_cols = p.cols_of(NUMERICAL)
with cl:
    with st.container(border=True):
        st.markdown("**Correlation heatmap**")
        if len(num_cols) < 2:
            st.info("Need at least two numerical columns.")
        else:
            method = st.radio("Method", ["pearson", "spearman"], horizontal=True, key="ov_corr")
            corr = correlation_matrix(df, num_cols, method)
            if corr.shape[0] < 2:
                st.info("Not enough varying numerical columns.")
            else:
                st.plotly_chart(px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                                          aspect="auto").update_layout(margin=dict(t=10, b=10)), key="ov_heat", **W)
            st.caption("Correlation does not establish causation.")
with ol:
    with st.container(border=True):
        st.markdown("**Potential outliers (IQR method)**")
        if p.outliers.empty:
            st.info("No numerical columns.")
        else:
            o = p.outliers[["Column", "Outliers", "Outlier %", "Lower bound", "Upper bound"]]
            st.dataframe(o.round(3), hide_index=True, **W)
            col = st.selectbox("Column", p.outliers["Column"].tolist(), key="ov_out")
            x = as_numeric(df[col])
            st.plotly_chart(px.box(x.dropna().to_frame(name=col), y=col, points="outliers", color_discrete_sequence=[ui.PURPLE]).update_layout(
                showlegend=False, yaxis_title=col, margin=dict(t=10, b=10), height=300), key="ov_outbox", **W)
            if st.toggle("Show flagged rows", key="ov_showout"):
                st.dataframe(df[outlier_mask(df[col])].head(200), **W)
            st.caption("Flag = below Q1 − 1.5·IQR or above Q3 + 1.5·IQR. Outliers are flagged for investigation, never removed automatically.")