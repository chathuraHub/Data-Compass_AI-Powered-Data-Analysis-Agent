"""Page 3 — Data Cleaning Studio: queue → preview → approve → log → undo."""
import io

import numpy as np
import pandas as pd
import streamlit as st

from core import state, ui
from core.state import cached_profile
from core.cleaning import (
    CleaningError, apply_operation, apply_operations, build_diff, describe_op,
)
from core.profiler import CATEGORICAL, NUMERICAL, is_text_dtype

if not state.has_data():
    ui.require_data()

ss = st.session_state
W = ui.stretch()
original = state.get_original_df()
working = state.get_working_df()
p = cached_profile(working)

ui.page_header("Data Cleaning Studio", "Inspect, approve and reverse cleaning steps. The original data is never overwritten.", icon_name="clean")
m = st.columns(4)
ui.kpi(m[0], "Original rows", f"{len(original):,}")
ui.kpi(m[1], "Working copy rows", f"{len(working):,}", f"{len(working) - len(original):+,} vs original")
ui.kpi(m[2], "Working copy version", f"v{len(ss.versions) - 1}")
ui.kpi(m[3], "Queued operations", len(ss.pending_ops))
st.write("")

# ------------------------------------------------------------ detected issues
st.subheader("Detected issues")
if not p.issues:
    st.success("No issues detected in the working copy.")
else:
    cols = st.columns(min(3, len(p.issues)))
    for i, issue in enumerate(p.issues):
        with cols[i % len(cols)].container(border=True):
            st.markdown(f"**{issue['title']}**")
            st.caption(issue["detail"])

# ------------------------------------------------------------ build operations
st.subheader("Cleaning operations")
t_miss, t_dup, t_type, t_cat, t_out = st.tabs(["Missing values", "Duplicates", "Data types", "Categories", "Outliers"])


def queue(op: dict) -> None:
    ss.pending_ops.append(op)
    st.toast("Added to queue")


with t_miss:
    with_missing = p.missing.loc[p.missing["Missing"] > 0, "Column"].tolist()
    if not with_missing:
        st.info("No missing values in the working copy.")
    else:
        cols = st.multiselect("Columns", with_missing, default=with_missing, key="cl_miss_cols")
        labels = {"Fill with median": "median", "Fill with mean": "mean", "Fill with mode (most frequent)": "mode",
                  "Drop rows": "drop_rows", "Leave as is": "leave"}
        choice = st.selectbox("Treatment", list(labels), key="cl_miss_strat")
        st.caption("Mean/median work on numeric columns only. Use mode for text columns.")
        if st.button("Add to queue", key="cl_miss_add"):
            if cols:
                queue({"type": "missing", "columns": cols, "strategy": labels[choice]})
            else:
                st.warning("Select at least one column.")

with t_dup:
    n_dup = p.overview["duplicate_rows"]
    st.write(f"**{n_dup:,}** fully duplicated rows in the working copy. Repeated values inside one column are normal and are not touched.")
    if st.button("Add 'remove duplicate rows' to queue", key="cl_dup_add", disabled=n_dup == 0):
        queue({"type": "duplicates", "keep": "first"})

with t_type:
    col = st.selectbox("Column", list(working.columns), key="cl_type_col")
    st.caption(f"Current dtype: `{working[col].dtype}` · feature type: {p.feature_types.get(col)}")
    targets = ["numeric", "integer", "datetime", "category", "string", "boolean"]
    default = {"Numerical": 0, "Date/time": 2}.get(p.feature_types.get(col), 0)
    target = st.selectbox("Convert to", targets, index=default, key="cl_type_target")
    if st.button("Add to queue", key="cl_type_add"):
        queue({"type": "dtype", "column": col, "target": target})

with t_cat:
    text_cols = [c for c in working.columns if is_text_dtype(working[c]) and p.feature_types.get(c) in (CATEGORICAL, "Text", "Other")]
    if not text_cols:
        st.info("No categorical columns to standardize.")
    else:
        default_cols = [c for c in p.inconsistent if c in text_cols]
        cols = st.multiselect("Columns", text_cols, default=default_cols, key="cl_cat_cols")
        a, b, c = st.columns(3)
        strip = a.checkbox("Trim spaces", True, key="cl_cat_strip")
        collapse = b.checkbox("Collapse repeated spaces", True, key="cl_cat_collapse")
        case = c.selectbox(
            "Standardize spelling", ["most_common", "none", "lower", "upper", "title"],
            format_func={"most_common": "Match most common spelling", "none": "Keep as is", "lower": "lowercase",
                         "upper": "UPPERCASE", "title": "Title Case"}.get,
            key="cl_cat_case", help="'Match most common spelling' merges 'SUV', 'suv ' and 'Suv' into whichever spelling is most frequent.")
        mapping: dict = {}
        if len(cols) == 1:
            with st.expander("Merge category values (optional)"):
                vals = sorted(working[cols[0]].dropna().astype(str).unique().tolist())[:300]
                merge = st.multiselect("Values to merge (after trim/case rules above)", vals, key="cl_cat_merge")
                new_name = st.text_input("Merge into", key="cl_cat_target")
                if merge and new_name:
                    mapping = {v: new_name for v in merge}
        else:
            st.caption("Select exactly one column to merge specific values.")
        if st.button("Add to queue", key="cl_cat_add"):
            if cols:
                queue({"type": "categories", "columns": cols, "strip": strip, "collapse_spaces": collapse,
                       "case": case, "mapping": mapping})
            else:
                st.warning("Select at least one column.")

with t_out:
    out_cols = p.outliers.loc[p.outliers["Outliers"] > 0, "Column"].tolist() if len(p.outliers) else []
    if not out_cols:
        st.info("No outliers flagged by the IQR rule.")
    else:
        cols = st.multiselect("Columns", out_cols, default=out_cols, key="cl_out_cols")
        action = st.selectbox("Treatment", ["keep", "cap", "remove"], format_func={
            "keep": "Keep (flag only)", "cap": "Cap at IQR bounds", "remove": "Remove rows"}.get, key="cl_out_action")
        st.caption("Outliers can be real. Only change them if you know they are errors.")
        if st.button("Add to queue", key="cl_out_add"):
            if cols:
                queue({"type": "outliers", "columns": cols, "action": action})

# ------------------------------------------------------------ queue + preview
st.subheader("Review & approve")
if not ss.pending_ops:
    st.info("Add one or more operations above to preview their effect.")
else:
    for i, op in enumerate(list(ss.pending_ops)):
        a, b = st.columns([12, 1])
        a.markdown(f"**{i + 1}.** {describe_op(op)}")
        if b.button("✕", key=f"rm_{i}", help="Remove from queue"):
            ss.pending_ops.pop(i)
            st.rerun()
    if st.button("Clear queue"):
        ss.pending_ops = []
        st.rerun()

    try:
        after, entries = apply_operations(working, ss.pending_ops)
    except CleaningError as e:
        st.error(f"Cannot preview: {e}")
        after, entries = None, []

    if after is not None:
        if not entries:
            st.warning("These operations would not change any data.")
        else:
            st.dataframe(pd.DataFrame(entries).rename(columns={
                "operation": "Operation", "column": "Column", "rows_changed": "Rows changed", "detail": "Detail"}),
                hide_index=True, **W)
        diff = build_diff(working, after)
        k = st.columns(3)
        k[0].metric("Rows before → after", f"{len(working):,} → {len(after):,}")
        k[1].metric("Rows removed", f"{diff['n_removed']:,}")
        k[2].metric("Rows with changed values", f"{diff['n_changed']:,}")

        b_col, a_col = st.columns(2)
        with b_col:
            st.markdown("**Before** (affected rows)")
            show = diff["before"] if len(diff["before"]) else diff["removed"]
            st.dataframe(show, **W)
        with a_col:
            st.markdown("**After** (changed cells highlighted)")
            if len(diff["after"]):
                mask = diff["mask"].reindex(columns=diff["after"].columns, fill_value=False)
                styles = pd.DataFrame(np.where(mask.to_numpy(), "background-color: #DDD6FE", ""),
                                      index=mask.index, columns=mask.columns)
                st.dataframe(diff["after"].style.apply(lambda _: styles, axis=None), **W)
            else:
                st.caption("No cell values changed (rows were removed or nothing happened).")
        if diff["n_removed"] and len(diff["before"]):
            with st.expander(f"Removed rows ({diff['n_removed']:,}, first 8 shown)"):
                st.dataframe(diff["removed"], **W)

        if st.button("✅ Approve & apply changes", type="primary", disabled=not entries):
            cur = working
            for op in ss.pending_ops:
                new, ents = apply_operation(cur, op)
                if ents:
                    state.push_version(new, ents, describe_op(op))
                    cur = new
            ss.pending_ops = []
            st.success("Changes applied to the working copy. The original is untouched.")
            st.rerun()

# ------------------------------------------------------------ history & versions
st.subheader("Transformation history")
if ss.log:
    log = pd.DataFrame(ss.log)
    log["Status"] = np.where(log["undone"], "undone", "active")
    log = log.rename(columns={"time": "Time", "version": "Version", "operation": "Operation", "column": "Column",
                              "rows_changed": "Rows changed", "detail": "Detail"})
    st.dataframe(log[["Time", "Version", "Operation", "Column", "Rows changed", "Detail", "Status"]],
                 hide_index=True, **W)
else:
    st.caption("No transformations applied yet.")

c1, c2, c3 = st.columns([2, 1, 1])
labels = [f"v{i} — {v['label']}" for i, v in enumerate(ss.versions)]
pick = c1.selectbox("Restore a previous version", labels, index=len(labels) - 1, key="cl_restore_pick")
if c1.button("Restore selected version", disabled=len(ss.versions) == 1):
    state.restore_version(labels.index(pick))
    st.rerun()
if c2.button("Undo last change", disabled=len(ss.versions) == 1):
    state.undo_last()
    st.rerun()
if c3.button("Reset to original", disabled=len(ss.versions) == 1):
    state.reset_to_original()
    st.rerun()

# ------------------------------------------------------------ downloads
st.subheader("Download")
base = (ss.file_name or "dataset").rsplit(".", 1)[0]
d1, d2, d3 = st.columns(3)
d1.download_button("⬇ Cleaned dataset (CSV)", working.to_csv(index=False).encode("utf-8"),
                   file_name=f"{base}_cleaned.csv", mime="text/csv")
try:
    buf = io.BytesIO()
    working.to_excel(buf, index=False, engine="openpyxl")
    d2.download_button("⬇ Cleaned dataset (Excel)", buf.getvalue(), file_name=f"{base}_cleaned.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
except Exception:
    d2.caption("Install openpyxl to enable Excel export.")
if ss.log:
    d3.download_button("⬇ Transformation log (CSV)", pd.DataFrame(ss.log).to_csv(index=False).encode("utf-8"),
                       file_name=f"{base}_cleaning_log.csv", mime="text/csv")

st.info("Choose which version the AI Analyst and Chart Builder use with the **Data used by** switch in the sidebar.")
