"""Shared data layer: file loading, session state, dataset versions and cached profiling.

The uploaded dataset is stored once as ``original_df`` and is never modified.
Every cleaning step creates a *new* version (a copy) on the ``versions`` stack;
the last version is the working copy.
"""
from __future__ import annotations

import datetime as dt
import uuid

import pandas as pd
import streamlit as st

from .profiler import Profile, profile_dataset

SRC_WORKING = "Working copy"   # internal key; the text shown to the user comes from working_label()
SRC_ORIGINAL = "Original data (empty rows/columns removed)"


# --------------------------------------------------------------------------- #
# Loading uploaded files
# --------------------------------------------------------------------------- #
def _dedupe_columns(cols) -> list[str]:
    seen: dict = {}
    out = []
    for c in cols:
        name = str(c).strip() or "unnamed"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    return out


def read_uploaded(file) -> pd.DataFrame:
    """Read an uploaded CSV / Excel file into a DataFrame."""
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
    else:
        df = None
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                file.seek(0)
                df = pd.read_csv(file, encoding=enc, low_memory=False)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise ValueError("Could not decode the CSV file. Save it as UTF-8 and try again.")
    if df.empty:
        raise ValueError("The file has no rows.")
    # Spreadsheet exports often end with thousands of blank rows or a trailing empty
    # column. They carry no information and would distort missing/duplicate counts.
    n_before = len(df)
    df = df.dropna(how="all")
    empty_cols = [str(c) for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=empty_cols)
    if df.empty:
        raise ValueError("The file has no usable rows.")
    df.columns = _dedupe_columns(df.columns)
    df = df.reset_index(drop=True)
    df.attrs["dropped_empty_rows"] = n_before - len(df)
    df.attrs["dropped_empty_cols"] = empty_cols
    return df


# --------------------------------------------------------------------------- #
# Cached profiling (so it is not recomputed on every Streamlit rerun)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Profiling dataset…", max_entries=8)
def cached_profile(df: pd.DataFrame) -> Profile:
    return profile_dataset(df)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def _now() -> str:
    return dt.datetime.now().strftime("%H:%M:%S")


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("session_id", uuid.uuid4().hex[:8])
    ss.setdefault("file_name", None)
    ss.setdefault("loaded_file_id", None)
    ss.setdefault("last_upload_id", None)
    ss.setdefault("original_df", None)
    ss.setdefault("versions", [])
    ss.setdefault("log", [])
    ss.setdefault("pending_ops", [])
    ss.setdefault("active_source", SRC_WORKING)
    ss.setdefault("analyst_history", [])
    ss.setdefault("thread_n", 0)
    ss.setdefault("load_notice", "")


def has_data() -> bool:
    return st.session_state.get("original_df") is not None


def _load_notice(df: pd.DataFrame) -> str:
    rows = df.attrs.get("dropped_empty_rows", 0)
    cols = df.attrs.get("dropped_empty_cols", [])
    parts = []
    if rows:
        parts.append(f"{rows:,} completely empty rows")
    if cols:
        parts.append(f"{len(cols)} empty column(s) ({', '.join(cols)})")
    return ("Ignored on load: " + " and ".join(parts) + ".") if parts else ""


def load_dataset(df: pd.DataFrame, name: str, file_id: str | None = None) -> None:
    ss = st.session_state
    ss.original_df = df
    ss.file_name = name
    ss.loaded_file_id = file_id
    ss.versions = [{"label": SRC_ORIGINAL, "df": df.copy(), "time": _now()}]
    ss.log = []
    ss.pending_ops = []
    ss.active_source = SRC_WORKING
    ss.analyst_history = []
    ss.thread_n = 0
    ss.load_notice = _load_notice(df)
    ss.pop("analyst_memory", None)
    for k in [k for k in ss.keys() if str(k).startswith("cb_")]:
        del ss[k]


# ----- access ---------------------------------------------------------------
def get_original_df() -> pd.DataFrame:
    return st.session_state.original_df


def get_working_df() -> pd.DataFrame:
    return st.session_state.versions[-1]["df"]


def get_active_df() -> pd.DataFrame:
    """Dataset used by the Overview, AI Analyst and Chart Builder."""
    if st.session_state.active_source == SRC_ORIGINAL:
        return get_original_df()
    return get_working_df()


def working_label() -> str:
    """Text shown for the working copy: changes once a cleaning step has been applied."""
    n = len(st.session_state.versions) - 1
    if n <= 0:
        return "Working copy (same as original, no cleaning yet)"
    return f"Cleaned working copy (v{n})"


def active_label() -> str:
    ss = st.session_state
    if ss.active_source == SRC_ORIGINAL:
        return SRC_ORIGINAL
    return working_label()


def dataset_key() -> str:
    ss = st.session_state
    return f"{ss.file_name}|{len(ss.versions)}|{ss.active_source}"


# ----- versioning -----------------------------------------------------------
def push_version(df: pd.DataFrame, entries: list[dict], label: str) -> None:
    ss = st.session_state
    ss.versions.append({"label": label, "df": df, "time": _now()})
    v = len(ss.versions) - 1
    for e in entries:
        ss.log.append({"time": _now(), "version": v, "undone": False, **e})


def undo_last() -> bool:
    ss = st.session_state
    if len(ss.versions) <= 1:
        return False
    popped = len(ss.versions) - 1
    ss.versions.pop()
    for e in ss.log:
        if e["version"] == popped:
            e["undone"] = True
    return True


def restore_version(index: int) -> None:
    ss = st.session_state
    index = max(0, min(index, len(ss.versions) - 1))
    for e in ss.log:
        if e["version"] > index:
            e["undone"] = True
    ss.versions = ss.versions[: index + 1]


def reset_to_original() -> None:
    restore_version(0)
