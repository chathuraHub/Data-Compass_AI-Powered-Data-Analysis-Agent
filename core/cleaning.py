"""Data-cleaning operations.

Every operation is a plain dict (so it can sit in a queue in session state) and
``apply_operation`` never mutates its input: it returns a new DataFrame plus
transformation-log entries (operation, column, rows changed, detail).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .profiler import as_numeric, iqr_stats, safe_duplicated, safe_to_datetime, is_text_dtype


class CleaningError(ValueError):
    pass


_TRUE = {"true", "yes", "y", "1", "t"}
_FALSE = {"false", "no", "n", "0", "f"}


def _entry(operation: str, column: str, rows: int, detail: str) -> dict:
    return {"operation": operation, "column": column, "rows_changed": int(rows), "detail": detail}


def _fmt(v) -> str:
    try:
        return f"{float(v):.4g}"
    except (TypeError, ValueError):
        return str(v)


# --------------------------------------------------------------------------- #
# Individual operations
# --------------------------------------------------------------------------- #
def _missing(df: pd.DataFrame, op: dict):
    cols, strat = op["columns"], op["strategy"]
    if not cols:
        raise CleaningError("Select at least one column.")
    out = df.copy()
    if strat == "leave":
        return out, []
    if strat == "drop_rows":
        mask = out[cols].isna().any(axis=1)
        out = out.loc[~mask]
        return out, [_entry("Drop rows with missing values", ", ".join(cols), mask.sum(),
                            f"Removed rows with a missing value in: {', '.join(cols)}")]
    entries = []
    for c in cols:
        n = int(out[c].isna().sum())
        if n == 0:
            continue
        if strat in ("mean", "median"):
            if not pd.api.types.is_numeric_dtype(out[c]):
                raise CleaningError(f"'{c}' is not numeric — use 'mode' or 'drop rows' instead.")
            fill = out[c].mean() if strat == "mean" else out[c].median()
        elif strat == "mode":
            m = out[c].mode(dropna=True)
            if m.empty:
                continue
            fill = m.iloc[0]
        else:
            raise CleaningError(f"Unknown strategy '{strat}'.")
        if isinstance(out[c].dtype, pd.CategoricalDtype):
            out[c] = out[c].astype(object)
        out[c] = out[c].fillna(fill)
        entries.append(_entry(f"Impute missing ({strat})", c, n, f"Filled {n} missing value(s) with {_fmt(fill)}"))
    return out, entries


def _duplicates(df: pd.DataFrame, op: dict):
    mask = safe_duplicated(df, keep=op.get("keep", "first"))
    n = int(mask.sum())
    out = df.loc[~mask].copy()
    return out, ([_entry("Remove duplicate rows", "(all columns)", n,
                         f"Removed {n} fully duplicated row(s), kept the first occurrence")] if n else [])


def _dtype(df: pd.DataFrame, op: dict):
    c, target = op["column"], op["target"]
    out = df.copy()
    s = out[c]
    n_nonnull = int(s.notna().sum())
    before_missing = int(s.isna().sum())
    if target in ("numeric", "integer"):
        new = as_numeric(s)
        if target == "integer":
            new = new.round().astype("Int64")
    elif target == "datetime":
        new = safe_to_datetime(s)
    elif target == "category":
        new = s.astype("category")
    elif target == "string":
        new = s.astype("string")
    elif target == "boolean":
        mapping = {**{k: True for k in _TRUE}, **{k: False for k in _FALSE}}
        new = s.astype("string").str.strip().str.lower().map(mapping).astype("boolean")
    else:
        raise CleaningError(f"Unknown target type '{target}'.")
    lost = int(new.isna().sum()) - before_missing
    if n_nonnull and lost > 0.5 * n_nonnull:
        raise CleaningError(
            f"Converting '{c}' to {target} would turn {lost} of {n_nonnull} values into missing. "
            "The column does not look like this type."
        )
    out[c] = new
    detail = f"Converted {s.dtype} → {new.dtype}"
    if lost > 0:
        detail += f"; {lost} value(s) could not be converted and became missing"
    return out, [_entry("Convert data type", c, n_nonnull, detail)]


def _categories(df: pd.DataFrame, op: dict):
    cols = op["columns"]
    if not cols:
        raise CleaningError("Select at least one column.")
    out = df.copy()
    entries = []
    mapping = {k: v for k, v in (op.get("mapping") or {}).items() if k != v}
    for c in cols:
        if not is_text_dtype(out[c]):
            raise CleaningError(f"'{c}' is not a text/category column.")
        orig = out[c].astype(object).astype("string")
        ser = orig
        if op.get("strip", True):
            ser = ser.str.strip()
        if op.get("collapse_spaces", True):
            ser = ser.str.replace(r"\s+", " ", regex=True)
        case = op.get("case", "none")
        if case == "most_common":
            key = ser.str.lower()
            canon = (
                pd.DataFrame({"k": key, "v": ser}).dropna()
                .groupby("k")["v"].agg(lambda x: x.value_counts().index[0])
            )
            ser = key.map(canon).astype("string")
        elif case == "lower":
            ser = ser.str.lower()
        elif case == "upper":
            ser = ser.str.upper()
        elif case == "title":
            ser = ser.str.title()
        if mapping:
            ser = ser.replace(mapping)
        changed = int((ser.fillna("\0") != orig.fillna("\0")).sum())
        out[c] = ser.astype(object)
        if changed:
            parts = [p for p in (
                "trim" if op.get("strip", True) else "",
                "collapse spaces" if op.get("collapse_spaces", True) else "",
                ("match most common spelling" if case == "most_common" else f"{case}case") if case != "none" else "",
                f"merge {len(mapping)} value(s)" if mapping else "") if p]
            entries.append(_entry("Standardize categories", c, changed, ", ".join(parts)))
    return out, entries


def _outliers(df: pd.DataFrame, op: dict):
    cols, action = op["columns"], op["action"]
    if action == "keep" or not cols:
        return df.copy(), []
    out = df.copy()
    entries = []
    drop = pd.Series(False, index=out.index)
    for c in cols:
        x = as_numeric(out[c])
        st = iqr_stats(x)
        if np.isnan(st["lower"]):
            continue
        flagged = (x < st["lower"]) | (x > st["upper"])
        n = int(flagged.sum())
        if not n:
            continue
        if action == "cap":
            out[c] = x.clip(st["lower"], st["upper"])
            entries.append(_entry("Cap outliers (IQR)", c, n,
                                  f"Clipped to [{_fmt(st['lower'])}, {_fmt(st['upper'])}]"))
        elif action == "remove":
            drop |= flagged.fillna(False)
            entries.append(_entry("Remove outlier rows (IQR)", c, n,
                                  f"Rows outside [{_fmt(st['lower'])}, {_fmt(st['upper'])}]"))
    if action == "remove":
        out = out.loc[~drop]
    return out, entries


_DISPATCH = {
    "missing": _missing,
    "duplicates": _duplicates,
    "dtype": _dtype,
    "categories": _categories,
    "outliers": _outliers,
}


def apply_operation(df: pd.DataFrame, op: dict):
    fn = _DISPATCH.get(op["type"])
    if fn is None:
        raise CleaningError(f"Unknown operation '{op['type']}'.")
    return fn(df, op)


def apply_operations(df: pd.DataFrame, ops: list[dict]):
    """Run operations in order. Returns (new_df, entries). Raises CleaningError."""
    out, all_entries = df, []
    for op in ops:
        out, entries = apply_operation(out, op)
        all_entries.extend(entries)
    return out, all_entries


def describe_op(op: dict) -> str:
    t = op["type"]
    if t == "missing":
        labels = {"leave": "Leave as is", "drop_rows": "Drop rows", "mean": "Fill with mean",
                  "median": "Fill with median", "mode": "Fill with mode"}
        return f"Missing values — {labels[op['strategy']]} · {', '.join(op['columns'])}"
    if t == "duplicates":
        return "Remove fully duplicated rows (keep first)"
    if t == "dtype":
        return f"Convert '{op['column']}' to {op['target']}"
    if t == "categories":
        extra = f" + merge {len(op.get('mapping') or {})}" if op.get("mapping") else ""
        return f"Standardize categories — {', '.join(op['columns'])}{extra}"
    if t == "outliers":
        return f"Outliers — {op['action']} · {', '.join(op['columns'])}"
    return t


# --------------------------------------------------------------------------- #
# Before / after preview
# --------------------------------------------------------------------------- #
def build_diff(before: pd.DataFrame, after: pd.DataFrame, limit: int = 8) -> dict:
    """Rows removed and rows whose values changed (aligned on the index)."""
    removed_idx = before.index.difference(after.index)
    common = after.index.intersection(before.index)
    cols = [c for c in after.columns if c in before.columns]
    changed_idx = pd.Index([])
    mask = pd.DataFrame(index=common, columns=cols, dtype=bool)
    if len(common) and cols:
        b = before.loc[common, cols].astype("string").fillna("∅")
        a = after.loc[common, cols].astype("string").fillna("∅")
        mask = (b != a)
        changed_idx = common[mask.any(axis=1).to_numpy()]
    shown = changed_idx[:limit]
    return {
        "n_removed": int(len(removed_idx)),
        "n_changed": int(len(changed_idx)),
        "removed": before.loc[removed_idx[:limit]],
        "before": before.loc[shown],
        "after": after.loc[shown],
        "mask": mask.loc[shown] if len(shown) else mask.iloc[0:0],
        "changed_cols": [c for c in cols if len(changed_idx) and bool(mask[c].any())],
    }
