"""Shared UI helpers: styling, icons, sidebar blocks, KPI cards, page headers, guards."""
from __future__ import annotations

import streamlit as st

from . import state

PURPLE = "#6D4AFF"
PALETTE = ["#6D4AFF", "#A78BFA", "#4C1D95", "#C4B5FD", "#8B5CF6", "#DDD6FE", "#7C3AED", "#5B21B6"]

APP_NAME = "Data Compass"
APP_TAGLINE = "Ask my CSV"

# --------------------------------------------------------------------------- #
# Icons (inline SVG, 24x24 outline style) so they work on every Streamlit version
# --------------------------------------------------------------------------- #
_ICONS = {
    "upload": '<path d="M4 14.9A7 7 0 1 1 15.7 8h1.8a4.5 4.5 0 0 1 2.5 8.2"/><polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/>',
    "understand": '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><line x1="9" y1="18" x2="15" y2="18"/><line x1="10" y1="22" x2="14" y2="22"/>',
    "clean": '<path d="M9.9 15.5A2 2 0 0 0 8.5 14.1l-6.1-1.6a.5.5 0 0 1 0-1L8.5 9.9A2 2 0 0 0 9.9 8.5l1.6-6.1a.5.5 0 0 1 1 0l1.6 6.1a2 2 0 0 0 1.4 1.4l6.1 1.6a.5.5 0 0 1 0 1l-6.1 1.6a2 2 0 0 0-1.4 1.4l-1.6 6.1a.5.5 0 0 1-1 0z"/><path d="M20 3v4"/><path d="M22 5h-4"/>',
    "ask": '<path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z"/>',
    "dashboard": '<rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/>',
    "bot": '<rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/><line x1="8" y1="16" x2="8.01" y2="16"/><line x1="16" y1="16" x2="16.01" y2="16"/>',
    "chart": '<line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/>',
    "table": '<rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/>',
    "columns": '<rect x="3" y="3" width="18" height="18" rx="2"/><line x1="12" y1="3" x2="12" y2="21"/>',
    "alert": '<path d="m21.7 18-8-14a2 2 0 0 0-3.5 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.7-3"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "copy": '<rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>',
    "sigma": '<path d="M18 7V5a1 1 0 0 0-1-1H6.5a.5.5 0 0 0-.4.8l4.5 6a2 2 0 0 1 0 2.4l-4.5 6a.5.5 0 0 0 .4.8H17a1 1 0 0 0 1-1v-2"/>',
    "tag": '<path d="M12.6 2.6A2 2 0 0 0 11.2 2H4a2 2 0 0 0-2 2v7.2a2 2 0 0 0 .6 1.4l8.7 8.7a2.4 2.4 0 0 0 3.4 0l6.6-6.6a2.4 2.4 0 0 0 0-3.4z"/><circle cx="7.5" cy="7.5" r=".5"/>',
    "calendar": '<rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/>',
    "toggle": '<rect x="1" y="5" width="22" height="14" rx="7"/><circle cx="16" cy="12" r="3"/>',
    "dots": '<circle cx="5" cy="12" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>',
    "info": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
    "spark": '<path d="M9.9 15.5A2 2 0 0 0 8.5 14.1l-6.1-1.6a.5.5 0 0 1 0-1L8.5 9.9A2 2 0 0 0 9.9 8.5l1.6-6.1a.5.5 0 0 1 1 0l1.6 6.1a2 2 0 0 0 1.4 1.4l6.1 1.6a.5.5 0 0 1 0 1l-6.1 1.6a2 2 0 0 0-1.4 1.4l-1.6 6.1a.5.5 0 0 1-1 0z"/>',
}

_BRAND_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="52" height="52" viewBox="0 0 64 64">'
    '<defs><linearGradient id="dcg" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#A78BFA"/><stop offset="1" stop-color="#6D4AFF"/></linearGradient></defs>'
    '<circle cx="32" cy="32" r="28" fill="none" stroke="url(#dcg)" stroke-width="4"/>'
    '<polygon points="45,19 36.2,36.2 27.8,27.8" fill="url(#dcg)"/>'
    '<polygon points="19,45 27.8,27.8 36.2,36.2" fill="#C4B5FD"/>'
    '<circle cx="32" cy="32" r="2.5" fill="#ffffff"/></svg>'
)


def icon(name: str, size: int = 20, color: str = "currentColor", stroke: float = 1.8) -> str:
    inner = _ICONS.get(name, _ICONS["info"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">{inner}</svg>'
    )


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
_CSS = """
<style>
.block-container {padding-top: 2.4rem; max-width: 1400px;}

/* ---- sidebar: collapse (<<) button sits in its own strip above the brand card */
section[data-testid="stSidebar"] > div:first-child {padding-top: 0;}
section[data-testid="stSidebar"][aria-expanded="true"] {width: 23rem !important; min-width: 23rem !important;}
div[data-testid="stSidebarHeader"] {position: absolute; top: .5rem; right: 1rem; width: auto; height: auto;
    min-height: 0; padding: 0; margin: 0; z-index: 10; background: transparent;}
div[data-testid="stSidebarCollapseButton"] button {color: #6D4AFF; background: rgba(109,74,255,.10); border-radius: 8px;}
div[data-testid="stSidebarCollapseButton"] button:hover {background: rgba(109,74,255,.2);}
div[data-testid="stSidebarUserContent"] {padding: 3rem 0.25rem 1rem 0.25rem;}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap: 0.55rem;}
section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] .stElementContainer {width: 100%; max-width: 100%;}

/* ---- cards / pills */
.dc-card {border: 1px solid rgba(109,74,255,.22); background: rgba(109,74,255,.06);
          border-radius: 14px; padding: 14px 16px; height: 100%;}
.dc-card-ic {width: 30px; height: 30px; border-radius: 9px; display: flex; align-items: center; justify-content: center;
             background: rgba(109,74,255,.14); color: #6D4AFF; margin-bottom: 8px;}
.dc-card .label {font-size: .78rem; opacity: .75; margin-bottom: 2px;}
.dc-card .value {font-size: 1.65rem; font-weight: 700; line-height: 1.15; white-space: nowrap;}
.dc-card .sub {font-size: .75rem; opacity: .7; margin-top: 3px;}
.dc-grid {display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin-bottom: 12px;}
.dc-grid .dc-card {box-sizing: border-box; min-height: 132px;}
@media (max-width: 900px) {.dc-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}}
.dc-pill {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem;
          background: rgba(109,74,255,.14); color: #6D4AFF; font-weight: 600;}
.dc-muted {opacity: .72; font-size: .88rem;}
div[data-testid="stMetric"] {background: rgba(109,74,255,.05); border-radius: 12px; padding: 8px 12px;}

/* ---- page header */
.dc-head {display: flex; align-items: center; gap: 14px; margin-bottom: 6px;}
.dc-head-ic {width: 46px; height: 46px; border-radius: 13px; display: flex; align-items: center; justify-content: center;
             background: linear-gradient(135deg, #8B5CF6, #6D4AFF); color: #fff; flex: none;}
.dc-title {font-size: 2.35rem; font-weight: 700; line-height: 1.15; letter-spacing: -.02em;}

/* ---- sidebar top card */
.dc-top {border: 1px solid rgba(109,74,255,.25); background: rgba(109,74,255,.07);
         border-radius: 16px; padding: 12px 9px 12px 9px; margin-bottom: 14px;
         box-sizing: border-box; width: 100%;}
.dc-brand {display: flex; align-items: center; gap: 12px; padding: 0 2px 10px 2px;
           border-bottom: 1px solid rgba(109,74,255,.2); margin-bottom: 12px;}
.dc-brand-logo {flex: none; line-height: 0;}
.dc-brand-text {min-width: 0;}
.dc-brand-name {font-size: 1.25rem; font-weight: 800; line-height: 1.15; letter-spacing: -.02em; white-space: nowrap;}
.dc-brand-tag {font-size: .95rem; font-weight: 600; color: #6D4AFF; white-space: nowrap;}
.dc-brand-agent {display: none;}
.dc-steps {display: flex; justify-content: space-between; align-items: flex-start;}
.dc-step {flex: 1; min-width: 0; display: flex; flex-direction: column; align-items: center; gap: 6px; text-align: center;
          font-size: .66rem; font-weight: 650; line-height: 1.2; border-left: 1px solid rgba(109,74,255,.18); padding: 0 2px;
          overflow-wrap: normal; word-break: normal;}
.dc-step:first-child {border-left: none;}
.dc-step-ic {width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #fff; flex: none;}

/* ---- sidebar content */
.dc-section {font-weight: 700; font-size: .95rem; margin: 6px 0 4px 0;}
.dc-file {display: flex; align-items: center; gap: 10px; margin: 8px 0 2px 0;}
.dc-file-ic {width: 30px; height: 30px; border-radius: 8px; display: flex; align-items: center; justify-content: center;
             background: rgba(109,74,255,.14); color: #6D4AFF; flex: none;}
.dc-file-name {font-weight: 700; font-size: .88rem; word-break: break-word; line-height: 1.2;}
.dc-file-meta {font-size: .76rem; opacity: .7;}
.dc-note {display: flex; gap: 8px; align-items: flex-start; background: rgba(109,74,255,.08);
          border: 1px solid rgba(109,74,255,.22); border-radius: 10px; padding: 8px 8px; margin: 8px 0;
          box-sizing: border-box; width: 100%;
          font-size: .78rem; line-height: 1.35;}
.dc-note svg {flex: none; margin-top: 1px; color: #6D4AFF;}
.dc-foot {display: flex; gap: 8px; align-items: flex-start; width: 100%; box-sizing: border-box;
          margin-top: 10px; padding: 10px 2px 0 2px; border-top: 1px solid rgba(109,74,255,.16);
          font-size: .78rem; font-weight: 400; line-height: 1.5; text-align: left; color: #4F4A67; opacity: .92;}
.dc-foot svg {flex: none; color: #6D4AFF; margin-top: 2px;}
.dc-agent-foot {width: 100%; box-sizing: border-box; margin-top: 9px; padding: 0 2px 2px 2px;
                text-align: right; font-size: .62rem; font-weight: 700; letter-spacing: .08em;
                text-transform: uppercase; color: #7C70A8; opacity: .9;}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 12px; border-color: rgba(109,74,255,.25);}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {border-radius: 12px; border: 1px dashed rgba(109,74,255,.4);}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def _version() -> tuple:
    try:
        return tuple(int(x) for x in st.__version__.split(".")[:2])
    except ValueError:
        return (1, 0)


def stretch() -> dict:
    """Full-width kwarg that works on both older and newer Streamlit releases."""
    return {"width": "stretch"} if _version() >= (1, 50) else {"use_container_width": True}


# --------------------------------------------------------------------------- #
# Sidebar blocks (note: HTML is kept free of blank lines and 4-space indents,
# otherwise Streamlit's markdown turns it into a code block)
# --------------------------------------------------------------------------- #
_STEPS = [("upload", "Upload Data", "#6D4AFF"), ("understand", "Understand It", "#8B5CF6"),
          ("clean", "Clean It", "#7C3AED"), ("ask", "Ask It", "#A78BFA")]


def sidebar_top() -> None:
    steps = "".join(
        f'<div class="dc-step"><span class="dc-step-ic" style="background:{c}">{icon(i, 18, "#fff")}</span>{label}</div>'
        for i, label, c in _STEPS
    )
    st.markdown(
        f'<div class="dc-top"><div class="dc-brand"><div class="dc-brand-logo">{_BRAND_SVG}</div>'
        f'<div class="dc-brand-text"><div class="dc-brand-name">{APP_NAME}</div>'
        f'<div class="dc-brand-tag">{APP_TAGLINE}</div>'
        f'</div></div>'
        f'<div class="dc-steps">{steps}</div></div>',
        unsafe_allow_html=True,
    )


def sidebar_section(title: str) -> None:
    st.markdown(f'<div class="dc-section">{title}</div>', unsafe_allow_html=True)


def file_card(name: str, rows: int, cols: int) -> None:
    st.markdown(
        f'<div class="dc-file"><span class="dc-file-ic">{icon("table", 17)}</span>'
        f'<div><div class="dc-file-name">{name}</div><div class="dc-file-meta">{rows:,} rows × {cols} columns</div></div></div>',
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    st.markdown(f'<div class="dc-note">{icon("info", 16)}<div>{text}</div></div>', unsafe_allow_html=True)


def sidebar_footer() -> None:
    st.markdown(
        f'<div class="dc-foot">{icon("spark", 18)}<div>Workspace to explore, clean, analyze and visualize your data.</div></div>'
        f'<div class="dc-agent-foot">AI Data Analysis Agent</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Page components
# --------------------------------------------------------------------------- #
def kpi(col, label: str, value, sub: str = "", icon_name: str | None = None) -> None:
    ic = f'<div class="dc-card-ic">{icon(icon_name, 18)}</div>' if icon_name else ""
    col.markdown(
        f'<div class="dc-card">{ic}<div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def kpi_grid(items: list) -> None:
    """items: (label, value, sub, icon_name) tuples, drawn as one grid of equal-sized cards."""
    cards = "".join(
        f'<div class="dc-card"><div class="dc-card-ic">{icon(ic, 18)}</div><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>'
        for label, value, sub, ic in items
    )
    st.markdown(f'<div class="dc-grid">{cards}</div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str, icon_name: str | None = None) -> None:
    ic = f'<span class="dc-head-ic">{icon(icon_name, 24, "#fff")}</span>' if icon_name else ""
    st.markdown(f'<div class="dc-head">{ic}<span class="dc-title">{title}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="dc-muted">{subtitle}</div>', unsafe_allow_html=True)
    st.write("")


def source_badge() -> None:
    if state.has_data():
        st.markdown(
            f'<span class="dc-pill">Using: {state.active_label()}</span> '
            f'<span class="dc-muted"> · {st.session_state.file_name}</span>',
            unsafe_allow_html=True,
        )


def require_data():
    """Return the active DataFrame or show an empty state and stop the page."""
    if not state.has_data():
        st.info(
            "No dataset loaded yet. Upload a CSV or Excel file in the sidebar to get started."
        )
        st.stop()
    return state.get_active_df()