"""Data Compass — entry point.

Run with:  streamlit run app.py
"""
from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from core import state, ui

st.set_page_config(page_title="Data Compass", page_icon="🧭", layout="wide")
state.init_state()
ui.inject_css()

ss = st.session_state

with st.sidebar:
    ui.sidebar_top()

    ui.sidebar_section("Upload dataset")
    file = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
    if file is not None:
        fid = getattr(file, "file_id", None) or f"{file.name}:{file.size}"
        if ss.last_upload_id != fid:
            try:
                state.load_dataset(state.read_uploaded(file), file.name, fid)
                ss.last_upload_id = fid
            except Exception as e:
                st.error(f"Could not read this file: {e}")

    if state.has_data():
        orig = state.get_original_df()
        ui.file_card(ss.file_name, orig.shape[0], orig.shape[1])
        if ss.load_notice:
            ui.note(ss.load_notice)
        with st.container(border=True):
            st.radio(
                "Data used by Overview, AI Analyst & Chart Builder",
                [state.SRC_WORKING, state.SRC_ORIGINAL],
                key="active_source",
                format_func=lambda v: state.working_label() if v == state.SRC_WORKING else v,
                help="After cleaning, choose whether to analyse the cleaned working copy or the original data (only empty rows/columns removed).",
            )

    ui.sidebar_footer()

pages = [
    st.Page("views/overview.py", title="Overview & EDA", icon=":material/dashboard:", default=True),
    st.Page("views/analyst.py", title="AI Data Analyst", icon=":material/smart_toy:"),
    st.Page("views/cleaning_studio.py", title="Data Cleaning Studio", icon=":material/auto_fix_high:"),
    st.Page("views/chart_builder.py", title="Chart Builder", icon=":material/bar_chart:"),
]
st.navigation(pages, position="top").run()
