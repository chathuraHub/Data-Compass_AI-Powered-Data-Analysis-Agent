"""Page 2 — AI Data Analyst (LangChain/LangGraph agent + validated pandas tools)."""
import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver

from core import state, ui
from core.state import cached_profile
from core.agent import groq_ready, run_analyst
from core.profiler import CATEGORICAL, DATETIME, NUMERICAL

df = ui.require_data()
profile = cached_profile(df)
ss = st.session_state
W = ui.stretch()

ui.page_header("AI Data Analyst", "Ask questions in plain English. Numbers come from validated pandas operations, not from the model's memory.", icon_name="bot")
ui.source_badge()

if not groq_ready():
    st.error("GROQ_API_KEY not found. Add it to your `.env` file and restart the app.")

if "analyst_memory" not in ss:
    ss.analyst_memory = InMemorySaver()
thread_id = f"{ss.session_id}:{state.dataset_key()}:{ss.thread_n}"


def render_message(msg: dict, idx: int) -> None:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
        return
    with st.chat_message("assistant"):
        if msg.get("error"):
            st.error(msg["error"])
        if msg["content"]:
            st.markdown(msg["content"])
        for j, r in enumerate(msg.get("results", [])):
            if r.chart is not None:
                st.plotly_chart(r.chart, key=f"an_{idx}_{j}", **W)
            with st.expander(f"📋 {r.title}", expanded=(j == 0 and r.chart is None)):
                st.dataframe(r.table, hide_index=True, **W)
            with st.expander("How this was calculated"):
                st.markdown(r.explanation)


# ----- suggestions built from the actual columns
num = profile.cols_of(NUMERICAL)
cat = profile.cols_of(CATEGORICAL)
dts = profile.cols_of(DATETIME)
ideas = []
if cat:
    ideas.append(f"How many rows are there for each {cat[0]}?")
if cat and num:
    ideas.append(f"Calculate the average {num[0]} for each {cat[0]}.")
if len(num) >= 2:
    ideas.append(f"What is the correlation between {num[0]} and {num[1]}?")
if dts and num:
    ideas.append(f"Which month has the highest total {num[0]}?")

if not ss.analyst_history and ideas:
    st.markdown("**Try asking**")
    cols = st.columns(len(ideas))
    for c, idea in zip(cols, ideas):
        if c.button(idea, key=f"idea_{idea}"):
            ss.pending_question = idea
            st.rerun()

top_l, top_r = st.columns([4, 1])
with top_r:
    if ss.analyst_history and st.button("Clear chat"):
        ss.analyst_history = []
        ss.pop("analyst_memory", None)
        ss.thread_n += 1
        st.rerun()

past = [m["content"] for m in ss.analyst_history if m["role"] == "user"]
if past:
    with st.expander("Recent analyses"):
        for i, qn in enumerate(reversed(past[-6:])):
            if st.button(qn, key=f"recent_{i}"):
                ss.pending_question = qn
                st.rerun()

for i, m in enumerate(ss.analyst_history):
    render_message(m, i)

question = st.chat_input("Ask anything about your data…")
if not question and ss.get("pending_question"):
    question = ss.pop("pending_question")

if question:
    ss.analyst_history.append({"role": "user", "content": question})
    render_message(ss.analyst_history[-1], len(ss.analyst_history) - 1)
    with st.spinner("Analysing…"):
        resp = run_analyst(question, df, profile.feature_types, ss.analyst_memory, thread_id)
    if resp.error:
        ss.thread_n += 1  # start a fresh memory thread so a failed call can't poison follow-ups
    msg = {"role": "assistant", "content": resp.answer, "results": resp.results, "error": resp.error}
    ss.analyst_history.append(msg)
    render_message(msg, len(ss.analyst_history) - 1)
