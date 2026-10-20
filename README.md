# Data Compass — Ask my CSV

Upload a CSV/Excel file, then explore, clean, query and chart it.

## Setup
```
python -m venv env
env\Scripts\activate        # Windows  (macOS/Linux: source env/bin/activate)
pip install -r requirements.txt
copy .env.example .env      # then put your GROQ_API_KEY inside
streamlit run app.py
```

## Layout
```
app.py                  entry point: sidebar upload + navigation
core/state.py           file loading, session state, versions/undo, cached profiling
core/profiler.py        feature-type inference, quality checks, summaries
core/query_engine.py    validated pandas queries (no model-written Python)
core/cleaning.py        cleaning operations + before/after diff
core/chart_engine.py    chart spec -> validated data -> Plotly figure
core/agent.py           LangChain tools + LangGraph agent + Groq model
core/ui.py              styling, KPI cards, guards
views/overview.py       Page 1 - Overview & EDA
views/analyst.py        Page 2 - AI Data Analyst
views/cleaning_studio.py Page 3 - Data Cleaning Studio
views/chart_builder.py  Page 4 - Chart Builder
.streamlit/config.toml  theme
```
