# Data Compass — AI-Powered Data Analysis Agent

Data Compass is a Streamlit application for exploring, cleaning, querying, and visualizing CSV and Excel datasets. It combines deterministic data processing with an AI analyst powered by LangChain and a language model accessed through the Groq API.

> **Important:** This project uses a pretrained, remotely hosted language model for inference. You do **not** need to download model weights, train a model, or use a GPU to run the application locally. A Groq API key is needed for the AI Analyst and natural-language chart generation; the other features can run without it.

## Features

- **Overview & EDA:** dataset statistics, column profiling, feature-type inference, missing values, duplicates, outliers, and visual summaries.
- **AI Data Analyst:** natural-language questions answered using LangChain tool calling and validated Pandas operations.
- **Data Cleaning Studio:** missing-value handling, deduplication, type conversion, category standardization, outlier treatment, preview, approval, version history, and undo.
- **Chart Builder:** manual or natural-language chart configuration, Plotly visualizations, and chart/data exports.

## Technology stack

Python, Streamlit, Pandas, NumPy, Plotly, Pydantic, LangChain, LangGraph, LangChain-Groq, Groq API, python-dotenv, and openpyxl.

## Prerequisites

1. **Python:** Python 3.11 or 3.12 is recommended. Verify with `python --version` (Windows) or `python3 --version` (macOS/Linux).
2. **Git:** Required if cloning the repository. Verify with `git --version`.
3. **Groq API key:** Required only for AI-powered features. Obtain a key from [Groq Console](https://console.groq.com/keys). Usage and model availability are subject to Groq's current policies.
4. **Internet connection:** Needed to install dependencies and make Groq API calls.

## Installation and local setup

### Step 1 — Clone the repository

Open a terminal (PowerShell, Command Prompt, or a terminal on macOS/Linux):

```bash
git clone https://github.com/chathuraHub/Data-Compass_AI-Powered-Data-Analysis-Agent.git
cd Data-Compass_AI-Powered-Data-Analysis-Agent
```

If you already downloaded the ZIP, extract it and open a terminal in the extracted project folder instead. The terminal should be in the directory containing `app.py` and `requirements.txt`.

### Step 2 — Create a virtual environment (recommended)

A virtual environment isolates this project's Python packages from packages used by other projects.

**Windows (PowerShell or Command Prompt):**

```powershell
python -m venv .venv
```

**macOS/Linux:**

```bash
python3 -m venv .venv
```

### Step 3 — Activate the virtual environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt (cmd):**

```bat
.venv\Scripts\activate.bat
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

Your terminal should now display `(.venv)` before the command prompt.

If PowerShell blocks activation, you can run the following in **that PowerShell session only**, then retry activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Alternatively, use Command Prompt activation. Do not change system-wide execution policies just for this project.

### Step 4 — Install dependencies

With the virtual environment activated, run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The existing `requirements.txt` contains the project's dependencies. Installation may take a few minutes.

### Step 5 — Configure the Groq API key

Create a file named **`.env`** in the repository root, alongside `app.py`:

```dotenv
GROQ_API_KEY=replace_with_your_actual_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

- Replace the placeholder with your real Groq API key.
- `GROQ_MODEL` is optional; the code defaults to `openai/gpt-oss-120b`.
- The model must be available to your Groq account. If it is unavailable, change `GROQ_MODEL` to a currently supported compatible model.
- Never publish your actual API key in GitHub, screenshots, or logs.
- The repository's existing `.gitignore` excludes `.env` from Git tracking.

**Note:** The old README refers to `.env.example`, but that file is not currently included in the repository. You can create `.env` directly as shown above.

### Step 6 — Start the application

From the repository root, run:

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL, typically:

```text
http://localhost:8501
```

Open that URL in your browser if it does not open automatically.

To stop the application, return to the terminal and press **Ctrl+C**.

To leave the virtual environment after stopping the application, run:

```bash
deactivate
```

## Reproduce the application workflow

There is no model-training stage in this repository. Reproduction means installing the dependencies, running the Streamlit application, and executing its data-analysis workflows using a CSV or Excel file.

1. **Upload data:** Open the application and upload a `.csv`, `.xlsx`, or `.xls` file through the uploader.
2. **Overview & EDA:** Inspect the dataset shape, missing values, duplicate rows, feature types, statistics, and outlier indicators.
3. **AI Data Analyst:** With `GROQ_API_KEY` configured, ask questions such as "How many rows are in this dataset?" or "What is the average of the Sales column by Region?" (adapt the names to the uploaded columns). The agent chooses validated tools and uses Pandas to compute results.
4. **Data Cleaning Studio:** Select a cleaning operation, add it to the queue, preview its effect, approve the changes, and inspect the transformation history. You can undo or restore versions.
5. **Choose the data source:** Select the original or working (cleaned) dataset for downstream analysis using the application's data-source control.
6. **Chart Builder:** Create a chart manually, or enter a natural-language request such as "Plot average sales by region" when the Groq API is configured. Download the chart as HTML or its underlying data as CSV.
7. **Export:** Download the cleaned dataset as CSV or Excel, or download the cleaning transformation log.

### Optional: Create a small sample CSV

To test the application without an existing dataset, create a file named `sample_sales.csv` with the following content:

```csv
Region,Product,Sales,Quantity
North,Laptop,5000,2
South,Phone,3000,3
North,Phone,7000,5
South,Laptop,9000,4
West,Tablet,4000,2
North,Laptop,5000,2
```

Upload it and try:

- **EDA:** Confirm that the dataset has 6 rows and 4 columns and includes one fully duplicated row.
- **AI Analyst:** Ask "What is the average sales value for each region?"
- **Cleaning:** Remove duplicate rows, preview the change, and approve it. The working dataset should then contain 5 rows.
- **Chart Builder:** Generate a bar chart of total Sales by Region using the working dataset.

The sample data is for smoke testing, not a formal benchmark.

## How the source files connect

```text
app.py                         Streamlit entry point, upload, and navigation
│
├── core/state.py              Data loading, session state, dataset versions, undo
├── core/ui.py                 Shared UI components and styling
│
└── views/
    ├── overview.py            EDA interface
    │   └── core/profiler.py    Data profiling, quality checks, statistics
    │
    ├── analyst.py             AI chat interface
    │   └── core/agent.py      Groq LLM, LangChain tools, agent and memory
    │       └── core/query_engine.py   Validated Pandas queries
    │
    ├── cleaning_studio.py     Cleaning interface, queue and approval
    │   └── core/cleaning.py   Cleaning operations and before/after differences
    │       └── core/state.py  Version management
    │
    └── chart_builder.py       Chart configuration interface
        └── core/chart_engine.py   Chart specs, validation, Plotly rendering
            └── core/query_engine.py   Data aggregation and filtering
```

The modules share data through Streamlit session state. The `views/` directory contains user-facing pages, while `core/` contains the processing logic.

## How the AI works

1. A user uploads a dataset. The app reads it into a Pandas DataFrame.
2. The user asks a question in the AI Data Analyst page.
3. LangChain sends the question and tool descriptions to the Groq-hosted language model.
4. The model selects a permitted tool, such as `run_query` or `column_summary`.
5. The tool performs the actual calculation locally using Pandas, with query validation where applicable.
6. The tool result is returned to the agent, which formulates a natural-language answer.
7. Streamlit displays the answer and any supported result table or chart.

**Groq** provides model inference. **LangChain** provides the agent/tool interface. **LangGraph** supports agent orchestration and conversation checkpoints. **Pandas** performs the actual dataset calculations. **Plotly** renders charts. The LLM is not trained or fine-tuned by this project.

## Troubleshooting

| Problem | What to do |
|---|---|
| `python` is not recognized | Install Python and ensure it is on PATH; on Windows, try `py -3.11` instead of `python`. |
| `No module named streamlit` | Activate `.venv` and run `python -m pip install -r requirements.txt`. |
| PowerShell blocks activation | Use Command Prompt or the session-only PowerShell command in Step 3. |
| `GROQ_API_KEY not found` | Check that `.env` is in the same folder as `app.py`, contains the correct variable name, and restart Streamlit. |
| Groq reports `model_not_found` | Check available models for your account and update `GROQ_MODEL` in `.env`. |
| Groq authentication or rate-limit error | Verify the API key and your account's limits; never paste the secret into public issues. |
| `Address already in use` | Run `python -m streamlit run app.py --server.port 8502` and open the URL shown. |
| CSV/Excel upload fails | Verify the file format and size; `.streamlit/config.toml` sets `maxUploadSize = 200` MB. |
| AI answer mentions a nonexistent column | Check the exact column names in the Overview page and rephrase the request. |

## Reproducibility notes

- Dependencies in the current `requirements.txt` are not fully version-pinned. Installing them at different times may yield different package versions. For a strictly reproducible environment, test a known-working installation and record its exact versions with `python -m pip freeze > requirements-lock.txt`.
- LLM outputs may vary between runs and depend on the model and service availability. `temperature=0` reduces randomness but does not guarantee identical responses.
- Session-state data and conversation checkpoints are in-memory; they are not a persistent database.
- Uploaded datasets are supplied by the user. The repository does not contain a required training dataset or pretrained model-weight file.
- This is an application-reproduction workflow, not a training-experiment reproduction protocol.

## Security

Keep your Groq API key in `.env` and out of Git. Before pushing, verify it is not tracked:

```bash
git ls-files .env
```

This should produce no output. If you have previously committed a real API key, removing it from the latest commit is not sufficient; revoke/rotate the key and clean the affected Git history as appropriate.

## License

No license is declared here. Add a `LICENSE` file if you intend to grant explicit reuse permissions.
