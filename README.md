# InsightX — Fintech Intelligence Platform
### IIT Bombay Techfest 2025-26

> A locally-running, AI-powered analytics platform for fintech transaction data — combining a quantised on-device LLM (Phi-3 Mini) with a deterministic pandas engine to deliver zero-latency, hallucination-resistant insights through a fully custom dark-mode GUI.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [Dataset](#dataset)
5. [Project Structure](#project-structure)
6. [Setup & Installation](#setup--installation)
7. [Running the App](#running-the-app)
8. [Usage Guide](#usage-guide)
9. [Query Categories](#query-categories)
10. [Keyboard Shortcuts](#keyboard-shortcuts)
11. [Technical Design Decisions](#technical-design-decisions)

---

## Overview

InsightX is a desktop intelligence platform designed for analysts working with large-scale UPI/fintech transaction datasets. It answers natural language questions about the data, generates charts, tables, and analytical reports — all without an internet connection or API key.

The core design principle is **accuracy over generation**: a deterministic pandas analytics engine handles all numerical computation, and the on-device Phi-3 Mini LLM is used only for narrative prose. This means the numbers are always correct even if the model produces a weak response.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      ui.py  (Frontend)                   │
│  InsightXApp (tkinter)                                   │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Sidebar  │  │  Chat Panel  │  │  Advanced         │  │
│  │ Quick    │  │  ttk.Tree-   │  │  Explorer         │  │
│  │ Actions  │  │  view Tables │  │  (7 chart types)  │  │
│  └──────────┘  └──────────────┘  └───────────────────┘  │
└─────────────────────────────┬───────────────────────────┘
                              │ engine.process_query()
┌─────────────────────────────▼───────────────────────────┐
│                    engine.py  (Backend)                   │
│                                                          │
│  ┌─────────────────────┐    ┌──────────────────────────┐ │
│  │  Query Router       │    │  Direct Analytics Engine  │ │
│  │  wants_stats_table  │───▶│  (Pure pandas — always    │ │
│  │  wants_summary      │    │   accurate, never fails)  │ │
│  │  wants_chart        │    └──────────────────────────┘ │
│  │  wants_report       │                                  │
│  │  detect_query_cat   │    ┌──────────────────────────┐ │
│  └─────────────────────┘    │  Phi-3 Mini 4K (GGUF)    │ │
│           │                 │  run_analyst()  — code    │ │
│           └────────────────▶│  run_narrator() — prose   │ │
│                             │  Triple-scored, fallback  │ │
│                             └──────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
                              │
              ┌───────────────▼──────────────┐
              │        data.csv              │
              │  250,000 UPI transactions    │
              │  ~29 MB · 18 columns         │
              └──────────────────────────────┘
```

**Key flow:** Every query first hits the deterministic pandas engine. The LLM is called in parallel only for prose narration, and its output is triple-scored and validated — if it fails the quality bar, the pandas-generated answer is used instead. For stats tables, the LLM is bypassed entirely.

---

## Features

### Chat Interface
- Natural language query input with typewriter response animation
- Multi-tab conversations — create, rename, delete, and switch chat sessions
- Prompt history navigation (↑/↓ in input box)
- Tab-autocomplete for common query patterns
- Full chat export to `.txt`
- Search across chat history with snippet preview and jump-to

### Analytics Engine
- **6 query categories** automatically detected: Descriptive, Comparative, Temporal, Segmentation, Correlation, Risk
- **Pure-pandas stats table** — generated entirely without the LLM, guaranteed to succeed
- **Triple-scored LLM narration** — runs up to 3 inference passes, keeps the highest-quality result, falls back to pandas prose if score < threshold
- Hallucination detection — rejects responses containing placeholder markers, vague phrases, or invented percentages
- Cross-tab analytics (e.g. Age Group × Device Type)

### Visualisation
- **7 chart types**: Bar, Grouped Bar, Line, Pie, Histogram, Scatter, Heatmap
- Interactive hover tooltips on all chart types
- Export charts as PNG / SVG / PDF
- **Advanced Explorer** — GUI-driven chart builder with X/Y axis selection, colour dimension, up to 3 AND-combined filters, sort order, top-N control, log scale, and value label toggles

### Data Tables
- `ttk.Treeview`-based renderer — full content, no character truncation
- Click-to-sort on any column (asc/desc toggle)
- Double-click any cell to copy its value
- Filter/search bar in expanded table view
- Export any table to CSV or copy as TSV

### Sidebar
- **Pinned queries** — one-click re-run of saved queries, persisted to disk
- **Custom Actions** — define your own labelled shortcut queries
- **Favourites** — star any response to save it
- **Data Snapshot** — live KPI dashboard (19 metrics) in a popup
- 40+ pre-built quick-access queries organised by category
- Session timer and query counter

### UI/UX
- 4 built-in themes: Default Dark, Midnight Blue, Forest Dark, Cyberpunk
- Zoom in/out (Ctrl+Scroll or Ctrl+±) with range 35%–100%
- Collapsible sidebar
- Toast notifications for all actions
- Fully keyboard-navigable

---

## Dataset

| Property | Value |
|---|---|
| File | `data.csv` |
| Size | ~29 MB |
| Rows | 250,000 transactions |
| Columns | 18 |

**Columns:**

| Column | Type | Description |
|---|---|---|
| `transaction_id` | ID | Unique transaction identifier |
| `timestamp` | Datetime | Transaction timestamp |
| `transaction_type` | Categorical | P2P, P2M, Bill Payment, Recharge |
| `merchant_category` | Categorical | Food, Grocery, Fuel, Entertainment, Shopping, Healthcare, Education, Transport, Utilities |
| `amount_inr` | Numeric | Transaction amount in Indian Rupees |
| `transaction_status` | Categorical | SUCCESS or FAILED |
| `fraud_flag` | Binary | 1 = flagged for review (not confirmed fraud) |
| `sender_age_group` | Categorical | 18-25, 26-35, 36-45, 46-55, 56+ |
| `receiver_age_group` | Categorical | Same groupings |
| `sender_state` | Categorical | Indian state of sender |
| `sender_bank` | Categorical | SBI, HDFC, ICICI, Axis, PNB, Kotak, IndusInd, Yes Bank |
| `receiver_bank` | Categorical | Same banks |
| `device_type` | Categorical | Android, iOS, Web |
| `network_type` | Categorical | 4G, 5G, WiFi, 3G |
| `hour_of_day` | Numeric | Derived from timestamp (0–23) |
| `day_of_week` | Categorical | Derived from timestamp (Monday–Sunday) |
| `is_weekend` | Binary | Derived from timestamp (0/1) |

---

## Project Structure

```
InsightX/
├── main.py                      # Entry point
├── engine.py                    # Analytics engine + LLM integration
├── ui.py                        # Full GUI (tkinter)
├── data.csv                     # Transaction dataset (250k rows)
├── Phi-3-mini-4k-instruct-q4.gguf  # Quantised LLM (Q4, ~2.3 GB)
├── requirements.txt             # Python dependencies
├── insightx_custom_actions.json # User-defined quick actions (auto-created)
├── insightx_pinned.json         # Pinned queries (auto-created)
├── insightx_favorites.json      # Starred responses (auto-created)
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.10 or higher
- A C++ compiler (required by `llama-cpp-python`)
  - **Windows**: [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with "Desktop development with C++"
  - **Linux**: `sudo apt install build-essential`
  - **macOS**: `xcode-select --install`

### Step 1 — Clone / download the project

Place all files in a single directory as shown in [Project Structure](#project-structure).

### Step 2 — Install dependencies

**CPU only (works on any machine):**
```bash
pip install -r requirements.txt
```

**GPU acceleration — NVIDIA CUDA:**
```bash
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python
pip install pandas numpy matplotlib
```

**GPU acceleration — Apple Silicon (Metal):**
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
pip install pandas numpy matplotlib
```

### Step 3 — Download the model

The model file `Phi-3-mini-4k-instruct-q4.gguf` must be in the project root directory. It can be downloaded from:

```
https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
```

Download the `q4` quantised variant (~2.3 GB).

---

## Running the App

```bash
python main.py
```

The app will load the model and dataset on startup (this takes 10–30 seconds depending on hardware). The status indicator in the bottom-left of the sidebar will turn green when ready.

---

## Usage Guide

### Asking Questions

Type any natural language question about the data into the input box and press **Ctrl+Enter** or click **Send**.

**Example queries:**
```
What is the average transaction amount for bill payments?
Compare failure rates by device type
Show a pie chart of transaction status
Which age group uses P2P transfers most?
Show fraud flag rate by sender bank as a table
Is there a relationship between network type and transaction success?
Show full statistics table
Generate full report
```

### Quick Actions Sidebar

Click any item in the left sidebar to instantly run a pre-built query. Hover over any item to reveal the 📌 pin button — pin it to the top of the sidebar for fast access.

### Advanced Explorer

Open with **Ctrl+R** or the sidebar button. Build custom charts with full control:
- Choose any X-axis column and Y-axis metric
- Add a colour/grouping dimension for grouped charts
- Apply up to 3 simultaneous AND filters
- Select chart type, sort order, and top-N limit

### Data Snapshot

Open with **Ctrl+D** — shows a live popup with 19 KPIs computed directly from the loaded dataset: success/failure rates, average amounts, top banks, devices, age groups, and more.

---

## Query Categories

InsightX automatically classifies each query and routes it to the appropriate analytics logic:

| Category | What it handles | Example |
|---|---|---|
| **Descriptive** | Averages, totals, counts, distributions | *"What is the average P2P amount?"* |
| **Comparative** | Side-by-side comparison across groups | *"Which device type has the highest failure rate?"* |
| **Temporal** | Time-based patterns, peaks, trends | *"What is the peak transaction hour?"* |
| **Segmentation** | Group-level breakdowns and profiles | *"Which age group drives the most P2P volume?"* |
| **Correlation** | Relationships between two variables | *"Does network type affect success rate?"* |
| **Risk** | Flagged transaction analysis, high-value profiling | *"What % of high-value transactions are flagged?"* |

---

## Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl+Enter` | Send message |
| `Ctrl+N` | New chat tab |
| `Ctrl+S` | Save / export chat |
| `Ctrl+F` | Search chat history |
| `Ctrl+R` | Open Advanced Explorer |
| `Ctrl+D` | Open Data Snapshot |
| `Ctrl+L` | Clear current chat |
| `Ctrl+/` | Show all shortcuts |
| `Ctrl++ / Ctrl+−` | Zoom in / out |
| `Ctrl+0` | Reset zoom |
| `Ctrl+Scroll` | Zoom in / out |
| `↑ / ↓` in input | Browse prompt history |
| `Tab` in input | Autocomplete suggestion |
| `Escape` | Clear input |
| `Home / End` | Scroll to top / bottom of chat |
| `Page Up / Down` | Scroll chat |

---

## Technical Design Decisions

**Why a local LLM instead of an API?**
InsightX runs entirely offline. No data leaves the machine — critical for fintech use cases where transaction data is sensitive. The Phi-3 Mini Q4 model fits in ~2.3 GB of RAM and runs reasonably fast on CPU.

**Why pandas-first instead of LLM-first?**
LLMs hallucinate numbers. The direct analytics engine computes everything from the DataFrame using deterministic pandas operations. The LLM is used only to narrate the result in prose. If narration quality fails the scoring threshold, the pandas-generated narrative is used instead.

**Why ttk.Treeview for tables instead of a Canvas grid?**
The original Canvas+Label grid approach hardcapped column widths at 34 characters, causing dict-format statistics values to be truncated mid-content. Treeview handles horizontal scrolling natively, auto-sizes columns in pixels, and integrates with the OS scrollbar for a native feel.

**Why triple-scored LLM inference?**
A single inference pass from a 4-bit quantised 3.8B parameter model on complex analytical queries is unreliable. Running up to 3 passes and keeping the highest-scoring output (measured by: presence of specific numbers, completeness vs table row count, absence of hallucination markers) significantly improves response quality without requiring a larger model.
