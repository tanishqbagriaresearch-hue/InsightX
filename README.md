# InsightX — Fintech Intelligence Platform
### Techfest 2025-26

InsightX is a desktop AI analytics application for fintech transaction data. It combines a local Phi-3 LLM with pandas-powered analytics to let you explore, visualise, and interrogate 250,000+ transaction records entirely on-device — no cloud, no API keys.

---

## Features

| Feature | Description |
|---|---|
| 🤖 AI Chat | Ask questions in plain English — the on-device Phi-3 model answers |
| 📊 Auto Charts | Bar, pie, line, histogram, scatter, heatmap — generated from your data |
| 📋 Data Tables | Sortable, filterable, expandable — export to CSV or copy as TSV |
| 📈 Advanced Explorer | Mix-and-match any column × metric × chart type |
| 📸 Live Snapshot | Real-time KPI dashboard computed directly from your dataset |
| 💬 Multi-Chat | Multiple named chat tabs, drag-to-reorder, rename, right-click menu |
| ⭐ Favorites | Star any response to save it for later |
| 📝 Notes | Per-tab notes panel |
| 🔍 Chat Search | Full-text search across your conversation history |
| 🔬 Report Generator | Markdown + plain-text analytical reports |
| 🔍 Zoom | Ctrl+Scroll or ＋/－ buttons to resize the entire UI |
| ⌨️ Keyboard Shortcuts | Ctrl+N, Ctrl+S, Ctrl+F, Ctrl+R, Ctrl+/ and more |

---

## Quick Start

### 1 — Clone / download the project

```
insightx/
├── main.py          ← run this
├── engine.py        ← AI + analytics engine
├── ui.py            ← Tkinter UI
├── data.csv         ← your transaction dataset (place here)
├── requirements.txt
└── README.md
```

### 2 — Place your dataset

Copy your CSV file into the project folder and rename it `data.csv`.  
Expected columns (case-insensitive, spaces OK):

```
transaction_id, timestamp, transaction_type, merchant_category,
amount_inr, transaction_status, sender_age_group, receiver_age_group,
sender_state, sender_bank, receiver_bank, device_type, network_type,
fraud_flag, hour_of_day, day_of_week, is_weekend
```

Missing columns are silently skipped.

### 3 — Run InsightX

```bash
python main.py
```

The launcher will:
1. Check all required packages
2. Offer to **auto-install** any that are missing (just type `yes`)
3. Suppress verbose library output
4. Launch the GUI

---

## Manual Installation

If you prefer to install packages yourself:

```bash
pip install -r requirements.txt
```

### tkinter (Python's GUI library)

`tkinter` is a Python built-in and **cannot** be pip-installed:

| Platform | Fix |
|---|---|
| **Windows** | Reinstall Python from [python.org](https://python.org) and tick **tcl/tk and IDLE** |
| **Ubuntu/Debian** | `sudo apt install python3-tk` |
| **macOS** | `brew install python-tk` or use the [python.org](https://python.org) macOS installer |

### PyTorch (GPU / CUDA)

The default pip install gives the CPU build. For GPU acceleration:

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Check your CUDA version: `nvcc --version`

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.10 – 3.11 |
| RAM | 8 GB | 16 GB |
| VRAM (GPU) | 4 GB (4-bit) | 6 GB+ |
| Storage | 5 GB (model cache) | 10 GB |
| OS | Windows 10 / Ubuntu 20.04 / macOS 12 | Latest |

> **CPU-only mode**: InsightX runs without a GPU, but model inference will be significantly slower (10-60 seconds per response). The analytics features (charts, tables, statistics) are instant regardless.

---

## Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Enter` / `Ctrl+Enter` | Send message |
| `Ctrl+N` | New chat tab |
| `Ctrl+S` | Save / download current chat |
| `Ctrl+F` | Search chat history |
| `Ctrl+R` | Open Advanced Explorer |
| `Ctrl+D` | Open Data Snapshot |
| `Ctrl+L` | Clear current chat |
| `Ctrl+/` | Show all shortcuts |
| `Ctrl+Scroll` | Zoom in / out |
| `Ctrl + =` / `Ctrl + -` | Zoom in / out |
| `↑` / `↓` in input | Browse prompt history |
| `Tab` in input | Autocomplete suggestion |
| `Escape` | Clear input |
| `Home` / `End` | Scroll to top / bottom of chat |

---

## Chat Panel — Right-Click Menu

Right-click any chat in the sidebar to:
- **✏ Rename** — give the chat a custom name
- **⬇ Download Chat** — save the full conversation as a `.txt` file
- **✕ Delete** — remove the chat (not available on Default)

You can also **drag chats** up and down to reorder them.

---

## Example Queries

```
What is the average transaction amount for bill payments?
Show a bar chart by sender bank
Compare failure rates between Android and iOS users
Show fraud flag rate by transaction type table
Which age group uses P2P transfers most frequently?
Generate full report
Summarize our conversation
Show total amount by merchant category
```

---

## Troubleshooting

**App hangs / slow responses**  
The Phi-3 model inference can be slow on CPU (30–90 s). The UI will show "Thinking…" — just wait. Analytics queries (charts, tables) are instant.

**`[InsightX] generate error`** errors in the console  
These are attention-weight shape mismatches in the cached Phi-3 model and are **silently suppressed**. The app retries with a pandas-based fallback. You will still get chart and table results; only the LLM narrative text may be empty.

**CUDA out of memory**  
InsightX uses 4-bit quantisation to minimise VRAM. If you still get OOM errors, reduce `MAX_HISTORY_TURNS` in `engine.py` (default: 4).

**`bitsandbytes` install fails**  
This package requires a CUDA-capable GPU. On CPU-only machines you can comment it out of `requirements.txt` — the model will load in `float32` instead.

**`flash-attn` warning**  
This is optional and purely a performance hint. InsightX falls back to eager attention automatically.

---

## File Structure

```
insightx/
├── main.py                        ← Entry point + auto-installer
├── engine.py                      ← Phi-3 LLM wrapper, pandas analytics, chart generation
├── ui.py                          ← Tkinter GUI (chat, sidebar, charts, tables)
├── data.csv                       ← Your dataset (not included)
├── requirements.txt               ← pip dependencies
├── README.md                      ← This file
├── insightx_custom_actions.json   ← Saved custom sidebar buttons (auto-created)
├── insightx_pinned.json           ← Pinned queries (auto-created)
├── insightx_notes.json            ← Per-chat notes (auto-created)
└── insightx_favorites.json        ← Starred responses (auto-created)
```

---

## Known Limitations

- The Phi-3-mini model is a 3.8B parameter model quantised to 4-bit. Responses may occasionally be factually incorrect or truncated — always verify numbers against the table/chart results (which are 100% pandas-computed).
- Multi-tab conversation history is in-memory only and resets on app restart.
- The zoom feature scales fonts globally; some UI elements may need a restart to re-render cleanly after aggressive zooming.

---

*Built for Techfest 2025-26 · Powered by Microsoft Phi-3-mini · Analytics by pandas*
