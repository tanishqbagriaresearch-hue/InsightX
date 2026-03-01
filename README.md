# InsightX

AI-powered transaction analytics dashboard. Load a CSV dataset and explore it through natural language queries, charts, tables, and automated reports.

Built with Python, Tkinter, llama.cpp, and Phi-3-mini.

---

## Setup (first time only)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the setup script
```bash
python setup.py
```

This installs any missing packages and checks for the AI model file.  
If the model is missing, it opens the download page automatically and shows a popup with instructions.

**Model to download:** `Phi-3-mini-4k-instruct-q4.gguf` (~2.4 GB)  
**Save it to:** the same folder as `engine.py`

> The model only needs to be downloaded once. After that, it loads from disk in a few seconds.

---

## Running the app

```bash
python main.py
```

---

## Loading your data

Use the file picker inside the app to load any CSV file.  
The app is pre-configured for UPI/payment transaction datasets with columns like:

- `transaction_type`, `merchant_category`, `amount_inr`
- `transaction_status`, `sender_bank`, `receiver_bank`
- `sender_state`, `device_type`, `fraud_flag`

It also works with other CSV formats — just load and ask questions in natural language.

---

## Requirements

- Python 3.9 or higher
- ~2.5 GB free disk space (for the model)
- ~3 GB free RAM
- GPU optional — runs fine on CPU

---

## Project structure

```
InsightX/
├── main.py          # Entry point
├── engine.py        # Analytics engine + AI model loading
├── ui.py            # Tkinter UI
├── setup.py         # First-run setup (installs deps, checks model)
├── requirements.txt
├── README.md
└── .gitignore
```

The GGUF model file is not included in the repository (2.4 GB).  
`setup.py` handles downloading it automatically.

---

## Troubleshooting

**App opens but AI chat doesn't work**  
→ The model file is missing or incomplete. Run `python setup.py` and follow the popup instructions.

**`llama-cpp-python` fails to install**  
→ Try: `pip install llama-cpp-python --prefer-binary`  
→ Or with CUDA: `set CMAKE_ARGS=-DLLAMA_CUDA=on && pip install llama-cpp-python`

**Charts and tables work but AI responses are empty**  
→ This is normal if the GGUF isn't loaded. All analytics features work without the AI model.
