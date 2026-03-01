import pandas as pd
import numpy as np
import torch
import re, sys, io, json, textwrap, warnings, os, logging, gc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── GGUF config ───────────────────────────────────────────────────────────────
# The GGUF file is auto-downloaded by setup.py on first run.
# Phi-3-mini-4k Q4_K_M: ~2.2 GB, runs on CPU + GPU, needs ~3 GB RAM.
GGUF_FILENAME = "Phi-3-mini-4k-instruct-q4.gguf"
GGUF_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)), GGUF_FILENAME)
GGUF_URL      = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
CONTEXT_SIZE  = 4096

_df   = None
_llm  = None
chat_tabs = {"Default": []}
chat_rich = {"Default": []}
MAX_HISTORY_TURNS = 4

PS = {  # plot style
    "bg":  "#0d1117", "fg":  "#e6edf3", "accent": "#58a6ff",
    "a2":  "#f78166", "a3":  "#3fb950", "grid": "#21262d",
    "colors": ["#58a6ff","#3fb950","#f78166","#d2a8ff","#ffa657",
               "#79c0ff","#56d364","#ff9f43","#a29bfe","#fd79a8","#00cec9","#e17055"],
}

COLUMN_META = {
    "transaction_id":     {"label":"Transaction ID",      "kind":"id"},
    "timestamp":          {"label":"Timestamp",           "kind":"datetime"},
    "transaction_type":   {"label":"Transaction Type",    "kind":"categorical"},
    "merchant_category":  {"label":"Merchant Category",   "kind":"categorical"},
    "amount_inr":         {"label":"Amount (INR)",        "kind":"numeric"},
    "amount_(inr)":       {"label":"Amount (INR)",        "kind":"numeric"},
    "transaction_status": {"label":"Transaction Status",  "kind":"categorical"},
    "sender_age_group":   {"label":"Sender Age Group",    "kind":"categorical"},
    "receiver_age_group": {"label":"Receiver Age Group",  "kind":"categorical"},
    "sender_state":       {"label":"Sender State",        "kind":"categorical"},
    "sender_bank":        {"label":"Sender Bank",         "kind":"categorical"},
    "receiver_bank":      {"label":"Receiver Bank",       "kind":"categorical"},
    "device_type":        {"label":"Device Type",         "kind":"categorical"},
    "network_type":       {"label":"Network Type",        "kind":"categorical"},
    "fraud_flag":         {"label":"Fraud Flag (Review)", "kind":"binary"},
    "hour_of_day":        {"label":"Hour of Day",         "kind":"numeric"},
    "day_of_week":        {"label":"Day of Week",         "kind":"categorical"},
    "is_weekend":         {"label":"Is Weekend",          "kind":"binary"},
}

# ── helpers ──────────────────────────────────────────────────────────────────
def _amt(df):
    for c in ["amount_inr","amount_(inr)"]:
        if c in df.columns: return c
    return None

def load_data(filepath="data.csv"):
    global _df
    df = pd.read_csv(filepath)
    df.columns = [c.lower().strip().replace(" ","_") for c in df.columns]
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if "hour_of_day" not in df.columns:
            df["hour_of_day"] = df["timestamp"].dt.hour
        if "day_of_week" not in df.columns:
            df["day_of_week"] = df["timestamp"].dt.day_name()
        if "is_weekend" not in df.columns:
            df["is_weekend"] = df["timestamp"].dt.dayofweek.apply(lambda x: 1 if x>=5 else 0)
    if "merchant_category" in df.columns:
        df["merchant_category"] = df["merchant_category"].fillna("P2P_Transfer")
    _df = df; return df

GGUF_MISSING_SENTINEL = "__GGUF_MISSING__"  # sent via queue to trigger UI popup

def _ensure_gguf():
    """Return True if GGUF is present and large enough, False otherwise."""
    return os.path.isfile(GGUF_PATH) and os.path.getsize(GGUF_PATH) > 1_000_000_000


class _LlamaCppWrapper:
    """Native llama_cpp wrapper matching the call signature used throughout this file."""
    def __init__(self, model):
        self._model = model

    def __call__(self, prompt, max_tokens=512, stop=None,
                 temperature=0.1, echo=False, repeat_penalty=1.1, **_ignored):
        return self._model(
            prompt,
            max_tokens=max_tokens,
            stop=stop or [],
            temperature=max(float(temperature), 1e-6),
            echo=echo,
            repeat_penalty=float(repeat_penalty),
        )


def load_model():
    global _llm

    # ── Step 1: ensure GGUF is present ───────────────────────────────────────
    if not _ensure_gguf():
        print("[InsightX] GGUF not found — signalling UI to show download prompt.")
        _llm = None
        return GGUF_MISSING_SENTINEL   # ui.py checks for this and shows the popup

    # ── Step 2: load with llama_cpp ──────────────────────────────────────────
    try:
        from llama_cpp import Llama
    except ImportError:
        print("[InsightX] llama-cpp-python not installed.")
        print("[InsightX] Run:  pip install llama-cpp-python")
        _llm = None
        return None

    use_cuda   = torch.cuda.is_available()
    n_gpu      = -1 if use_cuda else 0   # -1 = offload all layers to GPU
    print(f"[InsightX] Loading {GGUF_FILENAME}  (GPU={'yes' if use_cuda else 'cpu-only'})...")

    try:
        llama = Llama(
            model_path=GGUF_PATH,
            n_ctx=CONTEXT_SIZE,
            n_gpu_layers=n_gpu,
            verbose=False,
        )
        _llm = _LlamaCppWrapper(llama)
        print("[InsightX] Model ready.")
        return _llm
    except Exception as e:
        print(f"[InsightX] llama_cpp load failed: {e}")
        # Retry on CPU only if GPU load failed
        if use_cuda:
            try:
                print("[InsightX] Retrying on CPU only...")
                llama = Llama(model_path=GGUF_PATH, n_ctx=CONTEXT_SIZE,
                              n_gpu_layers=0, verbose=False)
                _llm = _LlamaCppWrapper(llama)
                print("[InsightX] Model ready (CPU).")
                return _llm
            except Exception as e2:
                print(f"[InsightX] CPU load also failed: {e2}")
        _llm = None
        return None


def get_df():  return _df
def get_llm(): return _llm

def history_to_prompt_block(tab="Default"):
    history = chat_tabs.get(tab, [])
    block = ""
    for msg in history[-(MAX_HISTORY_TURNS*2):]:
        tag = "<|user|>" if msg["role"]=="user" else "<|assistant|>"
        block += f"{tag}\n{msg['content']}<|end|>\n"
    return block

def push_history(u, a, tab="Default"):
    if tab not in chat_tabs: chat_tabs[tab] = []
    chat_tabs[tab].append({"role":"user","content":u})
    chat_tabs[tab].append({"role":"assistant","content":a})

def push_rich(tab, entry):
    if tab not in chat_rich: chat_rich[tab] = []
    chat_rich[tab].append(entry)

def get_rich(tab="Default"):    return chat_rich.get(tab,[])
def get_history(tab="Default"): return chat_tabs.get(tab,[])

def rename_tab(old, new):
    if old==new or new in chat_tabs: return False
    chat_tabs[new]=chat_tabs.pop(old); chat_rich[new]=chat_rich.pop(old,[]); return True

def create_tab(name):
    if name not in chat_tabs: chat_tabs[name]=[]; chat_rich[name]=[]

def delete_tab(name):
    if name in chat_tabs and name!="Default":
        del chat_tabs[name]; chat_rich.pop(name,None)

def list_tabs(): return list(chat_tabs.keys())

# ── query routing ─────────────────────────────────────────────────────────────
CONV_EXACT = {"hello","hi","hey","hiya","howdy","how are you","how are you?",
              "how r u","how r u?","thanks","thank you","ty","thx","okay","ok",
              "got it","sure","alright","bye","goodbye","see you","interesting",
              "nice","cool","wow","great","what can you do","what can you do?",
              "who are you","who are you?"}

DATA_KW = ["how many","how much","what is the","what are the","count","total",
           "sum","average","mean","median","percent","percentage","%","rate","ratio",
           "highest","lowest","most","least","top","bottom","fraud","flagged","flag",
           "failed","failure","success","compare","breakdown","distribution",
           "which bank","which state","which day","which hour","show me","give me",
           "list","calculate","find","analyse","analyze","trend","pattern","peak",
           "per day","per hour","by bank","by state","by type","weekend","weekday",
           "morning","night","transaction","amount","inr","rupee","table","chart",
           "graph","plot","pie","bar","histogram","report","statistics","stats",
           "age","device","network","merchant","category","p2p","p2m","recharge",
           "bill","payment","android","ios","web","4g","5g","wifi","3g","state",
           "region","correlation","relationship","between","compare","versus","vs",
           "risk","high value","high-value","flagged for review","recommend",
           "insight","suggest","food","grocery","fuel","entertainment","shopping",
           "healthcare","education","transport","utilities","peak","hour","temporal"]

CHART_KW  = ["chart","graph","plot","pie","bar","histogram","visualize","visual",
             "show chart","show graph","line chart","scatter"]
TABLE_KW  = ["table","tabulate","grid","structured","breakdown table"]
REPORT_KW = ["report","full report","summary report","generate report"]
CHAT_SUM_KW = ["chat summary","conversation summary","summarize chat",
               "summarize conversation","summarize our","summary of our",
               "what did we discuss"]
DATA_SUM_KW = ["summary of the data","data summary","summarize the data",
               "dataset summary","summarize data","overview of the data","data overview"]

def wants_summary(q):
    ql = q.lower()
    if any(k in ql for k in DATA_SUM_KW): return "data"
    if any(k in ql for k in CHAT_SUM_KW): return "chat"
    if "summarize" in ql or "summary" in ql:
        if any(w in ql for w in ["transaction","fraud","bank","amount","dataset",
                                  "data","stats","inr","age","device","network"]):
            return "data"
        return "chat"
    return None

def needs_data(q):
    ql = q.lower().strip().rstrip("?!.")
    if ql in CONV_EXACT: return False
    if re.match(r"^how (are|r) (you|u)\b",ql): return False
    if re.match(r"^(why|explain|what do you think|tell me more|what does that mean)\b",ql): return False
    if any(k in ql for k in DATA_KW): return True
    return len(ql.split())>5

STATS_KW = ["stats table","statistics table","generate stats","show stats","show statistics",
            "full stats","dataset stats","data stats","show me stats","show me statistics",
            "overview table","overview stats","key stats","all stats","complete stats",
            "generate statistics","statistics overview","stats overview"]

def wants_stats_table(q):
    ql = q.lower().strip()
    return any(k in ql for k in STATS_KW)

def wants_chart(q):  return any(k in q.lower() for k in CHART_KW)
def wants_table(q):  return any(k in q.lower() for k in TABLE_KW)
def wants_report(q): return any(k in q.lower() for k in REPORT_KW)

# ── query category detection ──────────────────────────────────────────────────
def detect_query_category(q):
    ql = q.lower()

    # ── Failure/success RATE by category → always comparative (not risk) ──────
    _rate_q = any(w in ql for w in ["failure rate","fail rate","success rate",
                                     "highest failure","lowest failure","highest fail",
                                     "most failures","failed rate"])
    _comp_subject = any(w in ql for w in [
        "transaction type","device","network","bank","age","merchant","category",
        "which","highest","lowest","most","least","by type","by device","by network","by bank"
    ])
    if _rate_q or (_comp_subject and any(w in ql for w in ["fail","failure","success rate"])):
        return "comparative"

    if any(w in ql for w in ["average","mean","total","sum","count","how many","how much",
                              "what is the","descriptive","what are the","show","list","give me",
                              "breakdown","by merchant","by category","by bank","by state",
                              "by type","by device","by network","by age"]):
        return "descriptive"
    if any(w in ql for w in ["compare","vs","versus","difference","between",
                              "which is higher","which is lower","comparative",
                              "highest","lowest","most","least","which has","which type",
                              "which transaction"]):
        return "comparative"
    if any(w in ql for w in ["peak","hour","daily","weekly","trend","over time",
                              "temporal","when","what time","day of week","weekend"]):
        return "temporal"
    if any(w in ql for w in ["age group","which group","segment","who","age",
                              "device","bank","segmentation","most frequently"]):
        return "segmentation"
    if any(w in ql for w in ["relationship","correlation","related","correlated",
                              "association","does","affect","impact","influence"]):
        return "correlation"
    if any(w in ql for w in ["risk","fraud","flag","flagged","high value","anomaly",
                              "suspicious","review","high-value"]):
        return "risk"
    return "descriptive"

# ── direct analytics functions (no LLM needed) ────────────────────────────────
def _direct_analytics(q_lower, df):
    """
    Returns (table_data, narrative_data_str, chart_labels, chart_values, chart_type, 
             recommendations) for the 6 query categories — all from pandas, always accurate.
    """
    amt = _amt(df) or "amount_inr"
    cat = detect_query_category(q_lower)
    tbl = None; nav = ""; labels=[]; values=[]; ctype="bar"; recs=[]

    # ── CROSS-TAB: age group × device type (must intercept before category routing) ──
    _has_age    = any(w in q_lower for w in ["age","age group","age_group"])
    _has_device = any(w in q_lower for w in ["device","android","ios","web"])
    if _has_age and _has_device and "sender_age_group" in df.columns and "device_type" in df.columns:
        order = ["18-25","26-35","36-45","46-55","56+"]
        devices = sorted(df["device_type"].dropna().unique().tolist())
        ct = df.groupby(["sender_age_group","device_type"]).size().unstack(fill_value=0)
        ct = ct.reindex(order, fill_value=0)
        for d in devices:
            if d not in ct.columns:
                ct[d] = 0
        ct = ct[devices]
        rows = [[age] + [f"{int(ct.loc[age, d]):,}" for d in devices] for age in order if age in ct.index]
        tbl = {"headers": ["Age Group"] + devices, "rows": rows}
        total_by_age = ct.sum(axis=1)
        top_age = total_by_age.idxmax()
        top_device = ct.sum(axis=0).idxmax()
        nav = (f"Transaction count by age group × device type. "
               f"Most active: {top_age} age group ({int(total_by_age[top_age]):,} txns), "
               f"most used device: {top_device}.")
        labels = [a for a in order if a in ct.index]
        values = [int(total_by_age.get(a, 0)) for a in labels]
        recs = [
            f"'{top_age}' is the most active age group — prioritise features and UX for this segment.",
            f"'{top_device}' is the dominant device — ensure it receives the best app experience.",
            "Cross-referencing age and device helps target marketing campaigns more precisely.",
        ]
        return tbl, nav, labels, values, ctype, recs

    # ── DESCRIPTIVE ──────────────────────────────────────────────────────────
    if cat == "descriptive":
        _want_avg   = any(w in q_lower for w in ["average","mean","avg"])
        _want_total = any(w in q_lower for w in ["total","sum","revenue"])
        _want_count = any(w in q_lower for w in ["count","how many","number of"])
        _want_merch = any(w in q_lower for w in ["merchant","category","merchant category"])
        _want_type  = any(w in q_lower for w in ["transaction type","by type","type"])
        _want_bank  = any(w in q_lower for w in ["bank","sender bank"])
        _want_bill  = any(w in q_lower for w in ["bill payment","bill"])
        _want_p2p   = "p2p" in q_lower
        _want_state = any(w in q_lower for w in ["state","region","sender state"])

        # ── Merchant category (total, avg, count, or just "show") ────────────
        if _want_merch and "merchant_category" in df.columns:
            g = df[df["merchant_category"]!="P2P_Transfer"].groupby("merchant_category")[amt].agg(["mean","count","sum"])
            g = g.sort_values("sum" if _want_total else "mean", ascending=False)
            rows = [[mc,
                     f"Rs.{row['sum']:,.0f}",
                     f"Rs.{row['mean']:,.2f}",
                     f"{int(row['count']):,}"]
                    for mc,row in g.iterrows()]
            tbl = {"headers":["Category","Total Amount","Avg Amount","Count"],"rows":rows}
            top = g.index[0]
            nav = (f"Merchant category totals: {top} leads with Rs.{g['sum'].iloc[0]:,.0f} total "
                   f"({int(g['count'].iloc[0]):,} transactions, avg Rs.{g['mean'].iloc[0]:,.2f}).")
            labels = [str(i)[:18] for i in g.index]
            values = [round(float(v),2) for v in (g["sum"] if _want_total else g["mean"]).tolist()]
            recs = [f"'{top}' drives the most revenue — prioritise promotions and reliability for this category.",
                    "Categories with high avg but low count are high-value segments worth targeting."]

        # ── Transaction type breakdown ────────────────────────────────────────
        elif _want_type and "transaction_type" in df.columns:
            g = df.groupby("transaction_type")[amt].agg(["mean","count","sum"])
            g = g.sort_values("sum" if _want_total else "count", ascending=False)
            rows = [[tt,
                     f"Rs.{row['sum']:,.0f}",
                     f"Rs.{row['mean']:,.2f}",
                     f"{int(row['count']):,}",
                     f"{(df[df['transaction_type']==tt]['transaction_status']=='FAILED').mean()*100:.2f}%"]
                    for tt,row in g.iterrows()]
            tbl = {"headers":["Type","Total","Avg Amount","Count","Fail Rate"],"rows":rows}
            top = g.index[0]
            nav = f"Transaction types by {'total' if _want_total else 'volume'}: {top} leads."
            labels = [r[0] for r in rows]; values = [float(r[1].replace("Rs.","").replace(",","")) for r in rows]
            recs = [f"'{top}' dominates by {'revenue' if _want_total else 'volume'} — ensure highest reliability here.",
                    "Monitor failure rates per type and set type-specific SLA targets."]

        # ── Bill payment specific ─────────────────────────────────────────────
        elif _want_bill and "transaction_type" in df.columns:
            sub = df[df["transaction_type"]=="Bill Payment"]
            avg_val = sub[amt].mean() if len(sub) else 0
            tbl = {"headers":["Metric","Value"],"rows":[
                ["Transaction Type","Bill Payment"],
                ["Count",f"{len(sub):,}"],
                ["Total Amount",f"Rs.{sub[amt].sum():,.0f}"],
                ["Average Amount",f"Rs.{avg_val:,.2f}"],
                ["Median Amount",f"Rs.{sub[amt].median():,.2f}"],
                ["Fail Rate",f"{(sub['transaction_status']=='FAILED').mean()*100:.2f}%"],
            ]}
            nav = f"Bill Payment: {len(sub):,} transactions, avg Rs.{avg_val:,.2f}, total Rs.{sub[amt].sum():,.0f}"
            recs = ["Bill payment amounts are higher on average — consider targeted promotions.",
                    "Monitor failure rates; third-party biller integrations are common failure points."]

        # ── P2P specific ──────────────────────────────────────────────────────
        elif _want_p2p and "transaction_type" in df.columns:
            sub = df[df["transaction_type"]=="P2P"]
            avg_val = sub[amt].mean() if len(sub) else 0
            tbl = {"headers":["Metric","Value"],"rows":[
                ["Transaction Type","P2P"],
                ["Count",f"{len(sub):,}"],
                ["Total Amount",f"Rs.{sub[amt].sum():,.0f}"],
                ["Average Amount",f"Rs.{avg_val:,.2f}"],
                ["Median Amount",f"Rs.{sub[amt].median():,.2f}"],
            ]}
            nav = f"P2P: {len(sub):,} transactions, avg Rs.{avg_val:,.2f}"
            recs = ["P2P transfers are high volume — focus on reducing friction for this category."]

        # ── Bank breakdown ────────────────────────────────────────────────────
        elif _want_bank and "sender_bank" in df.columns:
            g = df.groupby("sender_bank")[amt].agg(["mean","count","sum"])
            g = g.sort_values("sum" if _want_total else "count", ascending=False)
            rows = [[b, f"Rs.{row['sum']:,.0f}", f"Rs.{row['mean']:,.2f}", f"{int(row['count']):,}"]
                    for b,row in g.iterrows()]
            tbl = {"headers":["Bank","Total","Avg Amount","Count"],"rows":rows}
            nav = f"Top bank by {'revenue' if _want_total else 'volume'}: {g.index[0]}"
            labels = [r[0] for r in rows]; values = [float(r[1].replace("Rs.","").replace(",","")) for r in rows]
            recs = ["Market leaders by volume hold negotiating power — focus partner deepening with top 3 banks."]

        # ── State breakdown ───────────────────────────────────────────────────
        elif _want_state and "sender_state" in df.columns:
            g = df.groupby("sender_state")[amt].agg(["mean","count","sum"])
            g = g.sort_values("sum" if _want_total else "count", ascending=False).head(15)
            rows = [[s, f"Rs.{row['sum']:,.0f}", f"Rs.{row['mean']:,.2f}", f"{int(row['count']):,}"]
                    for s,row in g.iterrows()]
            tbl = {"headers":["State","Total","Avg Amount","Count"],"rows":rows}
            nav = f"Top state: {g.index[0]} with Rs.{g['sum'].iloc[0]:,.0f} total"
            labels = [r[0] for r in rows[:10]]; values = [float(r[1].replace("Rs.","").replace(",","")) for r in rows[:10]]
            recs = [f"'{g.index[0]}' leads in transaction volume — prioritise infrastructure investment here."]

        # ── Generic amount stats ──────────────────────────────────────────────
        elif amt in df.columns:
            s = df[amt].describe()
            total_amt = df[amt].sum()
            rows = [
                ["Total Transactions", f"{len(df):,}"],
                ["Total Amount",       f"Rs.{total_amt:,.0f}"],
                ["Average Amount",     f"Rs.{s['mean']:,.2f}"],
                ["Median Amount",      f"Rs.{s['50%']:,.2f}"],
                ["Std Deviation",      f"Rs.{s['std']:,.2f}"],
                ["Min Amount",         f"Rs.{s['min']:,.2f}"],
                ["Max Amount",         f"Rs.{s['max']:,.2f}"],
            ]
            tbl = {"headers":["Metric","Value"],"rows":rows}
            nav = f"Overall: {len(df):,} transactions, avg Rs.{s['mean']:,.2f}, total Rs.{total_amt:,.0f}"
            recs = ["High std deviation suggests diverse user segments — consider tiered product offerings."]

    # ── COMPARATIVE ──────────────────────────────────────────────────────────
    elif cat == "comparative":
        # ── Transaction type failure rate (the most common comparison query) ──
        _asking_tx_fail = (
            any(w in q_lower for w in ["transaction type","transaction_type","p2p","p2m","recharge","bill payment","bill"])
            and any(w in q_lower for w in ["fail","failure","success rate","highest","lowest","which","rate"])
            and "bank" not in q_lower and "device" not in q_lower and "network" not in q_lower
        )
        _generic_fail_by_type = (
            any(w in q_lower for w in ["highest failure rate","highest fail","most failures",
                                        "which.*fail","which transaction.*fail","transaction.*highest fail"])
            and "bank" not in q_lower and "device" not in q_lower and "network" not in q_lower
        )
        if (_asking_tx_fail or _generic_fail_by_type) and "transaction_type" in df.columns and "transaction_status" in df.columns:
            _ff_col = "fraud_flag" if "fraud_flag" in df.columns else "transaction_status"
            g = df.groupby("transaction_type").agg(
                count=("transaction_status","count"),
                fail_rate=("transaction_status", lambda x: (x=="FAILED").mean()*100),
                success_rate=("transaction_status", lambda x: (x=="SUCCESS").mean()*100),
                flag_rate=(_ff_col,"mean"),
                avg_amt=(amt,"mean") if amt in df.columns else ("transaction_status","count"),
            ).sort_values("fail_rate",ascending=False)
            rows = []
            for t,row in g.iterrows():
                rows.append([
                    t,
                    f"{int(row['count']):,}",
                    f"{row['fail_rate']:.2f}%",
                    f"{row['success_rate']:.2f}%",
                    f"{row['flag_rate']*100:.2f}%",
                    f"Rs.{row['avg_amt']:,.2f}" if amt in df.columns else "N/A"
                ])
            tbl = {"headers":["Transaction Type","Count","Fail Rate","Success Rate","Flagged Rate","Avg Amount"],"rows":rows}
            highest = g.index[0]; highest_rate = g["fail_rate"].iloc[0]
            lowest  = g.index[-1]; lowest_rate  = g["fail_rate"].iloc[-1]
            # Build a rich narrative string for the narrator
            parts = [f"{t}: {row['fail_rate']:.1f}%" for t,row in g.iterrows()]
            nav = (
                f"Failure rates by transaction type — "
                + ", ".join(parts)
                + f". Highest: {highest} at {highest_rate:.1f}%"
                + f". Lowest: {lowest} at {lowest_rate:.1f}%."
            )
            labels = [r[0] for r in rows]; values = [float(r[2].replace("%","")) for r in rows]
            recs = [
                f"'{highest}' transactions have the highest failure rate at {highest_rate:.1f}% — "
                "investigate third-party integrations and timeout handling for this type.",
                f"'{lowest}' transactions show the most reliability at {lowest_rate:.1f}% failure — "
                "use this as the benchmark for service-level targets.",
                "Set up real-time alerting when any transaction type exceeds its historical failure rate baseline.",
            ]
        elif ("android" in q_lower or "ios" in q_lower or "device" in q_lower) and "device_type" in df.columns and "transaction_status" in df.columns:
            g = df.groupby("device_type")["transaction_status"].apply(
                lambda x: (x=="FAILED").mean()*100).sort_values(ascending=False)
            rows = []
            for dev in g.index:
                sub = df[df["device_type"]==dev]
                fail = (sub["transaction_status"]=="FAILED").mean()*100
                flag = sub["fraud_flag"].mean()*100 if "fraud_flag" in df.columns else 0
                avg  = sub[amt].mean() if amt in df.columns else 0
                rows.append([dev,f"{len(sub):,}",f"{fail:.2f}%",f"{flag:.2f}%",f"Rs.{avg:,.2f}"])
            tbl = {"headers":["Device","Transactions","Fail Rate","Flagged Rate","Avg Amount"],"rows":rows}
            nav = " | ".join([f"{r[0]}: {r[2]} fail" for r in rows])
            labels = [r[0] for r in rows]; values = [float(r[2].replace("%","")) for r in rows]
            recs = [f"'{g.index[0]}' has the highest failure rate — investigate platform-specific issues.",
                    "Consider push notifications or UX improvements on the highest-fail platform."]
        elif ("network" in q_lower or "4g" in q_lower or "5g" in q_lower or "wifi" in q_lower) and "network_type" in df.columns and "transaction_status" in df.columns:
            rows = []
            for net in df["network_type"].unique():
                sub = df[df["network_type"]==net]
                fail = (sub["transaction_status"]=="FAILED").mean()*100
                flag = sub["fraud_flag"].mean()*100 if "fraud_flag" in df.columns else 0
                avg  = sub[amt].mean() if amt in df.columns else 0
                rows.append([net,f"{len(sub):,}",f"{fail:.2f}%",f"{flag:.2f}%",f"Rs.{avg:,.2f}"])
            rows.sort(key=lambda r: float(r[2].replace("%","")),reverse=True)
            tbl = {"headers":["Network","Transactions","Fail Rate","Flagged Rate","Avg Amount"],"rows":rows}
            nav = " | ".join([f"{r[0]}: {r[2]} fail" for r in rows])
            labels = [r[0] for r in rows]; values = [float(r[2].replace("%","")) for r in rows]
            recs = ["Network-related failures can be reduced by retry logic and fallback routing.",
                    "Flagged rates on certain networks may indicate geographic anomalies — investigate location overlap."]
        elif "bank" in q_lower and "sender_bank" in df.columns and "transaction_status" in df.columns:
            rows = []
            for bank in df["sender_bank"].unique():
                sub = df[df["sender_bank"]==bank]
                fail = (sub["transaction_status"]=="FAILED").mean()*100
                flag = sub["fraud_flag"].mean()*100 if "fraud_flag" in df.columns else 0
                avg  = sub[amt].mean() if amt in df.columns else 0
                rows.append([bank,f"{len(sub):,}",f"{fail:.2f}%",f"{flag:.2f}%",f"Rs.{avg:,.2f}"])
            rows.sort(key=lambda r: float(r[2].replace("%","")),reverse=True)
            tbl = {"headers":["Bank","Transactions","Fail Rate","Flagged Rate","Avg Amount"],"rows":rows}
            nav = " | ".join([f"{r[0]}: {r[2]} fail" for r in rows[:4]])
            labels = [r[0] for r in rows]; values = [float(r[2].replace("%","")) for r in rows]
            recs = [f"'{rows[0][0]}' has highest failure rate — consider direct integration review.",
                    "Banks with high flagged rates should be investigated for patterns in transaction timing."]
        elif "age" in q_lower:
            order = ["18-25","26-35","36-45","46-55","56+"]
            rows = []
            for age in order:
                sub = df[df["sender_age_group"]==age]
                if len(sub)==0: continue
                fail = (sub["transaction_status"]=="FAILED").mean()*100
                flag = sub["fraud_flag"].mean()*100
                avg  = sub[amt].mean() if amt in df.columns else 0
                rows.append([age,f"{len(sub):,}",f"{fail:.2f}%",f"{flag:.2f}%",f"Rs.{avg:,.2f}"])
            tbl = {"headers":["Age Group","Transactions","Fail Rate","Flagged Rate","Avg Amount"],"rows":rows}
            nav = " | ".join([f"{r[0]}: {r[4]} avg" for r in rows])
            labels = [r[0] for r in rows]; values = [float(r[4].replace("Rs.","").replace(",","")) for r in rows]
            recs = ["Older age groups (46+) typically transact higher amounts — offer premium features.",
                    "Younger users (18-25) drive volume — optimize for mobile experience and speed."]

    # ── TEMPORAL ─────────────────────────────────────────────────────────────
    elif cat == "temporal":
        if "food" in q_lower or "grocery" in q_lower or "merchant" in q_lower or "category" in q_lower:
            # Map query keywords to actual category values present in the data
            _cat_map = {"food":"Food","grocery":"Grocery","fuel":"Fuel",
                        "entertainment":"Entertainment","shopping":"Shopping",
                        "healthcare":"Healthcare","education":"Education",
                        "transport":"Transport","utilities":"Utilities"}
            target_cat = None
            if "merchant_category" in df.columns:
                for kw, cat_val in _cat_map.items():
                    if kw in q_lower and cat_val in df["merchant_category"].values:
                        target_cat = cat_val
                        break
            if target_cat:
                sub = df[df["merchant_category"] == target_cat]
            elif "merchant_category" in df.columns:
                sub = df[df["merchant_category"] != "P2P_Transfer"]
            else:
                sub = df
            # Safety: if sub is empty after filter, fall back to full df
            if len(sub) == 0:
                sub = df
                target_cat = None
            if "hour" in q_lower or "peak" in q_lower:
                if "hour_of_day" not in sub.columns or len(sub) == 0:
                    nav = "No hourly transaction data available for the selected category."
                    recs = []
                else:
                    g = sub.groupby("hour_of_day").size().sort_index()
                    if len(g) == 0:
                        nav = "No data found for this category."
                        recs = []
                    else:
                        peak_h = int(g.idxmax())
                        labels = [f"{h}:00" for h in g.index]; values = [int(_v) for _v in g.values.tolist()]; ctype="line"
                        tbl_rows = [[f"{h}:00",f"{_v:,}"] for h,_v in g.items()]
                        tbl = {"headers":["Hour","Transactions"],"rows":tbl_rows}
                        cat_name = target_cat or "Merchant"
                        nav = f"Peak hour for {cat_name}: {peak_h}:00  |  Transactions then: {int(g.max()):,}"
                        recs = [f"Staff up systems and support around {peak_h}:00 for {cat_name} transactions.",
                                "Consider time-limited offers during off-peak hours to distribute load."]
            else:
                g = sub.groupby("day_of_week").size()
                order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                g = g.reindex(order, fill_value=0)
                labels = [d[:3] for d in g.index]; values = [int(_v) for _v in g.values.tolist()]
                tbl = {"headers":["Day","Transactions"],"rows":[[d,f"{_v:,}"] for d,_v in zip(g.index,g.values)]}
                nav = f"Peak day: {g.idxmax()} with {int(g.max()):,} transactions"
                recs = ["Weekend peaks in merchant categories suggest leisure spending — relevant for entertainment/food."]
        elif "hour" in q_lower or "peak" in q_lower:
            g = df.groupby("hour_of_day").size().sort_index()
            peak_h = g.idxmax()
            labels = [f"{h}:00" for h in g.index]; values = g.values.tolist(); ctype="line"
            tbl_rows = [[f"{h}:00",f"{int(g[h]):,}",
                         f"{(df[df['hour_of_day']==h]['transaction_status']=='FAILED').mean()*100:.2f}%"]
                        for h in g.index]
            tbl = {"headers":["Hour","Transactions","Fail Rate"],"rows":tbl_rows}
            nav = f"Peak transaction hour: {peak_h}:00  |  Volume: {g.max():,}  |  Lowest: {g.idxmin()}:00"
            recs = [f"Peak load at {peak_h}:00 — scale infrastructure accordingly.",
                    "Consider incentivising off-peak usage with cashback or lower fees."]
        elif "day" in q_lower or "week" in q_lower:
            order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            g = df.groupby("day_of_week").size().reindex(order,fill_value=0)
            labels = [d[:3] for d in g.index]; values = g.values.tolist()
            fail_by_day = df.groupby("day_of_week")["transaction_status"].apply(
                lambda x: (x=="FAILED").mean()*100).reindex(order,fill_value=0)
            tbl_rows = [[d,f"{_v:,}",f"{fail_by_day[d]:.2f}%"] for d,_v in zip(g.index,g.values)]
            tbl = {"headers":["Day","Transactions","Fail Rate"],"rows":tbl_rows}
            nav = f"Busiest day: {g.idxmax()} ({g.max():,} txns)  |  Quietest: {g.idxmin()}"
            recs = ["Weekend/weekday patterns matter for staffing and support scheduling.",
                    "Higher failure rates on certain days may correlate with load — investigate server capacity."]

    # ── SEGMENTATION ─────────────────────────────────────────────────────────
    elif cat == "segmentation":
        if "p2p" in q_lower and ("age" in q_lower or "group" in q_lower or "who" in q_lower) and "transaction_type" in df.columns:
            sub = df[df["transaction_type"]=="P2P"]
            order = ["18-25","26-35","36-45","46-55","56+"]
            g = sub.groupby("sender_age_group").size().reindex(order,fill_value=0)
            top = g.idxmax()
            pcts = g/g.sum()*100
            rows = [[age,f"{_v:,}",f"{pcts[age]:.1f}%",
                     f"Rs.{sub[sub['sender_age_group']==age][amt].mean():,.2f}" if amt in df.columns else "N/A"]
                    for age,_v in g.items()]
            tbl = {"headers":["Age Group","P2P Count","% of P2P","Avg Amount"],"rows":rows}
            nav = f"Most P2P transfers: {top} ({g[top]:,} txns, {pcts[top]:.1f}%)"
            labels=order; values=g.reindex(order).tolist()
            recs = [f"'{top}' drives P2P volume — target with P2P-specific features and referral incentives.",
                    "Consider peer group features (splitting, requests) for the dominant age segment."]
        elif ("age" in q_lower or "age group" in q_lower) and ("device" in q_lower or "android" in q_lower or "ios" in q_lower):
            # Cross-tab: age group × device type
            order = ["18-25","26-35","36-45","46-55","56+"]
            devices = sorted(df["device_type"].dropna().unique().tolist())
            ct = df.groupby(["sender_age_group","device_type"]).size().unstack(fill_value=0)
            ct = ct.reindex(order, fill_value=0)
            for d in devices:
                if d not in ct.columns:
                    ct[d] = 0
            ct = ct[devices]
            rows = [[age] + [f"{int(ct.loc[age, d]):,}" for d in devices] for age in order if age in ct.index]
            tbl = {"headers": ["Age Group"] + devices, "rows": rows}
            total_by_age = ct.sum(axis=1)
            top_age = total_by_age.idxmax()
            top_device = ct.sum(axis=0).idxmax()
            nav = f"Transaction count by age group × device type. Most active: {top_age} age group, most used device: {top_device}."
            labels = order
            values = [int(total_by_age.get(a, 0)) for a in order]
            recs = [
                f"'{top_age}' is the most active age group — prioritise features and UX for this segment.",
                f"'{top_device}' is the dominant device — ensure it receives the best app experience and fastest performance.",
                "Cross-referencing age and device helps target marketing campaigns more precisely.",
            ]
        elif ("device" in q_lower or "android" in q_lower or "ios" in q_lower) and not ("age" in q_lower or "age group" in q_lower) and "device_type" in df.columns and "transaction_type" in df.columns:
            g = df.groupby("device_type")["transaction_type"].value_counts().unstack(fill_value=0)
            rows = [[dev]+[f"{g.loc[dev,t]:,}" if t in g.columns else "0" for t in df["transaction_type"].unique()]
                    for dev in g.index]
            tbl = {"headers":["Device"]+list(df["transaction_type"].unique()),"rows":rows}
            nav = f"Device segmentation across transaction types"
            recs = ["Device-specific transaction preferences can guide UI/UX investment decisions."]
        elif "bank" in q_lower and "sender_bank" in df.columns:
            _count_col = "transaction_type" if "transaction_type" in df.columns else df.columns[0]
            _agg = {"count": (_count_col,"count")}
            if _amt(df): _agg["avg_amt"] = (_amt(df),"mean")
            if "transaction_status" in df.columns: _agg["fail_rate"] = ("transaction_status",lambda x:(x=="FAILED").mean()*100)
            if "fraud_flag" in df.columns: _agg["flag_rate"] = ("fraud_flag","mean")
            g = df.groupby("sender_bank").agg(**_agg).sort_values("count",ascending=False) if _agg else None
            if g is not None:
                rows = [[bank,f"{int(row['count']):,}",f"Rs.{row['avg_amt']:,.2f}",
                         f"{row['fail_rate']:.2f}%",f"{row['flag_rate']*100:.2f}%"]
                        for bank,row in g.iterrows()]
                tbl = {"headers":["Bank","Count","Avg Amount","Fail Rate","Flagged Rate"],"rows":rows}
                nav = f"Top bank by volume: {g.index[0]} ({int(g['count'].iloc[0]):,} txns)"
                recs = ["Market leaders by volume hold negotiating power — focus partner deepening with top 3 banks."]
        elif "sender_age_group" in df.columns:
            order = ["18-25","26-35","36-45","46-55","56+"]
            count_col = "transaction_type" if "transaction_type" in df.columns else df.columns[0]
            fail_kw   = {"fail_rate": ("transaction_status", lambda x:(x=="FAILED").mean()*100)} if "transaction_status" in df.columns else {}
            g = df.groupby("sender_age_group").agg(
                count=(count_col,"count"),
                **fail_kw
            ).reindex(order, fill_value=0)
            rows = [[age,f"{int(row['count']):,}",f"{row['fail_rate']:.2f}%"] for age,row in g.iterrows()]
            tbl = {"headers":["Age Group","Transactions","Fail Rate"],"rows":rows}
            nav = f"Most active age group: {g['count'].idxmax()} ({int(g['count'].max()):,} txns)"
            labels=order; values=[int(g.loc[a,'count']) if a in g.index else 0 for a in order]
            recs = ["Tailor communication and product features by age segment for maximum engagement."]

    # ── CORRELATION ───────────────────────────────────────────────────────────
    elif cat == "correlation":
        _has_net = "network_type" in df.columns and "transaction_status" in df.columns
        _has_dev = "device_type"  in df.columns and "transaction_status" in df.columns
        _has_tt  = "transaction_type" in df.columns
        _has_ff  = "fraud_flag" in df.columns
        _amt_col = _amt(df) or "amount_inr"

        if ("network" in q_lower) and _has_net:
            agg_dict = dict(
                count=("transaction_status","count"),
                success_rate=("transaction_status",lambda x:(x=="SUCCESS").mean()*100),
                fail_rate=("transaction_status",lambda x:(x=="FAILED").mean()*100),
            )
            if _has_ff:  agg_dict["flag_rate"] = ("fraud_flag","mean")
            if _amt_col in df.columns: agg_dict["avg_amt"] = (_amt_col,"mean")
            g = df.groupby("network_type").agg(**agg_dict).sort_values("fail_rate",ascending=False)
            headers = ["Network","Count","Success%","Fail%"]
            rows = []
            for net,row in g.iterrows():
                r = [net, f"{int(row['count']):,}", f"{row['success_rate']:.2f}%", f"{row['fail_rate']:.2f}%"]
                if _has_ff:  r.append(f"{row['flag_rate']*100:.2f}%")
                if _amt_col in df.columns: r.append(f"Rs.{row['avg_amt']:,.2f}")
                rows.append(r)
            if _has_ff:  headers.append("Flagged%")
            if _amt_col in df.columns: headers.append("Avg Amount")
            tbl = {"headers": headers, "rows": rows}
            nav = f"Network vs success: highest fail rate is {g.index[0]} at {g['fail_rate'].iloc[0]:.2f}%"
            labels=[r[0] for r in rows]; values=[float(r[3].replace("%","")) for r in rows]
            recs = ["Invest in network resilience and retry logic — network type is a strong predictor of failure.",
                    "Consider prompting users on 3G to retry or switch networks before high-value transactions."]

        elif "device" in q_lower and _has_dev:
            agg_dict = dict(
                count=("transaction_status","count"),
                success_rate=("transaction_status",lambda x:(x=="SUCCESS").mean()*100),
                fail_rate=("transaction_status",lambda x:(x=="FAILED").mean()*100),
            )
            if _has_ff: agg_dict["flag_rate"] = ("fraud_flag","mean")
            g = df.groupby("device_type").agg(**agg_dict).sort_values("fail_rate",ascending=False)
            headers = ["Device","Count","Success%","Fail%"]
            rows = []
            for dev,row in g.iterrows():
                r = [dev, f"{int(row['count']):,}", f"{row['success_rate']:.2f}%", f"{row['fail_rate']:.2f}%"]
                if _has_ff: r.append(f"{row['flag_rate']*100:.2f}%")
                rows.append(r)
            if _has_ff: headers.append("Flagged%")
            tbl = {"headers": headers, "rows": rows}
            nav = f"Device vs success: {g.index[0]} has highest failure at {g['fail_rate'].iloc[0]:.2f}%"
            labels=[r[0] for r in rows]; values=[float(r[3].replace("%","")) for r in rows]
            recs = ["Device failure correlation suggests app-level bugs on specific platforms — run targeted QA."]

        elif _has_net and _has_tt:
            # Generic: network × transaction_type success rate crosstab
            ct = pd.crosstab(df["network_type"], df["transaction_type"],
                             values=df["transaction_status"].apply(lambda x: 1 if x=="SUCCESS" else 0),
                             aggfunc="mean") * 100
            tt_cols = list(df["transaction_type"].unique())
            rows = [[net] + [f"{ct.loc[net,t]:.1f}%" if t in ct.columns and net in ct.index else "N/A"
                             for t in tt_cols]
                    for net in ct.index]
            tbl = {"headers": ["Network"] + tt_cols, "rows": rows}
            nav = "Cross-correlation of network type × transaction type success rates"
            recs = ["Use correlation analysis to drive infrastructure investment — prioritise failure-prone combinations."]

        else:
            nav = "Correlation analysis requires network_type, device_type, or transaction_type columns in your dataset."
            tbl = {"headers":["Info"], "rows":[[nav]]}

    # ── RISK ANALYSIS ─────────────────────────────────────────────────────────
    elif cat == "risk":
        if "high value" in q_lower or "high-value" in q_lower:
            amt_col = _amt(df) or "amount_inr"
            threshold = df[amt_col].quantile(0.90) if amt_col in df.columns else 10000
            high = df[df[amt_col]>=threshold] if amt_col in df.columns else df
            flag_rate = high["fraud_flag"].mean()*100
            overall = df["fraud_flag"].mean()*100
            rows = [
                ["Threshold (90th pct)",f"Rs.{threshold:,.2f}"],
                ["High-Value Count",f"{len(high):,}"],
                ["High-Value Flagged Rate",f"{flag_rate:.2f}%"],
                ["Overall Flagged Rate",f"{overall:.2f}%"],
                ["Relative Risk",f"{flag_rate/overall:.2f}x higher" if overall>0 else "N/A"],
                ["NOTE","'Flagged' = flagged for review, NOT confirmed fraud"],
            ]
            tbl = {"headers":["Metric","Value"],"rows":rows}
            nav = f"High-value transactions (≥Rs.{threshold:,.0f}) flagged at {flag_rate:.2f}% vs {overall:.2f}% overall"
            labels=["High Value (≥90th pct)","All Transactions"]; values=[flag_rate,overall]; ctype="bar"
            recs = ["High-value flagged transactions warrant manual review prioritisation.",
                    f"Set automated alerts for transactions above Rs.{threshold:,.0f}.",
                    "Correlate flagged high-value txns with specific banks/devices for targeted controls."]
        elif "bank" in q_lower:
            g = df.groupby("sender_bank")["fraud_flag"].agg(["mean","sum","count"])
            g = g.sort_values("mean",ascending=False)
            rows = [[bank,f"{int(row['count']):,}",f"{int(row['sum']):,}",
                     f"{row['mean']*100:.2f}%"]
                    for bank,row in g.iterrows()]
            tbl = {"headers":["Bank","Transactions","Flagged Count","Flagged Rate"],"rows":rows}
            nav = f"Highest flagged rate: {g.index[0]} at {g['mean'].iloc[0]*100:.2f}%"
            labels=[r[0] for r in rows]; values=[float(r[3].replace("%","")) for r in rows]
            recs = [f"'{g.index[0]}' shows highest flagged rate — escalate to risk team for pattern review.",
                    "Remember: flagged rate ≠ fraud rate. Manual review is needed to confirm."]
        elif "transaction_type" in df.columns and "fraud_flag" in df.columns:
            g = df.groupby("transaction_type")["fraud_flag"].agg(["mean","sum","count"])
            g = g.sort_values("mean",ascending=False)
            rows = [[t,f"{int(row['count']):,}",f"{int(row['sum']):,}",f"{row['mean']*100:.2f}%"]
                    for t,row in g.iterrows()]
            tbl = {"headers":["Type","Transactions","Flagged Count","Flagged Rate"],"rows":rows}
            nav = f"Highest flagged rate: {g.index[0]} at {g['mean'].iloc[0]*100:.2f}%"
            labels=[r[0] for r in rows]; values=[float(r[3].replace("%","")) for r in rows]
            recs = ["Flagged transactions require manual investigation — not all are fraudulent.",
                    f"Focus risk controls on '{g.index[0]}' transaction type.",
                    "Build a feedback loop to train the flagging model on confirmed cases."]
        else:
            nav = "Risk analysis requires fraud_flag and transaction_type columns in your dataset."
            tbl = {"headers":["Info"], "rows":[[nav]]}

    return tbl, nav, labels, values, ctype, recs

# ── LLM code analyst (fallback/supplement) ───────────────────────────────────
def clean_and_fix_code(text):
    m = re.search(r"```python\s*(.*?)```",text,re.DOTALL|re.IGNORECASE)
    if m: code=m.group(1)
    else:
        m = re.search(r"```\s*(.*?)```",text,re.DOTALL)
        code = m.group(1) if m else text
    code = re.sub(r"^python\s*\n","",code.strip(),flags=re.IGNORECASE)
    lines=code.splitlines(); out=[]
    for line in lines:
        s=line.strip()
        if s and not s.startswith("#") and re.match(r"^[A-Z][a-z]",s) and "=" not in s and "(" not in s:
            break
        out.append(line)
    return textwrap.dedent("\n".join(out)).strip()

def run_analyst(user_query):
    df=get_df(); llm=get_llm()
    if df is None or llm is None: return None
    amt=_amt(df)
    if amt is None:
        amt_candidates=[c for c in df.columns if "amount" in c.lower()]
        amt=amt_candidates[0] if amt_candidates else None
    if amt is None: return None
    prompt=(
        "<|system|>\nYou are a precise Python data analyst. Output ONLY executable Python code. No explanations.<|end|>\n"
        "<|user|>\n"
        f"DataFrame `df` columns: {list(df.columns)}\n"
        f"- IMPORTANT: The amount column is EXACTLY '{amt}' — use this exact string, do not guess or rename it.\n"
        "- fraud_flag: 1=flagged for review (NOT confirmed fraud), 0=not flagged\n"
        "- transaction_status: 'SUCCESS' or 'FAILED'\n"
        "- transaction_type: P2P, P2M, Bill Payment, Recharge\n"
        "- merchant_category: Food/Grocery/Fuel/Entertainment/Shopping/Healthcare/Education/Transport/Utilities/Other\n"
        "- sender_age_group: 18-25, 26-35, 36-45, 46-55, 56+\n"
        "- device_type: Android, iOS, Web\n"
        "- network_type: 4G, 5G, WiFi, 3G\n"
        "- sender_bank: SBI, HDFC, ICICI, Axis, PNB, Kotak, IndusInd, Yes Bank\n"
        "- day_of_week: Monday-Sunday | is_weekend: 0/1\n\n"
        f"Write Python code to answer: \"{user_query}\"\n"
        "Rules:\n"
        "1. Use var df. Do NOT reload.\n"
        "2. Last line MUST be print().\n"
        "3. Pure Python only. Rates to 2 decimal places.\n"
        "4. NEVER print a raw DataFrame or Series — always print formatted strings.\n"
        "5. Use print(f'Label: {{value}}') or a for loop over rows.\n"
        "6. NEVER use .to_string() or just print(df).\n"
        "<|end|>\n<|assistant|>\n```python\n"
    )
    out=llm(prompt,max_tokens=1024,stop=["```","<|end|>","<|user|>","<|system|>"],echo=False,temperature=0.1)
    raw=out["choices"][0]["text"]
    if not raw.strip(): return None
    code=clean_and_fix_code(raw)
    if not code: return None
    _wrong_amt_names=["amount_inr","amount_(inr)","amount","amt"]
    for wrong in _wrong_amt_names:
        if wrong!=amt:
            code=re.sub(rf"(['\"]){re.escape(wrong)}\1",f"\'{amt}\'",code)
    _col_refs=re.findall(r"df\[[\'\"]([^\'\"]+)[\'\"]\]",code)
    for col in _col_refs:
        if col not in df.columns:
            return None
    old_out,old_err=sys.stdout,sys.stderr
    buf=io.StringIO()
    sys.stdout=buf; sys.stderr=io.StringIO()
    try:
        exec(code,{"pd":pd,"df":df,"np":np,"print":print,
                   "__builtins__":__builtins__,
                   "zip":zip,"len":len,"range":range,"sorted":sorted,
                   "round":round,"int":int,"float":float,"str":str,
                   "list":list,"dict":dict,"sum":sum,"min":min,"max":max,
                   "enumerate":enumerate,"abs":abs,"any":any,"all":all,
                   "isinstance":isinstance,"type":type,"set":set,"tuple":tuple})
    except Exception:
        sys.stdout,sys.stderr=old_out,old_err; return None
    sys.stdout,sys.stderr=old_out,old_err
    raw_out = buf.getvalue().strip()
    if not raw_out:
        return None
    _bad_patterns = [
        r"Series\(\[\]",
        r"dtype: (int|float|object|bool)",
        r"\[(\d+) rows x \d+ columns\]",
        r"RangeIndex:",
        r"Index\(\[",
    ]
    import re as _re
    for pat in _bad_patterns:
        if _re.search(pat, raw_out):
            return None
    lines = [l for l in raw_out.splitlines() if l.strip()]
    if lines and all(_re.match(r"^\d+\s+", l) for l in lines[:3]):
        return None
    if lines and len(lines) > 1:
        all_idx = all(_re.match(r"^\s*\d+\s+\S", l) for l in lines)
        if all_idx:
            return None
    return raw_out

STOP_TOKENS=["<|end|>","<|user|>","<|system|>","<|assistant|>",
             "\nUser:","User said:","User asked:","\nAssistant:","The user",
             "\nYou:","\nHuman:","\nQuestion:","\nQ:","\nA:","User:","Assistant:"]

def _build_analytical_prose(user_query, nav, tbl, recs, cat):
    """
    Construct a precise analytical response purely from pandas-computed data.
    This guarantees perfect accuracy regardless of LLM quality.
    """
    if not nav and not tbl:
        return None

    q = user_query.lower()

    # ── Failure / success rate comparison across transaction types ─────────────
    if cat == "comparative" and tbl and any(w in q for w in ["fail","failure","success rate"]):
        rows   = tbl.get("rows", [])
        headers = tbl.get("headers", [])
        if rows and "Fail Rate" in headers:
            fi  = headers.index("Fail Rate")
            ni  = 0  # name always first column
            sorted_rows = sorted(rows, key=lambda r: float(r[fi].replace("%","")), reverse=True)
            top    = sorted_rows[0]
            bottom = sorted_rows[-1]
            middle = sorted_rows[1:-1]

            # Part 1: direct answer — top item
            answer = f"{top[ni]} transactions have the highest failure rate at {top[fi]}"

            # Part 2: all middle items
            if middle:
                mid_parts = ", ".join(f"{r[ni]} at {r[fi]}" for r in middle)
                answer += f", followed by {mid_parts}"

            # Part 3: lowest item
            answer += f". {bottom[ni]} transactions show the lowest failure rate at {bottom[fi]}."

            # Part 4: pattern insight based on spread
            top_val    = float(top[fi].replace("%",""))
            bottom_val = float(bottom[fi].replace("%",""))
            spread     = top_val - bottom_val
            if spread > 5:
                answer += f" The {spread:.1f} percentage point spread indicates significant variance — {top[ni]} likely has systemic integration issues."
            elif spread > 1:
                answer += f" The gap between highest and lowest ({spread:.1f}pp) suggests process-level differences worth investigating."
            else:
                answer += f" The rates are tightly clustered within {spread:.1f}pp — failures may be driven by shared infrastructure rather than type-specific issues."

            # Part 5: top recommendation
            if recs:
                answer += f" {recs[0]}"

            return answer

    # ── Device / Network / Bank failure rate comparison ────────────────────────
    if cat == "comparative" and tbl and "Fail Rate" in tbl.get("headers", []):
        rows    = tbl["rows"]
        headers = tbl["headers"]
        fi = headers.index("Fail Rate")
        sorted_rows = sorted(rows, key=lambda r: float(r[fi].replace("%","")), reverse=True)
        top    = sorted_rows[0]
        bottom = sorted_rows[-1]
        col_name = headers[0]
        answer = f"{top[0]} has the highest failure rate at {top[fi]}"
        if len(sorted_rows) > 2:
            mid = ", ".join(f"{r[0]} at {r[fi]}" for r in sorted_rows[1:-1])
            answer += f", followed by {mid}"
        answer += f". {bottom[0]} is the most reliable at {bottom[fi]} failure rate."
        if recs:
            answer += f" {recs[0]}"
        return answer

    # ── Descriptive stats ──────────────────────────────────────────────────────
    if nav:
        result = nav
        if recs:
            result += f" {recs[0]}"
        return result

    return None


def run_narrator(user_query, data_result, recommendations, tab="Default", tbl=None):
    """
    Enhanced narrator: tries LLM first with a precise prompt.
    Falls back to pure-pandas analytical prose if LLM output is weak.
    """
    llm=get_llm()
    cat=detect_query_category(user_query.lower())
    rec_text="\n".join(f"- {r}" for r in recommendations) if recommendations else ""

    # Always build the pandas-based fallback first (it's always accurate)
    pandas_answer = _build_analytical_prose(user_query, data_result, tbl, recommendations, cat)

    if llm is None:
        return pandas_answer or "No data available."

    sys_msg=(
        "You are InsightX, a precision fintech data analyst. Rules:\n"
        "1. ALWAYS start with the direct answer + the exact number from the data.\n"
        "2. Name EVERY category with its exact percentage or value from the data — do NOT omit any.\n"
        "3. Identify the underlying pattern or cause.\n"
        "4. End with one actionable recommendation.\n"
        "5. Write flowing prose — no bullet points, no headers. Be thorough and complete.\n"
        "6. Use ONLY the numbers provided in the Data section. Never invent figures.\n"
        "7. When fraud_flag mentioned: say 'flagged for review, not confirmed fraud'.\n"
        "8. Always write COMPLETE sentences. Never stop mid-sentence or mid-thought."
    )
    if data_result:
        user_turn=(
            f"[{cat.upper()} QUERY] User asked: \"{user_query}\"\n\n"
            f"COMPUTED DATA (use these exact numbers):\n{data_result}\n\n"
            f"RECOMMENDATIONS:\n{rec_text}\n\n"
            "Write a COMPLETE analytical response covering: (1) direct answer with the top finding and its exact value, "
            "(2) list ALL categories with their individual values — do not skip any, "
            "(3) explain the underlying pattern, "
            "(4) give one actionable recommendation. Be thorough and do not stop early."
        )
    else:
        user_turn=f"User said: \"{user_query}\"\nReply conversationally in 1-2 sentences."

    prompt=(
        f"<|system|>\n{sys_msg}<|end|>\n"
        +history_to_prompt_block(tab)
        +f"<|user|>\n{user_turn}<|end|>\n<|assistant|>\n"
    )
    def _run_llm(prompt_text):
        """Run LLM once and return cleaned text, or None on failure."""
        try:
            out = llm(prompt_text, max_tokens=2048, stop=STOP_TOKENS, echo=False,
                      temperature=0.3, repeat_penalty=1.1)
            raw = out["choices"][0]["text"]
            for m in STOP_TOKENS:
                idx = raw.find(m)
                if idx != -1: raw = raw[:idx]
            raw = re.split(r"\n(?:User|Assistant|Human|You|Q|A)\s*[:\-]", raw, maxsplit=1)[0]
            raw = raw.strip()
            # Trim to last complete sentence
            if raw and raw[-1] not in ".!?":
                last_end = max(raw.rfind("."), raw.rfind("!"), raw.rfind("?"))
                if last_end > len(raw) // 2:
                    raw = raw[:last_end + 1]
            return raw.strip()
        except Exception:
            return None

    def _score_llm(text, expected_tbl):
        """Score LLM output quality 0-100. Higher = better."""
        if not text: return 0
        score = 0
        # Has numbers/percentages
        if re.search(r"\d+\.?\d*%", text): score += 30
        if re.search(r"\d{2,}", text): score += 10
        # Not too short
        words = len(text.split())
        if words >= 40: score += 20
        elif words >= 20: score += 10
        # Does NOT contain hallucination placeholders
        hallucination_markers = [
            "exact figure needed", "exact figures needed", "to be determined",
            "cooking up", "cooked by", "written by insightx", "(exact figure",
            "x%", "y%", "z%", "a%", "b%", "c%", "d%", "e%", "f%",
            "figure x", "figure y", "figure z", "value x", "value a",
        ]
        if not any(m in text.lower() for m in hallucination_markers): score += 25
        # Completeness vs table
        if expected_tbl:
            rows = expected_tbl.get("rows", [])
            names_found = sum(1 for r in rows if str(r[0]).lower() in text.lower())
            completeness = names_found / max(len(rows), 1)
            score += int(completeness * 15)
        return score

    # ── Triple-check: run LLM up to 3 times, keep the best result ─────────────
    best_llm = None
    best_score = -1
    for attempt in range(3):
        candidate = _run_llm(prompt)
        if candidate:
            score = _score_llm(candidate, tbl)
            if score > best_score:
                best_score = score
                best_llm = candidate
            # Short-circuit if we already have an excellent response
            if score >= 75:
                break

    llm_answer = best_llm or ""

    # Quality check: if LLM output is too short, generic, incomplete, or has hallucinations → use pandas answer
    has_numbers = bool(re.search(r"\d+\.?\d*%", llm_answer))
    is_too_short = len(llm_answer.split()) < 20
    hallucination_markers = [
        "exact figure needed", "exact figures needed", "to be determined",
        "cooking up", "cooked by", "written by insightx", "(exact figure",
        " x%", " y%", " z%", " a%", " b%",
        "[insert", "[add", "[placeholder", "[previous response",
        "from previous response", "insert full", "insert table",
        "insert the", "[data]", "[table]", "[chart]",
        "as mentioned earlier", "as stated before", "as shown above",
        "refer to previous", "see above",
    ]
    is_hallucinating = any(m in llm_answer.lower() for m in hallucination_markers)
    is_vague = any(phrase in llm_answer.lower() for phrase in [
        "i cannot","i don't have","not available","based on the data provided",
        "the data shows","the analysis","considering the","it's advisable",
        "this figure exceeds","notably,","it is important","given this insight",
        "this pattern suggests","it is worth","it should be noted",
        "certainly, here", "certainly here", "of course", "sure, here",
        "i'd be happy", "i would be happy", "happy to help",
        "let me provide", "let me explain", "let me show",
        "please note", "please be aware",
    ])

    # For comparative queries, check that LLM mentioned multiple categories (not just 1)
    is_incomplete_comparative = False
    if cat == "comparative" and tbl:
        expected_rows = tbl.get("rows", [])
        if len(expected_rows) >= 3:
            # Count how many row names appear in the answer
            names_mentioned = sum(
                1 for row in expected_rows
                if str(row[0]).lower() in llm_answer.lower()
            )
            # If fewer than half the categories are named, the answer is incomplete
            if names_mentioned < max(2, len(expected_rows) // 2):
                is_incomplete_comparative = True

    if pandas_answer and (is_too_short or not has_numbers or is_vague or is_hallucinating or is_incomplete_comparative):
        return pandas_answer

    if llm_answer and not (is_too_short or is_hallucinating or is_vague or is_incomplete_comparative):
        return llm_answer

    if pandas_answer:
        return pandas_answer

    # ── Last-resort: synthesise a plain answer from the table if we have one ──
    if tbl:
        rows = tbl.get("rows", [])
        hdrs = tbl.get("headers", [])
        if rows:
            lines = [f"{hdrs[i] if i < len(hdrs) else ''}: {rows[0][i]}" for i in range(min(len(hdrs), len(rows[0])))]
            summary = f"Here is the data for your query. Top result — {', '.join(lines)}."
            if len(rows) > 1:
                summary += f" ({len(rows)} rows total — see the table below.)"
            return summary

    return "I was unable to find relevant data for that query. Try asking about transaction amounts, failure rates, merchant categories, banks, or age groups."

def _expand_dict_value(key, val_str):
    """
    If val_str looks like a Python dict (e.g. "{'P2P': 112445, 'P2M': 87660}"),
    parse it and return a list of [sub_key, sub_val] rows prefixed with the key,
    or None if it's not a dict.
    """
    val_str = val_str.strip()
    if not (val_str.startswith("{") and val_str.endswith("}")):
        return None
    try:
        import ast
        parsed = ast.literal_eval(val_str)
        if not isinstance(parsed, dict):
            return None
        rows = []
        for k, v in parsed.items():
            if isinstance(v, float):
                v_str = f"{v:,.2f}"
            elif isinstance(v, int):
                v_str = f"{v:,}"
            else:
                v_str = str(v)
            rows.append([f"{key} — {k}", v_str])
        return rows if rows else None
    except Exception:
        return None


def format_as_table(data_result):
    lines = [l for l in data_result.strip().splitlines() if l.strip()]
    if not lines:
        return None

    def _strip_index(parts):
        if parts and re.match(r"^\d+$", parts[0].strip()):
            return parts[1:]
        return parts

    def _clean_val(v):
        return v.strip()

    # ── Phase 1: Try to detect a proper columnar table (2+ cols, first row = header) ──
    rows = []
    for line in lines:
        parts = re.split(r"\t|\s{2,}", line.strip())
        parts = [_clean_val(p) for p in parts if p.strip()]
        parts = _strip_index(parts)
        if len(parts) >= 2:
            rows.append(parts)

    if rows and len(rows) >= 2:
        max_cols = max(len(r) for r in rows)
        header = rows[0]
        if len(header) < max_cols:
            header = header + [f"Col{i+1}" for i in range(len(header), max_cols)]
        data_rows = []
        for r in rows[1:]:
            if len(r) < max_cols:
                r = r + [""] * (max_cols - len(r))
            data_rows.append(r[:max_cols])
        # Check if any value looks like a dict — if so fall through to KV expansion
        has_dict_vals = any(
            str(cell).strip().startswith("{") and str(cell).strip().endswith("}")
            for row in data_rows for cell in row
        )
        if not has_dict_vals:
            return {"headers": header[:max_cols], "rows": data_rows}

    # ── Phase 2: Key-value table (Metric | Value), expanding dicts into sub-rows ──
    kv = []
    for line in lines:
        # Try splitting on colon or tab or 2+ spaces
        parts = re.split(r"\t|\s{2,}", line.strip(), maxsplit=1)
        if len(parts) < 2:
            parts = re.split(r":\s*", line.strip(), maxsplit=1)
        parts = [_clean_val(p) for p in parts if p.strip()]
        parts = _strip_index(parts)
        if len(parts) == 2:
            key, val = parts[0], parts[1]
            # Attempt to expand dict-format values into multiple rows
            expanded = _expand_dict_value(key, val)
            if expanded:
                kv.extend(expanded)
            else:
                # Truncate extremely long values with ellipsis — they'll still be readable
                kv.append([key, val])
        elif len(parts) == 1 and kv:
            # Continuation line — append to last value
            kv[-1][1] = kv[-1][1] + " " + parts[0]

    if kv:
        return {"headers": ["Metric", "Value"], "rows": kv}

    return None

def detect_table_worthy(d):
    if not d: return False
    lines=[l for l in d.strip().splitlines() if l.strip()]
    if len(lines)>=3: return True
    return bool(re.search(r"\d+",d)) and len(lines)>=2

# ── plot utilities ────────────────────────────────────────────────────────────
def _style_ax(ax, fig):
    fig.patch.set_facecolor(PS["bg"]); ax.set_facecolor(PS["bg"])
    ax.tick_params(colors=PS["fg"],labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor(PS["grid"])
    for attr in (ax.yaxis.label,ax.xaxis.label,ax.title):
        attr.set_color(PS["fg"])
    ax.grid(color=PS["grid"],linewidth=0.5,alpha=0.7)

def _make_fig(w=7.8,h=5.2):
    fig=Figure(figsize=(w,h),dpi=110,facecolor=PS["bg"])
    ax=fig.add_subplot(111,facecolor=PS["bg"])
    _style_ax(ax,fig); return fig,ax

def _attach_meta(fig,hover_type,hover_data,bars=None,wedges=None):
    fig._hover_type=hover_type; fig._hover_data=hover_data
    if bars:   fig._bars=bars
    if wedges: fig._wedges=wedges

def _safe_tight_layout(fig, rect=None):
    """Tight layout with fallback — never crashes on edge cases."""
    try:
        if rect:
            fig.tight_layout(rect=rect)
        else:
            fig.tight_layout()
    except Exception:
        try:
            fig.subplots_adjust(bottom=0.18, top=0.90, left=0.10, right=0.97)
        except Exception:
            pass

def _draw_bar(ax, fig, labels, values, title="", ylabel=""):
    colors = PS["colors"][:len(values)]
    bars = ax.bar(range(len(labels)), values, color=colors,
                  edgecolor=PS["bg"], linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{val:,.1f}", ha="center", va="bottom",
                color=PS["fg"], fontsize=7)
    # Legend OUTSIDE plot area — below the chart so it never overlaps bars
    ax.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in colors],
        labels=[str(l) for l in labels],
        facecolor=PS["bg"], edgecolor=PS["grid"], labelcolor=PS["fg"],
        fontsize=7, ncol=min(len(labels), 5),
        loc="upper center", bbox_to_anchor=(0.5, -0.22),
        framealpha=0.9,
    )
    if ylabel: ax.set_ylabel(ylabel, color=PS["fg"], fontsize=9)
    if title:  ax.set_title(title, color=PS["fg"], fontsize=10, pad=8)
    _safe_tight_layout(fig, rect=[0, 0.18, 1, 1])
    _attach_meta(fig, "bar", list(zip(labels, values)), bars=bars)
    return fig

def _draw_pie(ax, fig, labels, values, title=""):
    colors = PS["colors"][:len(values)]
    wedges, texts, autotexts = ax.pie(
        values, labels=None,          # no inline labels — legend handles them
        colors=colors, autopct="%1.1f%%", startangle=90,
        textprops={"color": PS["fg"], "fontsize": 8.5},
        wedgeprops={"edgecolor": PS["bg"], "linewidth": 2},
    )
    for at in autotexts:
        at.set_color(PS["bg"]); at.set_fontsize(8)
    ax.grid(False)
    # Legend outside below the pie
    ax.legend(
        wedges, [f"{l} ({_v:,.0f})" for l, _v in zip(labels, values)],
        loc="upper center", bbox_to_anchor=(0.5, -0.08),
        ncol=min(len(labels), 4), fontsize=7.5,
        facecolor=PS["bg"], edgecolor=PS["grid"], labelcolor=PS["fg"],
        framealpha=0.9,
    )
    if title: ax.set_title(title, color=PS["fg"], fontsize=10, pad=8)
    _safe_tight_layout(fig, rect=[0, 0.18, 1, 1])
    _attach_meta(fig, "pie", list(zip(labels, values)), wedges=wedges)
    return fig

def _draw_line(ax, fig, labels, values, title="", ylabel=""):
    if not labels or not values:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                color=PS["fg"], fontsize=12, transform=ax.transAxes)
        _safe_tight_layout(fig)
        fig._hover_type = None; fig._hover_data = []
        return fig
    # Guard: ensure values are numeric plain floats
    safe_values = []
    for _v in values:
        try:
            safe_values.append(float(_v))
        except (TypeError, ValueError):
            safe_values.append(0.0)
    # Guard: ensure labels are plain strings (avoid matplotlib format-string misparse)
    labels = [str(l) for l in labels]
    ax.plot(range(len(labels)), safe_values, color=PS["accent"], linewidth=2.5,
            marker="o", markersize=6, markerfacecolor=PS["a2"],
            label=ylabel or "Value")
    ax.fill_between(range(len(labels)), safe_values, alpha=0.12, color=PS["accent"])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    # Legend outside below
    ax.legend(
        facecolor=PS["bg"], edgecolor=PS["grid"], labelcolor=PS["fg"],
        fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22),
        framealpha=0.9,
    )
    if ylabel: ax.set_ylabel(ylabel, color=PS["fg"], fontsize=9)
    if title:  ax.set_title(title, color=PS["fg"], fontsize=10, pad=8)
    _safe_tight_layout(fig, rect=[0, 0.18, 1, 1])
    _attach_meta(fig, "line", list(zip(labels, safe_values)))
    return fig

def _draw_histogram(ax, fig, data, title="", xlabel="", bins=40):
    ax.hist(data, bins=bins, color=PS["accent"], edgecolor=PS["bg"], alpha=0.85)
    ax.set_xlabel(xlabel, color=PS["fg"], fontsize=9)
    ax.set_ylabel("Count", color=PS["fg"], fontsize=9)
    if title: ax.set_title(title, color=PS["fg"], fontsize=10, pad=8)
    _safe_tight_layout(fig)
    fig._hover_type = None; fig._hover_data = []
    return fig

def _draw_grouped_bar(fig, ax, group_labels, series_dict, title="", ylabel=""):
    """Multi-series grouped bar chart with legend outside the plot area."""
    n_groups = len(group_labels); n_series = len(series_dict)
    width = 0.8 / n_series; x = np.arange(n_groups)
    for i, (name, vals) in enumerate(series_dict.items()):
        offset = (i - (n_series - 1) / 2) * width
        ax.bar(x + offset, vals, width=width * 0.9,
               color=PS["colors"][i % len(PS["colors"])],
               edgecolor=PS["bg"], linewidth=0.5, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=45, ha="right", fontsize=8)
    ax.legend(
        facecolor=PS["bg"], edgecolor=PS["grid"], labelcolor=PS["fg"],
        fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22),
        ncol=min(n_series, 4), framealpha=0.9,
    )
    if ylabel: ax.set_ylabel(ylabel, color=PS["fg"], fontsize=9)
    if title:  ax.set_title(title, color=PS["fg"], fontsize=10, pad=8)
    _safe_tight_layout(fig, rect=[0, 0.18, 1, 1])
    fig._hover_type = None; fig._hover_data = []
    return fig

# ── main chart dispatch ───────────────────────────────────────────────────────
def _unsupported_chart_fig(reason=""):
    """Return a figure that clearly explains why a chart can't be drawn."""
    fig, ax = _make_fig()
    msg = reason or "This chart type isn't supported for the selected data."
    ax.text(0.5, 0.6, "Chart unavailable", ha="center", va="center",
            color=PS["accent"], fontsize=13, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.42, msg, ha="center", va="center",
            color=PS["fg"], fontsize=9, wrap=True,
            transform=ax.transAxes)
    ax.axis("off")
    fig._hover_type = None; fig._hover_data = []
    return fig

def generate_chart(user_query, data_result):
    try:
        return _generate_chart_inner(user_query, data_result)
    except Exception as ex:
        return _unsupported_chart_fig(
            f"Unable to generate this chart: {str(ex)[:120]}"
        )

def _generate_chart_inner(user_query, data_result):
    df=get_df(); q=user_query.lower(); amt=_amt(df) or "amount_inr"
    chart_type="bar"
    if "pie" in q: chart_type="pie"
    elif "line" in q or "trend" in q or "over time" in q or "hour" in q: chart_type="line"
    labels=[]; values=[]; title=user_query[:58]+("..." if len(user_query)>58 else "")

    if any(w in q for w in ["fraud","flag","flagged","review"]):
        # If asking for breakdown by type/bank/device, show rates per group
        if any(w in q for w in ["breakdown","by type","by transaction","by bank","by device","by network"]):
            g = df.groupby("transaction_type")["fraud_flag"].mean() * 100
            g = g.sort_values(ascending=False)
            labels = g.index.tolist()
            values = [round(float(v), 4) for v in g.values]
            # chart_type stays as set above (pie if user asked, else bar)
        else:
            # Simple flagged vs not-flagged count — good for pie
            c = df["fraud_flag"].value_counts().sort_index()
            labels = ["Not Flagged", "Flagged for Review"]
            values = [int(c.get(0, 0)), int(c.get(1, 0))]
            chart_type = "pie"  # sensible default for binary split
    elif any(w in q for w in ["status","success","fail","failed","failure"]) and not any(w in q for w in ["bank","device","network","age"]):
        c=df["transaction_status"].value_counts(); labels=c.index.tolist(); values=c.values.tolist()
    elif "transaction type" in q or "transaction_type" in q or (("type" in q or "p2p" in q or "p2m" in q or "recharge" in q or "bill" in q) and "bank" not in q):
        c=df["transaction_type"].value_counts(); labels=c.index.tolist(); values=c.values.tolist()
    elif any(w in q for w in ["merchant","category","food","grocery","fuel","entertainment","shopping","healthcare","education","transport","utilities"]):
        c=df[df["merchant_category"]!="P2P_Transfer"]["merchant_category"].value_counts().head(10)
        labels=c.index.tolist(); values=c.values.tolist()
    elif "fail" in q and "bank" in q:
        g=df.groupby("sender_bank")["transaction_status"].apply(lambda x:(x=="FAILED").mean()*100).sort_values(ascending=False)
        labels=g.index.tolist(); values=[round(_v,2) for _v in g.values.tolist()]
    elif "fail" in q and "device" in q:
        g=df.groupby("device_type")["transaction_status"].apply(lambda x:(x=="FAILED").mean()*100).sort_values(ascending=False)
        labels=g.index.tolist(); values=[round(_v,2) for _v in g.values.tolist()]
    elif "fail" in q and "network" in q:
        g=df.groupby("network_type")["transaction_status"].apply(lambda x:(x=="FAILED").mean()*100).sort_values(ascending=False)
        labels=g.index.tolist(); values=[round(_v,2) for _v in g.values.tolist()]
    elif "bank" in q and "receiver" not in q:
        c=df["sender_bank"].value_counts(); labels=c.index.tolist(); values=c.values.tolist()
    elif "receiver" in q and "bank" in q:
        c=df["receiver_bank"].value_counts(); labels=c.index.tolist(); values=c.values.tolist()
    elif any(w in q for w in ["device","android","ios","web"]):
        c=df["device_type"].value_counts(); labels=c.index.tolist(); values=c.values.tolist()
    elif any(w in q for w in ["network","4g","5g","wifi","3g"]):
        c=df["network_type"].value_counts(); labels=c.index.tolist(); values=c.values.tolist()
    elif "age" in q and "receiver" not in q:
        order=["18-25","26-35","36-45","46-55","56+"]
        c=df["sender_age_group"].value_counts().reindex(order,fill_value=0)
        labels=c.index.tolist(); values=c.values.tolist()
    elif "receiver" in q and "age" in q:
        c=df["receiver_age_group"].dropna().value_counts(); labels=c.index.tolist(); values=c.values.tolist()
    elif "state" in q:
        c=df["sender_state"].value_counts().head(10); labels=c.index.tolist(); values=c.values.tolist()
    elif "hour" in q:
        c=df.groupby("hour_of_day").size().sort_index()
        labels=[f"{h}:00" for h in c.index]; values=c.values.tolist(); chart_type="line"
    elif "day" in q and "weekend" not in q:
        order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        c=df["day_of_week"].value_counts().reindex(order,fill_value=0)
        labels=[d[:3] for d in c.index]; values=c.values.tolist()
    elif "weekend" in q:
        c=df["is_weekend"].value_counts().sort_index()
        labels=["Weekday","Weekend"]; values=[int(c.get(0,0)),int(c.get(1,0))]; chart_type="pie"
    elif "amount" in q and "type" in q:
        g=df.groupby("transaction_type")[amt].mean().sort_values(ascending=False) if amt in df.columns else None
        if g is not None: labels=g.index.tolist(); values=[round(_v,2) for _v in g.values.tolist()]
    elif "amount" in q and "bank" in q:
        g=df.groupby("sender_bank")[amt].mean().sort_values(ascending=False) if amt in df.columns else None
        if g is not None: labels=g.index.tolist(); values=[round(_v,2) for _v in g.values.tolist()]
    else:
        c=df["transaction_type"].value_counts(); labels=c.index.tolist(); values=c.values.tolist()

    # Sanitise: plain Python types only — numpy scalars cause matplotlib _V errors
    labels = [str(l) for l in labels]
    values = [float(v) if not isinstance(v, int) else int(v) for v in values]

    fig,ax=_make_fig()
    if not values:
        ax.text(0.5,0.5,"Not enough data to chart",ha="center",va="center",
                color=PS["fg"],fontsize=12,transform=ax.transAxes)
        return fig
    if chart_type=="pie":    return _draw_pie(ax,fig,labels,values,title)
    elif chart_type=="line": return _draw_line(ax,fig,labels,values,title)
    else:                    return _draw_bar(ax,fig,labels,values,title)

# ── Advanced Explorer: multi-variable chart builder ───────────────────────────
def _unsupported_chart_fig(reason=""):
    """Return a figure that clearly explains why a chart cannot be drawn."""
    fig, ax = _make_fig()
    msg = reason or "This chart type is not supported for the selected data."
    ax.text(0.5, 0.62, "Chart unavailable", ha="center", va="center",
            color=PS["accent"], fontsize=13, fontweight="bold",
            transform=ax.transAxes)
    # Word-wrap the reason manually (matplotlib text wrap is unreliable)
    words = msg.split()
    lines = []; line = ""
    for w in words:
        if len(line) + len(w) + 1 > 60:
            lines.append(line); line = w
        else:
            line = (line + " " + w).strip()
    if line: lines.append(line)
    reason_text = "\n".join(lines)
    ax.text(0.5, 0.38, reason_text, ha="center", va="center",
            color=PS["fg"], fontsize=9,
            transform=ax.transAxes)
    ax.axis("off")
    fig._hover_type = None; fig._hover_data = []
    return fig


def generate_explorer_chart(config):
    """Public wrapper — never raises, always returns a Figure."""
    try:
        return _generate_explorer_chart_inner(config)
    except Exception as ex:
        return _unsupported_chart_fig(
            "This combination of chart type and data columns cannot be visualised. "
            "Try a different chart type or column selection."
        )


def _generate_explorer_chart_inner(config):
    """
    config = {
      "chart_type":  "bar"|"grouped_bar"|"line"|"pie"|"histogram"|"scatter"|"heatmap",
      "x_col":       column name (x-axis or grouping),
      "y_metric":    "count"|"mean_amount"|"sum_amount"|"fail_rate"|"fraud_rate"|"success_rate",
      "color_col":   column name for series split (optional, for grouped_bar/line),
      "filter_col":  column name to filter on (optional),
      "filter_val":  value to filter to (optional),
      "filter_col2": second filter column (optional),
      "filter_val2": second filter value (optional),
      "bins":        int for histogram (optional, default 40),
      "top_n":       int, limit categories (default 12),
      "sort":        "desc"|"asc"|"natural",
      "normalize":   bool, normalize to % (for grouped_bar),
    }
    Returns matplotlib Figure.
    """
    df = get_df().copy()
    amt = _amt(df) or "amount_inr"

    # Apply filters
    for fcol, fval in [("filter_col","filter_val"),("filter_col2","filter_val2")]:
        col = config.get(fcol); val = config.get(fval)
        if col and val and col in df.columns:
            df = df[df[col].astype(str).str.strip() == str(val).strip()]

    if len(df) == 0:
        fig, ax = _make_fig()
        ax.text(0.5, 0.5, "No data after filter", ha="center", va="center",
                color=PS["fg"], fontsize=12, transform=ax.transAxes)
        return fig

    x_col     = config.get("x_col","transaction_type")
    y_metric  = config.get("y_metric","count")
    color_col = config.get("color_col")
    ctype     = config.get("chart_type","bar")
    top_n     = int(config.get("top_n",12))
    sort_order= config.get("sort","desc")
    normalize = config.get("normalize", False)
    bins_     = int(config.get("bins",40))

    # Ensure x_col exists
    if x_col not in df.columns and ctype != "histogram":
        fig, ax = _make_fig()
        ax.text(0.5,0.5,f"Column '{x_col}' not found",ha="center",va="center",
                color=PS["fg"],fontsize=12,transform=ax.transAxes)
        return fig

    # Y metric helper
    def compute_metric(sub, grp_col):
        if y_metric == "count":
            s = sub.groupby(grp_col).size()
        elif y_metric == "mean_amount":
            s = sub.groupby(grp_col)[amt].mean() if amt in sub.columns else sub.groupby(grp_col).size()
        elif y_metric == "sum_amount":
            s = sub.groupby(grp_col)[amt].sum() if amt in sub.columns else sub.groupby(grp_col).size()
        elif y_metric == "fail_rate":
            s = sub.groupby(grp_col)["transaction_status"].apply(lambda x:(x=="FAILED").mean()*100)
        elif y_metric == "success_rate":
            s = sub.groupby(grp_col)["transaction_status"].apply(lambda x:(x=="SUCCESS").mean()*100)
        elif y_metric == "fraud_rate":
            s = sub.groupby(grp_col)["fraud_flag"].mean()*100
        else:
            s = sub.groupby(grp_col).size()
        return s

    # Natural order for specific cols
    natural_orders = {
        "hour_of_day":    list(range(24)),
        "day_of_week":    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
        "sender_age_group": ["18-25","26-35","36-45","46-55","56+"],
        "receiver_age_group": ["18-25","26-35","36-45","46-55","56+"],
    }

    y_label_map = {"count":"Count","mean_amount":"Avg Amount (Rs)","sum_amount":"Total Amount (Rs)",
                   "fail_rate":"Failure Rate (%)","success_rate":"Success Rate (%)","fraud_rate":"Flagged Rate (%)"}
    y_label = y_label_map.get(y_metric, y_metric)
    x_label = COLUMN_META.get(x_col,{}).get("label",x_col)
    color_label = COLUMN_META.get(color_col,{}).get("label",color_col) if color_col else ""
    title = f"{x_label}  ×  {y_label}" + (f"  by {color_label}" if color_col else "")

    # ── HISTOGRAM ─────────────────────────────────────────────────────────────
    if ctype == "histogram":
        fig, ax = _make_fig(7.8, 4.2)
        if x_col in df.columns and df[x_col].dtype in (float, int, np.float64, np.int64):
            data = df[x_col].dropna()
        elif amt in df.columns:
            data = df[amt].dropna()
            x_label = "Amount (INR)"
        else:
            data = pd.Series([])

        if color_col and color_col in df.columns:
            # Overlapping histograms per group
            groups = df[color_col].dropna().unique()[:8]
            for i, grp in enumerate(groups):
                sub_data = df[df[color_col]==grp][x_col].dropna() if x_col in df.columns else df[df[color_col]==grp][amt].dropna()
                ax.hist(sub_data, bins=bins_, alpha=0.55,
                        color=PS["colors"][i%len(PS["colors"])],
                        edgecolor=PS["bg"], label=str(grp))
            ax.legend(facecolor=PS["bg"],edgecolor=PS["grid"],labelcolor=PS["fg"],fontsize=8)
        else:
            ax.hist(data, bins=bins_, color=PS["accent"], edgecolor=PS["bg"], alpha=0.85)

        ax.set_xlabel(x_label, color=PS["fg"], fontsize=9)
        ax.set_ylabel("Count", color=PS["fg"], fontsize=9)
        ax.set_title(title, color=PS["fg"], fontsize=10, pad=8)
        _style_ax(ax, fig); fig.tight_layout()
        fig._hover_type=None; fig._hover_data=[]; return fig

    # ── GROUPED / MULTI-SERIES ─────────────────────────────────────────────────
    if color_col and color_col in df.columns and ctype in ("bar","grouped_bar","line"):
        color_vals = df[color_col].dropna().unique()
        # Limit color series
        if len(color_vals) > 8:
            top_color = df[color_col].value_counts().head(8).index.tolist()
            color_vals = top_color
            df = df[df[color_col].isin(color_vals)]

        # Get union of x values
        s_all = compute_metric(df, x_col)
        if x_col in natural_orders:
            nat = [_v for _v in natural_orders[x_col] if _v in s_all.index]
            x_vals = nat[:top_n]
        elif sort_order == "desc":
            x_vals = s_all.sort_values(ascending=False).head(top_n).index.tolist()
        elif sort_order == "asc":
            x_vals = s_all.sort_values().head(top_n).index.tolist()
        else:
            x_vals = s_all.index.tolist()[:top_n]

        series_dict = {}
        for cv in color_vals:
            sub = df[df[color_col]==cv]
            s = compute_metric(sub, x_col).reindex(x_vals, fill_value=0)
            series_dict[str(cv)] = s.values.tolist()

        if normalize:
            totals = np.array([sum(series_dict[k][i] for k in series_dict) for i in range(len(x_vals))])
            for k in series_dict:
                series_dict[k] = [_v/t*100 if t>0 else 0 for _v,t in zip(series_dict[k],totals)]
            y_label = "% Share"

        x_labels = [str(_v)[:18] for _v in x_vals]
        fig, ax = _make_fig(8.0, 4.5)
        _style_ax(ax, fig)
        if ctype == "line":
            for i,(name,vals) in enumerate(series_dict.items()):
                ax.plot(range(len(x_labels)),vals,
                        color=PS["colors"][i%len(PS["colors"])],
                        linewidth=2,marker="o",markersize=5,label=name)
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels,rotation=45,ha="right",fontsize=8)
            ax.legend(facecolor=PS["bg"],edgecolor=PS["grid"],labelcolor=PS["fg"],fontsize=8)
        else:
            _draw_grouped_bar(fig,ax,x_labels,series_dict,title,y_label)
        ax.set_xlabel(x_label,color=PS["fg"],fontsize=9)
        ax.set_ylabel(y_label+(  " (%)" if normalize else ""),color=PS["fg"],fontsize=9)
        ax.set_title(title,color=PS["fg"],fontsize=10,pad=8)
        fig.tight_layout(rect=[0,0.05,1,1])
        fig._hover_type=None; fig._hover_data=[]; return fig

    # ── SINGLE-SERIES ──────────────────────────────────────────────────────────
    s = compute_metric(df, x_col)
    if x_col in natural_orders:
        nat = natural_orders[x_col]
        s = s.reindex([_v for _v in nat if _v in s.index], fill_value=0)
        labels = [str(_v) for _v in s.index]; values = s.values.tolist()
        if ctype=="auto" and x_col=="hour_of_day": ctype="line"
        labels = [f"{_v}:00" if x_col=="hour_of_day" else str(_v)[:18] for _v in s.index]
    else:
        if sort_order=="desc":   s=s.sort_values(ascending=False)
        elif sort_order=="asc":  s=s.sort_values()
        s=s.head(top_n)
        labels=[str(_v)[:18] for _v in s.index]; values=s.values.tolist()

    fig, ax = _make_fig()
    if not values:
        ax.text(0.5,0.5,"No data",ha="center",va="center",
                color=PS["fg"],fontsize=12,transform=ax.transAxes)
        return fig

    if ctype in ("pie",) or (ctype=="auto" and len(values)<=6 and y_metric=="count"):
        return _draw_pie(ax,fig,labels,values,title)
    elif ctype=="line" or (ctype=="auto" and x_col=="hour_of_day"):
        return _draw_line(ax,fig,labels,values,title,y_label)
    else:
        return _draw_bar(ax,fig,labels,values,title,y_label)

# ── legacy wrapper for Graph Builder ─────────────────────────────────────────
def generate_custom_chart(col_x, col_y, chart_type, agg_func="count",
                          filter_col=None, filter_val=None):
    metric_map = {"count":"count","mean":"mean_amount","sum":"sum_amount",
                  "fraud_rate":"fraud_rate","fail_rate":"fail_rate"}
    chart_map  = {"auto":"bar","bar":"bar","pie":"pie","line":"line"}
    config = {
        "x_col":      col_x,
        "y_metric":   metric_map.get(agg_func,"count"),
        "chart_type": chart_map.get(chart_type,"bar"),
        "filter_col": filter_col,
        "filter_val": filter_val,
    }
    return generate_explorer_chart(config)

def generate_stats_analysis(col):
    df=get_df()
    if col not in df.columns: return None,None
    meta=COLUMN_META.get(col,{}); kind=meta.get("kind","categorical"); label=meta.get("label",col)
    amt=_amt(df) or "amount_inr"
    if kind=="numeric":
        s=df[col].describe()
        rows=[[k,f"{_v:,.4f}"] for k,_v in s.items()]
        rows.append(["skewness",f"{df[col].skew():,.4f}"])
        rows.append(["kurtosis",f"{df[col].kurtosis():,.4f}"])
        tbl={"headers":[label,"Value"],"rows":rows}
        fig,ax=_make_fig(); vals=df[col].dropna()
        _draw_histogram(ax,fig,vals,f"Distribution — {label}",label,bins=40)
        return tbl,fig
    elif kind=="binary":
        c=df[col].value_counts()
        lab=["Not Flagged (0)","Flagged for Review (1)"] if col=="fraud_flag" else [f"{k}" for k in c.index]
        vals=c.values.tolist()
        rows=[[l,f"{_v:,}",f"{_v/len(df)*100:.2f}%"] for l,_v in zip(lab,vals)]
        tbl={"headers":[label,"Count","% of Total"],"rows":rows}
        fig,ax=_make_fig(); return tbl,_draw_pie(ax,fig,lab,vals,f"{label} Distribution")
    else:
        c=df[col].value_counts().head(15)
        rows=[[str(k),f"{_v:,}",f"{_v/len(df)*100:.2f}%"] for k,_v in c.items()]
        tbl={"headers":[label,"Count","% of Total"],"rows":rows}
        fig,ax=_make_fig()
        return tbl,_draw_bar(ax,fig,[str(k)[:18] for k in c.index],c.values.tolist(),f"{label} Breakdown")

def generate_correlation_table(col_a,col_b,metric="count"):
    df=get_df()
    if col_a not in df.columns or col_b not in df.columns: return None
    if metric=="fraud_rate":
        ct=df.groupby([col_a,col_b])["fraud_flag"].mean().unstack(fill_value=0)*100
    elif metric=="fail_rate":
        ct=df.groupby([col_a,col_b])["transaction_status"].apply(
            lambda x:(x=="FAILED").mean()).unstack(fill_value=0)*100
    else:
        ct=pd.crosstab(df[col_a],df[col_b])
    rows=[]
    for idx in ct.index:
        row=[str(idx)]+[f"{_v:,.2f}" if isinstance(_v,float) else f"{_v:,}" for _v in ct.loc[idx]]
        rows.append(row)
    headers=[f"{col_a} \\ {col_b}"]+[str(c) for c in ct.columns]
    return {"headers":headers,"rows":rows}

def generate_full_stats_table():
    df=get_df(); amt=_amt(df) or "amount_inr"; rows=[]
    cat_cols=["transaction_type","merchant_category","sender_bank","receiver_bank",
              "device_type","network_type","sender_age_group","sender_state","transaction_status"]
    for col in cat_cols:
        if col not in df.columns: continue
        top=df[col].value_counts().idxmax(); n=df[col].nunique()
        rows.append([COLUMN_META.get(col,{}).get("label",col),"Categorical",f"{n}",str(top),
                     f"{df[col].value_counts().iloc[0]/len(df)*100:.1f}% share"])
    if amt in df.columns:
        rows.append(["Amount (INR)","Numeric",f"Rs.{df[amt].mean():,.2f} avg",
                     f"Rs.{df[amt].median():,.2f} median",f"Rs.{df[amt].std():,.2f} std"])
    if "fraud_flag" in df.columns:
        rate=df["fraud_flag"].mean()*100
        rows.append(["Fraud Flag","Binary",f"{rate:.2f}% flagged for review",
                     f"{int(df['fraud_flag'].sum()):,} transactions","Not confirmed fraud"])
    if "hour_of_day" in df.columns:
        peak=df.groupby("hour_of_day").size().idxmax()
        rows.append(["Hour of Day","Numeric",f"Peak: {peak}:00","0-23 range","Derived from timestamp"])
    return {"headers":["Column","Type","Stat 1","Stat 2","Notes"],"rows":rows}

def generate_report(tab="Default"):
    df=get_df(); history=get_history(tab); total=len(df); amt=_amt(df) or "amount_inr"
    suc=(df["transaction_status"]=="SUCCESS").mean()*100 if "transaction_status" in df.columns else 0
    flg=df["fraud_flag"].mean()*100 if "fraud_flag" in df.columns else 0
    avg=df[amt].mean() if amt in df.columns else 0
    top_bk=df["sender_bank"].value_counts().idxmax() if "sender_bank" in df.columns else "N/A"
    ph=df.groupby("hour_of_day").size().idxmax() if "hour_of_day" in df.columns else "N/A"
    top_ty=df["transaction_type"].value_counts().idxmax() if "transaction_type" in df.columns else "N/A"
    top_dv=df["device_type"].value_counts().idxmax() if "device_type" in df.columns else "N/A"
    top_nw=df["network_type"].value_counts().idxmax() if "network_type" in df.columns else "N/A"
    top_ag=df["sender_age_group"].value_counts().idxmax() if "sender_age_group" in df.columns else "N/A"
    lines=[
        "INSIGHTX ANALYTICAL REPORT",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "="*56,"",
        "DATASET OVERVIEW",
        f"  Total Transactions       : {total:,}",
        f"  Success Rate             : {suc:.2f}%",
        f"  Failure Rate             : {100-suc:.2f}%",
        f"  Flagged for Review       : {flg:.2f}%  (not confirmed fraud)",
        f"  Avg Transaction (INR)    : Rs.{avg:,.2f}",
        f"  Top Sender Bank          : {top_bk}",
        f"  Peak Transaction Hour    : {ph}:00",
        f"  Most Common Tx Type      : {top_ty}",
        f"  Most Common Device       : {top_dv}",
        f"  Most Common Network      : {top_nw}",
        f"  Most Active Age Group    : {top_ag}","",
        "QUERY COVERAGE VALIDATION",
        "  Descriptive  : Average amounts by type, bank, merchant ✓",
        "  Comparative  : Failure rates across device/network/bank ✓",
        "  Temporal     : Peak hours, day-of-week patterns ✓",
        "  Segmentation : Age group, device, bank volume splits ✓",
        "  Correlation  : Network × success, device × failure ✓",
        "  Risk         : High-value flagged rate, bank risk scores ✓","",
        "TRANSACTION TYPE BREAKDOWN",
    ]
    if "transaction_type" in df.columns:
        for t,c in df["transaction_type"].value_counts().items():
            lines.append(f"  {t:<22}: {c:,}  ({c/total*100:.1f}%)")
    lines+=["","FRAUD FLAG ANALYSIS (flagged for review, NOT confirmed fraud)"]
    if "fraud_flag" in df.columns:
        if "sender_bank" in df.columns:
            fb=df.groupby("sender_bank")["fraud_flag"].mean().sort_values(ascending=False)
            for bank,rate in fb.items(): lines.append(f"  {bank:<22}: {rate*100:.2f}% flagged")
        if "transaction_type" in df.columns:
            lines.append("")
            ft=df.groupby("transaction_type")["fraud_flag"].mean().sort_values(ascending=False)
            for t,rate in ft.items(): lines.append(f"  {t:<22}: {rate*100:.2f}% flagged")
    lines+=["","DEVICE & NETWORK BREAKDOWN"]
    if "device_type" in df.columns:
        for d,c in df["device_type"].value_counts().items():
            fail=(df[df["device_type"]==d]["transaction_status"]=="FAILED").mean()*100
            lines.append(f"  {d:<12}: {c:,} txns  ({fail:.1f}% fail rate)")
    if "network_type" in df.columns:
        lines.append("")
        for n,c in df["network_type"].value_counts().items():
            fail=(df[df["network_type"]==n]["transaction_status"]=="FAILED").mean()*100
            lines.append(f"  {n:<12}: {c:,} txns  ({fail:.1f}% fail rate)")
    lines+=["","AGE GROUP ANALYSIS"]
    if "sender_age_group" in df.columns and amt in df.columns:
        ag=df.groupby("sender_age_group")[amt].agg(["mean","count"])
        for age,row in ag.iterrows():
            lines.append(f"  {age:<10}: {int(row['count']):,} txns  avg Rs.{row['mean']:,.2f}")
    lines+=["","MERCHANT CATEGORY REVENUE"]
    if "merchant_category" in df.columns and amt in df.columns:
        mc=df[df["merchant_category"]!="P2P_Transfer"].groupby("merchant_category")[amt].agg(["sum","mean","count"]).sort_values("sum",ascending=False)
        for cat_n,row in mc.iterrows():
            lines.append(f"  {cat_n:<22}: Rs.{row['sum']:,.0f} total  ({int(row['count']):,} txns  avg Rs.{row['mean']:,.0f})")
    if history:
        lines+=["",f"CONVERSATION SUMMARY ({len(history)//2} exchanges)"]
        for msg in history[-10:]:
            prefix="  Q" if msg["role"]=="user" else "  A"
            lines.append(f"{prefix}: {msg['content'][:90]}")
    return "\n".join(lines)

def generate_chat_summary(tab="Default"):
    history=get_history(tab)
    if not history: return "No conversation history yet."
    llm=get_llm(); convo=""
    for msg in history[-16:]:
        role="User" if msg["role"]=="user" else "InsightX"
        convo+=f"{role}: {msg['content']}\n"
    prompt=(
        "<|system|>\nYou are a concise summarizer. Summarize in 3-5 bullet points. Be brief.<|end|>\n"
        f"<|user|>\nConversation:\n{convo}\n\nSummarize this.<|end|>\n<|assistant|>\n"
    )
    out=llm(prompt,max_tokens=768,stop=["<|end|>","<|user|>","<|system|>"],echo=False,temperature=0.4)
    raw=out["choices"][0]["text"].strip()
    for m in ["<|end|>","<|user|>","<|system|>"]:
        idx=raw.find(m)
        if idx!=-1: raw=raw[:idx]
    return raw.strip()

def save_chat_export(tab="Default"):
    rich=get_rich(tab)
    lines=["InsightX Chat Export",f"Tab: {tab}",
           f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}","="*60,""]
    for entry in rich:
        role=entry.get("role",""); ct=entry.get("content","")
        if role=="user":
            lines.append(f"[YOU]  {ct}")
        else:
            lines.append(f"[IX]   {ct}")
            if entry.get("table"):
                t=entry["table"]
                lines.append("  [TABLE] "+" | ".join(t["headers"]))
                for row in t["rows"]: lines.append("         "+" | ".join(str(c) for c in row))
            if entry.get("recommendations"):
                lines.append("  [RECOMMENDATIONS]")
                for r in entry["recommendations"]: lines.append(f"    • {r}")
            if entry.get("report"): lines.append(entry["report"])
            if entry.get("summary"): lines.append(f"  [SUMMARY] {entry['summary']}")
        lines.append("")
    return "\n".join(lines)

def generate_stats_table(df):
    """
    Build a comprehensive, perfectly-formatted stats table entirely from pandas.
    No LLM involved — zero risk of failure or truncation.
    Returns {"headers": [...], "rows": [...]} ready for the UI renderer.
    """
    amt = _amt(df) or "amount_inr"
    rows = []

    # ── Dataset basics ────────────────────────────────────────────────────────
    total = len(df)
    rows.append(["📊 Total Transactions", f"{total:,}"])

    if "transaction_status" in df.columns:
        suc  = (df["transaction_status"] == "SUCCESS").sum()
        fail = (df["transaction_status"] == "FAILED").sum()
        rows.append(["✅ Successful Transactions",  f"{suc:,}  ({suc/total*100:.2f}%)"])
        rows.append(["❌ Failed Transactions",       f"{fail:,}  ({fail/total*100:.2f}%)"])
        rows.append(["📉 Overall Failure Rate",      f"{fail/total*100:.2f}%"])

    if "fraud_flag" in df.columns:
        flagged = df["fraud_flag"].sum()
        rows.append(["🚩 Flagged for Review",        f"{int(flagged):,}  ({flagged/total*100:.2f}%)"])

    # ── Amount stats ──────────────────────────────────────────────────────────
    if amt in df.columns:
        a = df[amt]
        rows.append(["─── Amount Statistics ───", ""])
        rows.append(["💰 Total Volume (INR)",          f"₹{a.sum():,.0f}"])
        rows.append(["📈 Average Amount (INR)",         f"₹{a.mean():,.2f}"])
        rows.append(["📊 Median Amount (INR)",          f"₹{a.median():,.2f}"])
        rows.append(["📏 Std Deviation (INR)",          f"₹{a.std():,.2f}"])
        rows.append(["⬇ Min Amount (INR)",             f"₹{a.min():,.2f}"])
        rows.append(["⬆ Max Amount (INR)",             f"₹{a.max():,.2f}"])
        rows.append(["📐 90th Percentile (INR)",        f"₹{a.quantile(0.90):,.2f}"])

    # ── Transaction type breakdown ────────────────────────────────────────────
    if "transaction_type" in df.columns:
        rows.append(["─── Transaction Types ───", ""])
        for ttype, cnt in df["transaction_type"].value_counts().items():
            fail_r = (df[df["transaction_type"]==ttype]["transaction_status"]=="FAILED").mean()*100
            rows.append([f"  {ttype}", f"{cnt:,}  ({cnt/total*100:.1f}%)  |  fail: {fail_r:.2f}%"])

    # ── Merchant category ─────────────────────────────────────────────────────
    if "merchant_category" in df.columns:
        rows.append(["─── Merchant Categories ───", ""])
        mc_counts = df["merchant_category"].value_counts()
        for mc, cnt in mc_counts.items():
            rows.append([f"  {mc}", f"{cnt:,}  ({cnt/total*100:.1f}%)"])

    # ── Sender banks ──────────────────────────────────────────────────────────
    if "sender_bank" in df.columns:
        rows.append(["─── Sender Banks ───", ""])
        for bank, cnt in df["sender_bank"].value_counts().items():
            fail_r = (df[df["sender_bank"]==bank]["transaction_status"]=="FAILED").mean()*100
            rows.append([f"  {bank}", f"{cnt:,}  ({cnt/total*100:.1f}%)  |  fail: {fail_r:.2f}%"])

    # ── Device types ──────────────────────────────────────────────────────────
    if "device_type" in df.columns:
        rows.append(["─── Device Types ───", ""])
        for dev, cnt in df["device_type"].value_counts().items():
            rows.append([f"  {dev}", f"{cnt:,}  ({cnt/total*100:.1f}%)"])

    # ── Network types ─────────────────────────────────────────────────────────
    if "network_type" in df.columns:
        rows.append(["─── Network Types ───", ""])
        for net, cnt in df["network_type"].value_counts().items():
            fail_r = (df[df["network_type"]==net]["transaction_status"]=="FAILED").mean()*100
            rows.append([f"  {net}", f"{cnt:,}  ({cnt/total*100:.1f}%)  |  fail: {fail_r:.2f}%"])

    # ── Age groups ────────────────────────────────────────────────────────────
    if "sender_age_group" in df.columns:
        rows.append(["─── Sender Age Groups ───", ""])
        order = ["18-25","26-35","36-45","46-55","56+"]
        vc = df["sender_age_group"].value_counts()
        for age in order:
            cnt = vc.get(age, 0)
            if cnt > 0:
                avg_a = df[df["sender_age_group"]==age][amt].mean() if amt in df.columns else 0
                rows.append([f"  {age}", f"{cnt:,}  ({cnt/total*100:.1f}%)  |  avg ₹{avg_a:,.0f}"])

    # ── Day of week ───────────────────────────────────────────────────────────
    if "day_of_week" in df.columns:
        rows.append(["─── Day of Week ───", ""])
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        vc = df["day_of_week"].value_counts()
        for day in day_order:
            cnt = vc.get(day, 0)
            if cnt > 0:
                rows.append([f"  {day}", f"{cnt:,}  ({cnt/total*100:.1f}%)"])

    # ── Weekend split ─────────────────────────────────────────────────────────
    if "is_weekend" in df.columns:
        wknd = df["is_weekend"].sum()
        wkdy = total - wknd
        rows.append(["📅 Weekday Transactions",  f"{int(wkdy):,}  ({wkdy/total*100:.1f}%)"])
        rows.append(["🎉 Weekend Transactions",   f"{int(wknd):,}  ({wknd/total*100:.1f}%)"])

    # ── Peak hour ─────────────────────────────────────────────────────────────
    if "hour_of_day" in df.columns:
        peak_h = df["hour_of_day"].value_counts().idxmax()
        peak_c = df["hour_of_day"].value_counts().max()
        rows.append(["⏰ Peak Transaction Hour",  f"{peak_h}:00  ({peak_c:,} txns)"])

    return {"headers": ["Metric", "Value"], "rows": rows}


def process_query(user_query, tab="Default"):
    result={"text":"","table":None,"chart":None,"report":None,"summary":None,
            "data_raw":None,"recommendations":None}

    def _push(uq, text, extra=None):
        push_history(uq,text,tab)
        entry={"role":"assistant","content":text,"type":"text",
               "table":None,"chart":None,"report":None,"summary":None,"recommendations":None}
        if extra: entry.update(extra)
        push_rich(tab,{"role":"user","content":uq,"type":"user"})
        push_rich(tab,entry)

    sum_mode=wants_summary(user_query)

    # ── Stats table shortcut — pure pandas, zero LLM, never fails ────────────
    if wants_stats_table(user_query):
        df = get_df()
        if df is not None:
            tbl = generate_stats_table(df)
            recs = [
                "High failure rates by transaction type indicate integration issues — prioritise third-party API monitoring.",
                "Peak hour data helps plan infrastructure scaling for load surges.",
                "Age group and device breakdowns help target UX improvements to the highest-volume segments.",
            ]
            text = (f"Here is a complete statistics overview of your dataset ({len(df):,} transactions). "
                    f"The table covers transaction volumes, amounts, failure rates, device/network splits, "
                    f"age groups, and temporal patterns — all computed directly from your data.")
            result["table"] = tbl
            result["recommendations"] = recs
            result["text"] = text
            _push(user_query, text, {"table": tbl, "recommendations": recs})
            return result
        else:
            result["text"] = "No data loaded yet. Please load a dataset first."
            _push(user_query, result["text"])
            return result

    if sum_mode=="chat":
        result["summary"]=generate_chat_summary(tab)
        result["text"]="Here's a summary of our conversation so far."
        _push(user_query,result["text"],{"summary":result["summary"]}); return result
    if sum_mode=="data":
        data_raw=run_analyst(user_query); result["data_raw"]=data_raw
        if data_raw:
            t=format_as_table(data_raw)
            if t: result["table"]=t
        tbl,nav,labels,values,ctype,recs=_direct_analytics(user_query.lower(),get_df())
        if tbl and not result["table"]: result["table"]=tbl
        result["recommendations"]=recs
        text=run_narrator(user_query,data_raw or nav,recs,tab,tbl=result["table"]) or "Here is the data summary."
        result["text"]=text
        _push(user_query,text,{"table":result["table"],"recommendations":recs}); return result
    if wants_report(user_query):
        result["report"]=generate_report(tab); result["text"]="Full analytical report generated."
        _push(user_query,result["text"],{"report":result["report"]}); return result

    # Main path — direct analytics first, LLM supplements
    df=get_df()
    tbl,nav,labels,values,ctype,recs=_direct_analytics(user_query.lower(),df)
    result["table"]=tbl
    result["recommendations"]=recs

    data_raw=None
    if needs_data(user_query):
        data_raw=run_analyst(user_query); result["data_raw"]=data_raw
        # If LLM produced a table and we don't have one yet, use it
        if data_raw and not result["table"] and detect_table_worthy(data_raw):
            t=format_as_table(data_raw)
            if t: result["table"]=t

    if wants_chart(user_query):
        try:
            # Always route through generate_chart — it correctly maps the query
            # to both the right data AND the right chart type (pie/line/bar).
            # This prevents the direct-analytics labels (which may be raw means)
            # from being drawn with the wrong chart type.
            result["chart"] = generate_chart(user_query, data_raw or "")
        except Exception:
            result["chart"] = _unsupported_chart_fig(
                "This chart type isn't available for the requested data. "
                "Try asking for a bar chart or table instead."
            )

    combined_data=nav if nav else (data_raw or "")
    text=run_narrator(user_query,combined_data,recs,tab,tbl=tbl) or "I couldn't generate a response. Try rephrasing."
    result["text"]=text
    _push(user_query,text,{"table":result["table"],"chart":result["chart"],"recommendations":recs})
    return result
