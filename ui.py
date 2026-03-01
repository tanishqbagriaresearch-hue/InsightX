import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import threading
import queue
import re
import json
import os
import csv
import io
import math
import time
from datetime import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import engine

BG_DEEP     = "#080c12"
BG_PANEL    = "#0d1117"
BG_CARD     = "#161b22"
BG_INPUT    = "#1c2128"
BG_HOVER    = "#21262d"
BG_SELECTED = "#1f2937"
BG_TOAST    = "#0f172a"
BORDER      = "#30363d"
BORDER_LT   = "#3d4450"
BG_USER_BUBBLE = "#0e2035"

TEXT_PRIMARY   = "#e6edf3"
TEXT_SECONDARY = "#8b949e"
TEXT_MUTED     = "#484f58"

ACCENT_BLUE   = "#58a6ff"
ACCENT_GREEN  = "#3fb950"
ACCENT_RED    = "#f78166"
ACCENT_PURPLE = "#d2a8ff"
ACCENT_ORANGE = "#ffa657"
ACCENT_TEAL   = "#56d364"
ACCENT_YELLOW = "#e3b341"
ACCENT_PINK   = "#f778ba"

FONT_BODY  = ("Consolas", 10)
FONT_SMALL = ("Consolas", 9)
FONT_INPUT = ("Consolas", 11)
FONT_MONO  = ("Courier New", 9)

TYPEWRITER_DELAY    = 0.008
CUSTOM_ACTIONS_FILE = "insightx_custom_actions.json"
PINNED_FILE         = "insightx_pinned.json"
NOTES_FILE          = "insightx_notes.json"
FAVORITES_FILE      = "insightx_favorites.json"

ALL_COLUMNS = [
    ("transaction_type",   "Transaction Type"),
    ("merchant_category",  "Merchant Category"),
    ("amount_inr",         "Amount (INR)"),
    ("transaction_status", "Transaction Status"),
    ("sender_age_group",   "Sender Age Group"),
    ("receiver_age_group", "Receiver Age Group"),
    ("sender_state",       "Sender State"),
    ("sender_bank",        "Sender Bank"),
    ("receiver_bank",      "Receiver Bank"),
    ("device_type",        "Device Type"),
    ("network_type",       "Network Type"),
    ("fraud_flag",         "Fraud Flag (Review)"),
    ("day_of_week",        "Day of Week"),
    ("is_weekend",         "Is Weekend"),
]

ALL_COL_LABELS = [f"{c} — {l}" for c, l in ALL_COLUMNS]
COL_LABEL_MAP  = {f"{c} — {l}": c for c, l in ALL_COLUMNS}

Y_METRICS = [
    ("count",        "Count of Transactions"),
    ("mean_amount",  "Average Amount (INR)"),
    ("sum_amount",   "Total Amount (INR)"),
    ("fail_rate",    "Failure Rate (%)"),
    ("success_rate", "Success Rate (%)"),
    ("fraud_rate",   "Flagged-for-Review Rate (%)"),
    ("median_amount","Median Amount (INR)"),
    ("std_amount",   "Std Dev Amount (INR)"),
]

CHART_TYPES = [
    ("bar",         "Bar"),
    ("grouped_bar", "Grouped Bar"),
    ("line",        "Line"),
    ("pie",         "Pie"),
    ("histogram",   "Histogram"),
    ("scatter",     "Scatter"),
    ("heatmap",     "Heatmap"),
]

SORT_OPTIONS = [
    ("desc",    "High → Low"),
    ("asc",     "Low → High"),
    ("natural", "Natural order"),
    ("alpha",   "Alphabetical"),
]

THEMES = {
    "Default Dark": {
        "BG_DEEP": "#080c12", "BG_PANEL": "#0d1117", "BG_CARD": "#161b22",
        "ACCENT_BLUE": "#58a6ff", "TEXT_PRIMARY": "#e6edf3",
    },
    "Midnight Blue": {
        "BG_DEEP": "#050d1a", "BG_PANEL": "#091525", "BG_CARD": "#0f2035",
        "ACCENT_BLUE": "#4fc3f7", "TEXT_PRIMARY": "#e1f5fe",
    },
    "Forest Dark": {
        "BG_DEEP": "#060e09", "BG_PANEL": "#0a1a0d", "BG_CARD": "#112518",
        "ACCENT_BLUE": "#66bb6a", "TEXT_PRIMARY": "#e8f5e9",
    },
    "Cyberpunk": {
        "BG_DEEP": "#070010", "BG_PANEL": "#0e0020", "BG_CARD": "#150030",
        "ACCENT_BLUE": "#f000ff", "TEXT_PRIMARY": "#e8e0ff",
    },
}

PROMPT_HISTORY_MAX = 100
_prompt_history = []
_prompt_history_idx = -1


def _load_json_file(path, default):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return default


def _save_json_file(path, data):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def _load_custom_actions():
    return _load_json_file(CUSTOM_ACTIONS_FILE, [])


def _save_custom_actions(actions):
    _save_json_file(CUSTOM_ACTIONS_FILE, actions)


def _load_pinned():
    return _load_json_file(PINNED_FILE, [])


def _save_pinned(pins):
    _save_json_file(PINNED_FILE, pins)


def _load_notes():
    return _load_json_file(NOTES_FILE, {})


def _save_notes(notes):
    _save_json_file(NOTES_FILE, notes)


def _load_favorites():
    return _load_json_file(FAVORITES_FILE, [])


def _save_favorites(favs):
    _save_json_file(FAVORITES_FILE, favs)


def _make_text_bubble(parent, text, bg=BG_CARD, fg=TEXT_PRIMARY,
                      wraplength=700, font=None):
    if font is None:
        font = FONT_BODY
    chars_per_line = max(1, wraplength // 7)
    line_count = max(1, sum(
        max(1, (len(line) // chars_per_line) + 1)
        for line in text.split("\n")
    ))
    height = max(2, line_count)
    tw = tk.Text(parent, font=font, fg=fg, bg=bg,
                 relief="flat", bd=0, wrap="word", width=80, height=height,
                 padx=12, pady=8, cursor="xterm", state="normal",
                 selectbackground=ACCENT_BLUE, selectforeground=BG_DEEP,
                 inactiveselectbackground=ACCENT_BLUE)
    tw.insert("1.0", text)
    tw.configure(state="disabled")
    menu = tk.Menu(tw, tearoff=0, bg=BG_HOVER, fg=TEXT_PRIMARY,
                   activebackground=ACCENT_BLUE, activeforeground=BG_DEEP,
                   font=FONT_SMALL)
    menu.add_command(label="Copy", command=lambda: _copy_sel(tw))
    menu.add_command(label="Select All", command=lambda: _sel_all(tw))
    menu.add_separator()
    menu.add_command(label="Copy All Text", command=lambda: (
        tw.clipboard_clear(), tw.clipboard_append(tw.get("1.0", "end-1c"))))
    tw.bind("<Button-3>", lambda e: menu.tk_popup(e.x_root, e.y_root))
    tw.bind("<Control-c>", lambda e: _copy_sel(tw))
    tw.bind("<Control-a>", lambda e: _sel_all(tw))
    tw.bind("<Button-1>", lambda e: tw.configure(state="normal") or
            tw.after(1, lambda: tw.configure(state="disabled")))
    return tw


def _copy_sel(tw):
    try:
        sel = tw.get(tk.SEL_FIRST, tk.SEL_LAST)
        tw.clipboard_clear(); tw.clipboard_append(sel)
    except tk.TclError:
        tw.clipboard_clear(); tw.clipboard_append(tw.get("1.0", "end-1c"))


def _sel_all(tw):
    tw.configure(state="normal")
    tw.tag_add(tk.SEL, "1.0", "end-1c")
    tw.configure(state="disabled")


def _combo(parent, var, values, width=36, **kw):
    cb = ttk.Combobox(parent, textvariable=var, state="readonly",
                      values=values, font=FONT_SMALL, width=width, **kw)
    return cb


def _lbl(parent, text, small=False, color=None, bg=BG_PANEL):
    c = color or (TEXT_MUTED if small else TEXT_SECONDARY)
    return tk.Label(parent, text=text,
                    font=("Consolas", 7 if small else 9, "bold"),
                    fg=c, bg=bg)


def _sep(parent):
    tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=16, pady=5)


def _icon_btn(parent, text, cmd, fg=TEXT_SECONDARY, hover_fg=ACCENT_TEAL,
              bg=BG_PANEL, font=None, padx=8, pady=3):
    f = font or ("Consolas", 7)
    b = tk.Label(parent, text=text, font=f, fg=fg, bg=bg, padx=padx,
                 pady=pady, cursor="hand2")
    b.bind("<Button-1>", lambda e: cmd())
    b.bind("<Enter>", lambda e: b.configure(fg=hover_fg))
    b.bind("<Leave>", lambda e: b.configure(fg=fg))
    return b


class Toast:
    _instance = None

    @classmethod
    def show(cls, root, message, color=ACCENT_GREEN, duration=2200):
        if cls._instance and cls._instance.winfo_exists():
            try:
                cls._instance.destroy()
            except Exception:
                pass
        t = tk.Toplevel(root)
        cls._instance = t
        t.overrideredirect(True)
        t.attributes("-topmost", True)
        t.configure(bg=BG_TOAST)
        frame = tk.Frame(t, bg=BG_TOAST, highlightbackground=color,
                         highlightthickness=1)
        frame.pack(fill="both", expand=True)
        tk.Label(frame, text=message, font=("Consolas", 9, "bold"),
                 fg=color, bg=BG_TOAST, padx=18, pady=10).pack()
        rw = root.winfo_rootx() + root.winfo_width() // 2 - 150
        rh = root.winfo_rooty() + root.winfo_height() - 90
        t.geometry(f"300x42+{rw}+{rh}")
        t.after(duration, lambda: t.destroy() if t.winfo_exists() else None)


class DataSnapshotDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("InsightX — Live Data Snapshot")
        self.geometry("900x640")
        self.configure(bg=BG_PANEL)
        self.resizable(True, True)
        self._build()
        self.grab_set()
        self.transient(parent)

    def _build(self):
        hdr = tk.Frame(self, bg=BG_PANEL)
        hdr.pack(fill="x", padx=20, pady=(14, 4))
        tk.Label(hdr, text="📸  Live Data Snapshot",
                 font=("Trebuchet MS", 13, "bold"),
                 fg=ACCENT_BLUE, bg=BG_PANEL).pack(side="left")
        tk.Label(hdr, text="— key performance indicators computed live from dataset",
                 font=("Consolas", 8), fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left", padx=12)
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=16, pady=4)

        df = engine.get_df()
        if df is None:
            tk.Label(self, text="No data loaded yet.", font=FONT_BODY,
                     fg=TEXT_MUTED, bg=BG_PANEL).pack(pady=40)
            return

        outer = tk.Frame(self, bg=BG_PANEL)
        outer.pack(fill="both", expand=True, padx=16, pady=8)

        amt = engine._amt(df) or "amount_inr"

        kpis = self._compute_kpis(df, amt)
        self._render_kpi_grid(outer, kpis)

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=16, pady=6)
        bot = tk.Frame(self, bg=BG_PANEL)
        bot.pack(fill="x", padx=20, pady=(0, 12))
        _icon_btn(bot, "⬇ Export KPIs as CSV", self._export_csv,
                  fg=ACCENT_TEAL, bg=BG_PANEL, padx=10).pack(side="left")
        _icon_btn(bot, "✕ Close", self.destroy,
                  fg=TEXT_MUTED, hover_fg=ACCENT_RED, bg=BG_PANEL).pack(side="right")
        self._kpis = kpis

    def _compute_kpis(self, df, amt):
        total = len(df)
        suc = (df["transaction_status"] == "SUCCESS").mean() * 100 if "transaction_status" in df.columns else 0
        fail = 100 - suc
        flg = df["fraud_flag"].mean() * 100 if "fraud_flag" in df.columns else 0
        avg_amt = df[amt].mean() if amt in df.columns else 0
        total_vol = df[amt].sum() if amt in df.columns else 0
        med_amt = df[amt].median() if amt in df.columns else 0
        max_amt = df[amt].max() if amt in df.columns else 0
        top_bank = df["sender_bank"].value_counts().idxmax() if "sender_bank" in df.columns else "N/A"
        top_type = df["transaction_type"].value_counts().idxmax() if "transaction_type" in df.columns else "N/A"
        top_dev = df["device_type"].value_counts().idxmax() if "device_type" in df.columns else "N/A"
        top_net = df["network_type"].value_counts().idxmax() if "network_type" in df.columns else "N/A"
        top_state = df["sender_state"].value_counts().idxmax() if "sender_state" in df.columns else "N/A"
        top_age = df["sender_age_group"].value_counts().idxmax() if "sender_age_group" in df.columns else "N/A"
        weekend_pct = df["is_weekend"].mean() * 100 if "is_weekend" in df.columns else 0
        unique_states = df["sender_state"].nunique() if "sender_state" in df.columns else 0
        unique_banks = df["sender_bank"].nunique() if "sender_bank" in df.columns else 0

        if amt in df.columns and "transaction_type" in df.columns:
            p90 = df[amt].quantile(0.90)
            high_val_flg = df[df[amt] >= p90]["fraud_flag"].mean() * 100 if "fraud_flag" in df.columns else 0
        else:
            p90 = 0; high_val_flg = 0

        return [
            ("Total Transactions",    f"{total:,}",                  ACCENT_BLUE),
            ("Success Rate",          f"{suc:.2f}%",                  ACCENT_GREEN),
            ("Failure Rate",          f"{fail:.2f}%",                  ACCENT_RED),
            ("Flagged for Review",    f"{flg:.2f}%",                  ACCENT_ORANGE),
            ("Avg Transaction",       f"₹{avg_amt:,.2f}",             ACCENT_BLUE),
            ("Median Transaction",    f"₹{med_amt:,.2f}",             ACCENT_PURPLE),
            ("Total Volume",          f"₹{total_vol:,.0f}",           ACCENT_TEAL),
            ("Largest Transaction",   f"₹{max_amt:,.2f}",             ACCENT_YELLOW),
            ("90th Pct Threshold",    f"₹{p90:,.2f}",                 TEXT_SECONDARY),
            ("High-Value Flag Rate",  f"{high_val_flg:.2f}%",         ACCENT_RED),
            ("Top Bank",              str(top_bank),                   TEXT_PRIMARY),
            ("Top Tx Type",           str(top_type),                   TEXT_PRIMARY),
            ("Top Device",            str(top_dev),                    TEXT_PRIMARY),
            ("Top Network",           str(top_net),                    TEXT_PRIMARY),
            ("Top State",             str(top_state),                  TEXT_PRIMARY),
            ("Most Active Age Group", str(top_age),                    TEXT_PRIMARY),
            ("Weekend Transactions",  f"{weekend_pct:.1f}%",          ACCENT_PURPLE),
            ("Unique States",         str(unique_states),              TEXT_SECONDARY),
            ("Unique Banks",          str(unique_banks),               TEXT_SECONDARY),
        ]

    def _render_kpi_grid(self, parent, kpis):
        cols = 4
        for i, (label, value, color) in enumerate(kpis):
            r = i // cols
            c = i % cols
            card = tk.Frame(parent, bg=BG_CARD,
                            highlightbackground=BORDER, highlightthickness=1)
            card.grid(row=r, column=c, padx=5, pady=5, sticky="nsew")
            parent.columnconfigure(c, weight=1)
            tk.Label(card, text=label, font=("Consolas", 7, "bold"),
                     fg=TEXT_MUTED, bg=BG_CARD, padx=10, pady=4).pack(anchor="w", pady=(8, 2))
            tk.Label(card, text=value, font=("Consolas", 12, "bold"),
                     fg=color, bg=BG_CARD, padx=10, pady=4).pack(anchor="w", pady=(0, 8))

    def _export_csv(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            initialfile=f"insightx_kpis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        if path:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Metric", "Value"])
                for lbl, val, _ in self._kpis:
                    w.writerow([lbl, val])
            Toast.show(self.master, f"KPIs exported to {os.path.basename(path)}")


class NotesDialog(tk.Toplevel):
    def __init__(self, parent, tab_name):
        super().__init__(parent)
        self.title(f"InsightX — Notes: {tab_name}")
        self.geometry("560x420")
        self.configure(bg=BG_PANEL)
        self.resizable(True, True)
        self._tab = tab_name
        self._notes_store = _load_notes()
        self._build()
        self.grab_set()
        self.transient(parent)

    def _build(self):
        hdr = tk.Frame(self, bg=BG_PANEL)
        hdr.pack(fill="x", padx=18, pady=(14, 4))
        tk.Label(hdr, text=f"📝  Notes — {self._tab}",
                 font=("Trebuchet MS", 12, "bold"),
                 fg=ACCENT_YELLOW, bg=BG_PANEL).pack(side="left")
        tk.Label(hdr, text="Auto-saved per chat tab",
                 font=("Consolas", 7), fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left", padx=10)
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=14, pady=4)
        self._text = tk.Text(self, font=FONT_BODY, fg=TEXT_PRIMARY, bg=BG_INPUT,
                             insertbackground=ACCENT_BLUE, relief="flat", bd=0,
                             wrap="word", padx=14, pady=10,
                             selectbackground=ACCENT_BLUE, selectforeground=BG_DEEP)
        self._text.pack(fill="both", expand=True, padx=14, pady=4)
        self._text.insert("1.0", self._notes_store.get(self._tab, ""))
        self._text.bind("<KeyRelease>", self._auto_save)
        bot = tk.Frame(self, bg=BG_PANEL)
        bot.pack(fill="x", padx=18, pady=(4, 12))
        self._status = tk.Label(bot, text="", font=("Consolas", 7),
                                fg=ACCENT_GREEN, bg=BG_PANEL)
        self._status.pack(side="left")
        _icon_btn(bot, "⬇ Export .txt", self._export,
                  fg=ACCENT_TEAL, bg=BG_PANEL).pack(side="right")
        _icon_btn(bot, "🗑 Clear", self._clear,
                  fg=TEXT_MUTED, hover_fg=ACCENT_RED, bg=BG_PANEL).pack(side="right", padx=8)

    def _auto_save(self, event=None):
        self._notes_store[self._tab] = self._text.get("1.0", "end-1c")
        _save_notes(self._notes_store)
        self._status.configure(text="✓ Saved")
        self.after(1200, lambda: self._status.configure(text=""))

    def _clear(self):
        if messagebox.askyesno("Clear Notes", "Clear all notes for this tab?", parent=self):
            self._text.delete("1.0", "end")
            self._auto_save()

    def _export(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt"), ("All files", "*.*")],
            initialfile=f"insightx_notes_{self._tab}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._text.get("1.0", "end-1c"))
            Toast.show(self.master, "Notes exported!")


class FavoritesDialog(tk.Toplevel):
    def __init__(self, parent, on_select):
        super().__init__(parent)
        self.title("InsightX — Saved Favorites")
        self.geometry("540x400")
        self.configure(bg=BG_PANEL)
        self.resizable(True, True)
        self._on_select = on_select
        self._favs = _load_favorites()
        self._build()
        self.grab_set()
        self.transient(parent)

    def _build(self):
        hdr = tk.Frame(self, bg=BG_PANEL)
        hdr.pack(fill="x", padx=18, pady=(14, 4))
        tk.Label(hdr, text="⭐  Saved Favorites",
                 font=("Trebuchet MS", 12, "bold"),
                 fg=ACCENT_YELLOW, bg=BG_PANEL).pack(side="left")
        tk.Label(hdr, text="— click to re-run, × to remove",
                 font=("Consolas", 7), fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left", padx=10)
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=14, pady=4)

        outer = tk.Frame(self, bg=BG_PANEL)
        outer.pack(fill="both", expand=True, padx=14, pady=4)

        cv = tk.Canvas(outer, bg=BG_PANEL, highlightthickness=0)
        vs = ttk.Scrollbar(outer, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=vs.set)
        vs.pack(side="right", fill="y")
        cv.pack(side="left", fill="both", expand=True)

        self._inner = tk.Frame(cv, bg=BG_PANEL)
        win = cv.create_window((0, 0), window=self._inner, anchor="nw")
        self._inner.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>", lambda e: cv.itemconfig(win, width=e.width))

        self._render_list()

        if not self._favs:
            tk.Label(self._inner, text="No favorites saved yet.\nStar any response using the ⭐ button.",
                     font=FONT_BODY, fg=TEXT_MUTED, bg=BG_PANEL,
                     justify="center", pady=30).pack()

    def _render_list(self):
        for w in self._inner.winfo_children():
            w.destroy()
        for i, fav in enumerate(self._favs):
            row = tk.Frame(self._inner, bg=BG_CARD,
                           highlightbackground=BORDER, highlightthickness=1)
            row.pack(fill="x", padx=4, pady=3)
            tk.Label(row, text=fav.get("timestamp", ""),
                     font=("Consolas", 7), fg=TEXT_MUTED, bg=BG_CARD,
                     padx=10, pady=4).pack(anchor="w", pady=(6, 2))
            q_lbl = tk.Label(row, text=fav.get("query", ""),
                             font=("Consolas", 9, "bold"),
                             fg=ACCENT_BLUE, bg=BG_CARD, padx=10,
                             anchor="w", cursor="hand2", wraplength=440, justify="left")
            q_lbl.pack(anchor="w")
            resp_preview = fav.get("response", "")[:120].replace("\n", " ")
            tk.Label(row, text=resp_preview + ("…" if len(fav.get("response", "")) > 120 else ""),
                     font=("Consolas", 8), fg=TEXT_SECONDARY, bg=BG_CARD,
                     padx=10, pady=4, anchor="w", wraplength=440, justify="left").pack(anchor="w", pady=(2, 6))
            btns = tk.Frame(row, bg=BG_CARD)
            btns.pack(fill="x", padx=10, pady=(0, 6))
            _icon_btn(btns, "▶ Re-run", lambda f=fav: self._rerun(f),
                      fg=ACCENT_GREEN, bg=BG_CARD, padx=6).pack(side="left")
            _icon_btn(btns, "× Remove", lambda idx=i: self._remove(idx),
                      fg=TEXT_MUTED, hover_fg=ACCENT_RED, bg=BG_CARD, padx=6).pack(side="left", padx=6)
            q_lbl.bind("<Button-1>", lambda e, f=fav: self._rerun(f))
            q_lbl.bind("<Enter>", lambda e, w=q_lbl: w.configure(fg=TEXT_PRIMARY))
            q_lbl.bind("<Leave>", lambda e, w=q_lbl: w.configure(fg=ACCENT_BLUE))

    def _rerun(self, fav):
        self.destroy()
        self._on_select(fav.get("query", ""))

    def _remove(self, idx):
        self._favs.pop(idx)
        _save_favorites(self._favs)
        self._render_list()


class AdvancedExplorerDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("InsightX — Advanced Explorer")
        self.geometry("820x740")
        self.minsize(700, 620)
        self.configure(bg=BG_PANEL)
        self.resizable(True, True)
        self.result_fig = None
        self._build()
        self.grab_set()
        self.transient(parent)

    def _build(self):
        hdr = tk.Frame(self, bg=BG_PANEL)
        hdr.pack(fill="x", padx=20, pady=(14, 4))
        tk.Label(hdr, text="🔬  Advanced Explorer",
                 font=("Trebuchet MS", 13, "bold"),
                 fg=ACCENT_BLUE, bg=BG_PANEL).pack(side="left")
        tk.Label(hdr,
                 text="Mix & match any columns → bar, grouped bar, line, pie, histogram, heatmap",
                 font=("Consolas", 8), fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left", padx=14)

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=16, pady=(4, 0))

        bot = tk.Frame(self, bg=BG_PANEL)
        bot.pack(side="bottom", fill="x", padx=20, pady=(0, 12))
        self._status = tk.Label(bot, text="", font=("Consolas", 8),
                                fg=TEXT_SECONDARY, bg=BG_PANEL)
        self._status.pack(side="left", padx=4)
        cancel = tk.Label(bot, text="  Cancel  ", font=("Consolas", 9),
                          fg=TEXT_MUTED, bg=BG_PANEL, padx=10, pady=7, cursor="hand2",
                          highlightbackground=BORDER, highlightthickness=1)
        cancel.pack(side="right", padx=6)
        cancel.bind("<Button-1>", lambda e: self.destroy())
        go = tk.Label(bot, text="  ▶  Generate Chart  ",
                      font=("Consolas", 11, "bold"),
                      fg=BG_DEEP, bg=ACCENT_BLUE, padx=18, pady=7, cursor="hand2")
        go.pack(side="right", padx=4)
        go.bind("<Button-1>", lambda e: self._generate())
        go.bind("<Enter>", lambda e: go.configure(bg=ACCENT_PURPLE))
        go.bind("<Leave>", lambda e: go.configure(bg=ACCENT_BLUE))

        tk.Frame(self, bg=BORDER, height=1).pack(side="bottom", fill="x", padx=16, pady=(0, 4))

        examples_frame = tk.Frame(self, bg=BG_PANEL)
        examples_frame.pack(side="bottom", fill="x", padx=20, pady=(0, 4))
        tk.Label(examples_frame, text="Quick examples:", font=("Consolas", 7),
                 fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left")
        examples = [
            ("Age × Fail Rate",           {"x": "sender_age_group", "y": "fail_rate", "ct": "bar"}),
            ("Bank × Flagged (grouped)",   {"x": "sender_bank", "y": "fraud_rate", "color": "device_type", "ct": "grouped_bar"}),
            ("Amount hist by device",      {"x": "amount_inr", "y": "count", "color": "device_type", "ct": "histogram"}),
            ("Network × Type (grouped)",   {"x": "network_type", "y": "count", "color": "transaction_type", "ct": "grouped_bar"}),
            ("State × Revenue (bar)",      {"x": "sender_state", "y": "sum_amount", "ct": "bar"}),
            ("Age × Avg Spend (line)",     {"x": "sender_age_group", "y": "mean_amount", "ct": "line"}),
        ]
        for ex_name, ex_cfg in examples:
            b = tk.Label(examples_frame, text=ex_name, font=("Consolas", 7),
                         fg=ACCENT_BLUE, bg=BG_PANEL, cursor="hand2",
                         padx=5, pady=1,
                         highlightbackground=BORDER, highlightthickness=1)
            b.pack(side="left", padx=2)
            b.bind("<Button-1>", lambda e, cfg=ex_cfg: self._apply_example(cfg))
            b.bind("<Enter>", lambda e, w=b: w.configure(fg=TEXT_PRIMARY, bg=BG_HOVER))
            b.bind("<Leave>", lambda e, w=b: w.configure(fg=ACCENT_BLUE, bg=BG_PANEL))

        body = tk.Frame(self, bg=BG_PANEL)
        body.pack(fill="both", expand=True, padx=20, pady=4)
        left = tk.Frame(body, bg=BG_PANEL)
        right = tk.Frame(body, bg=BG_PANEL)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right.pack(side="left", fill="both", expand=True)

        _lbl(left, "X-AXIS  (Group / Dimension)").pack(anchor="w", pady=(6, 2))
        self._x_var = tk.StringVar(value=ALL_COL_LABELS[0])
        _combo(left, self._x_var, ALL_COL_LABELS, width=38).pack(anchor="w")

        _lbl(left, "Y-METRIC  (What to Measure)").pack(anchor="w", pady=(10, 2))
        self._y_var = tk.StringVar(value=f"{Y_METRICS[0][0]} — {Y_METRICS[0][1]}")
        _combo(left, self._y_var, [f"{k} — {v}" for k, v in Y_METRICS], width=38).pack(anchor="w")

        _lbl(left, "COLOR-BY  (Split into Groups, optional)").pack(anchor="w", pady=(10, 2))
        self._color_var = tk.StringVar(value="(none)")
        _combo(left, self._color_var, ["(none)"] + ALL_COL_LABELS, width=38).pack(anchor="w")
        tk.Label(left, text="→ creates Grouped Bar / multi-series Line / overlapping Histogram",
                 font=("Consolas", 7), fg=TEXT_MUTED, bg=BG_PANEL).pack(anchor="w")

        _lbl(left, "CHART TYPE").pack(anchor="w", pady=(10, 2))
        self._chart_var = tk.StringVar(value="bar")
        ct_grid = tk.Frame(left, bg=BG_PANEL)
        ct_grid.pack(anchor="w")
        for i, (val, lbl_text) in enumerate(CHART_TYPES):
            tk.Radiobutton(ct_grid, text=lbl_text, variable=self._chart_var, value=val,
                           font=FONT_SMALL, fg=TEXT_SECONDARY, bg=BG_PANEL,
                           selectcolor=BG_SELECTED, activebackground=BG_PANEL,
                           activeforeground=TEXT_PRIMARY).grid(row=i % 4, column=i // 4, sticky="w", padx=6)

        _lbl(left, "BINS  (for Histogram)", small=True).pack(anchor="w", pady=(8, 2))
        bin_row = tk.Frame(left, bg=BG_PANEL)
        bin_row.pack(anchor="w")
        self._bins_var = tk.StringVar(value="40")
        tk.Entry(bin_row, textvariable=self._bins_var, font=FONT_SMALL,
                 fg=TEXT_PRIMARY, bg=BG_INPUT, insertbackground=ACCENT_BLUE,
                 relief="flat", width=7).pack(side="left")
        tk.Label(bin_row, text="  bins", font=("Consolas", 8),
                 fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left")

        _lbl(left, "TITLE OVERRIDE  (optional)", small=True).pack(anchor="w", pady=(8, 2))
        self._title_var = tk.StringVar()
        tk.Entry(left, textvariable=self._title_var, font=FONT_SMALL,
                 fg=TEXT_PRIMARY, bg=BG_INPUT, insertbackground=ACCENT_BLUE,
                 relief="flat", width=38).pack(anchor="w")

        _lbl(right, "FILTER 1  (optional)").pack(anchor="w", pady=(6, 2))
        self._f1col_var = tk.StringVar(value="(none)")
        _combo(right, self._f1col_var, ["(none)"] + ALL_COL_LABELS, width=34).pack(anchor="w")
        self._f1val_var = tk.StringVar()
        f1row = tk.Frame(right, bg=BG_PANEL)
        f1row.pack(anchor="w", pady=(2, 0))
        tk.Label(f1row, text="= ", font=FONT_SMALL, fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left")
        tk.Entry(f1row, textvariable=self._f1val_var, font=FONT_SMALL,
                 fg=TEXT_PRIMARY, bg=BG_INPUT, insertbackground=ACCENT_BLUE,
                 relief="flat", width=22).pack(side="left")

        _lbl(right, "FILTER 2  (optional, AND logic)", small=True).pack(anchor="w", pady=(10, 2))
        self._f2col_var = tk.StringVar(value="(none)")
        _combo(right, self._f2col_var, ["(none)"] + ALL_COL_LABELS, width=34).pack(anchor="w")
        self._f2val_var = tk.StringVar()
        f2row = tk.Frame(right, bg=BG_PANEL)
        f2row.pack(anchor="w", pady=(2, 0))
        tk.Label(f2row, text="= ", font=FONT_SMALL, fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left")
        tk.Entry(f2row, textvariable=self._f2val_var, font=FONT_SMALL,
                 fg=TEXT_PRIMARY, bg=BG_INPUT, insertbackground=ACCENT_BLUE,
                 relief="flat", width=22).pack(side="left")

        _lbl(right, "FILTER 3  (optional, AND logic)", small=True).pack(anchor="w", pady=(10, 2))
        self._f3col_var = tk.StringVar(value="(none)")
        _combo(right, self._f3col_var, ["(none)"] + ALL_COL_LABELS, width=34).pack(anchor="w")
        self._f3val_var = tk.StringVar()
        f3row = tk.Frame(right, bg=BG_PANEL)
        f3row.pack(anchor="w", pady=(2, 0))
        tk.Label(f3row, text="= ", font=FONT_SMALL, fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left")
        tk.Entry(f3row, textvariable=self._f3val_var, font=FONT_SMALL,
                 fg=TEXT_PRIMARY, bg=BG_INPUT, insertbackground=ACCENT_BLUE,
                 relief="flat", width=22).pack(side="left")

        _lbl(right, "SORT ORDER").pack(anchor="w", pady=(10, 2))
        self._sort_var = tk.StringVar(value="desc — High → Low")
        _combo(right, self._sort_var, [f"{k} — {v}" for k, v in SORT_OPTIONS], width=26).pack(anchor="w")

        _lbl(right, "TOP-N  (max categories)", small=True).pack(anchor="w", pady=(8, 2))
        topn_row = tk.Frame(right, bg=BG_PANEL)
        topn_row.pack(anchor="w")
        self._topn_var = tk.StringVar(value="12")
        tk.Entry(topn_row, textvariable=self._topn_var, font=FONT_SMALL,
                 fg=TEXT_PRIMARY, bg=BG_INPUT, insertbackground=ACCENT_BLUE,
                 relief="flat", width=6).pack(side="left")
        tk.Label(topn_row, text="  categories max", font=("Consolas", 8),
                 fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left")

        _lbl(right, "OPTIONS", small=True).pack(anchor="w", pady=(8, 2))
        self._norm_var = tk.BooleanVar(value=False)
        tk.Checkbutton(right, text="Normalize to % share",
                       variable=self._norm_var, font=FONT_SMALL,
                       fg=TEXT_SECONDARY, bg=BG_PANEL, selectcolor=BG_SELECTED,
                       activebackground=BG_PANEL, activeforeground=TEXT_PRIMARY).pack(anchor="w")
        self._show_values_var = tk.BooleanVar(value=True)
        tk.Checkbutton(right, text="Show value labels on chart",
                       variable=self._show_values_var, font=FONT_SMALL,
                       fg=TEXT_SECONDARY, bg=BG_PANEL, selectcolor=BG_SELECTED,
                       activebackground=BG_PANEL, activeforeground=TEXT_PRIMARY).pack(anchor="w")
        self._log_scale_var = tk.BooleanVar(value=False)
        tk.Checkbutton(right, text="Logarithmic Y-axis scale",
                       variable=self._log_scale_var, font=FONT_SMALL,
                       fg=TEXT_SECONDARY, bg=BG_PANEL, selectcolor=BG_SELECTED,
                       activebackground=BG_PANEL, activeforeground=TEXT_PRIMARY).pack(anchor="w")

    def _apply_example(self, cfg):
        if "x" in cfg:
            for lbl in ALL_COL_LABELS:
                if lbl.startswith(cfg["x"]):
                    self._x_var.set(lbl)
                    break
        if "y" in cfg:
            for k, v in Y_METRICS:
                if k == cfg["y"]:
                    self._y_var.set(f"{k} — {v}")
                    break
        if "color" in cfg:
            for lbl in ALL_COL_LABELS:
                if lbl.startswith(cfg["color"]):
                    self._color_var.set(lbl)
                    break
        else:
            self._color_var.set("(none)")
        if "ct" in cfg:
            self._chart_var.set(cfg["ct"])

    def _generate(self):
        self._status.configure(text="⟳ Generating…", fg=ACCENT_ORANGE)
        self.update()
        try:
            x_label = self._x_var.get()
            x_col = COL_LABEL_MAP.get(x_label, x_label.split(" — ")[0])
            y_raw = self._y_var.get()
            y_metric = y_raw.split(" — ")[0]
            color_label = self._color_var.get()
            color_col = None if color_label == "(none)" else COL_LABEL_MAP.get(color_label, color_label.split(" — ")[0])
            chart_type = self._chart_var.get()
            sort_key = self._sort_var.get().split(" — ")[0]

            filters = []
            for col_var, val_var in [
                (self._f1col_var, self._f1val_var),
                (self._f2col_var, self._f2val_var),
                (self._f3col_var, self._f3val_var),
            ]:
                cl = col_var.get()
                vl = val_var.get().strip()
                if cl != "(none)" and vl:
                    fc = COL_LABEL_MAP.get(cl, cl.split(" — ")[0])
                    filters.append((fc, vl))

            try:
                top_n = int(self._topn_var.get())
            except ValueError:
                top_n = 12
            try:
                bins_ = int(self._bins_var.get())
            except ValueError:
                bins_ = 40

            normalize = self._norm_var.get()
            title_override = self._title_var.get().strip() or None
            show_values = self._show_values_var.get()
            log_scale = self._log_scale_var.get()

            config = {
                "x_col": x_col, "y_metric": y_metric,
                "color_col": color_col, "chart_type": chart_type,
                "filters": filters, "sort": sort_key,
                "top_n": top_n, "bins": bins_,
                "normalize": normalize,
                "title_override": title_override,
                "show_values": show_values,
                "log_scale": log_scale,
            }

            fig = engine.generate_explorer_chart(config)
            self.result_fig = fig
            self.result_config = config
            self._status.configure(text="✓ Done!", fg=ACCENT_GREEN)
            self.after(600, self.destroy)

        except Exception as ex:
            self._status.configure(
                text=f"Error: {str(ex)[:60]}",
                fg=ACCENT_RED
            )


class SearchDialog(tk.Toplevel):
    def __init__(self, parent, rich_data, on_jump):
        super().__init__(parent)
        self.title("InsightX — Search Chat History")
        self.geometry("520x460")
        self.configure(bg=BG_PANEL)
        self.resizable(True, True)
        self._rich = rich_data
        self._on_jump = on_jump
        self._build()
        self.grab_set()
        self.transient(parent)

    def _build(self):
        hdr = tk.Frame(self, bg=BG_PANEL)
        hdr.pack(fill="x", padx=18, pady=(14, 4))
        tk.Label(hdr, text="🔍  Search Chat History",
                 font=("Trebuchet MS", 12, "bold"),
                 fg=ACCENT_BLUE, bg=BG_PANEL).pack(side="left")
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=14, pady=4)
        inp_row = tk.Frame(self, bg=BG_PANEL)
        inp_row.pack(fill="x", padx=14, pady=6)
        self._search_var = tk.StringVar()
        entry = tk.Entry(inp_row, textvariable=self._search_var, font=FONT_INPUT,
                         fg=TEXT_PRIMARY, bg=BG_INPUT, insertbackground=ACCENT_BLUE,
                         relief="flat", bd=0)
        entry.pack(side="left", fill="x", expand=True, padx=10, pady=8)
        entry.bind("<KeyRelease>", self._do_search)
        entry.focus()
        tk.Label(inp_row, text="⌨ type to search",
                 font=("Consolas", 7), fg=TEXT_MUTED, bg=BG_PANEL).pack(side="right", padx=8)

        self._result_frame = tk.Frame(self, bg=BG_PANEL)
        self._result_frame.pack(fill="both", expand=True, padx=14, pady=4)

        cv = tk.Canvas(self._result_frame, bg=BG_PANEL, highlightthickness=0)
        vs = ttk.Scrollbar(self._result_frame, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=vs.set)
        vs.pack(side="right", fill="y")
        cv.pack(side="left", fill="both", expand=True)
        self._inner = tk.Frame(cv, bg=BG_PANEL)
        win = cv.create_window((0, 0), window=self._inner, anchor="nw")
        self._inner.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>", lambda e: cv.itemconfig(win, width=e.width))

        self._count_label = tk.Label(self, text="", font=("Consolas", 7),
                                     fg=TEXT_MUTED, bg=BG_PANEL)
        self._count_label.pack(pady=4)

    def _do_search(self, event=None):
        for w in self._inner.winfo_children():
            w.destroy()
        term = self._search_var.get().strip().lower()
        if not term:
            self._count_label.configure(text="")
            return
        hits = []
        for i, entry in enumerate(self._rich):
            text = entry.get("content", "")
            if term in text.lower():
                hits.append((i, entry))
        self._count_label.configure(text=f"{len(hits)} result(s) found")
        for idx, (i, entry) in enumerate(hits[:40]):
            role = entry.get("role", "")
            prefix = "You:" if role == "user" else "IX:"
            color = ACCENT_BLUE if role == "user" else ACCENT_GREEN
            text = entry.get("content", "")
            start = max(0, text.lower().find(term) - 40)
            snippet = ("…" if start > 0 else "") + text[start:start + 120] + ("…" if start + 120 < len(text) else "")
            card = tk.Frame(self._inner, bg=BG_CARD,
                            highlightbackground=BORDER, highlightthickness=1,
                            cursor="hand2")
            card.pack(fill="x", padx=4, pady=2)
            tk.Label(card, text=prefix, font=("Consolas", 8, "bold"),
                     fg=color, bg=BG_CARD, padx=10, pady=4).pack(anchor="w", pady=(6, 2))
            tk.Label(card, text=snippet, font=("Consolas", 8),
                     fg=TEXT_SECONDARY, bg=BG_CARD, padx=10, pady=4,
                     anchor="w", wraplength=440, justify="left").pack(anchor="w", pady=(0, 6))
            card.bind("<Button-1>", lambda e, pos=i: self._jump(pos))
            card.bind("<Enter>", lambda e, w=card: w.configure(bg=BG_HOVER))
            card.bind("<Leave>", lambda e, w=card: w.configure(bg=BG_CARD))

    def _jump(self, position):
        self.destroy()
        self._on_jump(position)


class InsightXApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("InsightX — Techfest 2025-26")
        self.geometry("1440x900")
        self.minsize(1100, 660)
        self.configure(bg=BG_DEEP)
        self._q                = queue.Queue()
        self._loading          = False
        self._active_tab       = "Default"
        self._tab_buttons      = {}
        self._custom_actions   = _load_custom_actions()
        self._pinned           = _load_pinned()
        self._favorites        = _load_favorites()
        self._zoom_level       = 1.0
        self._base_fonts       = None
        self._message_widgets  = {}
        self._session_start    = datetime.now()
        self._query_count      = 0
        self._sidebar_visible  = True
        self._sidebar_width    = 256
        self._dark_mode        = True
        self._input_history    = []
        self._input_hist_idx   = -1
        self._build_ui()
        self._start_engine_thread()
        self._bind_scroll()
        self._bind_keyboard_shortcuts()
        self.after(50, self._poll_queue)
        self.after(1000, self._update_session_timer)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _bind_keyboard_shortcuts(self):
        self.bind_all("<Control-n>", lambda e: self._new_tab())
        self.bind_all("<Control-s>", lambda e: self._save_chat())
        self.bind_all("<Control-f>", lambda e: self._open_search())
        self.bind_all("<Control-r>", lambda e: self._open_explorer())
        self.bind_all("<Control-d>", lambda e: self._open_data_snapshot())
        self.bind_all("<Control-p>", lambda e: self._toggle_pin_mode())
        self.bind_all("<Control-slash>", lambda e: self._show_shortcuts())
        self.bind_all("<Escape>", lambda e: self._clear_input())
        self.bind_all("<Control-l>", lambda e: self._clear_chat_confirm())
        self.bind_all("<Control-BackSpace>", lambda e: self._clear_input())
        self.input_box.bind("<Up>", self._history_up)
        self.input_box.bind("<Down>", self._history_down)
        self.input_box.bind("<Tab>", self._autocomplete)

    def _show_shortcuts(self):
        dlg = tk.Toplevel(self)
        dlg.title("Keyboard Shortcuts")
        dlg.geometry("420x480")
        dlg.configure(bg=BG_PANEL)
        dlg.grab_set()
        dlg.transient(self)
        tk.Label(dlg, text="⌨  Keyboard Shortcuts",
                 font=("Trebuchet MS", 12, "bold"),
                 fg=ACCENT_BLUE, bg=BG_PANEL, pady=14).pack()
        tk.Frame(dlg, bg=BORDER, height=1).pack(fill="x", padx=14)
        shortcuts = [
            ("Ctrl+Enter",     "Send message"),
            ("Ctrl+N",         "New chat tab"),
            ("Ctrl+S",         "Save chat"),
            ("Ctrl+F",         "Search chat history"),
            ("Ctrl+R",         "Open Advanced Explorer"),
            ("Ctrl+D",         "Open Data Snapshot"),
            ("Ctrl+L",         "Clear current chat"),
            ("Ctrl+/",         "Show shortcuts"),
            ("Ctrl++/−",       "Zoom in/out"),
            ("Ctrl+0",         "Reset zoom"),
            ("Ctrl+Scroll",    "Zoom in/out"),
            ("↑/↓ in input",   "Browse prompt history"),
            ("Tab in input",   "Autocomplete suggestion"),
            ("Escape",         "Clear input"),
            ("Home/End",       "Scroll to top/bottom"),
            ("Page Up/Down",   "Scroll chat"),
        ]
        for key, desc in shortcuts:
            row = tk.Frame(dlg, bg=BG_PANEL)
            row.pack(fill="x", padx=20, pady=2)
            tk.Label(row, text=key, font=("Consolas", 9, "bold"),
                     fg=ACCENT_BLUE, bg=BG_PANEL, width=18, anchor="w").pack(side="left")
            tk.Label(row, text=desc, font=("Consolas", 9),
                     fg=TEXT_SECONDARY, bg=BG_PANEL, anchor="w").pack(side="left")
        _icon_btn(dlg, "  Close  ", dlg.destroy,
                  fg=TEXT_MUTED, bg=BG_PANEL, padx=14).pack(pady=14)

    def _history_up(self, event=None):
        if not self._input_history:
            return "break"
        if self._input_hist_idx < len(self._input_history) - 1:
            self._input_hist_idx += 1
        self.input_var.set(self._input_history[-(self._input_hist_idx + 1)])
        self.input_box.icursor("end")
        return "break"

    def _history_down(self, event=None):
        if self._input_hist_idx <= 0:
            self._input_hist_idx = -1
            self.input_var.set("")
            return "break"
        self._input_hist_idx -= 1
        self.input_var.set(self._input_history[-(self._input_hist_idx + 1)])
        self.input_box.icursor("end")
        return "break"

    def _autocomplete(self, event=None):
        current = self.input_var.get().strip().lower()
        if not current:
            return "break"
        suggestions = [
            "show a bar chart by sender bank",
            "what is the average transaction amount",
            "compare failure rates by device type",
            "show fraud flag rate by sender bank",
            "generate full report",
            "summarize our conversation",
            "show total amount by merchant category table",
            "what percentage of high value transactions are flagged",
            "show a pie chart of transaction status",
            "compare p2p vs p2m transaction amounts",
            "show failure rate by network type",
            "which age group uses p2p transfers most",
            "show transactions by day of week",
        ]
        for s in suggestions:
            if s.startswith(current) and s != current:
                self.input_var.set(s)
                self.input_box.icursor("end")
                self.input_box.selection_range(len(current), "end")
                return "break"
        return "break"

    def _clear_input(self):
        self.input_var.set("")
        self._input_hist_idx = -1
        self.input_box.focus()

    def _clear_chat_confirm(self):
        if messagebox.askyesno("Clear Chat",
                               f"Clear all messages in '{self._active_tab}'?", parent=self):
            engine.chat_tabs[self._active_tab] = []
            engine.chat_rich[self._active_tab] = []
            for w in self._chat_inner.winfo_children():
                w.destroy()
            self._add_system_msg("Chat cleared.")
            Toast.show(self, "Chat cleared")

    def _toggle_sidebar(self):
        if self._sidebar_visible:
            self.sidebar.pack_forget()
            self._sidebar_visible = False
            self._toggle_sidebar_btn.configure(text="▶")
        else:
            self.sidebar.pack(side="left", fill="y", before=self.main_frame)
            self._sidebar_visible = True
            self._toggle_sidebar_btn.configure(text="◀")

    def _update_session_timer(self):
        elapsed = datetime.now() - self._session_start
        mins = int(elapsed.total_seconds() // 60)
        secs = int(elapsed.total_seconds() % 60)
        try:
            self._session_label.configure(text=f"⏱ {mins:02d}:{secs:02d}  •  {self._query_count} queries")
        except Exception:
            pass
        self.after(1000, self._update_session_timer)

    def _on_close(self):
        _save_pinned(self._pinned)
        _save_favorites(self._favorites)
        _save_custom_actions(self._custom_actions)
        self.destroy()

    def _bind_scroll(self):
        self.bind_all("<MouseWheel>", self._on_mousewheel)
        self.bind_all("<Button-4>", lambda e: self._on_mousewheel(e, -1))
        self.bind_all("<Button-5>", lambda e: self._on_mousewheel(e, 1))
        self.bind_all("<Shift-MouseWheel>", self._on_shift_mousewheel)
        self.bind_all("<Up>", lambda e: self._key_scroll(-1))
        self.bind_all("<Down>", lambda e: self._key_scroll(1))
        self.bind_all("<Prior>", lambda e: self._key_scroll(-10))
        self.bind_all("<Next>", lambda e: self._key_scroll(10))
        self.bind_all("<Home>", lambda e: self._chat_canvas.yview_moveto(0.0))
        self.bind_all("<End>", lambda e: self._chat_canvas.yview_moveto(1.0))
        self.bind_all("<Control-MouseWheel>", self._on_ctrl_mousewheel)
        self.bind_all("<Control-Button-4>", lambda e: self._zoom(-1))
        self.bind_all("<Control-Button-5>", lambda e: self._zoom(1))
        self.bind_all("<Control-equal>", lambda e: self._zoom(-1))
        self.bind_all("<Control-minus>", lambda e: self._zoom(1))
        self.bind_all("<Control-0>", lambda e: self._zoom_reset())

    def _on_ctrl_mousewheel(self, event):
        direction = int(-1 * event.delta / 120)
        self._zoom(direction)
        return "break"

    def _zoom(self, direction):
        step = 0.10
        # direction > 0 means zoom OUT (smaller), direction < 0 means zoom IN (bigger)
        new_zoom = round(self._zoom_level - direction * step, 2)
        new_zoom = max(0.40, min(2.0, new_zoom))
        if new_zoom == self._zoom_level:
            return
        self._zoom_level = new_zoom
        self._apply_zoom()
        self._update_zoom_label()

    def _zoom_reset(self):
        self._zoom_level = 1.0
        self._apply_zoom()
        self._update_zoom_label()

    def _apply_zoom(self):
        import tkinter.font as tkfont
        z = self._zoom_level
        # Store base sizes on first call so we always scale from the original
        if not hasattr(self, "_base_font_sizes"):
            self._base_font_sizes = {}
            for fname in tkfont.names(root=self):
                try:
                    fo = tkfont.Font(name=fname, exists=True, root=self)
                    sz = fo.actual("size")
                    if sz < 0:
                        sz = abs(sz)
                    if sz >= 6:
                        self._base_font_sizes[fname] = sz
                except Exception:
                    pass
        for fname, base_sz in self._base_font_sizes.items():
            try:
                fo = tkfont.Font(name=fname, exists=True, root=self)
                target = max(6, int(round(base_sz * z)))
                fo.configure(size=target)
            except Exception:
                pass
        self._last_zoom = z

    def _update_zoom_label(self):
        try:
            pct = int(round(self._zoom_level * 100))
            self._zoom_label.configure(text=f"🔍 {pct}%")
        except Exception:
            pass

    def _on_mousewheel(self, event, linux_dir=None):
        amt = linux_dir if linux_dir is not None else int(-1 * event.delta / 120)
        if amt == 0:
            return
        # If event originated inside a child Toplevel (e.g. expanded table dialog),
        # route scroll only to the widget under the cursor there — never the main canvas.
        try:
            top = event.widget.winfo_toplevel()
            if top is not self:
                try:
                    event.widget.yview_scroll(amt, "units")
                except Exception:
                    pass
                return "break"
        except Exception:
            pass
        if self._cursor_in_widget(event, self.sidebar):
            self._sb_canvas.yview_scroll(amt, "units")
            return "break"
        tc = self._find_table_canvas(event.widget)
        if tc:
            y0, y1 = tc.yview()
            if (y0 <= 0.0 and amt < 0) or (y1 >= 1.0 and amt > 0):
                self._chat_canvas.yview_scroll(amt, "units")
            else:
                tc.yview_scroll(amt, "units")
            return "break"
        self._chat_canvas.yview_scroll(amt, "units")
        return "break"

    def _on_shift_mousewheel(self, event, linux_dir=None):
        amt = linux_dir if linux_dir is not None else int(-1 * event.delta / 120)
        tc = self._find_table_canvas(event.widget)
        if tc:
            tc.xview_scroll(amt, "units")
            return "break"

    def _cursor_in_widget(self, event, widget):
        try:
            wx = widget.winfo_rootx(); wy = widget.winfo_rooty()
            ww = widget.winfo_width(); wh = widget.winfo_height()
            return wx <= event.x_root <= wx + ww and wy <= event.y_root <= wy + wh
        except Exception:
            return False

    def _find_table_canvas(self, widget):
        try:
            w = widget
            for _ in range(16):
                if w is None:
                    break
                if getattr(w, "_is_table_canvas", False):
                    return w
                w = getattr(w, "master", None)
        except Exception:
            pass
        return None

    def _key_scroll(self, units):
        if self.focus_get() == self.input_box:
            return
        self._chat_canvas.yview_scroll(units, "units")

    def _prop_scroll(self, widget):
        pass

    def _prop_sb_scroll(self, widget):
        pass

    def _build_ui(self):
        self._build_sidebar()
        self._build_main()

    def _build_sidebar(self):
        self.sidebar = tk.Frame(self, bg=BG_PANEL, width=self._sidebar_width)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        logo = tk.Frame(self.sidebar, bg=BG_PANEL, pady=10)
        logo.pack(fill="x")
        tk.Label(logo, text="IX", font=("Georgia", 26, "bold"),
                 fg=ACCENT_BLUE, bg=BG_PANEL).pack()
        tk.Label(logo, text="InsightX", font=("Trebuchet MS", 11, "bold"),
                 fg=TEXT_PRIMARY, bg=BG_PANEL).pack()
        tk.Label(logo, text="Fintech Intelligence  •  Techfest 2025-26",
                 font=("Consolas", 7), fg=TEXT_MUTED, bg=BG_PANEL).pack()

        self._session_label = tk.Label(logo, text="⏱ 00:00  •  0 queries",
                                       font=("Consolas", 7), fg=TEXT_MUTED, bg=BG_PANEL)
        self._session_label.pack(pady=(2, 0))

        tk.Frame(self.sidebar, bg=BORDER, height=1).pack(fill="x", padx=14, pady=4)

        ch_hdr = tk.Frame(self.sidebar, bg=BG_PANEL)
        ch_hdr.pack(fill="x", padx=10, pady=(2, 1))
        tk.Label(ch_hdr, text="CHATS", font=("Consolas", 8, "bold"),
                 fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left")
        add_tb = tk.Label(ch_hdr, text="+", font=("Consolas", 12, "bold"),
                          fg=ACCENT_BLUE, bg=BG_PANEL, cursor="hand2")
        add_tb.pack(side="right")
        add_tb.bind("<Button-1>", lambda e: self._new_tab())
        add_tb.bind("<Enter>", lambda e: add_tb.configure(fg=TEXT_PRIMARY))
        add_tb.bind("<Leave>", lambda e: add_tb.configure(fg=ACCENT_BLUE))

        self.tabs_frame = tk.Frame(self.sidebar, bg=BG_PANEL)
        self.tabs_frame.pack(fill="x", padx=6)
        self._render_tab_list()
        tk.Frame(self.sidebar, bg=BORDER, height=1).pack(fill="x", padx=14, pady=4)

        self._sb_canvas = tk.Canvas(self.sidebar, bg=BG_PANEL,
                                    highlightthickness=0, bd=0)
        sb_vs = ttk.Scrollbar(self.sidebar, orient="vertical",
                               command=self._sb_canvas.yview)
        self._sb_canvas.configure(yscrollcommand=sb_vs.set)
        sb_vs.pack(side="right", fill="y")
        self._sb_canvas.pack(side="left", fill="both", expand=True)

        self._actions_inner = tk.Frame(self._sb_canvas, bg=BG_PANEL)
        self._sb_win = self._sb_canvas.create_window(
            (0, 0), window=self._actions_inner, anchor="nw")
        self._actions_inner.bind("<Configure>", lambda e:
                                 self._sb_canvas.configure(scrollregion=self._sb_canvas.bbox("all")))
        self._sb_canvas.bind("<Configure>", lambda e:
                             self._sb_canvas.itemconfig(self._sb_win, width=e.width))

        self._render_actions()

        self.status_label = tk.Label(self.sidebar, text="● Loading…",
                                     font=("Consolas", 8),
                                     fg=ACCENT_ORANGE, bg=BG_PANEL)
        self.status_label.pack(side="bottom", pady=2)
        tk.Label(self.sidebar, text="Ctrl+/ for shortcuts",
                 font=("Consolas", 6), fg=TEXT_MUTED, bg=BG_PANEL).pack(side="bottom")

    def _sec_hdr(self, text, plus_cmd=None):
        row = tk.Frame(self._actions_inner, bg=BG_PANEL)
        row.pack(fill="x", padx=10, pady=(8, 1))
        tk.Label(row, text=text, font=("Consolas", 7, "bold"),
                 fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left")
        if plus_cmd:
            b = tk.Label(row, text="+", font=("Consolas", 11, "bold"),
                         fg=ACCENT_TEAL, bg=BG_PANEL, cursor="hand2")
            b.pack(side="right")
            b.bind("<Button-1>", lambda e: plus_cmd())

    def _sb_btn(self, label, query=None, cmd=None, color=None, bold=False, deletable=False, pinnable=False):
        row = tk.Frame(self._actions_inner, bg=BG_PANEL)
        row.pack(fill="x")
        fg = color or TEXT_SECONDARY
        fn = ("Consolas", 8, "bold") if bold else ("Consolas", 8)
        btn = tk.Label(row, text=label, font=fn, fg=fg, bg=BG_PANEL,
                       anchor="w", padx=14, pady=3, cursor="hand2")
        btn.pack(side="left", fill="x", expand=True)

        def _click(e):
            if cmd:
                cmd()
            elif query:
                self._quick(query)

        btn.bind("<Button-1>", _click)
        btn.bind("<Enter>", lambda e: btn.configure(bg=BG_HOVER, fg=TEXT_PRIMARY))
        btn.bind("<Leave>", lambda e: btn.configure(bg=BG_PANEL, fg=fg))
        self._prop_sb_scroll(btn)
        self._prop_sb_scroll(row)

        if deletable:
            d = tk.Label(row, text="×", font=("Consolas", 9), fg=TEXT_MUTED,
                         bg=BG_PANEL, padx=5, cursor="hand2")
            d.pack(side="right")
            d.bind("<Button-1>", lambda e, q=query: self._remove_custom_action(q))

        if pinnable and query:
            p = tk.Label(row, text="📌", font=("Consolas", 8), fg=TEXT_MUTED,
                         bg=BG_PANEL, padx=3, cursor="hand2")
            p.pack(side="right")
            p.bind("<Button-1>", lambda e, q=query, l=label: self._pin_query(q, l))
            p.bind("<Enter>", lambda e, w=p: w.configure(fg=ACCENT_ORANGE))
            p.bind("<Leave>", lambda e, w=p: w.configure(fg=TEXT_MUTED))

    def _pin_query(self, query, label):
        if not any(p["query"] == query for p in self._pinned):
            self._pinned.append({"query": query, "label": label,
                                  "pinned_at": datetime.now().isoformat()})
            _save_pinned(self._pinned)
            self._render_actions()
            Toast.show(self, f"📌 Pinned: {label[:30]}")

    def _render_actions(self):
        for w in self._actions_inner.winfo_children():
            w.destroy()

        if self._pinned:
            self._sec_hdr("📌 PINNED")
            for pin in self._pinned:
                row = tk.Frame(self._actions_inner, bg=BG_PANEL)
                row.pack(fill="x")
                btn = tk.Label(row, text=pin["label"][:30], font=("Consolas", 8),
                               fg=ACCENT_ORANGE, bg=BG_PANEL, anchor="w",
                               padx=14, pady=3, cursor="hand2")
                btn.pack(side="left", fill="x", expand=True)
                btn.bind("<Button-1>", lambda e, q=pin["query"]: self._quick(q))
                btn.bind("<Enter>", lambda e, w=btn: w.configure(bg=BG_HOVER, fg=TEXT_PRIMARY))
                btn.bind("<Leave>", lambda e, w=btn: w.configure(bg=BG_PANEL, fg=ACCENT_ORANGE))
                d = tk.Label(row, text="×", font=("Consolas", 9), fg=TEXT_MUTED,
                             bg=BG_PANEL, padx=5, cursor="hand2")
                d.pack(side="right")
                d.bind("<Button-1>", lambda e, q=pin["query"]: self._unpin(q))

        self._sec_hdr("OVERVIEW")
        for lbl, q in [
            ("📊  Full Stats Table", "show full statistics table"),
            ("📋  Full Report", "generate full report"),
            ("💬  Chat Summary", "summarize our conversation"),
            ("📸  Data Snapshot", None),
            ("⭐  View Favorites", None),
        ]:
            if lbl == "📸  Data Snapshot":
                self._sb_btn(lbl, cmd=self._open_data_snapshot, color=ACCENT_TEAL)
            elif lbl == "⭐  View Favorites":
                self._sb_btn(lbl, cmd=self._open_favorites, color=ACCENT_YELLOW)
            else:
                self._sb_btn(lbl, query=q, pinnable=True)

        self._sec_hdr("DATA EXPLORATION")
        self._sb_btn("🔬  Advanced Explorer ✦", cmd=self._open_explorer,
                     color=ACCENT_BLUE, bold=True)
        self._sb_btn("🔍  Search Chat", cmd=self._open_search, color=ACCENT_PURPLE)

        self._sec_hdr("QUERY TYPES — CORE DELIVERABLES")
        for lbl, q in [
            ("📝  Descriptive: Avg bill payment", "What is the average transaction amount for bill payments?"),
            ("⚖  Comparative: Android vs iOS", "How do failure rates compare between Android and iOS users?"),
            ("👥  Segmentation: P2P by age", "Which age group uses P2P transfers most frequently?"),
            ("🔗  Correlation: Network vs success", "Is there a relationship between network type and transaction success?"),
            ("🛡  Risk: High-value flagged %", "What percentage of high-value transactions are flagged for review?"),
        ]:
            self._sb_btn(lbl, query=q, pinnable=True)

        self._sec_hdr("CHARTS & QUICK VIEWS")
        for lbl, q in [
            ("🍕  Fraud Flag Pie", "show a pie chart of fraud flag"),
            ("🏦  Sender Bank Bar", "show a bar chart by sender bank"),
            ("📱  Device Type Chart", "show a bar chart by device type"),
            ("📡  Network Type Chart", "show a bar chart by network type"),
            ("💳  Transaction Type Chart", "show a bar chart by transaction type"),
            ("🛒  Merchant Category Chart", "show a bar chart by merchant category"),
            ("👤  Age Group Chart", "show a bar chart by sender age group"),
            ("📅  Day of Week Chart", "show transactions by day of week"),
            ("🗺  Top States Chart", "show a bar chart by sender state"),
            ("✅  Success vs Fail Pie", "show a pie chart of transaction status"),
            ("📶  Network Fail Rate Bar", "show failure rate by network type bar chart"),
        ]:
            self._sb_btn(lbl, query=q)

        self._sec_hdr("STATISTICAL ANALYSIS")
        for lbl, q in [
            ("💰  Amount Statistics", "show full statistics table for amount inr"),
            ("🏦  Bank Analysis", "compare failure rates by sender bank as a table"),
            ("📱  Device Analysis", "compare failure rates by device type"),
            ("📡  Network Analysis", "compare failure rates by network type"),
            ("👤  Age Group Analysis", "compare average transaction amount by sender age group"),
            ("🛡  Flagged Review Analysis", "show fraud flag rate by transaction type table"),
            ("🗺  State Analysis", "show top 10 states by total transaction count table"),
            ("💳  P2P vs P2M Analysis", "compare p2p vs p2m transaction amounts and counts"),
            ("📊  Merchant Revenue", "show total amount by merchant category table"),
            ("🔗  Age × Device Crosstab", "show transaction count by age group and device type"),
            ("🌐  Network × Status Table", "show failure rate by network type and transaction type"),
        ]:
            self._sb_btn(lbl, query=q, pinnable=True)

        self._sec_hdr("RISK & ANOMALY")
        for lbl, q in [
            ("🚨  High Value Flagged", "what percentage of high value transactions are flagged for review"),
            ("🏦  Bank Flagged Rates", "show flagged for review rate by sender bank as a table"),
            ("📱  Device Flagged Rates", "show flagged for review rate by device type"),
            ("👤  Age Group Risk", "show flagged for review rate by sender age group"),
            ("🌍  State Risk Analysis", "show flagged rate by sender state table"),
        ]:
            self._sb_btn(lbl, query=q, pinnable=True)

        self._sec_hdr("ADVANCED INSIGHTS")
        for lbl, q in [
            ("💵  High-Value Profiling", "show profile of high value transactions by bank and device"),
            ("🔄  Repeat Patterns", "show which transaction types have highest failure then retry rates"),
            ("🌐  Multi-Network Risk", "compare flagged rates across all network types and device combinations"),
        ]:
            self._sb_btn(lbl, query=q)

        self._sec_hdr("CUSTOM ACTIONS", plus_cmd=self._add_custom_action)
        for act in self._custom_actions:
            self._sb_btn(act["label"], query=act["query"], deletable=True)

        tk.Frame(self._actions_inner, bg=BG_PANEL, height=12).pack()

    def _unpin(self, query):
        self._pinned = [p for p in self._pinned if p["query"] != query]
        _save_pinned(self._pinned)
        self._render_actions()

    def _add_custom_action(self):
        label = simpledialog.askstring("New Quick Action", "Button label:", parent=self)
        if not label:
            return
        query = simpledialog.askstring("New Quick Action", "Query to send:", parent=self)
        if not query:
            return
        self._custom_actions.append({"label": label, "query": query})
        _save_custom_actions(self._custom_actions)
        self._render_actions()
        Toast.show(self, f"Custom action added: {label[:30]}")

    def _remove_custom_action(self, query):
        self._custom_actions = [a for a in self._custom_actions if a.get("query") != query]
        _save_custom_actions(self._custom_actions)
        self._render_actions()

    def _render_tab_list(self):
        for w in self.tabs_frame.winfo_children():
            w.destroy()
        self._tab_buttons = {}
        for name in engine.list_tabs():
            self._add_tab_button(name)

    def _add_tab_button(self, name):
        row = tk.Frame(self.tabs_frame, bg=BG_PANEL)
        row.pack(fill="x", pady=1)
        is_active = name == self._active_tab
        bg = BG_SELECTED if is_active else BG_PANEL
        fg = ACCENT_BLUE if is_active else TEXT_SECONDARY
        msg_count = len(engine.get_rich(name)) // 2
        display = f"💬 {name}" + (f"  [{msg_count}]" if msg_count > 0 else "")
        btn = tk.Label(row, text=display, font=("Consolas", 8),
                       fg=fg, bg=bg, anchor="w", padx=8, pady=4, cursor="hand2")
        btn.pack(side="left", fill="x", expand=True)
        btn.bind("<Button-1>", lambda e, n=name: self._tab_btn_click(e, n))
        btn.bind("<Double-Button-1>", lambda e, n=name: self._rename_tab_inline(n))
        btn.bind("<Enter>", lambda e, b=btn, n=name:
                 b.configure(bg=BG_HOVER if n != self._active_tab else BG_SELECTED))
        btn.bind("<Leave>", lambda e, b=btn, n=name:
                 b.configure(bg=BG_SELECTED if n == self._active_tab else BG_PANEL))

        # ── Right-click context menu ──────────────────────────────────────────
        def _show_ctx(e, n=name):
            ctx = tk.Menu(self, tearoff=0, bg=BG_HOVER, fg=TEXT_PRIMARY,
                          activebackground=ACCENT_BLUE, activeforeground=BG_DEEP,
                          font=("Consolas", 8))
            ctx.add_command(label="✏  Rename",
                            command=lambda: self._rename_tab_inline(n))
            ctx.add_command(label="⬇  Download Chat",
                            command=lambda: self._download_tab_chat(n))
            ctx.add_separator()
            if n != "Default":
                ctx.add_command(label="✕  Delete",
                                command=lambda: self._delete_tab(n))
            ctx.tk_popup(e.x_root, e.y_root)

        btn.bind("<Button-3>", _show_ctx)
        row.bind("<Button-3>", _show_ctx)

        # ── Drag-to-reorder ───────────────────────────────────────────────────
        btn.bind("<ButtonPress-1>",   lambda e, n=name: self._drag_start(e, n))
        btn.bind("<B1-Motion>",       lambda e, n=name: self._drag_motion(e, n))
        btn.bind("<ButtonRelease-1>", lambda e, n=name: self._drag_end(e, n))

        notes_count = len(_load_notes().get(name, "").strip())
        if notes_count > 0:
            tk.Label(row, text="📝", font=("Consolas", 7), fg=ACCENT_YELLOW,
                     bg=bg, padx=2).pack(side="right")
        if name != "Default":
            d = tk.Label(row, text="×", font=("Consolas", 9),
                         fg=TEXT_MUTED, bg=bg, cursor="hand2", padx=5)
            d.pack(side="right")
            d.bind("<Button-1>", lambda e, n=name: self._delete_tab(n))
        self._tab_buttons[name] = btn

    def _tab_btn_click(self, event, name):
        """Single click: switch tab (drag_end handles whether it was a drag or click)."""
        if not getattr(self, "_was_dragged", False):
            self._switch_tab(name)
        self._was_dragged = False

    def _drag_start(self, event, name):
        self._drag_tab_name = name
        self._drag_origin_y = event.y_root
        self._was_dragged   = False

    def _drag_motion(self, event, name):
        if getattr(self, "_drag_tab_name", None) != name:
            return
        if abs(event.y_root - getattr(self, "_drag_origin_y", event.y_root)) > 6:
            self._was_dragged = True
            btn = self._tab_buttons.get(name)
            if btn:
                btn.configure(bg="#2d4a6e")

    def _drag_end(self, event, name):
        if getattr(self, "_drag_tab_name", None) != name:
            return
        self._drag_tab_name = None
        if not self._was_dragged:
            return  # handled by _tab_btn_click
        # Find which tab the cursor is hovering over
        target = None
        for tname, tbtn in self._tab_buttons.items():
            try:
                bx = tbtn.winfo_rootx()
                by = tbtn.winfo_rooty()
                bw = tbtn.winfo_width()
                bh = tbtn.winfo_height()
                if by <= event.y_root <= by + bh:
                    target = tname
                    break
            except Exception:
                continue
        if target and target != name:
            tabs = engine.list_tabs()
            if name in tabs and target in tabs:
                ni, ti = tabs.index(name), tabs.index(target)
                tabs.insert(ti, tabs.pop(ni))
                new_chat = {t: engine.chat_tabs[t] for t in tabs}
                new_rich = {t: engine.chat_rich[t] for t in tabs}
                engine.chat_tabs.clear(); engine.chat_tabs.update(new_chat)
                engine.chat_rich.clear(); engine.chat_rich.update(new_rich)
        self._render_tab_list()

    def _download_tab_chat(self, tab_name):
        text = engine.save_chat_export(tab_name)
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt"), ("Markdown", "*.md"), ("All files", "*.*")],
            initialfile=f"insightx_{tab_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            Toast.show(self, f"Chat saved: {os.path.basename(path)}")

    def _rename_tab_inline(self, old_name):
        new_name = simpledialog.askstring("Rename Chat",
                                          f"New name for '{old_name}':",
                                          initialvalue=old_name, parent=self)
        if not new_name or new_name == old_name:
            return
        if engine.rename_tab(old_name, new_name):
            if self._active_tab == old_name:
                self._active_tab = new_name
                self.header_title.configure(text=f"InsightX — {new_name}")
            self._render_tab_list()
        else:
            messagebox.showwarning("Rename", f"'{new_name}' already exists.")

    def _build_main(self):
        self.main_frame = tk.Frame(self, bg=BG_DEEP)
        self.main_frame.pack(side="left", fill="both", expand=True)

        header = tk.Frame(self.main_frame, bg=BG_PANEL, height=52)
        header.pack(fill="x")
        header.pack_propagate(False)

        self._toggle_sidebar_btn = tk.Label(header, text="◀", font=("Consolas", 10),
                                            fg=TEXT_MUTED, bg=BG_PANEL, padx=8, cursor="hand2")
        self._toggle_sidebar_btn.pack(side="left", padx=4)
        self._toggle_sidebar_btn.bind("<Button-1>", lambda e: self._toggle_sidebar())

        self.header_title = tk.Label(header, text="InsightX — Default",
                                     font=("Trebuchet MS", 11, "bold"),
                                     fg=TEXT_PRIMARY, bg=BG_PANEL)
        self.header_title.pack(side="left", padx=10, pady=14)

        right_controls = tk.Frame(header, bg=BG_PANEL)
        right_controls.pack(side="right", padx=4, pady=8)

        for lbl, cmd, hover in [
            ("📝 Notes", self._open_notes, ACCENT_YELLOW),
            ("💾 Save", self._save_chat, ACCENT_TEAL),
            ("🔬 Explorer", self._open_explorer, ACCENT_BLUE),
            ("📸 Snapshot", self._open_data_snapshot, ACCENT_PURPLE),
            ("⭐ Favorites", self._open_favorites, ACCENT_YELLOW),
            ("🔍 Search", self._open_search, ACCENT_PINK),
        ]:
            b = tk.Label(right_controls, text=lbl, font=("Consolas", 8),
                         fg=TEXT_SECONDARY, bg=BG_PANEL, padx=7, pady=4, cursor="hand2")
            b.pack(side="right", padx=2)
            b.bind("<Button-1>", lambda e, c=cmd: c())
            b.bind("<Enter>", lambda e, w=b, h=hover: w.configure(fg=h))
            b.bind("<Leave>", lambda e, w=b: w.configure(fg=TEXT_SECONDARY))

        zoom_frame = tk.Frame(header, bg=BG_PANEL)
        zoom_frame.pack(side="right", padx=6, pady=8)
        for ztxt, zdir in [("－", 1), ("＋", -1)]:
            zb = tk.Label(zoom_frame, text=ztxt, font=("Consolas", 11, "bold"),
                          fg=ACCENT_BLUE, bg=BG_PANEL, padx=4, cursor="hand2")
            zb.pack(side="left")
            zb.bind("<Button-1>", lambda e, d=zdir: self._zoom(d))
            zb.bind("<Enter>", lambda e, w=zb: w.configure(fg=TEXT_PRIMARY))
            zb.bind("<Leave>", lambda e, w=zb: w.configure(fg=ACCENT_BLUE))
        self._zoom_label = tk.Label(zoom_frame, text="🔍 100%", font=("Consolas", 7),
                                    fg=TEXT_MUTED, bg=BG_PANEL, padx=3, cursor="hand2")
        self._zoom_label.pack(side="left")
        self._zoom_label.bind("<Button-1>", lambda e: self._zoom_reset())

        self.thinking_label = tk.Label(header, text="", font=("Consolas", 9),
                                       fg=ACCENT_ORANGE, bg=BG_PANEL)
        self.thinking_label.pack(side="right", padx=8)

        tk.Frame(self.main_frame, bg=BORDER, height=1).pack(fill="x")

        chat_frame = tk.Frame(self.main_frame, bg=BG_DEEP)
        chat_frame.pack(fill="both", expand=True)
        self._chat_canvas = tk.Canvas(chat_frame, bg=BG_DEEP, highlightthickness=0, bd=0)
        vs = ttk.Scrollbar(chat_frame, orient="vertical", command=self._chat_canvas.yview)
        self._chat_canvas.configure(yscrollcommand=vs.set)
        vs.pack(side="right", fill="y")
        self._chat_canvas.pack(side="left", fill="both", expand=True)

        self._chat_inner = tk.Frame(self._chat_canvas, bg=BG_DEEP)
        self._cwin = self._chat_canvas.create_window((0, 0), window=self._chat_inner, anchor="nw")
        self._chat_inner.bind("<Configure>", lambda e:
                              self._chat_canvas.configure(scrollregion=self._chat_canvas.bbox("all")))
        self._chat_canvas.bind("<Configure>", lambda e:
                               self._chat_canvas.itemconfig(self._cwin, width=e.width))

        self._scroll_to_bottom_btn = tk.Label(self.main_frame, text="⬇ Scroll to bottom",
                                              font=("Consolas", 7), fg=ACCENT_BLUE,
                                              bg=BG_PANEL, cursor="hand2", pady=2)
        self._scroll_to_bottom_btn.pack(side="bottom", fill="x")
        self._scroll_to_bottom_btn.bind("<Button-1>", lambda e: self._scroll_bottom())
        self._scroll_to_bottom_btn.pack_forget()

        self._chat_canvas.bind("<Configure>", lambda e: (
            self._chat_canvas.itemconfig(self._cwin, width=e.width),
            self._check_scroll_btn()
        ))
        self._chat_canvas.bind("<ButtonRelease-1>", lambda e: self._check_scroll_btn())

        self._build_input_bar()

    def _check_scroll_btn(self):
        try:
            y0, y1 = self._chat_canvas.yview()
            if y1 < 0.95:
                self._scroll_to_bottom_btn.pack(side="bottom", fill="x")
            else:
                self._scroll_to_bottom_btn.pack_forget()
        except Exception:
            pass

    def _build_input_bar(self):
        tk.Frame(self.main_frame, bg=BORDER, height=1).pack(fill="x", side="bottom")
        input_frame = tk.Frame(self.main_frame, bg=BG_PANEL, pady=8)
        input_frame.pack(fill="x", side="bottom")

        toolbar = tk.Frame(input_frame, bg=BG_PANEL)
        toolbar.pack(fill="x", padx=18, pady=(0, 4))

        suggested = [
            "Peak fraud rates?",
            "Bank failure rates",
            "Age group analysis",
            "Network vs success",
            "Top merchant revenue",
        ]
        tk.Label(toolbar, text="Try:", font=("Consolas", 7), fg=TEXT_MUTED,
                 bg=BG_PANEL, padx=0).pack(side="left")
        for sug in suggested:
            sb = tk.Label(toolbar, text=sug, font=("Consolas", 7),
                          fg=ACCENT_BLUE, bg=BG_PANEL, padx=6, cursor="hand2",
                          highlightbackground=BORDER, highlightthickness=1)
            sb.pack(side="left", padx=2)
            sb.bind("<Button-1>", lambda e, q=sug: self._quick(q))
            sb.bind("<Enter>", lambda e, w=sb: w.configure(fg=TEXT_PRIMARY, bg=BG_HOVER))
            sb.bind("<Leave>", lambda e, w=sb: w.configure(fg=ACCENT_BLUE, bg=BG_PANEL))

        self._char_counter = tk.Label(toolbar, text="", font=("Consolas", 7),
                                      fg=TEXT_MUTED, bg=BG_PANEL)
        self._char_counter.pack(side="right")

        inner = tk.Frame(input_frame, bg=BG_INPUT,
                         highlightbackground=BORDER_LT, highlightthickness=1)
        inner.pack(fill="x", padx=18)

        self.input_var = tk.StringVar()
        self.input_box = tk.Entry(inner, textvariable=self.input_var,
                                  font=FONT_INPUT, fg=TEXT_PRIMARY, bg=BG_INPUT,
                                  insertbackground=ACCENT_BLUE, relief="flat", bd=0)
        self.input_box.pack(side="left", fill="x", expand=True, padx=12, pady=9)
        self.input_box.bind("<Return>", self._on_send)
        self.input_box.bind("<Control-Return>", self._on_send)
        self.input_box.bind("<FocusIn>", lambda e: inner.configure(highlightbackground=ACCENT_BLUE))
        self.input_box.bind("<FocusOut>", lambda e: inner.configure(highlightbackground=BORDER_LT))
        self.input_var.trace_add("write", self._on_input_change)

        clear_btn = tk.Label(inner, text="✕", font=("Consolas", 10),
                             fg=TEXT_MUTED, bg=BG_INPUT, padx=6, cursor="hand2")
        clear_btn.pack(side="right")
        clear_btn.bind("<Button-1>", lambda e: self._clear_input())
        clear_btn.bind("<Enter>", lambda e: clear_btn.configure(fg=ACCENT_RED))
        clear_btn.bind("<Leave>", lambda e: clear_btn.configure(fg=TEXT_MUTED))

        send = tk.Label(inner, text="⏎ Send", font=("Consolas", 9, "bold"),
                        fg=ACCENT_BLUE, bg=BG_INPUT, padx=12, cursor="hand2")
        send.pack(side="right")
        send.bind("<Button-1>", self._on_send)
        send.bind("<Enter>", lambda e: send.configure(fg=TEXT_PRIMARY))
        send.bind("<Leave>", lambda e: send.configure(fg=ACCENT_BLUE))

        hint = tk.Label(input_frame,
                        text="↑/↓ Browse history  •  Tab Autocomplete  •  Ctrl+F Search  •  Ctrl+/ Shortcuts",
                        font=("Consolas", 7), fg=TEXT_MUTED, bg=BG_PANEL)
        hint.pack(pady=(2, 0))

    def _on_input_change(self, *args):
        txt = self.input_var.get()
        length = len(txt)
        if length > 0:
            self._char_counter.configure(text=f"{length}ch")
        else:
            self._char_counter.configure(text="")

    def _show_gguf_popup(self):
        """Open the HuggingFace download page and show an instructional popup.
        Called from the main thread via _poll_queue so tkinter is safe to use."""
        import webbrowser
        gguf_dir  = os.path.dirname(os.path.abspath(engine.__file__))
        dl_url    = ("https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf"
                     "/resolve/main/Phi-3-mini-4k-instruct-q4.gguf")
        webbrowser.open(dl_url)
        messagebox.showwarning(
            "Model Required — InsightX",
            "InsightX cannot run without the AI model file.\n\n"
            "A download has been started in your browser.\n\n"
            "File to download:\n"
            "  Phi-3-mini-4k-instruct-q4.gguf  (~2.4 GB)\n\n"
            f"Save it to:\n  {gguf_dir}\n\n"
            "Then restart the app:  python main.py",
        )
        self._add_system_msg(
            "⚠ AI model not found. Download started in browser — see popup for instructions. "
            "Charts and tables still work without the model."
        )

    def _start_engine_thread(self):
        self._set_status("⟳ Loading...", ACCENT_ORANGE)
        self._set_thinking("Loading model & data…")
        threading.Thread(target=self._init_engine, daemon=True).start()

    def _init_engine(self):
        try:
            if os.path.exists("data.csv"):
                engine.load_data()
        except Exception as ex:
            print(f"[InsightX] data.csv load warning: {ex}")
        try:
            result = engine.load_model()
            if result == engine.GGUF_MISSING_SENTINEL:
                self._q.put(("gguf_missing", None))
            else:
                self._q.put(("ready", None))
        except Exception as ex:
            self._q.put(("error", str(ex)))

    def _poll_queue(self):
        try:
            while True:
                msg, payload = self._q.get_nowait()
                if msg == "ready":
                    self._set_status("● Ready", ACCENT_GREEN)
                    self._set_thinking("")
                    df = engine.get_df()
                    row_count = f"{len(df):,}" if df is not None else "?"
                    self._add_system_msg(
                        f"InsightX ready — {row_count} transactions loaded.  "
                        f"Ask anything in natural language.  Ctrl+/ for shortcuts."
                    )
                    self._render_welcome_cards()
                elif msg == "gguf_missing":
                    self._set_status("● Model Missing", ACCENT_RED)
                    self._set_thinking("")
                    self._show_gguf_popup()
                elif msg == "error":
                    self._set_status("● Error", ACCENT_RED)
                    self._set_thinking("")
                    self._add_system_msg(f"Error loading: {payload}")
                elif msg == "result":
                    self._render_result(payload)
                    self._set_thinking("")
                    self._set_status("● Ready", ACCENT_GREEN)
                    self._loading = False
                    self.input_box.configure(state="normal")
                    self.input_box.focus()
        except queue.Empty:
            pass
        self.after(50, self._poll_queue)

    def _render_welcome_cards(self):
        card_frame = tk.Frame(self._chat_inner, bg=BG_DEEP)
        card_frame.pack(fill="x", padx=18, pady=8)
        tk.Label(card_frame, text="Quick Start — click any card to run:",
                 font=("Consolas", 8, "bold"), fg=TEXT_MUTED, bg=BG_DEEP).pack(anchor="w", pady=(0, 6))
        row1 = tk.Frame(card_frame, bg=BG_DEEP)
        row1.pack(fill="x")
        cards = [
            ("📊", "Full Statistics", "show full statistics table"),
            ("🍕", "Fraud Flag Pie", "show a pie chart of fraud flag"),
            ("🏦", "Bank Failure Rates", "compare failure rates by sender bank as a table"),
            ("📋", "Full Report", "generate full report"),
            ("🚨", "High-Value Risk", "what percentage of high value transactions are flagged for review"),
        ]
        for icon, label, query in cards:
            card = tk.Frame(row1, bg=BG_CARD,
                            highlightbackground=BORDER, highlightthickness=1,
                            cursor="hand2")
            card.pack(side="left", padx=4, pady=2, fill="y")
            tk.Label(card, text=icon, font=("Consolas", 14), bg=BG_CARD, pady=6).pack()
            tk.Label(card, text=label, font=("Consolas", 8, "bold"),
                     fg=TEXT_PRIMARY, bg=BG_CARD, padx=10, pady=6).pack()
            card.bind("<Button-1>", lambda e, q=query: self._quick(q))
            card.bind("<Enter>", lambda e, w=card: w.configure(bg=BG_HOVER))
            card.bind("<Leave>", lambda e, w=card: w.configure(bg=BG_CARD))
            for child in card.winfo_children():
                child.bind("<Button-1>", lambda e, q=query: self._quick(q))

    def _on_send(self, event=None):
        if self._loading:
            return
        text = self.input_var.get().strip()
        if not text:
            return
        self.input_var.set("")
        self._input_hist_idx = -1
        if not self._input_history or self._input_history[-1] != text:
            self._input_history.append(text)
            if len(self._input_history) > PROMPT_HISTORY_MAX:
                self._input_history.pop(0)
        self._query_count += 1
        self._add_user_msg(text)
        self._loading = True
        self.input_box.configure(state="disabled")
        self._set_thinking("Thinking…")
        self._set_status("⟳ Processing", ACCENT_ORANGE)
        threading.Thread(target=self._run_query, args=(text,), daemon=True).start()

    def _run_query(self, query):
        try:
            result = engine.process_query(query, self._active_tab)
            self._q.put(("result", result))
        except Exception as _exc:
            import traceback; traceback.print_exc()
            self._q.put(("result", {
                "text": (
                    f"⚠ Error: {_exc}\n\n"
                    "Sorry, I wasn't able to process that request. "
                    "Try rephrasing — for example: 'Show failure rate by bank' or "
                    "'What is the average transaction amount by device type?'"
                ),
                "table": None, "chart": None, "report": None,
                "summary": None, "data_raw": None, "recommendations": None
            }))

    def _quick(self, query):
        if self._loading:
            return
        self.input_var.set(query)
        self._on_send()

    def _save_chat(self):
        text = engine.save_chat_export(self._active_tab)
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt"), ("Markdown", "*.md"), ("All files", "*.*")],
            initialfile=f"insightx_{self._active_tab}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            Toast.show(self, f"Chat saved: {os.path.basename(path)}")

    def _open_explorer(self):
        dlg = AdvancedExplorerDialog(self)
        self.wait_window(dlg)
        if getattr(dlg, "result_fig", None) is not None:
            cfg = getattr(dlg, "result_config", {})
            title = (f"Explorer: {cfg.get('x_col', '?')} × {cfg.get('y_metric', '?')}"
                     + (f" by {cfg.get('color_col')}" if cfg.get("color_col") else ""))
            self._inject_chart(dlg.result_fig, label=title)

    def _inject_chart(self, fig, label="[Advanced Explorer]"):
        result = {"text": f"Chart: {label}",
                  "table": None, "chart": fig, "report": None,
                  "summary": None, "data_raw": None, "recommendations": None}
        self._add_user_msg(label)
        self._render_result(result)
        engine.push_rich(self._active_tab, {"role": "user", "content": label, "type": "user"})
        engine.push_rich(self._active_tab,
                         {"role": "assistant", "content": f"Chart: {label}", "type": "text",
                          "chart": fig, "table": None, "report": None, "summary": None,
                          "recommendations": None})

    def _open_data_snapshot(self):
        DataSnapshotDialog(self)

    def _open_notes(self):
        NotesDialog(self, self._active_tab)

    def _open_favorites(self):
        FavoritesDialog(self, self._quick)

    def _open_search(self):
        rich = engine.get_rich(self._active_tab)
        if not rich:
            Toast.show(self, "No chat history to search", color=ACCENT_ORANGE)
            return
        SearchDialog(self, rich, self._jump_to_message_index)

    def _jump_to_message_index(self, pos):
        self._scroll_bottom()

    def _toggle_pin_mode(self):
        Toast.show(self, "📌 Click any sidebar button's pin icon to pin a query")

    def _add_user_msg(self, text):
        outer = tk.Frame(self._chat_inner, bg=BG_DEEP)
        outer.pack(fill="x", padx=18, pady=(8, 1))
        self._prop_scroll(outer)
        right = tk.Frame(outer, bg=BG_DEEP)
        right.pack(side="right")
        name_row = tk.Frame(right, bg=BG_DEEP)
        name_row.pack(anchor="e")
        tk.Label(name_row, text="You", font=("Consolas", 8, "bold"),
                 fg=ACCENT_BLUE, bg=BG_DEEP).pack(side="left")
        tk.Label(name_row, text=f"  {datetime.now().strftime('%H:%M')}",
                 font=("Consolas", 7), fg=TEXT_MUTED, bg=BG_DEEP).pack(side="left")
        bubble = tk.Frame(right, bg=BG_USER_BUBBLE,
                          highlightbackground=ACCENT_BLUE, highlightthickness=1)
        bubble.pack(anchor="e")
        tw = _make_text_bubble(bubble, text, bg=BG_USER_BUBBLE, fg=TEXT_PRIMARY)
        tw.pack(padx=1, pady=1)
        self._prop_scroll(tw)
        self._scroll_bottom()

    def _add_system_msg(self, text):
        outer = tk.Frame(self._chat_inner, bg=BG_DEEP)
        outer.pack(fill="x", padx=18, pady=4)
        self._prop_scroll(outer)
        tk.Label(outer, text=text, font=("Consolas", 8),
                 fg=TEXT_MUTED, bg=BG_DEEP, justify="center").pack()
        self._scroll_bottom()

    def _render_result(self, result):
        outer = tk.Frame(self._chat_inner, bg=BG_DEEP)
        outer.pack(fill="x", padx=18, pady=(1, 8))
        self._prop_scroll(outer)
        left = tk.Frame(outer, bg=BG_DEEP)
        left.pack(side="left", anchor="n", fill="x", expand=True)
        self._prop_scroll(left)

        name_row = tk.Frame(left, bg=BG_DEEP)
        name_row.pack(anchor="w")
        tk.Label(name_row, text="IX", font=("Georgia", 10, "bold"),
                 fg=ACCENT_BLUE, bg=BG_DEEP).pack(side="left")
        tk.Label(name_row, text=" InsightX", font=("Consolas", 8, "bold"),
                 fg=TEXT_SECONDARY, bg=BG_DEEP).pack(side="left")
        ts = tk.Label(name_row, text=f"  {datetime.now().strftime('%H:%M')}",
                      font=("Consolas", 7), fg=TEXT_MUTED, bg=BG_DEEP)
        ts.pack(side="left")

        action_row = tk.Frame(name_row, bg=BG_DEEP)
        action_row.pack(side="right")
        main_text = result.get("text", "")

        star_btn = tk.Label(action_row, text="☆", font=("Consolas", 9),
                            fg=TEXT_MUTED, bg=BG_DEEP, padx=4, cursor="hand2")
        star_btn.pack(side="left")
        star_btn.bind("<Button-1>", lambda e, t=main_text: self._star_response(t))
        star_btn.bind("<Enter>", lambda e: star_btn.configure(fg=ACCENT_YELLOW))
        star_btn.bind("<Leave>", lambda e: star_btn.configure(fg=TEXT_MUTED))

        if result.get("summary"):
            self._render_text_bubble(left, result["summary"], color=ACCENT_PURPLE, animate=True)
        if result.get("report"):
            self._render_report_block(left, result["report"])
        if main_text and not result.get("report") and not result.get("summary"):
            self._render_text_bubble(left, main_text, animate=True)
        if result.get("table"):
            self._render_table(left, result["table"])
        if result.get("chart"):
            self._render_chart(left, result["chart"])
        if result.get("recommendations"):
            self._render_recommendations(left, result["recommendations"])

    def _star_response(self, text):
        last_user = None
        rich = engine.get_rich(self._active_tab)
        for entry in reversed(rich):
            if entry.get("role") == "user":
                last_user = entry.get("content", "")
                break
        self._favorites.append({
            "query": last_user or "Unknown query",
            "response": text,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "tab": self._active_tab,
        })
        _save_favorites(self._favorites)
        Toast.show(self, "⭐ Saved to favorites!")

    def _render_result_static(self, result):
        outer = tk.Frame(self._chat_inner, bg=BG_DEEP)
        outer.pack(fill="x", padx=18, pady=(1, 8))
        self._prop_scroll(outer)
        left = tk.Frame(outer, bg=BG_DEEP)
        left.pack(side="left", anchor="n", fill="x", expand=True)

        name_row = tk.Frame(left, bg=BG_DEEP)
        name_row.pack(anchor="w")
        tk.Label(name_row, text="IX", font=("Georgia", 10, "bold"),
                 fg=ACCENT_BLUE, bg=BG_DEEP).pack(side="left")
        tk.Label(name_row, text=" InsightX", font=("Consolas", 8, "bold"),
                 fg=TEXT_SECONDARY, bg=BG_DEEP).pack(side="left")

        main_text = result.get("text", "")
        if result.get("summary"):
            self._render_text_bubble(left, result["summary"], color=ACCENT_PURPLE, animate=False)
        if result.get("report"):
            self._render_report_block(left, result["report"])
        if main_text and not result.get("report") and not result.get("summary"):
            self._render_text_bubble(left, main_text, animate=False)
        if result.get("table"):
            self._render_table(left, result["table"])
        if result.get("chart"):
            self._render_chart(left, result["chart"])
        if result.get("recommendations"):
            self._render_recommendations(left, result["recommendations"])

    def _render_text_bubble(self, parent, text, color=TEXT_PRIMARY, animate=True):
        bubble = tk.Frame(parent, bg=BG_CARD,
                          highlightbackground=BORDER, highlightthickness=1)
        bubble.pack(anchor="w", pady=(3, 1), fill="x")
        self._prop_scroll(bubble)
        toolbar = tk.Frame(bubble, bg=BG_CARD)
        toolbar.pack(fill="x", padx=5, pady=(3, 0))
        c = tk.Label(toolbar, text="⎘ Copy All", font=("Consolas", 7),
                     fg=TEXT_MUTED, bg=BG_CARD, cursor="hand2")
        c.pack(side="right")
        c.bind("<Button-1>", lambda e, t=text: self._clip(t))
        c.bind("<Enter>", lambda e: c.configure(fg=ACCENT_TEAL))
        c.bind("<Leave>", lambda e: c.configure(fg=TEXT_MUTED))
        tw = _make_text_bubble(bubble, text, bg=BG_CARD, fg=color)
        tw.pack(anchor="w", fill="x", padx=1, pady=(0, 4))
        self._prop_scroll(tw)
        if animate:
            self._typewrite(tw, text)

    def _render_recommendations(self, parent, recs):
        if not recs:
            return
        frame = tk.Frame(parent, bg="#0f1e12",
                         highlightbackground=ACCENT_GREEN, highlightthickness=1)
        frame.pack(anchor="w", pady=(3, 1), fill="x", padx=1)
        self._prop_scroll(frame)
        title_bar = tk.Frame(frame, bg="#0f1e12")
        title_bar.pack(fill="x")
        tk.Label(title_bar, text="💡  Recommendations & Insights",
                 font=("Consolas", 8, "bold"),
                 fg=ACCENT_GREEN, bg="#0f1e12", padx=10, pady=4).pack(side="left")
        tk.Label(title_bar, text=f"{len(recs)} insights",
                 font=("Consolas", 7), fg=TEXT_MUTED, bg="#0f1e12", padx=6).pack(side="left")
        copy_text = "\n".join(f"• {r}" for r in recs)
        cb = tk.Label(title_bar, text="⎘ Copy", font=("Consolas", 7),
                      fg=TEXT_MUTED, bg="#0f1e12", padx=7, cursor="hand2")
        cb.pack(side="right")
        cb.bind("<Button-1>", lambda e, t=copy_text: self._clip(t))
        cb.bind("<Enter>", lambda e: cb.configure(fg=ACCENT_TEAL))
        cb.bind("<Leave>", lambda e: cb.configure(fg=TEXT_MUTED))
        for i, rec in enumerate(recs):
            row = tk.Frame(frame, bg="#0f1e12")
            row.pack(fill="x", padx=10,
                     pady=(2 if i == 0 else 0, 2 if i == len(recs) - 1 else 0))
            self._prop_scroll(row)
            tk.Label(row, text="▸", font=("Consolas", 9, "bold"),
                     fg=ACCENT_GREEN, bg="#0f1e12").pack(side="left", padx=(0, 6))
            lbl = tk.Label(row, text=rec, font=("Consolas", 9), fg="#a8d5b5",
                           bg="#0f1e12", anchor="w", wraplength=820, justify="left")
            lbl.pack(side="left", fill="x", expand=True)
            self._prop_scroll(lbl)

    def _render_table(self, parent, table_data):
        """Render table using ttk.Treeview — full content visible, no truncation."""
        headers = table_data.get("headers", [])
        rows    = table_data.get("rows", [])
        if not headers:
            return

        outer = tk.Frame(parent, bg=BG_CARD,
                         highlightbackground=BORDER_LT, highlightthickness=1)
        outer.pack(anchor="w", pady=(3, 1), fill="x", padx=1)

        # ── title bar ────────────────────────────────────────────────────────
        title_bar = tk.Frame(outer, bg=BG_HOVER)
        title_bar.pack(fill="x")
        tk.Label(title_bar, text="⊞  Data Table", font=("Consolas", 8, "bold"),
                 fg=ACCENT_BLUE, bg=BG_HOVER, padx=10, pady=4).pack(side="left")
        tk.Label(title_bar, text=f"{len(rows)} rows × {len(headers)} cols",
                 font=("Consolas", 7), fg=TEXT_MUTED, bg=BG_HOVER, padx=6).pack(side="left")

        copy_text = "\t".join(headers) + "\n" + \
                    "\n".join("\t".join(str(c) for c in r) for r in rows)
        for lbl_text, handler in [
            ("⤢ Expand",  lambda e, td=table_data: self._expand_table(td)),
            ("⬇ CSV",     lambda e, td=table_data: self._export_table_csv(td)),
            ("⎘ TSV",     lambda e, t=copy_text: self._clip(t)),
        ]:
            b = tk.Label(title_bar, text=lbl_text, font=("Consolas", 7),
                         fg=TEXT_MUTED, bg=BG_HOVER, padx=7, cursor="hand2")
            b.pack(side="right")
            b.bind("<Button-1>", handler)
            b.bind("<Enter>", lambda e, w=b: w.configure(fg=ACCENT_TEAL))
            b.bind("<Leave>", lambda e, w=b: w.configure(fg=TEXT_MUTED))

        # ── Treeview style ────────────────────────────────────────────────────
        style = ttk.Style()
        style.theme_use("default")
        tv_style = "InsightX.Treeview"
        style.configure(tv_style,
                        background=BG_CARD, foreground=TEXT_PRIMARY,
                        fieldbackground=BG_CARD, rowheight=26,
                        font=("Consolas", 9), borderwidth=0)
        style.configure(f"{tv_style}.Heading",
                        background=BG_HOVER, foreground=ACCENT_BLUE,
                        font=("Consolas", 9, "bold"), relief="flat",
                        borderwidth=1)
        style.map(tv_style,
                  background=[("selected", BG_SELECTED)],
                  foreground=[("selected", TEXT_PRIMARY)])
        style.map(f"{tv_style}.Heading",
                  background=[("active", BORDER_LT)])

        # ── compute column pixel widths ───────────────────────────────────────
        col_widths = self._calc_col_widths(table_data)
        total_w = sum(col_widths)
        MAX_TBL_H = 18  # max visible rows before scroll kicks in

        # ── scrollbars + treeview frame ───────────────────────────────────────
        tv_frame = tk.Frame(outer, bg=BG_CARD)
        tv_frame.pack(fill="x", expand=False, padx=2, pady=(0, 2))

        v_scroll = ttk.Scrollbar(tv_frame, orient="vertical")
        h_scroll = ttk.Scrollbar(tv_frame, orient="horizontal")

        tv = ttk.Treeview(tv_frame, columns=headers, show="headings",
                          style=tv_style,
                          yscrollcommand=v_scroll.set,
                          xscrollcommand=h_scroll.set,
                          height=min(len(rows), MAX_TBL_H),
                          selectmode="browse")

        v_scroll.config(command=tv.yview)
        h_scroll.config(command=tv.xview)

        v_scroll.pack(side="right", fill="y")
        h_scroll.pack(side="bottom", fill="x")
        tv.pack(side="left", fill="both", expand=True)

        # ── configure columns ─────────────────────────────────────────────────
        for h, w in zip(headers, col_widths):
            tv.heading(h, text=h,
                       command=lambda _h=h: self._tv_sort(tv, table_data, _h, {}))
            tv.column(h, width=w, minwidth=50, anchor="w", stretch=False)

        # ── tag colours ───────────────────────────────────────────────────────
        tv.tag_configure("even",  background=BG_CARD)
        tv.tag_configure("odd",   background="#13181f")
        tv.tag_configure("fail",  foreground=ACCENT_RED)
        tv.tag_configure("succ",  foreground=ACCENT_GREEN)
        tv.tag_configure("warn",  foreground=ACCENT_ORANGE)

        # ── insert rows ───────────────────────────────────────────────────────
        sort_state = {"col": None, "rev": False}
        self._tv_populate(tv, rows, headers)

        # ── scroll passthrough to chat canvas ─────────────────────────────────
        def _tbl_scroll(e, d=None):
            amt = d if d is not None else int(-1 * e.delta / 120)
            y0, y1 = tv.yview()
            if (y0 <= 0.0 and amt < 0) or (y1 >= 1.0 and amt > 0):
                self._chat_canvas.yview_scroll(amt, "units")
            else:
                tv.yview_scroll(amt, "units")
            return "break"

        tv.bind("<MouseWheel>", _tbl_scroll)
        tv.bind("<Button-4>", lambda e: _tbl_scroll(e, -1))
        tv.bind("<Button-5>", lambda e: _tbl_scroll(e,  1))

        # ── copy cell on double-click ─────────────────────────────────────────
        def _on_dbl(e):
            item = tv.identify_row(e.y)
            col  = tv.identify_column(e.x)
            if item and col:
                try:
                    ci   = int(col.lstrip("#")) - 1
                    vals = tv.item(item, "values")
                    if 0 <= ci < len(vals):
                        self._clip(str(vals[ci]))
                        Toast.show(self, "⎘ Cell copied")
                except Exception:
                    pass
        tv.bind("<Double-Button-1>", _on_dbl)

    def _tv_populate(self, tv, rows, headers):
        """Clear and re-populate a Treeview, applying colour tags."""
        for iid in tv.get_children():
            tv.delete(iid)
        for ri, row in enumerate(rows):
            vals = [str(c) for c in row]
            tags = ["even" if ri % 2 == 0 else "odd"]
            # semantic colouring based on cell content
            for val in vals:
                vl = val.lower()
                if "failed" in vl:
                    tags = ["fail"]; break
                if "success" in vl:
                    tags = ["succ"]; break
                if "%" in val:
                    try:
                        pct = float(val.replace("%",""))
                        if pct >= 50:
                            tags = ["succ"]; break
                        elif pct >= 20:
                            tags = ["warn"]; break
                    except Exception:
                        pass
            tv.insert("", "end", values=vals, tags=tags)

    def _tv_sort(self, tv, table_data, col, state):
        """Sort Treeview by column; toggle asc/desc."""
        headers = table_data["headers"]
        rows    = list(table_data["rows"])
        ci = headers.index(col) if col in headers else 0
        rev = state.get(col, False)

        def _key(r):
            v = str(r[ci]).replace(",","").replace("%","").replace("Rs.","").replace("₹","").strip()
            try:
                return float(v)
            except ValueError:
                return v

        rows.sort(key=_key, reverse=rev)
        state[col] = not rev
        self._tv_populate(tv, rows, headers)
        # update heading indicator
        for h in headers:
            arrow = ""
            if h == col:
                arrow = " ▼" if rev else " ▲"
            tv.heading(h, text=h + arrow)

    def _export_table_csv(self, table_data):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV file", "*.csv"), ("All files", "*.*")],
            initialfile=f"insightx_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        if path:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(table_data["headers"])
                w.writerows(table_data["rows"])
            Toast.show(self, f"Table exported: {os.path.basename(path)}")

    def _expand_table(self, table_data):
        headers = table_data.get("headers", [])
        rows    = table_data.get("rows", [])

        dlg = tk.Toplevel(self)
        dlg.title("InsightX — Full Table View")
        dlg.geometry("1200x700")
        dlg.configure(bg=BG_PANEL)
        dlg.resizable(True, True)

        # ── header bar ────────────────────────────────────────────────────────
        hdr = tk.Frame(dlg, bg=BG_PANEL)
        hdr.pack(fill="x", padx=14, pady=(12, 4))
        tk.Label(hdr, text="⊞  Full Table",
                 font=("Trebuchet MS", 12, "bold"),
                 fg=ACCENT_BLUE, bg=BG_PANEL).pack(side="left")
        tk.Label(hdr, text=f"{len(rows)} rows × {len(headers)} cols",
                 font=("Consolas", 8), fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left", padx=10)

        for lbl, handler in [
            ("⬇ Export CSV", lambda: self._export_table_csv(table_data)),
            ("⎘ Copy TSV",   lambda: self._clip(
                "\t".join(headers) + "\n" +
                "\n".join("\t".join(str(c) for c in r) for r in rows))),
        ]:
            b = tk.Label(hdr, text=lbl, font=("Consolas", 8),
                         fg=TEXT_MUTED, bg=BG_PANEL, padx=10, cursor="hand2")
            b.pack(side="right")
            b.bind("<Button-1>", lambda e, h=handler: h())
            b.bind("<Enter>", lambda e, w=b: w.configure(fg=ACCENT_TEAL))
            b.bind("<Leave>", lambda e, w=b: w.configure(fg=TEXT_MUTED))

        tk.Frame(dlg, bg=BORDER, height=1).pack(fill="x", padx=12, pady=4)

        # ── search bar ────────────────────────────────────────────────────────
        search_var = tk.StringVar()
        sf = tk.Frame(dlg, bg=BG_PANEL)
        sf.pack(fill="x", padx=14, pady=(0, 6))
        tk.Label(sf, text="🔍 Filter:", font=("Consolas", 8),
                 fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left")
        tk.Entry(sf, textvariable=search_var, font=FONT_SMALL,
                 fg=TEXT_PRIMARY, bg=BG_INPUT,
                 insertbackground=ACCENT_BLUE,
                 relief="flat", width=40).pack(side="left", padx=6)
        row_count_lbl = tk.Label(sf, text=f"{len(rows)} rows shown",
                                  font=("Consolas", 7), fg=TEXT_MUTED, bg=BG_PANEL)
        row_count_lbl.pack(side="left", padx=8)

        # ── Treeview style ────────────────────────────────────────────────────
        style = ttk.Style()
        style.theme_use("default")
        pop_style = "InsightXPop.Treeview"
        style.configure(pop_style,
                        background=BG_CARD, foreground=TEXT_PRIMARY,
                        fieldbackground=BG_CARD, rowheight=26,
                        font=("Consolas", 9), borderwidth=0)
        style.configure(f"{pop_style}.Heading",
                        background=BG_HOVER, foreground=ACCENT_BLUE,
                        font=("Consolas", 9, "bold"), relief="flat")
        style.map(pop_style,
                  background=[("selected", BG_SELECTED)],
                  foreground=[("selected", TEXT_PRIMARY)])

        # ── Treeview frame ────────────────────────────────────────────────────
        outer = tk.Frame(dlg, bg=BG_PANEL)
        outer.pack(fill="both", expand=True, padx=12, pady=4)

        v_scroll = ttk.Scrollbar(outer, orient="vertical")
        h_scroll = ttk.Scrollbar(outer, orient="horizontal")

        col_widths = self._calc_col_widths(table_data)
        tv = ttk.Treeview(outer, columns=headers, show="headings",
                          style=pop_style,
                          yscrollcommand=v_scroll.set,
                          xscrollcommand=h_scroll.set,
                          selectmode="browse")

        v_scroll.config(command=tv.yview)
        h_scroll.config(command=tv.xview)
        v_scroll.pack(side="right", fill="y")
        h_scroll.pack(side="bottom", fill="x")
        tv.pack(side="left", fill="both", expand=True)

        sort_state = {}

        for h, w in zip(headers, col_widths):
            tv.heading(h, text=h,
                       command=lambda _h=h: _sort_col(_h))
            tv.column(h, width=w, minwidth=50, anchor="w", stretch=False)

        tv.tag_configure("even", background=BG_CARD)
        tv.tag_configure("odd",  background="#13181f")
        tv.tag_configure("fail", foreground=ACCENT_RED)
        tv.tag_configure("succ", foreground=ACCENT_GREEN)
        tv.tag_configure("warn", foreground=ACCENT_ORANGE)

        current_rows = list(rows)

        def _populate(data):
            for iid in tv.get_children():
                tv.delete(iid)
            for ri, row in enumerate(data):
                vals = [str(c) for c in row]
                tags = ["even" if ri % 2 == 0 else "odd"]
                for val in vals:
                    vl = val.lower()
                    if "failed" in vl:   tags = ["fail"]; break
                    if "success" in vl:  tags = ["succ"]; break
                    if "%" in val:
                        try:
                            pct = float(val.replace("%",""))
                            if pct >= 50:   tags = ["succ"]; break
                            elif pct >= 20: tags = ["warn"]; break
                        except Exception: pass
                tv.insert("", "end", values=vals, tags=tags)
            row_count_lbl.configure(text=f"{len(data)} rows shown")

        def _filter(*_):
            ft = search_var.get().lower()
            if ft:
                filtered = [r for r in rows
                            if any(ft in str(c).lower() for c in r)]
            else:
                filtered = list(rows)
            _populate(filtered)

        def _sort_col(col):
            ci  = headers.index(col) if col in headers else 0
            rev = sort_state.get(col, False)

            def _key(r):
                v = str(r[ci]).replace(",","").replace("%","").replace("Rs.","").replace("₹","").strip()
                try: return float(v)
                except ValueError: return v

            data = [tv.item(iid, "values") for iid in tv.get_children()]
            data.sort(key=lambda r: _key(r), reverse=rev)
            sort_state[col] = not rev
            for iid in tv.get_children():
                tv.delete(iid)
            for ri, vals in enumerate(data):
                tags = ["even" if ri % 2 == 0 else "odd"]
                tv.insert("", "end", values=vals, tags=tags)
            for h in headers:
                arrow = (" ▼" if rev else " ▲") if h == col else ""
                tv.heading(h, text=h + arrow)

        search_var.trace_add("write", _filter)
        _populate(rows)

        # ── Isolated scroll — only scrolls this dialog's treeview ─────────────
        def _dlg_wheel(e, d=None):
            amt = d if d is not None else int(-1 * e.delta / 120)
            tv.yview_scroll(amt, "units")
            return "break"
        for w in (tv, dlg, outer):
            w.bind("<MouseWheel>", _dlg_wheel)
            w.bind("<Button-4>",   lambda e: _dlg_wheel(e, -1))
            w.bind("<Button-5>",   lambda e: _dlg_wheel(e,  1))

        # double-click copies cell
        def _on_dbl(e):
            item = tv.identify_row(e.y)
            col  = tv.identify_column(e.x)
            if item and col:
                try:
                    ci   = int(col.lstrip("#")) - 1
                    vals = tv.item(item, "values")
                    if 0 <= ci < len(vals):
                        self._clip(str(vals[ci]))
                        Toast.show(self, "⎘ Cell copied")
                except Exception:
                    pass
        tv.bind("<Double-Button-1>", _on_dbl)

    def _calc_col_widths(self, table_data):
        """Return pixel widths for each column (uncapped, based on content length)."""
        CHAR_PX = 7   # approximate pixels per character in Consolas 8
        MIN_PX  = 60
        MAX_PX  = 400
        widths = [max(len(h) * CHAR_PX + 16, MIN_PX) for h in table_data["headers"]]
        for row in table_data["rows"]:
            for i, cell in enumerate(row):
                if i < len(widths):
                    cell_px = min(len(str(cell)) * CHAR_PX + 16, MAX_PX)
                    widths[i] = max(widths[i], cell_px)
        return widths

    def _render_chart(self, parent, fig):
        frame = tk.Frame(parent, bg=BG_CARD,
                         highlightbackground=BORDER_LT, highlightthickness=1)
        frame.pack(anchor="w", pady=(3, 1), fill="x", padx=1)
        title_bar = tk.Frame(frame, bg=BG_HOVER)
        title_bar.pack(fill="x")
        tk.Label(title_bar, text="📈  Chart", font=("Consolas", 8, "bold"),
                 fg=ACCENT_BLUE, bg=BG_HOVER, padx=10, pady=3).pack(side="left")
        for lbl, cmd in [
            ("⬇ PNG", lambda f=fig: self._save_chart(f, "png")),
            ("⬇ SVG", lambda f=fig: self._save_chart(f, "svg")),
            ("⬇ PDF", lambda f=fig: self._save_chart(f, "pdf")),
        ]:
            sb = tk.Label(title_bar, text=lbl, font=("Consolas", 7),
                          fg=TEXT_MUTED, bg=BG_HOVER, padx=7, cursor="hand2")
            sb.pack(side="right")
            sb.bind("<Button-1>", lambda e, c=cmd: c())
            sb.bind("<Enter>", lambda e, w=sb: w.configure(fg=ACCENT_TEAL))
            sb.bind("<Leave>", lambda e, w=sb: w.configure(fg=TEXT_MUTED))
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.configure(bg=BG_CARD)
        widget.pack(padx=7, pady=7)
        self._prop_scroll(widget)
        self._attach_hover(fig, canvas)

    def _attach_hover(self, fig, canvas):
        hover_type = getattr(fig, "_hover_type", None)
        hover_data = getattr(fig, "_hover_data", [])
        if not hover_type or not hover_data:
            return
        ax = fig.axes[0]
        annot = ax.annotate("", xy=(0, 0), xytext=(14, 14),
                            textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.45", fc="#1c2128", ec="#3d4450", lw=1),
                            fontsize=8.5, color=TEXT_PRIMARY,
                            arrowprops=dict(arrowstyle="->", color=ACCENT_BLUE))
        annot.set_visible(False)

        def on_hover(event):
            if event.inaxes != ax:
                annot.set_visible(False)
                canvas.draw_idle()
                return
            visible = False
            if hover_type == "bar":
                for i, bar in enumerate(getattr(fig, "_bars", [])):
                    if bar.contains(event)[0]:
                        lbl, val = hover_data[i]
                        annot.xy = (bar.get_x() + bar.get_width() / 2, bar.get_height())
                        annot.set_text(f"{lbl}\n{val:,.2f}")
                        annot.set_visible(True)
                        visible = True
                        break
            elif hover_type == "pie":
                for i, wedge in enumerate(getattr(fig, "_wedges", [])):
                    if wedge.contains(event)[0]:
                        lbl, val = hover_data[i]
                        total = sum(v for _, v in hover_data)
                        pct = val / total * 100 if total else 0
                        annot.xy = (event.xdata or 0, event.ydata or 0)
                        annot.set_text(f"{lbl}\n{val:,.0f}  ({pct:.1f}%)")
                        annot.set_visible(True)
                        visible = True
                        break
            elif hover_type == "line":
                if event.xdata is not None:
                    xi = int(round(event.xdata))
                    if 0 <= xi < len(hover_data):
                        lbl, val = hover_data[xi]
                        annot.xy = (xi, val)
                        annot.set_text(f"{lbl}\n{val:,.0f}")
                        annot.set_visible(True)
                        visible = True
            if not visible:
                annot.set_visible(False)
            canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_hover)

    def _save_chart(self, fig, fmt="png"):
        ext_map = {"png": ("PNG image", "*.png"),
                   "svg": ("SVG image", "*.svg"),
                   "pdf": ("PDF file", "*.pdf")}
        ftype, fext = ext_map.get(fmt, ("PNG image", "*.png"))
        path = filedialog.asksaveasfilename(
            defaultextension=f".{fmt}",
            filetypes=[(ftype, fext), ("All files", "*.*")],
            initialfile=f"insightx_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{fmt}"
        )
        if path:
            dpi = 200 if fmt == "png" else 150
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor(), format=fmt)
            Toast.show(self, f"Chart saved: {os.path.basename(path)}")

    def _render_report_block(self, parent, report_text):
        container = tk.Frame(parent, bg=BG_CARD,
                             highlightbackground=ACCENT_BLUE, highlightthickness=1)
        container.pack(anchor="w", pady=(3, 1), fill="x", padx=1)
        title_bar = tk.Frame(container, bg=BG_HOVER)
        title_bar.pack(fill="x")
        tk.Label(title_bar, text="📋  ANALYTICAL REPORT", font=("Consolas", 8, "bold"),
                 fg=ACCENT_BLUE, bg=BG_HOVER, padx=10, pady=5).pack(side="left")
        for lbl, cmd in [("⎘ Copy", lambda t=report_text: self._clip(t)),
                         ("⬇ Save .txt", lambda t=report_text: self._save_report(t, "txt")),
                         ("⬇ Save .md", lambda t=report_text: self._save_report(t, "md"))]:
            b = tk.Label(title_bar, text=lbl, font=("Consolas", 7),
                         fg=TEXT_MUTED, bg=BG_HOVER, padx=6, cursor="hand2")
            b.pack(side="right")
            b.bind("<Button-1>", lambda e, c=cmd: c())
            b.bind("<Enter>", lambda e, w=b: w.configure(fg=ACCENT_TEAL))
            b.bind("<Leave>", lambda e, w=b: w.configure(fg=TEXT_MUTED))
        tw = tk.Text(container, font=FONT_MONO, fg=TEXT_PRIMARY, bg=BG_CARD,
                     relief="flat", bd=0,
                     height=min(report_text.count("\n") + 1, 35),
                     wrap="none", state="normal", padx=12, pady=7,
                     selectbackground=ACCENT_BLUE, selectforeground=BG_DEEP)
        tw.insert("1.0", report_text)
        tw.pack(fill="x")
        self._prop_scroll(tw)

    def _save_report(self, text, fmt="txt"):
        path = filedialog.asksaveasfilename(
            defaultextension=f".{fmt}",
            filetypes=[(f"{fmt.upper()} file", f"*.{fmt}"), ("All files", "*.*")],
            initialfile=f"insightx_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{fmt}"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                if fmt == "md":
                    f.write("# InsightX Analytical Report\n\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write("```\n")
                    f.write(text)
                    f.write("\n```\n")
                else:
                    f.write(text)
            Toast.show(self, f"Report saved: {os.path.basename(path)}")

    def _typewrite(self, tw, text):
        chunk = max(1, len(text) // 80)

        def writer(i=0):
            if i <= len(text):
                tw.configure(state="normal")
                tw.delete("1.0", "end")
                tw.insert("1.0", text[:i])
                tw.configure(state="disabled")
                lc = int(tw.index("end-1c").split(".")[0])
                tw.configure(height=max(2, lc))
                if i < len(text):
                    tw.after(int(TYPEWRITER_DELAY * 1000), writer, min(i + chunk, len(text)))
                else:
                    self._scroll_bottom()

        writer()

    def _clip(self, text):
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update()
        Toast.show(self, "⎘ Copied to clipboard")

    def _scroll_bottom(self):
        self._chat_canvas.update_idletasks()
        self._chat_canvas.yview_moveto(1.0)

    def _set_status(self, text, color):
        self.status_label.configure(text=text, fg=color)

    def _set_thinking(self, text):
        self.thinking_label.configure(text=text)

    def _switch_tab(self, name):
        self._active_tab = name
        self.header_title.configure(text=f"InsightX — {name}")
        self._render_tab_list()
        self._rebuild_chat_for_tab(name)

    def _rebuild_chat_for_tab(self, tab):
        for w in self._chat_inner.winfo_children():
            w.destroy()
        rich = engine.get_rich(tab)
        if not rich:
            self._add_system_msg("New conversation started.")
            return
        for entry in rich:
            if entry.get("role") == "user":
                self._add_user_msg(entry["content"])
            else:
                self._render_result_static({
                    "text": entry.get("content", ""),
                    "table": entry.get("table"),
                    "chart": entry.get("chart"),
                    "report": entry.get("report"),
                    "summary": entry.get("summary"),
                    "recommendations": entry.get("recommendations"),
                })
        self._scroll_bottom()

    def _new_tab(self):
        existing = set(engine.list_tabs())
        i = len(existing) + 1
        name = f"Chat {i}"
        while name in existing:
            i += 1
            name = f"Chat {i}"
        engine.create_tab(name)
        self._render_tab_list()
        self._switch_tab(name)

    def _delete_tab(self, name):
        if messagebox.askyesno("Delete Tab", f"Delete chat '{name}'?", parent=self):
            engine.delete_tab(name)
            self._render_tab_list()
            if self._active_tab == name:
                self._switch_tab("Default")
