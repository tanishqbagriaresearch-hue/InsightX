"""
InsightX UI — ui.py  (Advanced Explorer edition)
New features:
  • Advanced Explorer dialog — mix & match ANY columns:
      - X axis (group by any column)
      - Y metric (count / avg amount / total amount / fail rate / fraud rate / success rate)
      - Color-by (split into multi-series grouped bar or multi-line)
      - Chart types: bar, grouped_bar, line, pie, histogram
      - Histogram with optional color-by overlay
      - Filter 1 + Filter 2 (AND logic)
      - Sort (desc / asc / natural)
      - Normalize to % share
      - Top-N cap
      - Bin count (for histograms)
      - Live preview status
  • Recommendations panel rendered below each answer
  • All 6 query categories get structured 4-part responses:
      direct answer + supporting stats + contextual insight + recommendation
  • Text bubbles stay selectable (click-drag + Ctrl+C)
  • Sidebar mouse-wheel scroll isolated from chat scroll
  • Save Chat button in header
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import threading
import queue
import re
import json
import os
from datetime import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import engine

# ── Colours ──────────────────────────────────────────────────────────────────
BG_DEEP     = "#090d13"
BG_PANEL    = "#0d1117"
BG_CARD     = "#161b22"
BG_INPUT    = "#1c2128"
BG_HOVER    = "#21262d"
BG_SELECTED = "#1f2937"
BORDER      = "#30363d"
BORDER_LT   = "#3d4450"

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

FONT_BODY  = ("Consolas", 10)
FONT_SMALL = ("Consolas", 9)
FONT_INPUT = ("Consolas", 11)
FONT_MONO  = ("Courier New", 9)

TYPEWRITER_DELAY    = 0.010
CUSTOM_ACTIONS_FILE = "insightx_custom_actions.json"

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
    ("hour_of_day",        "Hour of Day"),
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
]

CHART_TYPES = [
    ("bar",         "Bar"),
    ("grouped_bar", "Grouped Bar"),
    ("line",        "Line"),
    ("pie",         "Pie"),
    ("histogram",   "Histogram"),
]

SORT_OPTIONS = [
    ("desc",    "High → Low"),
    ("asc",     "Low → High"),
    ("natural", "Natural order"),
]


# ── helpers ───────────────────────────────────────────────────────────────────
def _load_custom_actions():
    if os.path.exists(CUSTOM_ACTIONS_FILE):
        try:
            with open(CUSTOM_ACTIONS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _save_custom_actions(actions):
    with open(CUSTOM_ACTIONS_FILE, "w") as f:
        json.dump(actions, f, indent=2)


# ── selectable text bubble ────────────────────────────────────────────────────
def _make_text_bubble(parent, text, bg=BG_CARD, fg=TEXT_PRIMARY,
                      wraplength=700, font=None):
    if font is None:
        font = FONT_BODY
    chars_per_line = max(1, wraplength // 7)
    line_count = max(1, sum(
        max(1, (len(line) // chars_per_line) + 1)
        for line in text.split("\n")
    ))
    # No height cap — the widget expands to fit ALL content, never truncates
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


# ── combo helper ──────────────────────────────────────────────────────────────
def _combo(parent, var, values, width=36, **kw):
    cb = ttk.Combobox(parent, textvariable=var, state="readonly",
                      values=values, font=FONT_SMALL, width=width, **kw)
    return cb


def _lbl(parent, text, small=False, color=None):
    c = color or (TEXT_MUTED if small else TEXT_SECONDARY)
    return tk.Label(parent, text=text,
                    font=("Consolas", 7 if small else 9, "bold"),
                    fg=c, bg=BG_PANEL)


def _sep(parent):
    tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=16, pady=5)


# ═══════════════════════════════════════════════════════════════════════════════
#  Advanced Explorer Dialog
# ═══════════════════════════════════════════════════════════════════════════════
class AdvancedExplorerDialog(tk.Toplevel):
    """
    Full mix-and-match chart builder:
      X axis × Y metric × optional Color-by × optional Filters × chart type.
    Supports histograms with colour-group overlay, grouped bars, multi-line charts.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.title("InsightX — Advanced Explorer")
        self.geometry("720x640")
        self.configure(bg=BG_PANEL)
        self.resizable(True, True)
        self.result_fig = None
        self._build()
        self.grab_set()
        self.transient(parent)

    # ── build UI ──────────────────────────────────────────────────────────────
    def _build(self):
        # ── Header ──────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=BG_PANEL)
        hdr.pack(fill="x", padx=20, pady=(14, 4))
        tk.Label(hdr, text="🔬  Advanced Explorer",
                 font=("Trebuchet MS", 13, "bold"),
                 fg=ACCENT_BLUE, bg=BG_PANEL).pack(side="left")
        tk.Label(hdr,
                 text="Mix & match any columns → histogram, bar, grouped bar, line, pie",
                 font=("Consolas", 8), fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left", padx=14)

        _sep(self)

        # ── Two-column layout ────────────────────────────────────────────────
        body = tk.Frame(self, bg=BG_PANEL)
        body.pack(fill="both", expand=True, padx=20)
        left  = tk.Frame(body, bg=BG_PANEL)
        right = tk.Frame(body, bg=BG_PANEL)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right.pack(side="left", fill="both", expand=True)

        # ── LEFT COLUMN ──────────────────────────────────────────────────────
        # X Axis
        _lbl(left, "X-AXIS  (Group / Dimension)").pack(anchor="w", pady=(6,2))
        self._x_var = tk.StringVar(value=ALL_COL_LABELS[0])
        _combo(left, self._x_var, ALL_COL_LABELS, width=38).pack(anchor="w")

        # Y Metric
        _lbl(left, "Y-METRIC  (What to Measure)").pack(anchor="w", pady=(10,2))
        self._y_var = tk.StringVar(value=f"{Y_METRICS[0][0]} — {Y_METRICS[0][1]}")
        _combo(left, self._y_var, [f"{k} — {v}" for k, v in Y_METRICS], width=38).pack(anchor="w")

        # Color-by
        _lbl(left, "COLOR-BY  (Split into Groups, optional)").pack(anchor="w", pady=(10,2))
        self._color_var = tk.StringVar(value="(none)")
        _combo(left, self._color_var, ["(none)"] + ALL_COL_LABELS, width=38).pack(anchor="w")
        tk.Label(left, text="→ creates Grouped Bar / multi-series Line / overlapping Histogram",
                 font=("Consolas", 7), fg=TEXT_MUTED, bg=BG_PANEL).pack(anchor="w")

        # Chart type
        _lbl(left, "CHART TYPE").pack(anchor="w", pady=(10,2))
        ct_frame = tk.Frame(left, bg=BG_PANEL)
        ct_frame.pack(anchor="w")
        self._chart_var = tk.StringVar(value="bar")
        for val, lbl_text in CHART_TYPES:
            tk.Radiobutton(ct_frame, text=lbl_text, variable=self._chart_var, value=val,
                           font=FONT_SMALL, fg=TEXT_SECONDARY, bg=BG_PANEL,
                           selectcolor=BG_SELECTED, activebackground=BG_PANEL,
                           activeforeground=TEXT_PRIMARY).pack(side="left", padx=5)

        # Histogram bins
        _lbl(left, "BINS  (for Histogram)", small=True).pack(anchor="w", pady=(8,2))
        bin_row = tk.Frame(left, bg=BG_PANEL)
        bin_row.pack(anchor="w")
        self._bins_var = tk.StringVar(value="40")
        tk.Entry(bin_row, textvariable=self._bins_var,
                 font=FONT_SMALL, fg=TEXT_PRIMARY, bg=BG_INPUT,
                 insertbackground=ACCENT_BLUE, relief="flat", width=7).pack(side="left")
        tk.Label(bin_row, text="  bins", font=("Consolas",8), fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left")

        # ── RIGHT COLUMN ─────────────────────────────────────────────────────
        # Filter 1
        _lbl(right, "FILTER 1  (optional)").pack(anchor="w", pady=(6,2))
        self._f1col_var = tk.StringVar(value="(none)")
        _combo(right, self._f1col_var, ["(none)"] + ALL_COL_LABELS, width=34).pack(anchor="w")
        self._f1val_var = tk.StringVar()
        f1row = tk.Frame(right, bg=BG_PANEL)
        f1row.pack(anchor="w", pady=(2,0))
        tk.Label(f1row, text="= ", font=FONT_SMALL, fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left")
        tk.Entry(f1row, textvariable=self._f1val_var,
                 font=FONT_SMALL, fg=TEXT_PRIMARY, bg=BG_INPUT,
                 insertbackground=ACCENT_BLUE, relief="flat", width=22).pack(side="left")

        # Filter 2
        _lbl(right, "FILTER 2  (optional, AND logic)", small=True).pack(anchor="w", pady=(10,2))
        self._f2col_var = tk.StringVar(value="(none)")
        _combo(right, self._f2col_var, ["(none)"] + ALL_COL_LABELS, width=34).pack(anchor="w")
        self._f2val_var = tk.StringVar()
        f2row = tk.Frame(right, bg=BG_PANEL)
        f2row.pack(anchor="w", pady=(2,0))
        tk.Label(f2row, text="= ", font=FONT_SMALL, fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left")
        tk.Entry(f2row, textvariable=self._f2val_var,
                 font=FONT_SMALL, fg=TEXT_PRIMARY, bg=BG_INPUT,
                 insertbackground=ACCENT_BLUE, relief="flat", width=22).pack(side="left")

        # Sort
        _lbl(right, "SORT ORDER").pack(anchor="w", pady=(10,2))
        self._sort_var = tk.StringVar(value="desc — High → Low")
        _combo(right, self._sort_var, [f"{k} — {v}" for k, v in SORT_OPTIONS], width=26).pack(anchor="w")

        # Top-N
        _lbl(right, "TOP-N  (max categories)", small=True).pack(anchor="w", pady=(8,2))
        topn_row = tk.Frame(right, bg=BG_PANEL)
        topn_row.pack(anchor="w")
        self._topn_var = tk.StringVar(value="12")
        tk.Entry(topn_row, textvariable=self._topn_var,
                 font=FONT_SMALL, fg=TEXT_PRIMARY, bg=BG_INPUT,
                 insertbackground=ACCENT_BLUE, relief="flat", width=6).pack(side="left")
        tk.Label(topn_row, text="  categories max", font=("Consolas",8),
                 fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left")

        # Normalize
        _lbl(right, "NORMALIZE", small=True).pack(anchor="w", pady=(8,2))
        self._norm_var = tk.BooleanVar(value=False)
        tk.Checkbutton(right, text="Normalize to % share (stacked %)",
                       variable=self._norm_var,
                       font=FONT_SMALL, fg=TEXT_SECONDARY, bg=BG_PANEL,
                       selectcolor=BG_SELECTED, activebackground=BG_PANEL,
                       activeforeground=TEXT_PRIMARY).pack(anchor="w")

        # ── Bottom: buttons + status ─────────────────────────────────────────
        _sep(self)
        bot = tk.Frame(self, bg=BG_PANEL)
        bot.pack(fill="x", padx=20, pady=(0, 14))

        # Example combos quick-fill
        examples_frame = tk.Frame(bot, bg=BG_PANEL)
        examples_frame.pack(side="left")
        tk.Label(examples_frame, text="Quick examples:", font=("Consolas",7),
                 fg=TEXT_MUTED, bg=BG_PANEL).pack(anchor="w")
        examples = [
            ("Age × Fail Rate",           {"x":"sender_age_group","y":"fail_rate","ct":"bar"}),
            ("Hour × Count (line)",        {"x":"hour_of_day","y":"count","ct":"line"}),
            ("Bank × Flagged (grouped)",   {"x":"sender_bank","y":"fraud_rate","color":"device_type","ct":"grouped_bar"}),
            ("Amount histogram by device", {"x":"amount_inr","y":"count","color":"device_type","ct":"histogram"}),
            ("Network × Type (grouped)",   {"x":"network_type","y":"count","color":"transaction_type","ct":"grouped_bar"}),
        ]
        eq_row = tk.Frame(examples_frame, bg=BG_PANEL)
        eq_row.pack()
        for ex_name, ex_cfg in examples:
            b = tk.Label(eq_row, text=ex_name, font=("Consolas",7),
                         fg=ACCENT_BLUE, bg=BG_PANEL, cursor="hand2",
                         padx=5, pady=1,
                         highlightbackground=BORDER, highlightthickness=1)
            b.pack(side="left", padx=2)
            b.bind("<Button-1>", lambda e, cfg=ex_cfg: self._apply_example(cfg))
            b.bind("<Enter>", lambda e, w=b: w.configure(bg=BG_HOVER))
            b.bind("<Leave>", lambda e, w=b: w.configure(bg=BG_PANEL))

        btn_frame = tk.Frame(bot, bg=BG_PANEL)
        btn_frame.pack(side="right")

        gen = tk.Label(btn_frame, text="  🔬 Generate  ",
                       font=("Consolas", 10, "bold"),
                       fg=BG_DEEP, bg=ACCENT_BLUE, padx=12, pady=7, cursor="hand2")
        gen.pack(side="left", padx=6)
        gen.bind("<Button-1>", lambda e: self._generate())
        gen.bind("<Enter>", lambda e: gen.configure(bg="#7abfff"))
        gen.bind("<Leave>", lambda e: gen.configure(bg=ACCENT_BLUE))

        cancel = tk.Label(btn_frame, text="  Cancel  ",
                          font=("Consolas", 10),
                          fg=TEXT_SECONDARY, bg=BG_HOVER, padx=12, pady=7, cursor="hand2")
        cancel.pack(side="left")
        cancel.bind("<Button-1>", lambda e: self.destroy())

        self._status = tk.Label(self, text="", font=("Consolas", 8),
                                fg=ACCENT_ORANGE, bg=BG_PANEL)
        self._status.pack(pady=(0, 4))

    def _apply_example(self, cfg):
        # X axis
        for label in ALL_COL_LABELS:
            if label.startswith(cfg.get("x","") + " —"):
                self._x_var.set(label); break
        # Y metric
        for k, v in Y_METRICS:
            if k == cfg.get("y",""):
                self._y_var.set(f"{k} — {v}"); break
        # Color-by
        col = cfg.get("color")
        if col:
            for label in ALL_COL_LABELS:
                if label.startswith(col + " —"):
                    self._color_var.set(label); break
        else:
            self._color_var.set("(none)")
        # Chart type
        self._chart_var.set(cfg.get("ct","bar"))

    def _parse_col(self, var_val):
        """Parse 'column — Label' or '(none)' → col name or None."""
        v = var_val.strip()
        if v == "(none)" or not v:
            return None
        return COL_LABEL_MAP.get(v, v.split(" — ")[0].strip())

    def _generate(self):
        self._status.configure(text="Building chart…", fg=ACCENT_ORANGE)
        self.update()
        try:
            x_col    = self._parse_col(self._x_var.get())
            y_metric = self._y_var.get().split(" — ")[0].strip()
            color_col= self._parse_col(self._color_var.get())
            ct       = self._chart_var.get()
            f1col    = self._parse_col(self._f1col_var.get())
            f1val    = self._f1val_var.get().strip() or None
            f2col    = self._parse_col(self._f2col_var.get())
            f2val    = self._f2val_var.get().strip() or None
            sort_raw = self._sort_var.get().split(" — ")[0].strip()
            try:  top_n = int(self._topn_var.get())
            except: top_n = 12
            try:  bins_ = int(self._bins_var.get())
            except: bins_ = 40
            normalize= self._norm_var.get()

            config = {
                "chart_type":  ct,
                "x_col":       x_col or "transaction_type",
                "y_metric":    y_metric,
                "color_col":   color_col,
                "filter_col":  f1col,
                "filter_val":  f1val,
                "filter_col2": f2col,
                "filter_val2": f2val,
                "sort":        sort_raw,
                "top_n":       top_n,
                "bins":        bins_,
                "normalize":   normalize,
            }

            fig = engine.generate_explorer_chart(config)
            self.result_fig = fig
            self.result_config = config
            self._status.configure(text="✓ Done!", fg=ACCENT_GREEN)
            self.after(600, self.destroy)

        except Exception as ex:
            self._status.configure(text=f"Error: {ex}", fg=ACCENT_RED)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Application
# ═══════════════════════════════════════════════════════════════════════════════
class InsightXApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("InsightX — Techfest 2025-26")
        self.geometry("1400x860")
        self.minsize(1000, 640)
        self.configure(bg=BG_DEEP)
        self._q            = queue.Queue()
        self._loading      = False
        self._active_tab   = "Default"
        self._tab_buttons  = {}
        self._custom_actions = _load_custom_actions()
        self._zoom_level   = 1.0   # 1.0 = 100%, range 0.4 – 1.0
        self._base_fonts   = None  # set after _build_ui
        self._build_ui()
        self._start_engine_thread()
        self._bind_scroll()
        self.after(50, self._poll_queue)

    # ── Scroll Handling ────────────────────────────────────────────────────────
    def _bind_scroll(self):
        """
        Universal scroll: bind_all intercepts every scroll event regardless
        of which widget the cursor is over. Routes to sidebar, table, or chat.
        Works on every element including scrollbars, charts, labels — everywhere.
        """
        self.bind_all("<MouseWheel>",       self._on_mousewheel)
        self.bind_all("<Button-4>",         lambda e: self._on_mousewheel(e, -1))
        self.bind_all("<Button-5>",         lambda e: self._on_mousewheel(e,  1))
        self.bind_all("<Shift-MouseWheel>", self._on_shift_mousewheel)
        self.bind_all("<Up>",    lambda e: self._key_scroll(-1))
        self.bind_all("<Down>",  lambda e: self._key_scroll(1))
        self.bind_all("<Prior>", lambda e: self._key_scroll(-10))
        self.bind_all("<Next>",  lambda e: self._key_scroll(10))
        self.bind_all("<Home>",  lambda e: self._chat_canvas.yview_moveto(0.0))
        self.bind_all("<End>",   lambda e: self._chat_canvas.yview_moveto(1.0))
        # Ctrl+scroll = zoom in/out
        self.bind_all("<Control-MouseWheel>", self._on_ctrl_mousewheel)
        self.bind_all("<Control-Button-4>",   lambda e: self._zoom(-1))
        self.bind_all("<Control-Button-5>",   lambda e: self._zoom(1))
        self.bind_all("<Control-equal>",      lambda e: self._zoom(-1))
        self.bind_all("<Control-minus>",      lambda e: self._zoom(1))
        self.bind_all("<Control-0>",          lambda e: self._zoom_reset())

    def _on_ctrl_mousewheel(self, event):
        """Ctrl+scroll zooms the app in or out."""
        direction = int(-1 * event.delta / 120)
        self._zoom(direction)
        return "break"

    def _zoom(self, direction):
        """direction: +1 = zoom out (smaller), -1 = zoom in (larger)."""
        step = 0.10
        new_zoom = round(self._zoom_level - direction * step, 2)
        new_zoom = max(0.35, min(1.0, new_zoom))
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
        """Scale all fonts proportionally to the zoom level."""
        z = self._zoom_level
        # Base font sizes at zoom=1.0
        _base = {
            "body":  10, "small": 9,  "input": 11, "mono":  9,
            "logo":  26, "brand": 11, "title": 11,
        }
        def _sz(base): return max(6, int(round(base * z)))
        # Update module-level font tuples (Tkinter re-reads them on next widget creation)
        # and retag all existing text widgets
        import tkinter.font as tkfont
        try:
            for fname in tkfont.names(root=self):
                fo = tkfont.Font(name=fname, exists=True, root=self)
                fam = fo.actual("family")
                base_size = fo.actual("size")
                if base_size < 0: base_size = abs(base_size)
                if base_size == 0: continue
                # Only scale our known app fonts — skip tiny system fonts
                if base_size >= 7:
                    target = max(6, int(round(base_size * z / max(0.01, getattr(self, "_last_zoom", 1.0)))))
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
        if amt == 0: return
        if self._cursor_in_widget(event, self.sidebar):
            self._sb_canvas.yview_scroll(amt, "units"); return "break"
        tc = self._find_table_canvas(event.widget)
        if tc:
            y0, y1 = tc.yview()
            if (y0 <= 0.0 and amt < 0) or (y1 >= 1.0 and amt > 0):
                self._chat_canvas.yview_scroll(amt, "units")
            else:
                tc.yview_scroll(amt, "units")
            return "break"
        self._chat_canvas.yview_scroll(amt, "units"); return "break"

    def _on_shift_mousewheel(self, event, linux_dir=None):
        amt = linux_dir if linux_dir is not None else int(-1 * event.delta / 120)
        tc = self._find_table_canvas(event.widget)
        if tc:
            tc.xview_scroll(amt, "units"); return "break"

    def _cursor_in_widget(self, event, widget):
        try:
            wx = widget.winfo_rootx(); wy = widget.winfo_rooty()
            ww = widget.winfo_width();  wh = widget.winfo_height()
            return wx <= event.x_root <= wx + ww and wy <= event.y_root <= wy + wh
        except Exception: return False

    def _find_table_canvas(self, widget):
        try:
            w = widget
            for _ in range(16):
                if w is None: break
                if getattr(w, "_is_table_canvas", False): return w
                w = getattr(w, "master", None)
        except Exception: pass
        return None

    def _key_scroll(self, units):
        if self.focus_get() == self.input_box: return
        self._chat_canvas.yview_scroll(units, "units")

    # No-ops for backward compat — bind_all handles everything now
    def _prop_scroll(self, widget): pass
    def _prop_sb_scroll(self, widget): pass

    # ── UI Construction ────────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_sidebar()
        self._build_main()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    def _build_sidebar(self):
        self.sidebar = tk.Frame(self, bg=BG_PANEL, width=248)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Logo
        logo = tk.Frame(self.sidebar, bg=BG_PANEL, pady=10)
        logo.pack(fill="x")
        tk.Label(logo, text="IX", font=("Georgia", 26, "bold"),
                 fg=ACCENT_BLUE, bg=BG_PANEL).pack()
        tk.Label(logo, text="InsightX", font=("Trebuchet MS", 11, "bold"),
                 fg=TEXT_PRIMARY, bg=BG_PANEL).pack()
        tk.Label(logo, text="Fintech Intelligence  •  Techfest 2025-26",
                 font=("Consolas", 7), fg=TEXT_MUTED, bg=BG_PANEL).pack()
        tk.Frame(self.sidebar, bg=BORDER, height=1).pack(fill="x", padx=14, pady=4)

        # Chats
        ch_hdr = tk.Frame(self.sidebar, bg=BG_PANEL)
        ch_hdr.pack(fill="x", padx=10, pady=(2,1))
        tk.Label(ch_hdr, text="CHATS", font=("Consolas",8,"bold"),
                 fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left")
        add_tb = tk.Label(ch_hdr, text="+", font=("Consolas",12,"bold"),
                          fg=ACCENT_BLUE, bg=BG_PANEL, cursor="hand2")
        add_tb.pack(side="right")
        add_tb.bind("<Button-1>", lambda e: self._new_tab())

        self.tabs_frame = tk.Frame(self.sidebar, bg=BG_PANEL)
        self.tabs_frame.pack(fill="x", padx=6)
        self._render_tab_list()
        tk.Frame(self.sidebar, bg=BORDER, height=1).pack(fill="x", padx=14, pady=4)

        # Scrollable actions
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

        self.status_label = tk.Label(self.sidebar, text="● Ready",
                                     font=("Consolas", 8),
                                     fg=ACCENT_GREEN, bg=BG_PANEL)
        self.status_label.pack(side="bottom", pady=6)

    def _sec_hdr(self, text, plus_cmd=None):
        row = tk.Frame(self._actions_inner, bg=BG_PANEL)
        row.pack(fill="x", padx=10, pady=(8,1))
        tk.Label(row, text=text, font=("Consolas",7,"bold"),
                 fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left")
        if plus_cmd:
            b = tk.Label(row, text="+", font=("Consolas",11,"bold"),
                         fg=ACCENT_TEAL, bg=BG_PANEL, cursor="hand2")
            b.pack(side="right")
            b.bind("<Button-1>", lambda e: plus_cmd())

    def _sb_btn(self, label, query=None, cmd=None, color=None, bold=False, deletable=False):
        row = tk.Frame(self._actions_inner, bg=BG_PANEL)
        row.pack(fill="x")
        fg = color or TEXT_SECONDARY
        fn = ("Consolas", 8, "bold") if bold else ("Consolas", 8)
        btn = tk.Label(row, text=label, font=fn, fg=fg, bg=BG_PANEL,
                       anchor="w", padx=14, pady=3, cursor="hand2")
        btn.pack(side="left", fill="x", expand=True)
        def _click(e):
            if cmd: cmd()
            elif query: self._quick(query)
        btn.bind("<Button-1>", _click)
        btn.bind("<Enter>",  lambda e: btn.configure(bg=BG_HOVER, fg=TEXT_PRIMARY))
        btn.bind("<Leave>",  lambda e: btn.configure(bg=BG_PANEL, fg=fg))
        self._prop_sb_scroll(btn); self._prop_sb_scroll(row)
        if deletable:
            d = tk.Label(row, text="×", font=("Consolas",9), fg=TEXT_MUTED,
                         bg=BG_PANEL, padx=5, cursor="hand2")
            d.pack(side="right")
            d.bind("<Button-1>", lambda e, q=query: self._remove_custom_action(q))

    def _render_actions(self):
        for w in self._actions_inner.winfo_children():
            w.destroy()

        # ── OVERVIEW
        self._sec_hdr("OVERVIEW")
        for lbl, q in [
            ("📊  Full Stats Table",   "show full statistics table"),
            ("📋  Full Report",         "generate full report"),
            ("💬  Chat Summary",        "summarize our conversation"),
        ]: self._sb_btn(lbl, query=q)

        # ── EXPLORERS
        self._sec_hdr("DATA EXPLORATION")
        self._sb_btn("🔬  Advanced Explorer ✦", cmd=self._open_explorer,
                     color=ACCENT_BLUE, bold=True)
        self._sb_btn("📈  Graph Builder",       cmd=self._open_explorer,
                     color=ACCENT_PURPLE)

        # ── QUERY CATEGORIES (the 6 deliverables)
        self._sec_hdr("QUERY TYPES — CORE DELIVERABLES")
        for lbl, q in [
            ("📝  Descriptive: Avg bill payment", "What is the average transaction amount for bill payments?"),
            ("⚖  Comparative: Android vs iOS",   "How do failure rates compare between Android and iOS users?"),
            ("⏱  Temporal: Peak food hours",      "What are the peak transaction hours for food delivery?"),
            ("👥  Segmentation: P2P by age",       "Which age group uses P2P transfers most frequently?"),
            ("🔗  Correlation: Network vs success","Is there a relationship between network type and transaction success?"),
            ("🛡  Risk: High-value flagged %",     "What percentage of high-value transactions are flagged for review?"),
        ]: self._sb_btn(lbl, query=q)

        # ── CHARTS
        self._sec_hdr("CHARTS & QUICK VIEWS")
        for lbl, q in [
            ("🍕  Fraud Flag Pie",         "show a pie chart of fraud flag breakdown"),
            ("🏦  Sender Bank Bar",         "show a bar chart by sender bank"),
            ("📱  Device Type Chart",       "show a bar chart by device type"),
            ("📡  Network Type Chart",      "show a bar chart by network type"),
            ("💳  Transaction Type Chart",  "show a bar chart by transaction type"),
            ("🛒  Merchant Category Chart", "show a bar chart by merchant category"),
            ("👤  Age Group Chart",         "show a bar chart by sender age group"),
            ("🕐  Peak Hours Line",         "show transactions by hour as a line chart"),
            ("📅  Day of Week Chart",       "show transactions by day of week"),
            ("🗺  Top States Chart",        "show a bar chart by sender state"),
            ("✅  Success vs Fail Pie",     "show a pie chart of transaction status"),
            ("📶  Network Fail Rate",       "show failure rate by network type chart"),
        ]: self._sb_btn(lbl, query=q)

        # ── STATISTICAL ANALYSIS
        self._sec_hdr("STATISTICAL ANALYSIS")
        for lbl, q in [
            ("💰  Amount Statistics",       "show full statistics table for amount inr"),
            ("🏦  Bank Analysis",           "compare failure rates by sender bank as a table"),
            ("📱  Device Analysis",         "compare failure rates by device type"),
            ("📡  Network Analysis",        "compare failure rates by network type"),
            ("👤  Age Group Analysis",      "compare average transaction amount by sender age group"),
            ("🛡  Flagged Review Analysis", "show fraud flag rate by transaction type table"),
            ("🗺  State Analysis",          "show top 10 states by total transaction count table"),
            ("🕐  Hourly Analysis",         "show transaction count by hour table"),
            ("📅  Weekend vs Weekday",      "compare weekend vs weekday transaction stats"),
            ("💳  P2P vs P2M Analysis",     "compare p2p vs p2m transaction amounts and counts"),
            ("📊  Merchant Revenue",        "show total amount by merchant category table"),
            ("🔗  Age × Device Crosstab",   "show transaction count by age group and device type"),
            ("🌐  Network × Status Table",  "show failure rate by network type and transaction type"),
        ]: self._sb_btn(lbl, query=q)

        # ── RISK
        self._sec_hdr("RISK & ANOMALY")
        for lbl, q in [
            ("🚨  High Value Flagged",   "what percentage of high value transactions are flagged for review"),
            ("🏦  Bank Flagged Rates",   "show flagged for review rate by sender bank as a table"),
            ("📱  Device Flagged Rates", "show flagged for review rate by device type"),
            ("🕐  Peak Flagged Hours",   "which hours have the highest flagged for review rate"),
            ("👤  Age Group Risk",       "show flagged for review rate by sender age group"),
        ]: self._sb_btn(lbl, query=q)

        # ── CUSTOM
        self._sec_hdr("CUSTOM ACTIONS", plus_cmd=self._add_custom_action)
        for act in self._custom_actions:
            self._sb_btn(act["label"], query=act["query"], deletable=True)

        tk.Frame(self._actions_inner, bg=BG_PANEL, height=12).pack()

    def _add_custom_action(self):
        label = simpledialog.askstring("New Quick Action", "Button label:", parent=self)
        if not label: return
        query = simpledialog.askstring("New Quick Action", "Query to send:", parent=self)
        if not query: return
        self._custom_actions.append({"label": label, "query": query})
        _save_custom_actions(self._custom_actions)
        self._render_actions()

    def _remove_custom_action(self, query):
        self._custom_actions = [a for a in self._custom_actions if a.get("query") != query]
        _save_custom_actions(self._custom_actions)
        self._render_actions()

    # ── Tab Management ─────────────────────────────────────────────────────────
    def _render_tab_list(self):
        for w in self.tabs_frame.winfo_children(): w.destroy()
        self._tab_buttons = {}
        for name in engine.list_tabs():
            self._add_tab_button(name)

    def _add_tab_button(self, name):
        row = tk.Frame(self.tabs_frame, bg=BG_PANEL)
        row.pack(fill="x", pady=1)
        is_active = name == self._active_tab
        bg = BG_SELECTED if is_active else BG_PANEL
        fg = ACCENT_BLUE if is_active else TEXT_SECONDARY
        btn = tk.Label(row, text=f"💬 {name}", font=("Consolas",8),
                       fg=fg, bg=bg, anchor="w", padx=8, pady=4, cursor="hand2")
        btn.pack(side="left", fill="x", expand=True)
        btn.bind("<Button-1>",        lambda e, n=name: self._switch_tab(n))
        btn.bind("<Double-Button-1>", lambda e, n=name: self._rename_tab_inline(n))
        btn.bind("<Enter>",  lambda e, b=btn, n=name:
                 b.configure(bg=BG_HOVER if n!=self._active_tab else BG_SELECTED))
        btn.bind("<Leave>",  lambda e, b=btn, n=name:
                 b.configure(bg=BG_SELECTED if n==self._active_tab else BG_PANEL))
        if name != "Default":
            d = tk.Label(row, text="×", font=("Consolas",9),
                         fg=TEXT_MUTED, bg=bg, cursor="hand2", padx=5)
            d.pack(side="right")
            d.bind("<Button-1>", lambda e, n=name: self._delete_tab(n))
        self._tab_buttons[name] = btn

    def _rename_tab_inline(self, old_name):
        new_name = simpledialog.askstring("Rename Chat",
                                          f"New name for '{old_name}':",
                                          initialvalue=old_name, parent=self)
        if not new_name or new_name == old_name: return
        if engine.rename_tab(old_name, new_name):
            if self._active_tab == old_name:
                self._active_tab = new_name
                self.header_title.configure(text=f"InsightX — {new_name}")
            self._render_tab_list()
        else:
            messagebox.showwarning("Rename", f"'{new_name}' already exists.")

    # ── Main Area ──────────────────────────────────────────────────────────────
    def _build_main(self):
        self.main_frame = tk.Frame(self, bg=BG_DEEP)
        self.main_frame.pack(side="left", fill="both", expand=True)

        header = tk.Frame(self.main_frame, bg=BG_PANEL, height=50)
        header.pack(fill="x"); header.pack_propagate(False)
        self.header_title = tk.Label(header, text="InsightX — Default",
                                     font=("Trebuchet MS", 11, "bold"),
                                     fg=TEXT_PRIMARY, bg=BG_PANEL)
        self.header_title.pack(side="left", padx=18, pady=12)

        for lbl, cmd, hover in [
            ("💾 Save Chat",      self._save_chat,    ACCENT_TEAL),
            ("🔬 Advanced Explorer", self._open_explorer, ACCENT_BLUE),
        ]:
            b = tk.Label(header, text=lbl, font=("Consolas",8),
                         fg=TEXT_SECONDARY, bg=BG_PANEL, padx=10, cursor="hand2")
            b.pack(side="right", padx=4)
            b.bind("<Button-1>", lambda e, c=cmd: c())
            b.bind("<Enter>", lambda e, w=b, h=hover: w.configure(fg=h))
            b.bind("<Leave>", lambda e, w=b: w.configure(fg=TEXT_SECONDARY))

        # ── Zoom controls ──────────────────────────────────────────────────────
        zoom_frame = tk.Frame(header, bg=BG_PANEL)
        zoom_frame.pack(side="right", padx=8)
        for ztxt, zdir in [("－", 1), ("＋", -1)]:
            zb = tk.Label(zoom_frame, text=ztxt, font=("Consolas", 11, "bold"),
                          fg=ACCENT_BLUE, bg=BG_PANEL, padx=5, cursor="hand2")
            zb.pack(side="left")
            zb.bind("<Button-1>", lambda e, d=zdir: self._zoom(d))
            zb.bind("<Enter>", lambda e, w=zb: w.configure(fg=TEXT_PRIMARY))
            zb.bind("<Leave>", lambda e, w=zb: w.configure(fg=ACCENT_BLUE))
        self._zoom_label = tk.Label(zoom_frame, text="🔍 100%", font=("Consolas", 7),
                                    fg=TEXT_MUTED, bg=BG_PANEL, padx=4, cursor="hand2")
        self._zoom_label.pack(side="left")
        self._zoom_label.bind("<Button-1>", lambda e: self._zoom_reset())
        tk.Label(zoom_frame, text="Ctrl+Scroll", font=("Consolas", 6),
                 fg=TEXT_MUTED, bg=BG_PANEL, padx=2).pack(side="left")

        self.thinking_label = tk.Label(header, text="", font=("Consolas",9),
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
        self._cwin = self._chat_canvas.create_window((0,0), window=self._chat_inner, anchor="nw")
        self._chat_inner.bind("<Configure>", lambda e:
            self._chat_canvas.configure(scrollregion=self._chat_canvas.bbox("all")))
        self._chat_canvas.bind("<Configure>", lambda e:
            self._chat_canvas.itemconfig(self._cwin, width=e.width))

        self._build_input_bar()

    def _build_input_bar(self):
        tk.Frame(self.main_frame, bg=BORDER, height=1).pack(fill="x", side="bottom")
        input_frame = tk.Frame(self.main_frame, bg=BG_PANEL, pady=10)
        input_frame.pack(fill="x", side="bottom")
        inner = tk.Frame(input_frame, bg=BG_INPUT,
                         highlightbackground=BORDER_LT, highlightthickness=1)
        inner.pack(fill="x", padx=18)
        self.input_var = tk.StringVar()
        self.input_box = tk.Entry(inner, textvariable=self.input_var,
                                  font=FONT_INPUT, fg=TEXT_PRIMARY, bg=BG_INPUT,
                                  insertbackground=ACCENT_BLUE, relief="flat", bd=0)
        self.input_box.pack(side="left", fill="x", expand=True, padx=12, pady=9)
        self.input_box.bind("<Return>", self._on_send)
        self.input_box.bind("<FocusIn>",  lambda e: inner.configure(highlightbackground=ACCENT_BLUE))
        self.input_box.bind("<FocusOut>", lambda e: inner.configure(highlightbackground=BORDER_LT))
        send = tk.Label(inner, text="⏎", font=("Consolas",13),
                        fg=ACCENT_BLUE, bg=BG_INPUT, padx=12, cursor="hand2")
        send.pack(side="right")
        send.bind("<Button-1>", self._on_send)
        tk.Label(input_frame,
                 text="Ask anything in natural language  •  6 query categories  •  4-part structured answers  •  Drag to select text",
                 font=("Consolas",7), fg=TEXT_MUTED, bg=BG_PANEL).pack(pady=(2,0))

    # ── Engine Threading ───────────────────────────────────────────────────────
    def _start_engine_thread(self):
        self._set_status("⟳ Loading...", ACCENT_ORANGE)
        self._set_thinking("Loading model & data…")
        threading.Thread(target=self._init_engine, daemon=True).start()

    def _init_engine(self):
        try:
            engine.load_data(); engine.load_model()
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
                    self._add_system_msg("InsightX ready — 250,000 transactions loaded.  Ask anything.")
                elif msg == "error":
                    self._set_status("● Error", ACCENT_RED)
                    self._set_thinking("")
                    self._add_system_msg(f"Error: {payload}")
                elif msg == "result":
                    self._render_result(payload)
                    self._set_thinking("")
                    self._set_status("● Ready", ACCENT_GREEN)
                    self._loading = False
                    self.input_box.configure(state="normal")
        except queue.Empty:
            pass
        self.after(50, self._poll_queue)

    def _on_send(self, event=None):
        if self._loading: return
        text = self.input_var.get().strip()
        if not text: return
        self.input_var.set("")
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
        except Exception as ex:
            self._q.put(("result", {"text": f"Error: {ex}", "table": None,
                                    "chart": None, "report": None, "summary": None,
                                    "data_raw": None, "recommendations": None}))

    def _quick(self, query):
        if self._loading: return
        self.input_var.set(query)
        self._on_send()

    # ── Save Chat ──────────────────────────────────────────────────────────────
    def _save_chat(self):
        text = engine.save_chat_export(self._active_tab)
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text file","*.txt"),("All files","*.*")],
            initialfile=f"insightx_{self._active_tab}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        if path:
            with open(path,"w",encoding="utf-8") as f: f.write(text)
            messagebox.showinfo("Saved", f"Chat saved to:\n{path}")

    # ── Advanced Explorer ──────────────────────────────────────────────────────
    def _open_explorer(self):
        dlg = AdvancedExplorerDialog(self)
        self.wait_window(dlg)
        if getattr(dlg, "result_fig", None) is not None:
            cfg = getattr(dlg, "result_config", {})
            title = (f"Explorer: {cfg.get('x_col','?')} × {cfg.get('y_metric','?')}"
                     + (f" by {cfg.get('color_col')}" if cfg.get("color_col") else ""))
            self._inject_chart(dlg.result_fig, label=title)

    def _inject_chart(self, fig, label="[Advanced Explorer]"):
        result = {"text": f"Chart: {label}",
                  "table": None, "chart": fig, "report": None,
                  "summary": None, "data_raw": None, "recommendations": None}
        self._add_user_msg(label)
        self._render_result(result)
        engine.push_rich(self._active_tab,
                         {"role":"user","content":label,"type":"user"})
        engine.push_rich(self._active_tab,
                         {"role":"assistant","content":f"Chart: {label}","type":"text",
                          "chart":fig,"table":None,"report":None,"summary":None,
                          "recommendations":None})

    # ── Message Rendering ──────────────────────────────────────────────────────
    def _add_user_msg(self, text):
        outer = tk.Frame(self._chat_inner, bg=BG_DEEP)
        outer.pack(fill="x", padx=18, pady=(8,1))
        self._prop_scroll(outer)
        right = tk.Frame(outer, bg=BG_DEEP)
        right.pack(side="right")
        tk.Label(right, text="You", font=("Consolas",8,"bold"),
                 fg=ACCENT_BLUE, bg=BG_DEEP).pack(anchor="e")
        bubble = tk.Frame(right, bg=BG_SELECTED,
                          highlightbackground=BORDER, highlightthickness=1)
        bubble.pack(anchor="e")
        tw = _make_text_bubble(bubble, text, bg=BG_SELECTED, fg=TEXT_PRIMARY)
        tw.pack(padx=1, pady=1)
        self._prop_scroll(tw)
        self._scroll_bottom()

    def _add_system_msg(self, text):
        outer = tk.Frame(self._chat_inner, bg=BG_DEEP)
        outer.pack(fill="x", padx=18, pady=4)
        self._prop_scroll(outer)
        tk.Label(outer, text=text, font=("Consolas",8),
                 fg=TEXT_MUTED, bg=BG_DEEP, justify="center").pack()
        self._scroll_bottom()

    def _render_result(self, result):
        outer = tk.Frame(self._chat_inner, bg=BG_DEEP)
        outer.pack(fill="x", padx=18, pady=(1,8))
        self._prop_scroll(outer)
        left = tk.Frame(outer, bg=BG_DEEP)
        left.pack(side="left", anchor="n", fill="x", expand=True)
        self._prop_scroll(left)

        name_row = tk.Frame(left, bg=BG_DEEP)
        name_row.pack(anchor="w")
        tk.Label(name_row, text="IX", font=("Georgia",10,"bold"),
                 fg=ACCENT_BLUE, bg=BG_DEEP).pack(side="left")
        tk.Label(name_row, text=" InsightX", font=("Consolas",8,"bold"),
                 fg=TEXT_SECONDARY, bg=BG_DEEP).pack(side="left")
        tk.Label(name_row, text=f"  {datetime.now().strftime('%H:%M')}",
                 font=("Consolas",7), fg=TEXT_MUTED, bg=BG_DEEP).pack(side="left")

        main_text = result.get("text","")

        if result.get("summary"):
            self._render_text_bubble(left, result["summary"], color=ACCENT_PURPLE, animate=True)
        if result.get("report"):
            self._render_report_block(left, result["report"])

        # ── Always show the main text paragraph BEFORE the table/chart ────────
        if main_text and not result.get("report") and not result.get("summary"):
            self._render_text_bubble(left, main_text, animate=True)

        if result.get("table"):
            self._render_table(left, result["table"])
        if result.get("chart"):
            self._render_chart(left, result["chart"])
        # No duplicate analysis paragraph — text was already shown above
        if result.get("recommendations"):
            self._render_recommendations(left, result["recommendations"])
        self._scroll_bottom()

    def _render_result_static(self, result):
        outer = tk.Frame(self._chat_inner, bg=BG_DEEP)
        outer.pack(fill="x", padx=18, pady=(1,8))
        self._prop_scroll(outer)
        left = tk.Frame(outer, bg=BG_DEEP)
        left.pack(side="left", anchor="n", fill="x", expand=True)
        name_row = tk.Frame(left, bg=BG_DEEP)
        name_row.pack(anchor="w")
        tk.Label(name_row, text="IX", font=("Georgia",10,"bold"),
                 fg=ACCENT_BLUE, bg=BG_DEEP).pack(side="left")
        tk.Label(name_row, text=" InsightX", font=("Consolas",8,"bold"),
                 fg=TEXT_SECONDARY, bg=BG_DEEP).pack(side="left")

        main_text = result.get("text","")

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
        # No duplicate analysis paragraph
        if result.get("recommendations"):
            self._render_recommendations(left, result["recommendations"])

    # ── Text bubble ────────────────────────────────────────────────────────────
    def _render_text_bubble(self, parent, text, color=TEXT_PRIMARY, animate=True):
        bubble = tk.Frame(parent, bg=BG_CARD,
                          highlightbackground=BORDER, highlightthickness=1)
        bubble.pack(anchor="w", pady=(3,1), fill="x")
        self._prop_scroll(bubble)
        toolbar = tk.Frame(bubble, bg=BG_CARD)
        toolbar.pack(fill="x", padx=5, pady=(3,0))
        c = tk.Label(toolbar, text="⎘ Copy All", font=("Consolas",7),
                     fg=TEXT_MUTED, bg=BG_CARD, cursor="hand2")
        c.pack(side="right")
        c.bind("<Button-1>", lambda e, t=text: self._clip(t))
        c.bind("<Enter>", lambda e: c.configure(fg=ACCENT_TEAL))
        c.bind("<Leave>", lambda e: c.configure(fg=TEXT_MUTED))
        tw = _make_text_bubble(bubble, text, bg=BG_CARD, fg=color)
        tw.pack(anchor="w", fill="x", padx=1, pady=(0,4))
        self._prop_scroll(tw)
        if animate:
            self._typewrite(tw, text)

    # ── Analysis paragraph (shown below table/chart) ───────────────────────────
    def _render_analysis_paragraph(self, parent, text, animate=True):
        """
        Renders a dedicated 'Analysis' panel below any table or chart.
        This ensures users always get a real written summary of what they're
        looking at — not just raw numbers.
        """
        # Don't double-render if the text is very short (conversational replies)
        if not text or len(text.split()) < 8:
            return

        frame = tk.Frame(parent, bg="#0d1a2a",
                         highlightbackground=ACCENT_BLUE, highlightthickness=1)
        frame.pack(anchor="w", pady=(3,1), fill="x", padx=1)
        self._prop_scroll(frame)

        # Header
        hdr = tk.Frame(frame, bg="#0d1a2a")
        hdr.pack(fill="x")
        tk.Label(hdr, text="📊  Analysis",
                 font=("Consolas",8,"bold"),
                 fg=ACCENT_BLUE, bg="#0d1a2a", padx=10, pady=4).pack(side="left")
        cb = tk.Label(hdr, text="⎘ Copy", font=("Consolas",7),
                      fg=TEXT_MUTED, bg="#0d1a2a", padx=7, cursor="hand2")
        cb.pack(side="right")
        cb.bind("<Button-1>", lambda e, t=text: self._clip(t))
        cb.bind("<Enter>", lambda e: cb.configure(fg=ACCENT_TEAL))
        cb.bind("<Leave>", lambda e: cb.configure(fg=TEXT_MUTED))

        # Text content
        tw = _make_text_bubble(frame, text, bg="#0d1a2a",
                               fg="#a8c8f0", wraplength=860)
        tw.pack(anchor="w", fill="x", padx=10, pady=(2,8))
        self._prop_scroll(tw)
        if animate:
            self._typewrite(tw, text)

    # ── Recommendations panel ──────────────────────────────────────────────────
    def _render_recommendations(self, parent, recs):
        if not recs: return
        frame = tk.Frame(parent, bg="#0f1e12",
                         highlightbackground=ACCENT_GREEN, highlightthickness=1)
        frame.pack(anchor="w", pady=(3,1), fill="x", padx=1)
        self._prop_scroll(frame)

        title_bar = tk.Frame(frame, bg="#0f1e12")
        title_bar.pack(fill="x")
        tk.Label(title_bar, text="💡  Recommendations & Insights",
                 font=("Consolas",8,"bold"),
                 fg=ACCENT_GREEN, bg="#0f1e12", padx=10, pady=4).pack(side="left")
        copy_text = "\n".join(f"• {r}" for r in recs)
        cb = tk.Label(title_bar, text="⎘ Copy", font=("Consolas",7),
                      fg=TEXT_MUTED, bg="#0f1e12", padx=7, cursor="hand2")
        cb.pack(side="right")
        cb.bind("<Button-1>", lambda e, t=copy_text: self._clip(t))
        cb.bind("<Enter>", lambda e: cb.configure(fg=ACCENT_TEAL))
        cb.bind("<Leave>", lambda e: cb.configure(fg=TEXT_MUTED))

        for i, rec in enumerate(recs):
            row = tk.Frame(frame, bg="#0f1e12")
            row.pack(fill="x", padx=10, pady=(2 if i==0 else 0, 2 if i==len(recs)-1 else 0))
            self._prop_scroll(row)
            tk.Label(row, text="▸", font=("Consolas",9,"bold"),
                     fg=ACCENT_GREEN, bg="#0f1e12").pack(side="left", padx=(0,6))
            lbl = tk.Label(row, text=rec, font=("Consolas",9), fg="#a8d5b5",
                           bg="#0f1e12", anchor="w", wraplength=820, justify="left")
            lbl.pack(side="left", fill="x", expand=True)
            self._prop_scroll(lbl)   # ensure scroll passes through recommendation labels too

    # ── Table ──────────────────────────────────────────────────────────────────
    def _render_table(self, parent, table_data):
        outer = tk.Frame(parent, bg=BG_CARD,
                         highlightbackground=BORDER_LT, highlightthickness=1)
        outer.pack(anchor="w", pady=(3,1), fill="x", padx=1)

        # ── Title bar ──────────────────────────────────────────────────────────
        title_bar = tk.Frame(outer, bg=BG_HOVER)
        title_bar.pack(fill="x")
        tk.Label(title_bar, text="⊞  Data Table", font=("Consolas",8,"bold"),
                 fg=ACCENT_BLUE, bg=BG_HOVER, padx=10, pady=3).pack(side="left")
        tk.Label(title_bar,
                 text="scroll ↕  •  shift+scroll ↔  •  drag scrollbar for columns",
                 font=("Consolas",6), fg=TEXT_MUTED, bg=BG_HOVER).pack(side="left", padx=4)

        copy_text = "\t".join(table_data["headers"]) + "\n" + \
                    "\n".join("\t".join(str(c) for c in r) for r in table_data["rows"])
        for lbl, handler in [
            ("⤢ Expand",   lambda e, td=table_data: self._expand_table(td)),
            ("⎘ Copy TSV", lambda e, t=copy_text:   self._clip(t)),
        ]:
            b = tk.Label(title_bar, text=lbl, font=("Consolas",7),
                         fg=TEXT_MUTED, bg=BG_HOVER, padx=7, cursor="hand2")
            b.pack(side="right")
            b.bind("<Button-1>", handler)
            b.bind("<Enter>", lambda e, w=b: w.configure(fg=ACCENT_TEAL))
            b.bind("<Leave>", lambda e, w=b: w.configure(fg=TEXT_MUTED))

        # ── Scrollbars ────────────────────────────────────────────────────────
        row_h   = 26
        hdr_h   = 32
        max_vis = 12
        table_h = min(len(table_data["rows"]) * row_h + hdr_h,
                      max_vis * row_h + hdr_h)

        h_scroll = ttk.Scrollbar(outer, orient="horizontal")
        v_scroll = ttk.Scrollbar(outer, orient="vertical")
        canvas = tk.Canvas(outer, bg=BG_CARD, highlightthickness=0, height=table_h,
                           xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        canvas._is_table_canvas = True   # marker for smart-scroll detection

        h_scroll.config(command=canvas.xview)
        v_scroll.config(command=canvas.yview)
        v_scroll.pack(side="right",  fill="y")
        h_scroll.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)

        # Smart vertical scroll: at boundary → fall through to page
        def _tbl_vert(e, d=None):
            amt = d if d is not None else int(-1 * e.delta / 120)
            y0, y1 = canvas.yview()
            if (y0 <= 0.0 and amt < 0) or (y1 >= 1.0 and amt > 0):
                self._chat_canvas.yview_scroll(amt, "units")
            else:
                canvas.yview_scroll(amt, "units")
            return "break"

        def _tbl_horiz(e, d=None):
            amt = d if d is not None else int(-1 * e.delta / 120)
            canvas.xview_scroll(amt, "units")
            return "break"

        canvas.bind("<MouseWheel>",       _tbl_vert)
        canvas.bind("<Button-4>",         lambda e: _tbl_vert(e, -1))
        canvas.bind("<Button-5>",         lambda e: _tbl_vert(e,  1))
        canvas.bind("<Shift-MouseWheel>", _tbl_horiz)

        # ── Table content ─────────────────────────────────────────────────────
        inner  = tk.Frame(canvas, bg=BG_CARD)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        col_widths = self._calc_col_widths(table_data)

        hdr_row = tk.Frame(inner, bg=BG_HOVER)
        hdr_row.pack(fill="x", side="top")
        for i, h in enumerate(table_data["headers"]):
            tk.Label(hdr_row, text=h, font=("Consolas",8,"bold"),
                     fg=ACCENT_BLUE, bg=BG_HOVER,
                     width=col_widths[i], anchor="w", padx=9, pady=5).pack(side="left")

        for r_idx, row in enumerate(table_data["rows"]):
            bg = BG_CARD if r_idx % 2 == 0 else BG_PANEL
            data_row = tk.Frame(inner, bg=bg)
            data_row.pack(fill="x", side="top")
            data_row.bind("<MouseWheel>",       _tbl_vert)
            data_row.bind("<Button-4>",         lambda e: _tbl_vert(e, -1))
            data_row.bind("<Button-5>",         lambda e: _tbl_vert(e,  1))
            data_row.bind("<Shift-MouseWheel>", _tbl_horiz)
            for c_idx, cell_val in enumerate(row):
                w = col_widths[c_idx] if c_idx < len(col_widths) else 14
                ct = str(cell_val)
                is_num = bool(re.match(r"^[\d,.\-% ]+$", ct.strip()))
                fg = ACCENT_GREEN if is_num else TEXT_PRIMARY
                cl = tk.Label(data_row, text=ct, font=FONT_MONO,
                              fg=fg, bg=bg, width=w, anchor="w", padx=9, pady=3)
                cl.pack(side="left")
                cl.bind("<Button-1>",         lambda e, t=ct: self._clip(t))
                cl.bind("<MouseWheel>",       _tbl_vert)
                cl.bind("<Button-4>",         lambda e: _tbl_vert(e, -1))
                cl.bind("<Button-5>",         lambda e: _tbl_vert(e,  1))
                cl.bind("<Shift-MouseWheel>", _tbl_horiz)

        def _on_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(win_id,
                              width=max(inner.winfo_reqwidth(), canvas.winfo_width()))
        inner.bind("<Configure>",  _on_configure)
        canvas.bind("<Configure>", _on_configure)

    def _expand_table(self, table_data):
        """Full-screen expanded table with search/filter and unrestricted scroll."""
        win = tk.Toplevel(self)
        win.title("InsightX — Expanded Table")
        win.geometry("1100x680")
        win.configure(bg=BG_DEEP)
        win.resizable(True, True)

        # Header bar
        hdr = tk.Frame(win, bg=BG_PANEL)
        hdr.pack(fill="x")
        tk.Label(hdr, text="⊞  Expanded Data Table",
                 font=("Consolas",10,"bold"),
                 fg=ACCENT_BLUE, bg=BG_PANEL, padx=14, pady=8).pack(side="left")
        nrows = len(table_data["rows"])
        ncols = len(table_data["headers"])
        tk.Label(hdr, text=f"{nrows} rows × {ncols} columns",
                 font=("Consolas",8), fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left", padx=8)

        copy_text = "\t".join(table_data["headers"]) + "\n" + \
                    "\n".join("\t".join(str(c) for c in r) for r in table_data["rows"])
        for lbl, cmd in [("⎘ Copy TSV", lambda: self._clip(copy_text)),
                         ("✕ Close",    win.destroy)]:
            b = tk.Label(hdr, text=lbl, font=("Consolas",8),
                         fg=TEXT_SECONDARY, bg=BG_PANEL, padx=12, cursor="hand2")
            b.pack(side="right", padx=4)
            b.bind("<Button-1>", lambda e, c=cmd: c())
            b.bind("<Enter>",    lambda e, w=b: w.configure(fg=ACCENT_TEAL))
            b.bind("<Leave>",    lambda e, w=b: w.configure(fg=TEXT_SECONDARY))

        tk.Frame(win, bg=BORDER, height=1).pack(fill="x")

        # Search bar
        sbar = tk.Frame(win, bg=BG_PANEL, pady=5)
        sbar.pack(fill="x", padx=12)
        tk.Label(sbar, text="🔍 Filter:", font=("Consolas",8),
                 fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left")
        search_var = tk.StringVar()
        tk.Entry(sbar, textvariable=search_var, font=FONT_SMALL,
                 fg=TEXT_PRIMARY, bg=BG_INPUT,
                 insertbackground=ACCENT_BLUE, relief="flat", width=44).pack(
                     side="left", padx=8)
        count_lbl = tk.Label(sbar, text=f"Showing {nrows} rows",
                             font=("Consolas",7), fg=TEXT_MUTED, bg=BG_PANEL)
        count_lbl.pack(side="left", padx=8)

        # Canvas + scrollbars
        body = tk.Frame(win, bg=BG_DEEP)
        body.pack(fill="both", expand=True, padx=6, pady=(2,6))
        h_sb  = ttk.Scrollbar(body, orient="horizontal")
        v_sb  = ttk.Scrollbar(body, orient="vertical")
        cv    = tk.Canvas(body, bg=BG_CARD, highlightthickness=0,
                          xscrollcommand=h_sb.set, yscrollcommand=v_sb.set)
        h_sb.config(command=cv.xview)
        v_sb.config(command=cv.yview)
        v_sb.pack(side="right",  fill="y")
        h_sb.pack(side="bottom", fill="x")
        cv.pack(side="left", fill="both", expand=True)

        inn   = tk.Frame(cv, bg=BG_CARD)
        wid   = cv.create_window((0, 0), window=inn, anchor="nw")
        cw    = self._calc_col_widths(table_data)

        def _rebuild(ft=""):
            for c in inn.winfo_children(): c.destroy()
            # Header
            hr = tk.Frame(inn, bg=BG_HOVER); hr.pack(fill="x")
            for i, h in enumerate(table_data["headers"]):
                tk.Label(hr, text=h, font=("Consolas",9,"bold"),
                         fg=ACCENT_BLUE, bg=BG_HOVER,
                         width=cw[i], anchor="w", padx=10, pady=6).pack(side="left")
            visible = 0
            ftl = ft.lower()
            for ri, row in enumerate(table_data["rows"]):
                if ftl and not any(ftl in str(c).lower() for c in row):
                    continue
                visible += 1
                bg = BG_CARD if ri % 2 == 0 else BG_PANEL
                dr = tk.Frame(inn, bg=bg); dr.pack(fill="x")
                for ci, cell_val in enumerate(row):
                    w  = cw[ci] if ci < len(cw) else 14
                    ct = str(cell_val)
                    is_num = bool(re.match(r"^[\d,.\-% ]+$", ct.strip()))
                    clr = ACCENT_YELLOW if (ftl and ftl in ct.lower()) \
                          else (ACCENT_GREEN if is_num else TEXT_PRIMARY)
                    lw = tk.Label(dr, text=ct, font=FONT_MONO,
                                  fg=clr, bg=bg, width=w, anchor="w", padx=10, pady=4)
                    lw.pack(side="left")
                    lw.bind("<Button-1>", lambda e, t=ct: self._clip(t))
            count_lbl.configure(text=f"Showing {visible} / {nrows} rows")
            cv.update_idletasks()
            cv.configure(scrollregion=cv.bbox("all"))

        _rebuild()
        inn.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>",  lambda e: cv.itemconfig(
            wid, width=max(inn.winfo_reqwidth(), cv.winfo_width())))
        search_var.trace_add("write", lambda *_: _rebuild(search_var.get()))

        def _ev(e, d=None):
            amt = d if d is not None else int(-1 * e.delta / 120)
            cv.yview_scroll(amt, "units")
        def _eh(e, d=None):
            amt = d if d is not None else int(-1 * e.delta / 120)
            cv.xview_scroll(amt, "units")
        for w in (cv, win):
            w.bind("<MouseWheel>",       _ev)
            w.bind("<Button-4>",         lambda e: _ev(e, -1))
            w.bind("<Button-5>",         lambda e: _ev(e,  1))
            w.bind("<Shift-MouseWheel>", _eh)

    def _calc_col_widths(self, table_data):
        widths = [max(len(h), 7) for h in table_data["headers"]]
        for row in table_data["rows"]:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], min(len(str(cell)), 34))
        return [w+2 for w in widths]



    # ── Chart ──────────────────────────────────────────────────────────────────
    def _render_chart(self, parent, fig):
        frame = tk.Frame(parent, bg=BG_CARD,
                         highlightbackground=BORDER_LT, highlightthickness=1)
        frame.pack(anchor="w", pady=(3,1), fill="x", padx=1)
        title_bar = tk.Frame(frame, bg=BG_HOVER)
        title_bar.pack(fill="x")
        tk.Label(title_bar, text="📈  Chart", font=("Consolas",8,"bold"),
                 fg=ACCENT_BLUE, bg=BG_HOVER, padx=10, pady=3).pack(side="left")
        sb = tk.Label(title_bar, text="⬇ Save PNG", font=("Consolas",7),
                      fg=TEXT_MUTED, bg=BG_HOVER, padx=7, cursor="hand2")
        sb.pack(side="right")
        sb.bind("<Button-1>", lambda e, f=fig: self._save_chart(f))
        sb.bind("<Enter>", lambda e: sb.configure(fg=ACCENT_TEAL))
        sb.bind("<Leave>", lambda e: sb.configure(fg=TEXT_MUTED))
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
        if not hover_type or not hover_data: return
        ax = fig.axes[0]
        annot = ax.annotate("", xy=(0,0), xytext=(14,14),
                            textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.45", fc="#1c2128", ec="#3d4450", lw=1),
                            fontsize=8.5, color=TEXT_PRIMARY,
                            arrowprops=dict(arrowstyle="->", color=ACCENT_BLUE))
        annot.set_visible(False)
        def on_hover(event):
            if event.inaxes != ax:
                annot.set_visible(False); canvas.draw_idle(); return
            visible = False
            if hover_type == "bar":
                for i, bar in enumerate(getattr(fig,"_bars",[])):
                    if bar.contains(event)[0]:
                        lbl, val = hover_data[i]
                        annot.xy = (bar.get_x()+bar.get_width()/2, bar.get_height())
                        annot.set_text(f"{lbl}\n{val:,.2f}")
                        annot.set_visible(True); visible = True; break
            elif hover_type == "pie":
                for i, wedge in enumerate(getattr(fig,"_wedges",[])):
                    if wedge.contains(event)[0]:
                        lbl, val = hover_data[i]
                        total = sum(v for _,v in hover_data)
                        pct = val/total*100 if total else 0
                        annot.xy = (event.xdata or 0, event.ydata or 0)
                        annot.set_text(f"{lbl}\n{val:,.0f}  ({pct:.1f}%)")
                        annot.set_visible(True); visible = True; break
            elif hover_type == "line":
                if event.xdata is not None:
                    xi = int(round(event.xdata))
                    if 0 <= xi < len(hover_data):
                        lbl, val = hover_data[xi]
                        annot.xy = (xi, val)
                        annot.set_text(f"{lbl}\n{val:,.0f}")
                        annot.set_visible(True); visible = True
            if not visible: annot.set_visible(False)
            canvas.draw_idle()
        fig.canvas.mpl_connect("motion_notify_event", on_hover)

    def _save_chart(self, fig):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image","*.png"),("All files","*.*")],
            initialfile=f"insightx_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        if path:
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            messagebox.showinfo("Saved", f"Chart saved to {path}")

    # ── Report ─────────────────────────────────────────────────────────────────
    def _render_report_block(self, parent, report_text):
        container = tk.Frame(parent, bg=BG_CARD,
                             highlightbackground=ACCENT_BLUE, highlightthickness=1)
        container.pack(anchor="w", pady=(3,1), fill="x", padx=1)
        title_bar = tk.Frame(container, bg=BG_HOVER)
        title_bar.pack(fill="x")
        tk.Label(title_bar, text="📋  ANALYTICAL REPORT", font=("Consolas",8,"bold"),
                 fg=ACCENT_BLUE, bg=BG_HOVER, padx=10, pady=5).pack(side="left")
        for lbl, cmd in [("⎘ Copy", lambda t=report_text: self._clip(t)),
                         ("⬇ Save", lambda t=report_text: self._save_report(t))]:
            b = tk.Label(title_bar, text=lbl, font=("Consolas",7),
                         fg=TEXT_MUTED, bg=BG_HOVER, padx=6, cursor="hand2")
            b.pack(side="right")
            b.bind("<Button-1>", lambda e, c=cmd: c())
            b.bind("<Enter>", lambda e, w=b: w.configure(fg=ACCENT_TEAL))
            b.bind("<Leave>", lambda e, w=b: w.configure(fg=TEXT_MUTED))
        tw = tk.Text(container, font=FONT_MONO, fg=TEXT_PRIMARY, bg=BG_CARD,
                     relief="flat", bd=0,
                     height=min(report_text.count("\n")+1, 30),
                     wrap="none", state="normal", padx=12, pady=7,
                     selectbackground=ACCENT_BLUE, selectforeground=BG_DEEP)
        tw.insert("1.0", report_text)
        tw.pack(fill="x")
        self._prop_scroll(tw)

    def _save_report(self, text):
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text file","*.txt"),("All files","*.*")],
            initialfile=f"insightx_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        if path:
            with open(path,"w") as f: f.write(text)
            messagebox.showinfo("Saved", f"Report saved to {path}")

    # ── Typewriter ─────────────────────────────────────────────────────────────
    def _typewrite(self, tw, text):
        # Fast typewriter: chunks of characters instead of one at a time
        chunk = max(1, len(text) // 80)
        def writer(i=0):
            if i <= len(text):
                tw.configure(state="normal")
                tw.delete("1.0","end")
                tw.insert("1.0", text[:i])
                tw.configure(state="disabled")
                lc = int(tw.index("end-1c").split(".")[0])
                # No height cap — grow to fit ALL content
                tw.configure(height=max(2, lc))
                if i < len(text):
                    tw.after(int(TYPEWRITER_DELAY*1000), writer, min(i+chunk, len(text)))
                else:
                    self._scroll_bottom()
        writer()

    def _clip(self, text):
        self.clipboard_clear(); self.clipboard_append(text); self.update()

    def _scroll_bottom(self):
        self._chat_canvas.update_idletasks()
        self._chat_canvas.yview_moveto(1.0)

    def _set_status(self, text, color):  self.status_label.configure(text=text, fg=color)
    def _set_thinking(self, text):       self.thinking_label.configure(text=text)

    # ── Tab Switch ─────────────────────────────────────────────────────────────
    def _switch_tab(self, name):
        self._active_tab = name
        self.header_title.configure(text=f"InsightX — {name}")
        self._render_tab_list()
        self._rebuild_chat_for_tab(name)

    def _rebuild_chat_for_tab(self, tab):
        for w in self._chat_inner.winfo_children(): w.destroy()
        rich = engine.get_rich(tab)
        if not rich:
            self._add_system_msg("New conversation started.")
            return
        for entry in rich:
            if entry.get("role") == "user":
                self._add_user_msg(entry["content"])
            else:
                self._render_result_static({
                    "text":            entry.get("content",""),
                    "table":           entry.get("table"),
                    "chart":           entry.get("chart"),
                    "report":          entry.get("report"),
                    "summary":         entry.get("summary"),
                    "recommendations": entry.get("recommendations"),
                })
        self._scroll_bottom()

    def _new_tab(self):
        name = f"Chat {len(engine.list_tabs())+1}"
        engine.create_tab(name)
        self._render_tab_list()
        self._switch_tab(name)

    def _delete_tab(self, name):
        engine.delete_tab(name)
        self._render_tab_list()
        if self._active_tab == name:
            self._switch_tab("Default")
