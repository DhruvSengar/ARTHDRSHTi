# =============================================================================
#  ARTH DRISHTI — India Economic Analytics
#  arth_drishti_ui.py  |  Tkinter Front-End  (v4 — Pure Descriptive)
#
#  Changes vs v3:
#   • All forecasting language removed (no Holt's, ARIMA, 2029, violet CI)
#   • Welcome panel: "2000 vs 2024 Snapshot" replaces forecast dashboard
#   • show_forecast → show_trend_analysis  (historical decomposition only)
#   • Every stats panel rebuilt around detailed_stats(), compute_cagr(),
#     rolling_volatility(), and get_economic_insights()
#   • "Trend Forecast" menu item → "📉  Historical Trends"
#   • "forecast" tag repurposed as "stat" (teal, for computed stats)
#   • Data Tables panel: clearly describes 3 separate chart windows
#   • Features panel: updated feature names (Savings_Investment_Balance etc.)
#   • Shock explanations from ma.SHOCKS injected into GDP & CPI panels
#   • ACCENT_FC repurposed: teal accent for derived statistics
# =============================================================================

import tkinter as tk
from tkinter import scrolledtext
import sys
import io

import main_analysis as ma

# ── Colour palette ─────────────────────────────────────────────────────────────
BG_PAGE   = "#FFFFFF"
BG_SIDE   = "#F1F5F9"
BG_HEADER = "#F8FAFC"
ACCENT    = "#2563EB"   # blue  — titles, active menu
ACCENT2   = "#16A34A"   # green — section headers, status dot
ACCENT_ST = "#0D9488"   # teal  — computed statistics (replaces forecast violet)
TEXT_HI   = "#0F172A"
TEXT_MID  = "#334155"
TEXT_LO   = "#64748B"
BORDER    = "#CBD5E1"
BTN_HOV   = "#DBEAFE"
BTN_ACT   = "#BFDBFE"

FONT_BRAND = ("Georgia", 17, "bold")
FONT_TITLE = ("Georgia", 13, "bold")
FONT_MENU  = ("Helvetica", 10, "bold")
FONT_MONO  = ("Courier New", 10)
FONT_BODY  = ("Helvetica", 10)
FONT_SMALL = ("Helvetica", 9)
FONT_H2    = ("Georgia", 11, "bold")


# ── Navigation menu ────────────────────────────────────────────────────────────
def _menu_items(app):
    return [
        ("📈  GDP Analysis",           app.show_gdp),
        ("📊  GDP / GNI / NNI",        app.show_gdp_gni_nni),
        ("🔥  Inflation Analysis",     app.show_inflation),
        ("⚖️  Trade Balance",          app.show_trade),
        ("💰  Savings & Investment",   app.show_savings),
        ("🏙️  Population",            app.show_population),
        ("📉  Historical Trends",      app.show_trend_analysis),
        ("🔗  Correlation Heatmap",    app.show_heatmap),
        ("📅  Decade Comparison",      app.show_decade_comparison),
        ("📐  Normalised Trends",      app.show_normalised_comparison),
        ("🏛️  Fiscal & Household",    app.show_fiscal_household),
        ("📋  Data Tables",            app.show_data_tables),
        ("🔭  Features & PCA/LDA",     app.show_features_pca),
        ("💡  Economic Insights",      app.show_insights),
        ("🚪  Exit",                   app.quit_app),
    ]


# =============================================================================
#  Main application class
# =============================================================================

class ArthDrishtiApp:

    def __init__(self, root):
        self.root = root
        self._build_window()
        self._build_sidebar()
        self._build_main_panel()
        self._show_welcome()

    # =========================================================================
    #  Window & Layout builders
    # =========================================================================

    def _build_window(self):
        self.root.title("Arth Drishti — India Economic Analytics")
        self.root.configure(bg=BG_PAGE)
        self.root.geometry("1140x760")
        self.root.minsize(900, 560)
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth()  - 1140) // 2
        y = (self.root.winfo_screenheight() - 760)  // 2
        self.root.geometry(f"+{x}+{y}")

    def _build_sidebar(self):
        self.sidebar = tk.Frame(self.root, bg=BG_SIDE, width=240)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        # Brand block
        tk.Label(self.sidebar, text="अर्थ दृष्टि",
                 font=("Georgia", 14, "bold"), bg=BG_SIDE, fg=ACCENT
                 ).pack(pady=(16, 1))
        tk.Label(self.sidebar, text="Arth Drishti",
                 font=FONT_BRAND, bg=BG_SIDE, fg=TEXT_HI
                 ).pack()
        tk.Label(self.sidebar, text="India Economic Analytics",
                 font=("Helvetica", 9, "italic"), bg=BG_SIDE, fg=TEXT_LO
                 ).pack(pady=(1, 8))

        tk.Frame(self.sidebar, bg=BORDER, height=1).pack(fill=tk.X, padx=14, pady=4)
        tk.Label(self.sidebar, text="NAVIGATION",
                 font=("Helvetica", 7, "bold"), bg=BG_SIDE, fg=TEXT_LO
                 ).pack(anchor=tk.W, padx=18, pady=(0, 3))

        # Menu buttons
        self._btn_refs   = []
        self._active_btn = None

        for label, handler in _menu_items(self):
            btn = tk.Button(
                self.sidebar, text=label, font=FONT_MENU,
                bg=BG_SIDE, fg=TEXT_MID,
                activebackground=BTN_HOV, activeforeground=TEXT_HI,
                relief=tk.FLAT, anchor=tk.W, padx=14, pady=5,
                cursor="hand2",
                command=lambda h=handler: self._handle_menu(h),
            )
            btn.pack(fill=tk.X, padx=8, pady=1)
            btn.bind("<Enter>", lambda e, b=btn:
                     b.config(bg=BTN_HOV) if b is not self._active_btn else None)
            btn.bind("<Leave>", lambda e, b=btn:
                     b.config(bg=BTN_ACT if b is self._active_btn else BG_SIDE))
            self._btn_refs.append(btn)

        # Exit button styled red
        self._btn_refs[-1].config(fg="#DC2626")

        # Status bar at bottom of sidebar
        sf = tk.Frame(self.sidebar, bg=BG_SIDE)
        sf.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=10)
        tk.Frame(sf, bg=BORDER, height=1).pack(fill=tk.X, pady=(0, 5))
        self.status_dot = tk.Label(sf, text="●", font=("Helvetica", 9),
                                   bg=BG_SIDE, fg=ACCENT2)
        self.status_dot.pack(side=tk.LEFT)
        self.status_lbl = tk.Label(sf, text="Data loaded",
                                   font=FONT_SMALL, bg=BG_SIDE, fg=TEXT_LO)
        self.status_lbl.pack(side=tk.LEFT, padx=3)

    def _build_main_panel(self):
        self.main = tk.Frame(self.root, bg=BG_PAGE)
        self.main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Top header bar
        header = tk.Frame(self.main, bg=BG_HEADER, pady=10)
        header.pack(fill=tk.X)
        self.header_title = tk.Label(
            header, text="Welcome",
            font=FONT_TITLE, bg=BG_HEADER, fg=TEXT_HI)
        self.header_title.pack(side=tk.LEFT, padx=20)
        tk.Frame(self.main, bg=ACCENT, height=2).pack(fill=tk.X)

        # Scrollable text area
        content = tk.Frame(self.main, bg=BG_PAGE)
        content.pack(fill=tk.BOTH, expand=True, padx=18, pady=14)

        self.text_area = scrolledtext.ScrolledText(
            content, font=FONT_MONO,
            bg=BG_PAGE, fg=TEXT_MID,
            insertbackground=TEXT_HI,
            relief=tk.FLAT, borderwidth=0,
            padx=12, pady=12,
            state=tk.DISABLED, wrap=tk.WORD,
        )
        self.text_area.pack(fill=tk.BOTH, expand=True)

        # Text tags for semantic colouring
        self.text_area.tag_configure("title",   foreground=ACCENT,    font=FONT_TITLE)
        self.text_area.tag_configure("section", foreground=ACCENT2,   font=("Helvetica", 10, "bold"))
        self.text_area.tag_configure("key",     foreground="#7C3AED", font=("Courier New", 10, "bold"))
        self.text_area.tag_configure("stat",    foreground=ACCENT_ST, font=("Courier New", 10, "bold"))
        self.text_area.tag_configure("muted",   foreground=TEXT_LO,   font=FONT_SMALL)
        self.text_area.tag_configure("warn",    foreground="#DC2626", font=("Courier New", 9, "bold"))
        self.text_area.tag_configure("mono",    foreground=TEXT_MID,  font=FONT_MONO)
        self.text_area.tag_configure("up",      foreground="#16A34A", font=("Courier New", 10, "bold"))
        self.text_area.tag_configure("down",    foreground="#DC2626", font=("Courier New", 10, "bold"))
        self.text_area.tag_configure("h2",      foreground=ACCENT,    font=FONT_H2)
        self.text_area.tag_configure("shock",   foreground="#EA580C", font=("Helvetica", 9, "italic"))

        # Bottom status bar
        bar = tk.Frame(self.main, bg=BG_HEADER, pady=5)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Frame(bar, bg=BORDER, height=1).pack(fill=tk.X)
        self.action_lbl = tk.Label(
            bar, text="Select a section from the left menu",
            font=FONT_BODY, bg=BG_HEADER, fg=TEXT_LO)
        self.action_lbl.pack(side=tk.LEFT, padx=14, pady=4)

    # =========================================================================
    #  Helper utilities
    # =========================================================================

    def _set_status(self, msg, colour=ACCENT2):
        self.status_lbl.config(text=msg)
        self.status_dot.config(fg=colour)

    def _update_header(self, title):
        self.header_title.config(text=title)

    def _clear_text(self):
        self.text_area.config(state=tk.NORMAL)
        self.text_area.delete("1.0", tk.END)
        self.text_area.config(state=tk.DISABLED)

    def _write(self, text, tag=None):
        """Append text to the scrollable panel, optionally styled."""
        self.text_area.config(state=tk.NORMAL)
        if tag:
            self.text_area.insert(tk.END, text, tag)
        else:
            self.text_area.insert(tk.END, text)
        self.text_area.config(state=tk.DISABLED)
        self.text_area.see(tk.END)

    def _divider(self):
        self._write("  " + "─" * 58 + "\n")

    def _kv(self, label: str, value: str, val_tag: str = "key"):
        """Write a labelled key-value line: '  label  :  value'."""
        self._write(f"  {label:<22}:  ", "mono")
        self._write(f"{value}\n", val_tag)

    def _highlight_active(self, index: int):
        for i, btn in enumerate(self._btn_refs):
            if i == index:
                btn.config(bg=BTN_ACT, fg=ACCENT)
                self._active_btn = btn
            elif i == len(self._btn_refs) - 1:
                btn.config(bg=BG_SIDE, fg="#DC2626")
            else:
                btn.config(bg=BG_SIDE, fg=TEXT_MID)

    def _handle_menu(self, handler):
        items = _menu_items(self)
        for i, (_, h) in enumerate(items):
            if h == handler:
                self._highlight_active(i)
                break
        handler()

    def _launch_chart(self, plot_fn, after_fn=None):
        """
        Schedule a chart function to run after a short delay so the UI
        can update the status indicator before blocking on matplotlib.
        Captures stdout so console output can be shown in the text panel.
        """
        self._set_status("Opening chart…", "#F59E0B")
        buf = io.StringIO()

        def _run():
            old = sys.stdout
            sys.stdout = buf
            result = None
            try:
                result = plot_fn()
            except Exception as exc:
                sys.stdout = old
                self._write(f"\n⚠  Error: {exc}\n", "warn")
            finally:
                sys.stdout = old
            self._set_status("Ready", ACCENT2)
            if after_fn:
                after_fn(result, buf.getvalue())

        self.root.after(50, _run)

    def _write_shock_notes(self):
        """Write the standard 2008 / 2020 economic shock explanations."""
        self._write("  Historical Shocks\n", "section")
        for yr, (short, desc) in ma.SHOCKS.items():
            self._write(f"  {yr} — {short}\n", "warn")
            for line in desc.split("\n"):
                self._write(f"  {line.strip()}\n", "shock")
        self._write("\n")

    def _write_detailed_stats(self, indicator: str, label: str,
                               scale: float = 1.0, unit: str = ""):
        """
        Pull detailed_stats() for one indicator and write a formatted block.
        """
        d = ma.detailed_stats(indicator, scale)
        if not d:
            return
        cagr_str = f"{d['cagr']*100:.2f}%" if d["cagr"] is not None else "n/a"
        self._write("  Descriptive Statistics\n", "section")
        self._kv("Mean",              f"{d['mean']:.3f} {unit}")
        self._kv("Median",            f"{d['median']:.3f} {unit}")
        self._kv("Std Deviation",     f"{d['std']:.3f} {unit}")
        self._kv("Min",               f"{d['min']:.3f} {unit}")
        self._kv("Max",               f"{d['max']:.3f} {unit}")
        self._kv("Skewness",          f"{d['skewness']:.3f}", "stat")
        self._kv("Kurtosis (excess)", f"{d['kurtosis']:.3f}", "stat")
        self._kv("Total change",      f"{d['total_change_pct']:.1f}%",  "stat")
        if d["cagr"] is not None:
            self._kv("CAGR (2000–2024)", cagr_str, "stat")
        self._kv("Avg volatility",    f"{d['volatility_avg']:.4f}  (roll-3 std)", "stat")
        self._write("\n")

    # =========================================================================
    #  WELCOME PAGE
    # =========================================================================

    def _show_welcome(self):
        self._clear_text()
        self._update_header("Arth Drishti — India Economic Analytics")

        self._write("\n  Welcome to Arth Drishti\n", "title")
        self._divider()
        self._write(
            "  A college-level economic analytics project built on\n"
            "  World Bank open data for India (2000 – 2024).\n"
            "  All analysis is purely descriptive — historical data only.\n\n"
        )

        # Dataset overview
        self._write("  Dataset Summary\n", "section")
        self._write(ma.summary_stats() + "\n\n")

        # 2000 → 2024 snapshot table (replaces the old forecast dashboard)
        self._write("  2000 → 2024 Snapshot\n", "section")
        self._divider()
        try:
            rows = ma.get_welcome_forecasts()   # returns (label, v2000, v2024, unit)
            self._write(
                f"  {'Indicator':<14} {'2000':>14}  {'2024':>14}  {'Change':>9}  Unit\n",
                "mono")
            self._write("  " + "-" * 62 + "\n", "muted")
            for label, v2000, v2024, unit in rows:
                delta = v2024 - v2000
                sign  = "▲" if delta >= 0 else "▼"
                tag   = "up" if delta >= 0 else "down"
                pct   = f"{delta/abs(v2000)*100:+.1f}%" if v2000 != 0 else "  n/a"
                self._write(f"  {label:<14} ", "mono")
                self._write(f"{v2000:>14.3f}  ", "muted")
                self._write(f"{v2024:>14.3f}", "key")
                self._write(f"  {sign} {pct:>7}  {unit}\n", tag)
        except Exception as e:
            self._write(f"  (snapshot load error: {e})\n", "warn")

        self._write("\n")
        self._divider()

        # Section index
        self._write("  Available Sections\n", "section")
        for label, _ in _menu_items(self)[:-1]:
            self._write(f"  {label}\n", "key")

        self._write("\n")
        self._write("  📌 Click any option on the left to explore.\n", "muted")
        self._write(
            "  ■ Teal values = computed statistics  "
            "│  Purple = raw data values\n\n", "muted")

    # =========================================================================
    #  GDP ANALYSIS
    # =========================================================================

    def show_gdp(self):
        self._clear_text()
        self._update_header("📈  GDP Analysis")
        self._write("\n  GDP Analysis\n", "title")
        self._divider()

        self._write("  What is GDP?\n", "section")
        self._write(
            "  Gross Domestic Product is the total market value of all goods\n"
            "  and services produced within India's borders in a given year.\n"
            "  It is the primary measure of economic size and activity.\n\n"
        )

        # Pull data
        gdp   = ma.get_series("GDP – Gross Domestic Product (USD)") / 1e12
        gr    = ma.get_series("GDP Growth Rate (Annual %)")
        gdppc = ma.get_series("GDP per Capita (USD)")
        cagr  = ma.compute_cagr("GDP – Gross Domestic Product (USD)", 1e12)

        # Historical key facts
        self._write("  Key Historical Facts\n", "section")
        self._kv("GDP in 2000",      f"${gdp.iloc[0]:.2f} Trillion")
        self._kv("GDP in 2024",      f"${gdp.iloc[-1]:.2f} Trillion")
        self._kv("CAGR 2000–2024",   f"{cagr*100:.2f}%", "stat")
        self._kv("Peak growth year", f"{gr.max():.2f}%  ({ma._YEAR_PRESENT[int(gr.values.argmax())]})")
        self._kv("COVID-19 trough",  f"{gr.min():.2f}%  (2020)", "down")
        self._kv("GDP per capita '00", f"${gdppc.iloc[0]:,.0f}")
        self._kv("GDP per capita '24", f"${gdppc.iloc[-1]:,.0f}")
        self._write("\n")

        # Full descriptive stats
        self._write_detailed_stats(
            "GDP – Gross Domestic Product (USD)", "GDP", 1e12, "T USD")

        # Shock context
        self._write_shock_notes()

        # Chart guide
        self._write("  Chart Panels\n", "section")
        self._write(
            "  ① GDP level (Trillion USD) with shock markers\n"
            "  ② Annual growth rate — green bars positive, red negative,\n"
            "     orange dashed line = historical mean growth\n"
            "  ③ GDP per Capita (USD) — wealth per person over time\n\n"
        )
        self._launch_chart(ma.plot_gdp_analysis)

    # =========================================================================
    #  GDP / GNI / NNI COMPARISON
    # =========================================================================

    def show_gdp_gni_nni(self):
        self._clear_text()
        self._update_header("📊  GDP / GNI / NNI Comparison")
        self._write("\n  GDP vs GNI vs NNI — Multi-Aggregate Comparison\n", "title")
        self._divider()

        self._write("  Key Definitions\n", "section")
        self._write(
            "  GDP  = Total output produced within India's geographic borders.\n"
            "  GNI  = GDP + net factor income earned abroad\n"
            "         (remittances, foreign investment returns, etc.).\n"
            "  NNI  = GNI − capital consumption (depreciation of assets).\n"
            "         The most accurate measure of sustainable national income.\n\n"
        )

        self._write("  Why compare all three?\n", "section")
        self._write(
            "  The GNI − GDP gap reveals whether India is a net earner or\n"
            "  net payer to the rest of the world. A widening gap signals\n"
            "  growing integration with global capital markets.\n"
            "  NNI filters out capital that is merely replacing worn-out assets,\n"
            "  giving the truest picture of new wealth created each year.\n\n"
        )

        # 2024 snapshot
        gdp = ma.get_series("GDP – Gross Domestic Product (USD)") / 1e12
        gni = ma.get_series("GNI – Gross National Income (USD)")  / 1e12
        nni = ma.get_series("NNI – Adj. Net National Income (USD)") / 1e12

        self._write("  2024 Values\n", "section")
        self._kv("GDP",           f"${gdp.iloc[-1]:.2f} T")
        self._kv("GNI",           f"${gni.iloc[-1]:.2f} T")
        self._kv("NNI (Adj.)",    f"${nni.iloc[-1]:.2f} T")
        self._kv("GNI / GDP",     f"{gni.iloc[-1]/gdp.iloc[-1]*100:.1f}%", "stat")
        self._kv("NNI / GDP",     f"{nni.iloc[-1]/gdp.iloc[-1]*100:.1f}%", "stat")
        self._write("\n")

        self._write("  Chart Panels\n", "section")
        self._write(
            "  ① Absolute levels (USD Trillion) — overlaid lines\n"
            "  ② Year-on-year % change for each aggregate\n"
            "  ③ GNI & NNI as % of GDP — income retained in India\n\n"
        )
        self._launch_chart(ma.plot_gdp_gni_nni_comparison)

    # =========================================================================
    #  INFLATION ANALYSIS
    # =========================================================================

    def show_inflation(self):
        self._clear_text()
        self._update_header("🔥  Inflation Analysis")
        self._write("\n  CPI Inflation Analysis\n", "title")
        self._divider()

        self._write("  What is CPI?\n", "section")
        self._write(
            "  The Consumer Price Index tracks the average change in prices\n"
            "  paid by households for a basket of goods and services.\n"
            "  High CPI erodes purchasing power; the RBI targets 4% (±2%).\n\n"
        )

        # Full descriptive stats
        self._write_detailed_stats(
            "CPI Inflation – Consumer Prices (Annual %)", "CPI", 1.0, "%")

        # RBI context
        cpi = ma.get_series("CPI Inflation – Consumer Prices (Annual %)")
        years_above_6 = sum(1 for v in cpi.values if v > 6)
        years_below_4 = sum(1 for v in cpi.values if v < 4)

        self._write("  Policy Context\n", "section")
        self._kv("RBI comfort target", "4%")
        self._kv("RBI upper bound",    "6%")
        self._kv("Years above 6%",     str(years_above_6), "down")
        self._kv("Years below 4%",     str(years_below_4), "up")
        self._write("\n")

        # Shock notes
        self._write_shock_notes()

        self._write("  Chart Panels\n", "section")
        self._write(
            "  ① CPI trend with RBI target bands (4% and 6%)\n"
            "  ② Distribution histogram — shape, mean, spread\n"
            "  ③ Boxplot with skewness and kurtosis annotation\n\n"
        )
        self._launch_chart(ma.plot_inflation_analysis)

    # =========================================================================
    #  TRADE BALANCE
    # =========================================================================

    def show_trade(self):
        self._clear_text()
        self._update_header("⚖️  Trade Balance")
        self._write("\n  Trade Balance Analysis\n", "title")
        self._divider()

        self._write("  What is Trade Balance?\n", "section")
        self._write(
            "  Trade Balance = Exports − Imports (as % of GDP).\n"
            "  A negative balance (deficit) means India imports more than\n"
            "  it exports — structurally driven by oil, gold, and electronics.\n\n"
        )

        tb  = ma.get_series("Trade Balance (Exports minus Imports %)")
        exp = ma.get_series("Exports of Goods & Services (% of GDP)")
        imp = ma.get_series("Imports of Goods & Services (% of GDP)")

        self._write("  Historical Statistics\n", "section")
        self._kv("Avg Trade Balance", f"{tb.mean():.2f}% of GDP")
        self._kv("Min Balance",       f"{tb.min():.2f}%  ({ma._YEAR_PRESENT[int(tb.values.argmin())]})", "down")
        self._kv("Max Balance",       f"{tb.max():.2f}%  ({ma._YEAR_PRESENT[int(tb.values.argmax())]})", "up")
        self._kv("Avg Exports",       f"{exp.mean():.2f}% of GDP")
        self._kv("Avg Imports",       f"{imp.mean():.2f}% of GDP")
        self._kv("Exports 2000",      f"{exp.iloc[0]:.1f}%")
        self._kv("Exports 2024",      f"{exp.iloc[-1]:.1f}%", "stat")
        self._kv("Imports 2000",      f"{imp.iloc[0]:.1f}%")
        self._kv("Imports 2024",      f"{imp.iloc[-1]:.1f}%", "stat")
        self._write("\n")

        self._write("  Structural Note\n", "section")
        self._write(
            "  India's persistent trade deficit reflects its heavy dependence\n"
            "  on imported crude oil (≈ 85% of domestic demand) and gold.\n"
            "  Export diversification — software services, pharma, textiles —\n"
            "  has partially offset this but the deficit remains.\n\n"
        )

        self._write("  Chart Panels\n", "section")
        self._write(
            "  ① Exports vs Imports overlay — green/red fill shows surplus/deficit\n"
            "  ② Trade Balance bars — mean line for structural reference\n\n"
        )
        self._launch_chart(ma.plot_trade_balance)

    # =========================================================================
    #  SAVINGS & INVESTMENT
    # =========================================================================

    def show_savings(self):
        self._clear_text()
        self._update_header("💰  Savings & Investment")
        self._write("\n  Savings & Investment\n", "title")
        self._divider()

        self._write("  Why does this matter?\n", "section")
        self._write(
            "  High domestic savings provide a pool of capital that can be\n"
            "  channelled into investment without relying on foreign borrowing.\n"
            "  Gross Capital Formation (investment) drives the physical capacity\n"
            "  of the economy and is a key long-run GDP growth driver.\n\n"
        )

        sav = ma.get_series("Gross Savings Rate (% of GDP)")
        inv = ma.get_series("Gross Capital Formation / Investment (% of GDP)")
        d_sav = ma.detailed_stats("Gross Savings Rate (% of GDP)")
        d_inv = ma.detailed_stats("Gross Capital Formation / Investment (% of GDP)")

        self._write("  Savings Statistics\n", "section")
        self._kv("Mean savings rate",  f"{d_sav['mean']:.2f}% of GDP")
        self._kv("Peak savings",       f"{d_sav['max']:.2f}%  ({ma._YEAR_PRESENT[int(sav.values.argmax())]})")
        self._kv("2024 savings rate",  f"{sav.iloc[-1]:.2f}%", "stat")
        self._kv("Volatility (roll-3)", f"{d_sav['volatility_avg']:.3f}", "stat")
        self._write("\n")

        self._write("  Investment Statistics\n", "section")
        self._kv("Mean investment",    f"{d_inv['mean']:.2f}% of GDP")
        self._kv("Peak investment",    f"{d_inv['max']:.2f}%  ({ma._YEAR_PRESENT[int(inv.values.argmax())]})")
        self._kv("2024 investment",    f"{inv.iloc[-1]:.2f}%", "stat")
        self._kv("Savings–Inv gap '24",
                 f"{sav.iloc[-1] - inv.iloc[-1]:+.2f}% of GDP", "stat")
        self._write("\n")

        self._write("  Interpretation\n", "section")
        gap = sav.iloc[-1] - inv.iloc[-1]
        if gap > 0:
            self._write(
                f"  India currently saves more than it invests ({gap:+.1f}% GDP surplus).\n"
                "  This domestic surplus can support investment without external capital.\n\n"
            )
        else:
            self._write(
                f"  India's investment exceeds savings ({gap:+.1f}% GDP gap).\n"
                "  The shortfall is met by foreign capital inflows (FDI, FII).\n\n"
            )

        self._write("  Chart Panels\n", "section")
        self._write(
            "  ① Grouped bars — savings vs investment side by side each year\n"
            "  ② Scatter plot with OLS trend line — correlation between\n"
            "     savings and investment (Pearson r annotated)\n\n"
        )
        self._launch_chart(ma.plot_savings_investment)

    # =========================================================================
    #  POPULATION & URBANISATION
    # =========================================================================

    def show_population(self):
        self._clear_text()
        self._update_header("🏙️  Population & Urbanisation")
        self._write("\n  Population & Urbanisation\n", "title")
        self._divider()

        self._write("  Context\n", "section")
        self._write(
            "  India surpassed China in 2023 to become the world's most\n"
            "  populous nation. Rapid urbanisation is restructuring the economy\n"
            "  towards services, manufacturing, and formal employment.\n\n"
        )

        pop  = ma.get_series("Total Population") / 1e9
        urb  = ma.get_series("Urban Population (% of Total)")
        cagr = ma.compute_cagr("Total Population", 1e9)
        vol  = ma.rolling_volatility("Urban Population (% of Total)", window=3)

        self._write("  Population\n", "section")
        self._kv("2000",              f"{pop.iloc[0]:.3f} Billion")
        self._kv("2024",              f"{pop.iloc[-1]:.3f} Billion")
        self._kv("CAGR 2000–2024",    f"{cagr*100:.2f}%", "stat")
        self._kv("Total increase",    f"{(pop.iloc[-1]-pop.iloc[0]):.3f} Billion people", "stat")
        self._write("\n")

        self._write("  Urbanisation\n", "section")
        self._kv("Urban share 2000",  f"{urb.iloc[0]:.1f}%")
        self._kv("Urban share 2024",  f"{urb.iloc[-1]:.1f}%")
        self._kv("Percentage point Δ",f"+{urb.iloc[-1]-urb.iloc[0]:.1f} pp", "stat")
        self._kv("Avg volatility",    f"{float(vol.mean()):.4f}  (roll-3 std)", "stat")
        self._write("\n")

        self._write("  Demographic Dividend Note\n", "section")
        self._write(
            "  India's large and young population can drive productivity if\n"
            "  education, healthcare, and jobs keep pace — a demographic dividend.\n"
            "  Urbanisation accelerates this by concentrating labour in\n"
            "  high-productivity sectors (services, manufacturing).\n\n"
        )

        self._write("  Chart\n", "section")
        self._write(
            "  Dual-axis: bars = total population (Bn, left axis);\n"
            "             line = urban share % (right axis);\n"
            "             shaded band = ±1 rolling-3 std of urban share.\n"
            "  2023 crossover with China marked.\n\n"
        )
        self._launch_chart(ma.plot_population)

    # =========================================================================
    #  HISTORICAL TREND ANALYSIS  (replaces old "Trend Forecast" section)
    # =========================================================================

    def show_trend_analysis(self):
        self._clear_text()
        self._update_header("📉  Historical Trend Analysis")
        self._write("\n  GDP Historical Trend Decomposition\n", "title")
        self._divider()

        self._write("  What this section shows\n", "section")
        self._write(
            "  A purely historical analysis of India's GDP trajectory.\n"
            "  No forecasting is performed. Instead we decompose the 25-year\n"
            "  record into structural trends, decade averages, and volatility\n"
            "  bands to understand how growth has evolved over time.\n\n"
        )

        self._write("  Analytical Methods\n", "section")
        self._write(
            "  1. 5-Year Rolling Mean  — smooths short-term noise to reveal\n"
            "     the underlying trend direction of GDP and growth rate.\n\n"
            "  2. ±1 Rolling-3 Std Band  — volatility envelope showing how\n"
            "     stable or turbulent each period was around its trend.\n\n"
            "  3. Decade Average Lines  — structural benchmarks that capture\n"
            "     how each 10-year era performed relative to the others.\n\n"
        )

        self._write("  Decade Averages (GDP Growth %)\n", "section")

        # Pull the results by capturing stdout from plot_forecasting
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            result = ma.plot_forecasting()
        finally:
            sys.stdout = old

        if result:
            self._kv("2000s avg growth", f"{result['d2000s_gr']:.2f}%", "stat")
            self._kv("2010s avg growth", f"{result['d2010s_gr']:.2f}%", "stat")
            self._kv("2020s avg growth", f"{result['d2020s_gr']:.2f}%", "stat")
        self._write("\n")

        # Console log from plot_forecasting
        log = buf.getvalue().strip()
        if log:
            self._write("  Console Output\n", "section")
            self._write(log + "\n\n", "mono")

        self._write_shock_notes()

        self._write("  Chart Panels\n", "section")
        self._write(
            "  ① GDP level with 5-year rolling mean overlay\n"
            "     — deviation fill shows gap between actual and trend\n"
            "  ② Annual growth rate bars + rolling mean + volatility band\n"
            "     + decade average horizontal lines\n\n"
        )

    # =========================================================================
    #  CORRELATION HEATMAP
    # =========================================================================

    def show_heatmap(self):
        self._clear_text()
        self._update_header("🔗  Correlation Heatmap")
        self._write("\n  Correlation Heatmap\n", "title")
        self._divider()

        self._write("  How to read this chart\n", "section")
        self._write(
            "  Each cell shows the Pearson r between two indicators over\n"
            "  the full 2000–2024 period.  Range: −1 to +1.\n\n"
            "  r > +0.6  = strong positive  — they rise and fall together\n"
            "  r < −0.6  = strong negative  — when one rises, the other falls\n"
            "  |r| < 0.3 = weak / no linear relationship\n\n"
        )

        self._write("  Indicators Covered\n", "section")
        self._write(
            "  GDP level, GDP Growth, CPI Inflation, Gross Savings,\n"
            "  Gross Capital Formation, Exports, Imports, Trade Balance,\n"
            "  HFCE (Household Consumption), Tax Revenue.\n\n"
        )

        self._write("  Strong correlations will be printed below after the\n"
                    "  chart renders (|r| ≥ 0.6 threshold).\n\n", "muted")

        def _after(result, log):
            if log.strip():
                self._write("  Strong Correlations Found\n", "section")
                for line in log.strip().split("\n"):
                    if "↔" in line or "correlated" in line.lower():
                        self._write(line + "\n", "stat")
                    elif line.strip():
                        self._write(line + "\n", "muted")
                self._write("\n")
            self._set_status("Ready", ACCENT2)

        self._launch_chart(ma.plot_correlation_heatmap, after_fn=_after)

    # =========================================================================
    #  DECADE COMPARISON
    # =========================================================================

    def show_decade_comparison(self):
        self._clear_text()
        self._update_header("📅  Decade-Wise Economic Comparison")
        self._write("\n  Decade-Wise Economic Comparison\n", "title")
        self._divider()

        self._write("  Three Economic Eras\n", "section")
        self._write(
            "  2000s  (2000–2009)  — IT boom, early liberalisation,\n"
            "                        high savings & investment rates.\n"
            "  2010s  (2010–2019)  — digital growth, demonetisation (2016),\n"
            "                        GST reform (2017), moderated growth.\n"
            "  2020s  (2020–2024)  — COVID shock (2020), sharp recovery,\n"
            "                        PLI schemes, infrastructure push.\n\n"
        )

        self._write("  Indicators Compared\n", "section")
        self._write(
            "  GDP Growth %, CPI Inflation %, Savings % GDP,\n"
            "  Investment % GDP, Exports % GDP, Urban Population %.\n\n"
        )

        self._write("  Chart Panels\n", "section")
        self._write(
            "  Left  → Grouped bar chart with actual values labelled\n"
            "  Right → Z-score heatmap (actual values annotated):\n"
            "          colour = which decade was highest/lowest\n"
            "          relative to each indicator's own mean.\n\n"
        )
        self._launch_chart(ma.plot_decade_comparison)

    # =========================================================================
    #  NORMALISED TRENDS
    # =========================================================================

    def show_normalised_comparison(self):
        self._clear_text()
        self._update_header("📐  Normalised Multi-Indicator Trends")
        self._write("\n  Z-Score Normalised Multi-Indicator Comparison\n", "title")
        self._divider()

        self._write("  Why normalise?\n", "section")
        self._write(
            "  Indicators operate on completely different scales:\n"
            "  GDP (trillions of USD) vs CPI (single-digit %) vs\n"
            "  Population (1.4 billion). Plotting them raw is meaningless.\n\n"
            "  Z-score transformation: z = (value − mean) / std\n"
            "  This puts every indicator on the same dimensionless axis,\n"
            "  making trend direction, shocks, and turning points comparable.\n\n"
        )

        self._write("  Panel 1 — Z-Score Time Series\n", "section")
        self._write(
            "  All 6 indicators plotted as σ deviations from their own mean.\n"
            "  z = 0 means the indicator was at its historical average.\n"
            "  The COVID shock (2020) is starkly visible across all series.\n"
            "  Shaded bands mark ±1σ (historically average range) and\n"
            "  the GFC (2007–2009) and COVID (2019–2021) shock periods.\n\n"
        )

        self._write("  Panel 2 — Rolling 5-Year Pearson Correlation\n", "section")
        self._write(
            "  For each indicator, the 5-year rolling Pearson r with\n"
            "  GDP Growth is computed and plotted over time.\n"
            "  This shows which indicators consistently co-move with\n"
            "  economic growth — and which decouple in certain periods.\n"
            "  Dashed lines at ±0.5 mark the moderate-correlation threshold.\n\n"
        )
        self._launch_chart(ma.plot_normalised_comparison)

    # =========================================================================
    #  FISCAL & HOUSEHOLD
    # =========================================================================

    def show_fiscal_household(self):
        self._clear_text()
        self._update_header("🏛️  Fiscal & Household Expenditure")
        self._write("\n  Fiscal & Household Expenditure Analysis\n", "title")
        self._divider()

        self._write("  GDP Expenditure Identity\n", "section")
        self._write(
            "  GDP = Household Consumption (HFCE)\n"
            "      + Gross Capital Formation (Investment)\n"
            "      + Government Expenditure\n"
            "      + Net Exports (Exports − Imports)\n\n"
            "  Tracking these components shows what drives India's growth\n"
            "  and how the balance has shifted over 24 years.\n\n"
        )

        hfce = ma.get_series("HFCE – Household Final Consumption (% of GDP)")
        inv  = ma.get_series("Gross Capital Formation / Investment (% of GDP)")
        tax  = ma.get_series("Tax Revenue (% of GDP)")

        self._write("  2024 Snapshot  (% of GDP)\n", "section")
        self._kv("HFCE (Consumption)",  f"{hfce.iloc[-1]:.1f}%")
        self._kv("Investment",          f"{inv.iloc[-1]:.1f}%")
        self._kv("Tax Revenue",         f"{tax.iloc[-1]:.1f}%")
        self._kv("Inv / Tax ratio",     f"{inv.iloc[-1]/tax.iloc[-1]:.2f}x", "stat")
        self._write("\n")

        self._write("  2000 Snapshot  (% of GDP)\n", "section")
        self._kv("HFCE (Consumption)",  f"{hfce.iloc[0]:.1f}%")
        self._kv("Investment",          f"{inv.iloc[0]:.1f}%")
        self._kv("Tax Revenue",         f"{tax.iloc[0]:.1f}%")
        self._write("\n")

        self._write("  Trend Note\n", "section")
        hfce_chg = hfce.iloc[-1] - hfce.iloc[0]
        inv_chg  = inv.iloc[-1]  - inv.iloc[0]
        self._write(
            f"  Household consumption share: {hfce_chg:+.1f} pp  (2000→2024)\n"
            f"  Investment share:            {inv_chg:+.1f} pp  (2000→2024)\n\n"
        )

        self._write("  Chart Panels\n", "section")
        self._write(
            "  Left  → Stacked area: HFCE / Investment / Govt (residual)\n"
            "           Govt = GDP − HFCE − Investment − Exports (approximation)\n"
            "  Right → Bar chart: Tax Revenue vs Investment side by side\n"
            "           Shows fiscal space and capital-expenditure capacity.\n\n"
        )
        self._launch_chart(ma.plot_fiscal_household_comparison)

    # =========================================================================
    #  DATA TABLES
    # =========================================================================

    def show_data_tables(self):
        self._clear_text()
        self._update_header("📋  Data Tables — Raw vs Cleaned")
        self._write("\n  Data Tables — Raw CSV vs Cleaned DataFrame\n", "title")
        self._divider()

        self._write("  Three Separate Chart Windows\n", "section")
        self._write(
            "  ① Raw CSV table  — first 10 rows, 7 columns\n"
            "     Red cells = NaN or dirty token (N/A, TBD, error, ?, --)\n\n"
            "  ② Cleaned DataFrame table  — same slice after all 7 steps\n"
            "     All values are float64; alternating green rows\n\n"
            "  ③ Quality comparison summary  — side-by-side metrics\n"
            "     (total NaNs, dirty tokens, duplicates, dtypes)\n\n"
        )

        # Text-panel info summaries
        try:
            _, info_raw = ma.get_raw_table_info()
            _, info_cln = ma.get_clean_table_info()
            self._write("  Raw CSV — Quality Metrics\n", "section")
            self._write(info_raw + "\n\n")
            self._divider()
            self._write("  Cleaned DataFrame — Quality Metrics\n", "section")
            self._write(info_cln + "\n\n")
        except Exception as e:
            self._write(f"  Error loading info: {e}\n\n", "warn")

        self._divider()
        self._write("  7-Step Cleaning Pipeline\n", "section")
        self._write(
            "  1. Remove duplicate rows  (by Category + Indicator)\n"
            "  2. Replace dirty tokens   → NaN  (N/A, TBD, error, ?, --)\n"
            "  3. Coerce year columns    → numeric float64\n"
            "  4. Fix wrong-sign values  (GDP/population cannot be negative)\n"
            "  5. Clamp percentage cols  → ±200 plausible range\n"
            "  6. IQR×3 outlier removal  per row (time-series context)\n"
            "  7. Linear interpolation   + ffill/bfill + row-mean fill\n"
            "     → zero remaining NaNs guaranteed\n\n"
        )
        self._write("  Opening 3 chart windows…\n", "muted")
        self._launch_chart(ma.plot_data_tables)

    # =========================================================================
    #  FEATURES & PCA / LDA
    # =========================================================================

    def show_features_pca(self):
        self._clear_text()
        self._update_header("🔭  Feature Engineering & Dimensionality Reduction")
        self._write("\n  Feature Engineering + PCA & LDA\n", "title")
        self._divider()

        # Phase 4 — Extraction
        self._write("  Phase 4 — Time-Series Feature Extraction\n", "section")
        self._write(
            "  For each of 8 core indicators, four derived columns are built:\n\n"
            "  • 3-year rolling mean    — smoothed trend (reduces year-to-year noise)\n"
            "  • 3-year rolling std     — local volatility (how stable each period was)\n"
            "  • Lag-1 value            — previous year's value (auto-correlation)\n"
            "  • Year-on-year change    — annual momentum / acceleration\n\n"
        )

        # Phase 5 — Engineering
        self._write("  Phase 5 — Derived Economic Ratios\n", "section")
        self._write(
            "  Savings_Investment_Balance   = Savings − Investment\n"
            "      Positive = domestic capital surplus; negative = external reliance\n\n"
            "  Inflation_Adjusted_Growth    = GDP Growth − CPI\n"
            "      Approximates real (purchasing-power) growth\n\n"
            "  Investment_Productivity      = GDP Growth / Investment Rate\n"
            "      Output generated per unit of capital invested\n\n"
            "  Trade_Openness               = (Exports + Imports) / 2\n"
            "      Degree of global economic integration\n\n"
            "  GDP_Acceleration             = year-on-year Δ in growth rate\n"
            "      Is the economy speeding up or slowing down?\n\n"
            "  Urban_Momentum               = 3-year rolling slope of urban share\n"
            "      Speed of the rural-to-urban structural shift\n\n"
            "  Export_Intensity             = Export share × GDP (USD bn)\n"
            "      Absolute scale of exports, not just the GDP share\n\n"
        )

        # Feature matrix summary from ma
        self._write("  Feature Matrix Summary\n", "section")
        self._write(ma.get_feature_summary() + "\n\n")

        # PCA explanation
        self._write("  Phase 5b — PCA  (Principal Component Analysis)\n", "section")
        self._write(
            "  PCA is an unsupervised technique that finds the directions\n"
            "  of maximum variance in the 25-year feature matrix.\n\n"
            "  PC1 — 'Growth & Scale' axis:\n"
            "      Dominated by GDP level, Investment, Savings.\n"
            "      Years with large, growing economies score high on PC1.\n\n"
            "  PC2 — 'Inflation & External' axis:\n"
            "      Loaded on CPI, Exports, Imports, Trade Openness.\n"
            "      Separates years with high inflation or trade stress.\n\n"
            "  Scree plot shows variance explained per component.\n"
            "  2D scatter: each dot = one year, coloured by decade.\n"
            "  Loadings heatmap: which features drive each component.\n\n"
        )

        # LDA explanation
        self._write("  Phase 5b — LDA  (Linear Discriminant Analysis)\n", "section")
        self._write(
            "  LDA is supervised by decade label (2000s / 2010s / 2020s).\n"
            "  It finds the linear combination of features that maximally\n"
            "  separates the three decades — purely for decade profiling.\n"
            "  Well-separated clusters confirm India's economic structure\n"
            "  changed meaningfully across each decade.\n\n"
        )

        self._write("  Computing feature matrix and running PCA/LDA…\n\n", "muted")

        def _after(result, log):
            if result:
                ev = result["pca_explained"]
                self._write("  PCA Results\n", "section")
                self._kv("PC1 variance",    f"{ev[0]*100:.1f}%", "stat")
                self._kv("PC2 variance",    f"{ev[1]*100:.1f}%", "stat")
                self._kv("PC1+PC2 total",   f"{sum(ev[:2])*100:.1f}%", "stat")
                self._kv("Total features",  str(result["n_features"]))
                self._write("\n")

                dropped = result.get("dropped_cols", [])
                self._write("  Columns Dropped  (near-zero variance)\n", "section")
                if dropped:
                    for c in dropped:
                        self._write(f"  ✗  {c}\n", "warn")
                else:
                    self._write("  None — all features retained\n", "muted")

                low_inf = result.get("low_inf_cols", [])
                self._write("\n  Low-Influence Features  (bottom 25% PC loadings)\n", "section")
                for c in low_inf:
                    self._write(f"  ↓  {c}\n", "muted")
                self._write(
                    "\n  These features contribute least to PC1 and PC2.\n"
                    "  In a production pipeline they could be pruned to\n"
                    "  reduce dimensionality without significant info loss.\n\n",
                    "muted"
                )
            # Log from console
            if log.strip():
                for line in log.strip().split("\n"):
                    self._write("  " + line + "\n", "muted")
            self._set_status("Ready", ACCENT2)

        self._launch_chart(ma.plot_pca_lda, after_fn=_after)

    # =========================================================================
    #  ECONOMIC INSIGHTS  (new section — surfaces get_economic_insights())
    # =========================================================================

    def show_insights(self):
        self._clear_text()
        self._update_header("💡  Economic Insights Summary")
        self._write("\n  Key Economic Insights — India 2000–2024\n", "title")
        self._divider()
        self._write(
            "  A synthesis of the most important findings across all\n"
            "  indicators in this dataset. Combines quantitative statistics\n"
            "  with qualitative economic context.\n\n"
        )

        self._write("  Full Descriptive Statistics Table\n", "section")
        self._write(ma.get_detailed_stats_text() + "\n\n")

        self._divider()
        self._write(ma.get_economic_insights() + "\n\n")

        self._write("  ■ Teal values = derived / computed statistics\n", "muted")
        self._write("  ■ All figures from World Bank open data (2000–2024)\n\n", "muted")

    # =========================================================================
    #  Exit
    # =========================================================================

    def quit_app(self):
        self.root.destroy()


# =============================================================================
#  Entry point
# =============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app  = ArthDrishtiApp(root)
    root.mainloop()
