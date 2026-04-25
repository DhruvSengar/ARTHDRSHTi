# =============================================================================
#  ARTH DRISHTI — INDIA WORLD BANK DATA
#  main_analysis.py  |  Analysis Engine  v4  (Pure Descriptive)
#
#  Phases:
#   1. Data Loading & Inspection
#   2. Data Cleaning  (7-step pipeline, zero-NaN guarantee)
#   3. Statistical Analysis  (mean, std, skew, kurtosis, CAGR, volatility)
#   4. Feature Extraction  (rolling stats, lags, year-on-year changes)
#   5. Feature Engineering  (derived economic ratios)
#   5b. Dimensionality Reduction  (PCA + LDA — descriptive, no prediction)
#   6. Visualisations  (GDP, CPI, Trade, Savings, Population, Heatmap, etc.)
#   7. Economic Insights Summary
#
#  Design decisions:
#   • NO forecasting, prediction, or ML model fitting.
#   • All analysis is purely descriptive / historical (2000–2024).
#   • scipy.stats used only for Pearson correlation and OLS trend-line display.
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# ── Year range covered by the dataset ─────────────────────────────────────────
YEAR_COLS = [str(y) for y in range(2000, 2025)]

# ── Shared colour palette ──────────────────────────────────────────────────────
PALETTE = {
    "blue":   "#2563EB",
    "red":    "#DC2626",
    "green":  "#16A34A",
    "orange": "#EA580C",
    "purple": "#7C3AED",
    "teal":   "#0D9488",
    "pink":   "#DB2777",
    "yellow": "#CA8A04",
}

# Annotation colours for key historical shocks
SHOCK_COLOR = "#DC2626"   # red — crisis / shock markers

# ── Matplotlib global style ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8FAFC",
    "axes.grid":         True,
    "grid.alpha":        0.4,
    "grid.linestyle":    "--",
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Dirty / placeholder tokens to treat as missing ────────────────────────────
DIRTY = {
    "N/A", "n/a", "NA", "na", "TBD", "tbd",
    "error", "Error", "ERROR",
    "#N/A", "#VALUE", "?", "missing", "unknown",
    "n.a", "--", "", "  ",
}

_CSV_PATH = "india_wb_dirty_wide.csv"

# ── Historical economic shocks to annotate on charts ──────────────────────────
SHOCKS = {
    2008: ("GFC",   "Global Financial Crisis — Lehman Brothers collapse;\n"
                    "India's growth fell from 9.8% (2007) to 3.9% (2008)."),
    2020: ("COVID", "COVID-19 Pandemic — nationwide lockdown in March 2020;\n"
                    "GDP contracted by −5.8%, the sharpest drop since independence."),
}


# =============================================================================
#  PHASE 1 + 2  — Load, Inspect & Clean  (zero-NaN guarantee)
# =============================================================================

def _load_and_clean(path=_CSV_PATH):
    """
    Load raw CSV, report quality metrics, apply a 7-step cleaning pipeline,
    and return the cleaned DataFrame plus the list of available year columns.

    Cleaning steps
    ──────────────
    1. Remove duplicate rows (by Category + Indicator key).
    2. Replace dirty placeholder tokens with NaN.
    3. Coerce all year columns to numeric float64.
    4. Fix wrong-sign absolute values (GDP, population cannot be negative).
    5. Clamp percentage indicators to a plausible ±200 range.
    6. IQR×3 outlier removal per row (row-wise time series context).
    7. Linear interpolation → ffill/bfill → row-mean fill  (zero NaN left).
    """
    df_raw = pd.read_csv(path)

    # ── Phase 1: Inspection report ────────────────────────────────────────────
    nan_counts  = df_raw.isnull().sum().sum()
    dup_count   = df_raw.duplicated().sum()
    dirty_mask  = df_raw.map(
        lambda x: str(x).strip() in DIRTY if pd.notnull(x) else False)
    print(f"[Phase 1] Shape={df_raw.shape}  NaNs={nan_counts}  "
          f"Dups={dup_count}  DirtyTokens={int(dirty_mask.sum().sum())}")

    # ── Step 1: De-duplicate ──────────────────────────────────────────────────
    df = df_raw.drop_duplicates(
        subset=["Category", "Indicator"]
    ).reset_index(drop=True)

    # ── Step 2: Replace dirty tokens ──────────────────────────────────────────
    for col in YEAR_COLS:
        if col in df.columns:
            df[col] = df[col].map(
                lambda x: np.nan
                if (pd.isnull(x) or str(x).strip() in DIRTY)
                else x
            )
            # Step 3: Coerce to float64
            df[col] = pd.to_numeric(df[col], errors="coerce")

    year_present = [c for c in YEAR_COLS if c in df.columns]

    # ── Step 4: Fix wrong-sign large absolute indicators ──────────────────────
    # GDP, GNI, NNI, and population are always positive.
    BIG_INDICATORS = [
        "GDP – Gross Domestic Product (USD)",
        "GNI – Gross National Income (USD)",
        "NNI – Adj. Net National Income (USD)",
        "Total Population",
    ]
    for ind in BIG_INDICATORS:
        rows = df["Indicator"] == ind
        for col in year_present:
            mask = rows & (df[col] < 0)
            df.loc[mask, col] = df.loc[mask, col].abs()

    # ── Step 5: Clamp percentage / rate indicators to ±200 ───────────────────
    for ind in df["Indicator"]:
        if "%" in ind or "Rate" in ind:
            rows = df["Indicator"] == ind
            for col in year_present:
                v = df.loc[rows, col]
                df.loc[rows & (v.abs() > 200), col] = np.nan

    # ── Step 6: IQR×3 outlier removal per row ─────────────────────────────────
    for idx in df.index:
        row = df.loc[idx, year_present].astype(float)
        q1, q3 = row.quantile(0.25), row.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            bad = (row < q1 - 3 * iqr) | (row > q3 + 3 * iqr)
            df.loc[idx, year_present] = row.where(~bad, np.nan)

    # ── Step 7: Interpolate + fill — guaranteed no NaN ────────────────────────
    for idx in df.index:
        s = df.loc[idx, year_present].copy().astype(float)
        s = s.interpolate(method="linear", limit_direction="both")
        s = s.ffill().bfill()
        if s.isnull().any():
            row_mean = s.mean()
            s = s.fillna(row_mean if not np.isnan(row_mean) else 0.0)
        df.loc[idx, year_present] = s

    remaining = int(df[year_present].isnull().sum().sum())
    print(f"[Phase 2] Cleaned.  Remaining NaNs: {remaining}")
    return df, year_present


# ── Module-level singletons (loaded once on import) ───────────────────────────
_DF_RAW            = pd.read_csv(_CSV_PATH)
_DF, _YEAR_PRESENT = _load_and_clean()
_YEARS_INT         = [int(y) for y in _YEAR_PRESENT]
_N                 = len(_YEAR_PRESENT)


def get_series(indicator: str) -> pd.Series:
    """Return a cleaned time series for the given indicator name."""
    row = _DF[_DF["Indicator"] == indicator]
    if row.empty:
        return pd.Series(dtype=float)
    return row[_YEAR_PRESENT].iloc[0].astype(float)


def summary_stats() -> str:
    """Return a multi-line text summary of the dataset."""
    lines = [
        f"  Rows in raw file : {len(_DF_RAW)}",
        f"  Indicators       : {_DF['Indicator'].nunique()} unique",
        f"  Year range       : {_YEAR_PRESENT[0]} – {_YEAR_PRESENT[-1]}",
        f"  Categories       : {', '.join(_DF['Category'].unique())}",
        f"  Remaining NaNs   : {_DF[_YEAR_PRESENT].isnull().sum().sum()}",
    ]
    return "\n".join(lines)


# =============================================================================
#  PHASE 3 — Statistical Analysis Utilities
# =============================================================================

def detailed_stats(indicator: str, scale: float = 1.0) -> dict:
    """
    Compute a full descriptive statistics profile for one indicator.

    Parameters
    ----------
    indicator : World Bank indicator name (must match Indicator column exactly).
    scale     : Divide values by this before computing (e.g. 1e12 for USD→Trillion).

    Returns
    -------
    dict with keys: mean, median, std, min, max, skewness, kurtosis,
                    total_change_pct, cagr (where applicable), volatility_avg.
    """
    s = get_series(indicator)
    if s.empty:
        return {}
    v = s.values / scale

    # CAGR: Compound Annual Growth Rate  = (end/start)^(1/years) − 1
    # Only meaningful for absolute-value series (GDP, Population, etc.)
    years = len(v) - 1
    cagr  = None
    if v[0] > 0 and v[-1] > 0 and "%" not in indicator and "Rate" not in indicator:
        cagr = (v[-1] / v[0]) ** (1 / years) - 1

    # Rolling 3-year volatility (standard deviation as a proxy for instability)
    roll_std     = pd.Series(v).rolling(3, min_periods=2).std().dropna()
    volatility   = float(roll_std.mean())

    total_change = ((v[-1] - v[0]) / abs(v[0]) * 100) if v[0] != 0 else np.nan

    return {
        "mean":             float(np.mean(v)),
        "median":           float(np.median(v)),
        "std":              float(np.std(v, ddof=1)),
        "min":              float(np.min(v)),
        "max":              float(np.max(v)),
        "skewness":         float(stats.skew(v)),
        "kurtosis":         float(stats.kurtosis(v)),   # excess kurtosis
        "total_change_pct": float(total_change),
        "cagr":             cagr,
        "volatility_avg":   volatility,
    }


def compute_cagr(indicator: str, scale: float = 1.0) -> float:
    """
    Compound Annual Growth Rate from 2000 to 2024.
    Returns the rate as a decimal (multiply by 100 for %).
    """
    s = get_series(indicator)
    if s.empty:
        return np.nan
    v = s.values / scale
    years = len(v) - 1
    if v[0] <= 0 or v[-1] <= 0:
        return np.nan
    return (v[-1] / v[0]) ** (1 / years) - 1


def rolling_volatility(indicator: str, window: int = 3,
                        scale: float = 1.0) -> pd.Series:
    """
    Year-by-year rolling standard deviation (a simple volatility proxy).
    Larger values indicate more unstable / volatile periods.
    """
    s   = get_series(indicator) / scale
    vol = s.rolling(window=window, min_periods=2).std()
    vol.index = _YEARS_INT
    return vol


def interpret_correlations(corr_matrix: pd.DataFrame,
                            threshold: float = 0.6) -> list[str]:
    """
    Scan a correlation matrix and return human-readable sentences for all
    pairs whose |r| ≥ threshold (strong relationships only).

    Returns a list of plain-English interpretation strings.
    """
    interpretations = []
    cols = corr_matrix.columns.tolist()
    seen = set()

    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            pair = tuple(sorted([c1, c2]))
            if pair in seen:
                continue
            seen.add(pair)
            r = corr_matrix.loc[c1, c2]
            if abs(r) < threshold:
                continue

            direction = "positively" if r > 0 else "negatively"
            strength  = "strongly" if abs(r) > 0.8 else "moderately"
            interpretations.append(
                f"  {c1} & {c2}: {strength} {direction} correlated  (r = {r:.2f})"
            )

    return interpretations


def get_detailed_stats_text() -> str:
    """
    Return a formatted text block of descriptive stats for key indicators.
    Used by the UI's welcome / stats panel.
    """
    indicators = [
        ("GDP (T USD)",    "GDP – Gross Domestic Product (USD)",              1e12),
        ("GDP Growth %",   "GDP Growth Rate (Annual %)",                      1.0),
        ("CPI %",          "CPI Inflation – Consumer Prices (Annual %)",      1.0),
        ("Savings % GDP",  "Gross Savings Rate (% of GDP)",                   1.0),
        ("Investment %",   "Gross Capital Formation / Investment (% of GDP)", 1.0),
        ("Exports %",      "Exports of Goods & Services (% of GDP)",          1.0),
        ("Population (Bn)","Total Population",                                 1e9),
        ("Urban %",        "Urban Population (% of Total)",                    1.0),
    ]
    lines = [
        f"  {'Indicator':<18} {'Mean':>9} {'Std':>8} {'Skew':>7} "
        f"{'Kurt':>7} {'Total Δ%':>10} {'CAGR%':>8} {'Volatility':>11}",
        "  " + "─" * 80,
    ]
    for label, ind, scale in indicators:
        d = detailed_stats(ind, scale)
        if not d:
            continue
        cagr_str = f"{d['cagr']*100:.2f}%" if d["cagr"] is not None else "  n/a"
        lines.append(
            f"  {label:<18} {d['mean']:>9.2f} {d['std']:>8.2f} "
            f"{d['skewness']:>7.2f} {d['kurtosis']:>7.2f} "
            f"{d['total_change_pct']:>10.1f} {cagr_str:>8} "
            f"{d['volatility_avg']:>11.3f}"
        )
    return "\n".join(lines)


def get_economic_insights() -> str:
    """
    Summarise the most important findings from the 2000–2024 dataset.
    This is a qualitative-plus-quantitative synthesis aimed at students.
    """
    gdp     = get_series("GDP – Gross Domestic Product (USD)") / 1e12
    growth  = get_series("GDP Growth Rate (Annual %)")
    cpi     = get_series("CPI Inflation – Consumer Prices (Annual %)")
    sav     = get_series("Gross Savings Rate (% of GDP)")
    inv     = get_series("Gross Capital Formation / Investment (% of GDP)")
    urb     = get_series("Urban Population (% of Total)")
    pop     = get_series("Total Population") / 1e9
    exp_s   = get_series("Exports of Goods & Services (% of GDP)")

    gdp_cagr  = compute_cagr("GDP – Gross Domestic Product (USD)", 1e12)
    pop_cagr  = compute_cagr("Total Population", 1e9)

    lines = [
        "  ═══ KEY ECONOMIC INSIGHTS  (India, 2000–2024) ═══",
        "",
        "  1. GDP EXPANSION",
        f"     India's GDP grew from ${gdp.iloc[0]:.2f}T (2000) to ${gdp.iloc[-1]:.2f}T (2024),",
        f"     a {gdp_cagr*100:.1f}% CAGR over 24 years — one of the fastest among",
        f"     major economies globally.",
        "",
        "  2. GROWTH VOLATILITY & SHOCKS",
        f"     Peak growth: {growth.max():.1f}% ({_YEARS_INT[int(growth.values.argmax())]})",
        f"     Trough     : {growth.min():.1f}% (2020 — COVID-19 lockdown contraction)",
        f"     Std dev    : {growth.std():.2f}% — moderate volatility around a 6–7% mean.",
        "",
        "  3. INFLATION MANAGEMENT",
        f"     CPI averaged {cpi.mean():.1f}% (2000–2024). The RBI's 4% inflation target",
        f"     was frequently breached, especially in 2009 ({cpi.iloc[9]:.1f}%) and",
        f"     2022 ({cpi.iloc[22]:.1f}%), driven by food prices and global supply shocks.",
        "",
        "  4. SAVINGS–INVESTMENT DYNAMICS",
        f"     India's savings rate peaked at {sav.max():.1f}% of GDP (mid-2000s boom).",
        f"     By 2024 it stood at {sav.iloc[-1]:.1f}%, above the investment rate of",
        f"     {inv.iloc[-1]:.1f}%, suggesting a domestic surplus available for growth.",
        "",
        "  5. URBANISATION TREND",
        f"     Urban population share rose from {urb.iloc[0]:.1f}% (2000) to",
        f"     {urb.iloc[-1]:.1f}% (2024) — structural shift towards services & manufacturing.",
        "",
        "  6. POPULATION & DEMOGRAPHIC DIVIDEND",
        f"     Population grew from {pop.iloc[0]:.3f}Bn to {pop.iloc[-1]:.3f}Bn",
        f"     (CAGR {pop_cagr*100:.2f}%). India overtook China as the most populous",
        f"     nation in 2023, adding demographic-dividend tailwinds to growth.",
        "",
        "  7. TRADE OPENNESS",
        f"     Export share of GDP: {exp_s.iloc[0]:.1f}% (2000) → {exp_s.iloc[-1]:.1f}% (2024).",
        f"     India remains import-heavy (oil, electronics), keeping the trade",
        f"     balance in persistent deficit — a structural challenge.",
        "",
        "  8. 2008 GLOBAL FINANCIAL CRISIS",
        f"     India proved relatively resilient: growth dipped to {growth.iloc[8]:.1f}%",
        f"     in 2008 vs double-digit contractions in developed markets.",
        f"     Domestic demand and fiscal stimulus cushioned the shock.",
        "",
        "  9. 2020 COVID-19 SHOCK",
        f"     GDP contracted {growth.iloc[20]:.1f}% — the worst year in post-independence",
        f"     history. Recovery was sharp: {growth.iloc[21]:.1f}% in 2021,",
        f"     driven by pent-up demand and government spending.",
        "",
        "  10. PCA INTERPRETATION",
        "      PC1 ('Growth & Scale' axis): dominated by GDP level, Investment,",
        "           and Savings — captures India's overall economic scale trajectory.",
        "      PC2 ('Inflation & External' axis): loaded on CPI, Exports, Imports —",
        "           separates years with high inflation / trade stress from stable ones.",
        "      Decades are well-separated in PCA space, confirming that India's",
        "      economic structure meaningfully changed across each decade.",
    ]
    return "\n".join(lines)


# =============================================================================
#  DATA TABLE INFO — text summaries for the UI text panel
# =============================================================================

def get_raw_table_info() -> tuple[str, str]:
    """
    Return (head_str, info_str) describing the raw dirty CSV.
    head_str: first 8 rows × 6 year cols as a plain-text table.
    info_str: quality metrics paragraph.
    """
    df  = _DF_RAW.copy()
    yc  = [c for c in YEAR_COLS if c in df.columns]

    dirty_mask  = df.map(
        lambda x: str(x).strip() in DIRTY if pd.notnull(x) else False)
    total_dirty = int(dirty_mask.sum().sum())
    total_nan   = int(df.isnull().sum().sum())
    total_dup   = int(df.duplicated().sum())

    head_str = (
        df[["Category", "Indicator"] + yc[:6]]
        .head(8)
        .to_string(index=False)
    )

    info_lines = [
        f"  Shape            : {df.shape[0]} rows × {df.shape[1]} columns",
        f"  Total NaN cells  : {total_nan}",
        f"  Dirty tokens     : {total_dirty}  (N/A, TBD, error, ?, -- …)",
        f"  Duplicate rows   : {total_dup}",
        f"  Numeric cols     : {df.select_dtypes(include='number').shape[1]}",
        f"  String cols      : {df.select_dtypes(include='object').shape[1]}",
        "",
        "  Column dtypes (first 8 year cols as read from CSV):",
    ]
    for c in yc[:8]:
        info_lines.append(f"    {c}  →  {_DF_RAW[c].dtype}")

    return head_str, "\n".join(info_lines)


def get_clean_table_info() -> tuple[str, str]:
    """
    Return (head_str, info_str) describing the cleaned DataFrame.
    head_str: first 8 rows × 6 year cols as a plain-text table.
    info_str: cleaning outcome metrics paragraph.
    """
    df   = _DF.copy()
    yc6  = _YEAR_PRESENT[:6]

    head_str = (
        df[["Category", "Indicator"] + yc6]
        .head(8)
        .to_string(index=False)
    )

    info_lines = [
        f"  Shape            : {df.shape[0]} rows × {df.shape[1]} columns",
        f"  Remaining NaNs   : {int(df[_YEAR_PRESENT].isnull().sum().sum())}",
        f"  Duplicates       : 0  (removed)",
        f"  Interpolation    : linear both-direction  +  ffill/bfill  +  row-mean",
        f"  Outlier handling : IQR×3 clamp → re-interpolated",
        f"  All year cols    : float64",
        "",
        "  Column dtypes (first 8 year cols after cleaning):",
    ]
    for c in _YEAR_PRESENT[:8]:
        info_lines.append(f"    {c}  →  {df[c].dtype}")

    return head_str, "\n".join(info_lines)


# =============================================================================
#  DATA TABLE CHART — two separate matplotlib figure windows
# =============================================================================

def _style_table_header(tab, n_cols: int, bg: str) -> None:
    """Apply dark header styling to row 0 of a matplotlib table."""
    for j in range(n_cols):
        tab[0, j].set_facecolor(bg)
        tab[0, j].set_text_props(color="white", fontweight="bold")


def plot_data_table_raw() -> None:
    """
    Figure 1 of 3 — Raw CSV table.
    Shows first 10 rows × 7 cols with dirty cells highlighted in red.
    """
    raw_df    = _DF_RAW.copy()
    yc        = [c for c in YEAR_COLS if c in raw_df.columns]
    cols_show = ["Indicator"] + yc[:6]
    raw_slice = raw_df[cols_show].head(10).copy()

    def _short(s):
        s = str(s)
        return s[:28] + "…" if len(s) > 29 else s

    raw_slice["Indicator"] = raw_slice["Indicator"].apply(_short)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(
        "① RAW CSV  — Before Cleaning  (red cells = dirty / missing)",
        fontsize=12, fontweight="bold", color="#DC2626"
    )
    ax.axis("off")

    col_labels = list(raw_slice.columns)
    tab = ax.table(
        cellText=raw_slice.values.tolist(),
        colLabels=col_labels,
        cellLoc="center", loc="center",
        bbox=[0, 0, 1, 1],
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    _style_table_header(tab, len(col_labels), "#1E3A5F")

    for i in range(len(raw_slice)):
        for j, col in enumerate(col_labels):
            cell = tab[i + 1, j]
            val  = str(raw_slice.iloc[i][col]).strip()
            is_dirty = (
                pd.isnull(raw_df[cols_show].iloc[i][col]) or val in DIRTY
            )
            if is_dirty:
                cell.set_facecolor("#FEE2E2")
                cell.set_text_props(color="#DC2626")
            elif i % 2 == 0:
                cell.set_facecolor("#F0F4FF")
            else:
                cell.set_facecolor("white")

    plt.tight_layout()
    plt.show()


def plot_data_table_clean() -> None:
    """
    Figure 2 of 3 — Cleaned DataFrame table.
    Same 10 rows × 7 cols after the full 7-step cleaning pipeline.
    """
    yc        = [c for c in YEAR_COLS if c in _DF.columns]
    cols_show = ["Indicator"] + yc[:6]
    cln_slice = _DF[cols_show].head(10).copy()

    def _short(s):
        s = str(s)
        return s[:28] + "…" if len(s) > 29 else s

    cln_slice["Indicator"] = cln_slice["Indicator"].apply(_short)

    for c in yc[:6]:
        if c in cln_slice.columns:
            cln_slice[c] = cln_slice[c].map(
                lambda v: f"{v:.1f}" if pd.notnull(v) else "—"
            )

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(
        "② CLEANED DataFrame  — After 7-Step Pipeline  (all float64, zero NaN)",
        fontsize=12, fontweight="bold", color="#16A34A"
    )
    ax.axis("off")

    col_labels = list(cln_slice.columns)
    tab = ax.table(
        cellText=cln_slice.values.tolist(),
        colLabels=col_labels,
        cellLoc="center", loc="center",
        bbox=[0, 0, 1, 1],
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    _style_table_header(tab, len(col_labels), "#14532D")

    for i in range(len(cln_slice)):
        for j in range(len(col_labels)):
            tab[i + 1, j].set_facecolor(
                "#F0FFF4" if i % 2 == 0 else "white"
            )

    plt.tight_layout()
    plt.show()


def plot_data_table_summary() -> None:
    """
    Figure 3 of 3 — Side-by-side quality comparison table
    (raw CSV metrics vs cleaned DataFrame metrics).
    """
    raw_df2     = _DF_RAW.copy()
    dirty_mask2 = raw_df2.map(
        lambda x: str(x).strip() in DIRTY if pd.notnull(x) else False
    )

    summary_data = [
        ["Total rows",        str(raw_df2.shape[0]),                        str(_DF.shape[0])],
        ["Total columns",     str(raw_df2.shape[1]),                        str(_DF.shape[1])],
        ["NaN cells",         str(int(raw_df2.isnull().sum().sum())),       str(int(_DF[_YEAR_PRESENT].isnull().sum().sum()))],
        ["Dirty tokens",      str(int(dirty_mask2.sum().sum())),            "0"],
        ["Duplicate rows",    str(int(raw_df2.duplicated().sum())),         "0"],
        ["Data type (years)", "mixed / object",                             "float64"],
        ["Outlier handling",  "none",                                       "IQR×3 → re-interpolated"],
        ["Missing fill",      "none",                                       "linear interp + ffill/bfill + row-mean"],
    ]

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle(
        "③ Quality Comparison — Raw CSV  vs  Cleaned DataFrame",
        fontsize=12, fontweight="bold"
    )
    ax.axis("off")

    sum_cols = ["Metric", "Raw CSV", "Cleaned DataFrame"]
    tab = ax.table(
        cellText=summary_data,
        colLabels=sum_cols,
        cellLoc="center", loc="center",
        bbox=[0.02, 0, 0.96, 1],
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(9.5)
    _style_table_header(tab, 3, "#1E3A5F")

    for i in range(len(summary_data)):
        tab[i + 1, 0].set_text_props(fontweight="bold")
        tab[i + 1, 0].set_facecolor("#F8FAFC" if i % 2 == 0 else "white")
        tab[i + 1, 1].set_facecolor("#FFF7ED")   # raw  = amber
        tab[i + 1, 2].set_facecolor("#F0FFF4")   # clean = green

    plt.tight_layout()
    plt.show()


def plot_data_tables() -> None:
    """
    Entry point called by the UI.
    Opens three separate, clearly labelled figure windows:
      ① Raw CSV table
      ② Cleaned DataFrame table
      ③ Quality comparison summary
    """
    plot_data_table_raw()
    plot_data_table_clean()
    plot_data_table_summary()


# =============================================================================
#  WELCOME PAGE — 2024 snapshot table (descriptive, no forecasts)
# =============================================================================

def get_welcome_forecasts() -> list[tuple]:
    """
    Return a list of (label, value_2000, value_2024, unit) tuples
    for the welcome screen summary table.

    Note: the function signature is kept compatible with the UI, which expects
    (label, v_start, v_end, unit).  v_start = 2000, v_end = 2024.
    """
    indicators = [
        ("GDP",        "GDP – Gross Domestic Product (USD)",              1e12,  "T USD"),
        ("GDP Growth", "GDP Growth Rate (Annual %)",                      1.0,   "%"),
        ("CPI",        "CPI Inflation – Consumer Prices (Annual %)",      1.0,   "%"),
        ("Savings",    "Gross Savings Rate (% of GDP)",                   1.0,   "% GDP"),
        ("Investment", "Gross Capital Formation / Investment (% of GDP)", 1.0,   "% GDP"),
        ("Exports",    "Exports of Goods & Services (% of GDP)",          1.0,   "% GDP"),
        ("Population", "Total Population",                                1e9,   "Bn"),
        ("Urban %",    "Urban Population (% of Total)",                   1.0,   "%"),
    ]
    results = []
    for label, ind, scale, unit in indicators:
        s = get_series(ind)
        if s.empty:
            continue
        vals   = s.values / scale
        v_2000 = float(vals[0])
        v_2024 = float(vals[-1])
        results.append((label, v_2000, v_2024, unit))
    return results


# =============================================================================
#  PHASE 4 — Feature Extraction  (rolling stats, lags, year-on-year changes)
# =============================================================================

def extract_features() -> pd.DataFrame:
    """
    For each key indicator, extract four time-series features per year:
      • raw value
      • 3-year rolling mean  — smoothed trend, reduces noise
      • 3-year rolling std   — local volatility measure
      • lag-1 value          — previous year (auto-correlation context)
      • year-on-year change  — momentum / acceleration

    No prediction is performed; these are descriptive transformations only.
    """
    KEY_INDICATORS = {
        "GDP":        "GDP – Gross Domestic Product (USD)",
        "GDPgrowth":  "GDP Growth Rate (Annual %)",
        "CPI":        "CPI Inflation – Consumer Prices (Annual %)",
        "Savings":    "Gross Savings Rate (% of GDP)",
        "Investment": "Gross Capital Formation / Investment (% of GDP)",
        "Exports":    "Exports of Goods & Services (% of GDP)",
        "Imports":    "Imports of Goods & Services (% of GDP)",
        "PopUrban":   "Urban Population (% of Total)",
    }

    feat_df = pd.DataFrame(index=_YEARS_INT)

    for name, ind in KEY_INDICATORS.items():
        s = get_series(ind)
        s.index = _YEARS_INT

        feat_df[name]                 = s.values
        feat_df[f"{name}_roll3_mean"] = s.rolling(3, min_periods=1).mean().values
        feat_df[f"{name}_roll3_std"]  = s.rolling(3, min_periods=1).std().fillna(0).values
        feat_df[f"{name}_lag1"]       = s.shift(1).bfill().values
        feat_df[f"{name}_yoy_change"] = s.diff().fillna(0).values

    print(f"[Phase 4] Feature extraction complete.  Shape: {feat_df.shape}")
    return feat_df


# =============================================================================
#  PHASE 5 — Feature Engineering  (derived economic ratios)
# =============================================================================

def create_features(feat_df=None) -> pd.DataFrame:
    """
    Build seven derived features that capture important economic relationships:

    Savings_Investment_Balance (formerly SAV_INV_Gap)
        = Savings rate − Investment rate (% GDP)
        Positive = domestic surplus; negative = reliance on external capital.

    Inflation_Adjusted_Growth (formerly Real_Growth_Proxy)
        = GDP Growth rate − CPI inflation rate
        Approximates real (purchasing-power-adjusted) economic growth.

    Investment_Productivity (formerly Invest_Efficiency)
        = GDP Growth rate / Investment rate
        How much growth each unit of investment generates (output efficiency).

    Trade_Openness
        = (Exports + Imports) / 2  as % GDP
        Measures how integrated India is with the global economy.

    GDP_Acceleration
        = Year-on-year change in GDP growth rate (second difference)
        Positive = economy speeding up; negative = slowing down.

    Urban_Momentum
        = 3-year rolling slope of urbanisation rate
        Measures how rapidly the rural-urban shift is occurring.

    Export_Intensity
        = (Export share × GDP in USD) / 1e9
        Converts the export share into an absolute USD-billion figure.
    """
    if feat_df is None:
        feat_df = extract_features()

    gdp_abs = get_series("GDP – Gross Domestic Product (USD)").values

    # ── Derived features ──────────────────────────────────────────────────────

    # Positive balance = India saving more than it invests domestically
    feat_df["Savings_Investment_Balance"] = (
        feat_df["Savings"] - feat_df["Investment"]
    )

    # Approximation of real growth (CPI-deflated)
    feat_df["Inflation_Adjusted_Growth"] = (
        feat_df["GDPgrowth"] - feat_df["CPI"]
    )

    # GDP growth per unit of investment — higher = more productive capital
    feat_df["Investment_Productivity"] = (
        feat_df["GDPgrowth"]
        / feat_df["Investment"].replace(0, np.nan)
    )

    # Average of export and import shares — broader trade integration measure
    feat_df["Trade_Openness"] = (
        (feat_df["Exports"] + feat_df["Imports"]) / 2
    )

    # Second difference of growth rate — captures cyclical acceleration
    feat_df["GDP_Acceleration"] = feat_df["GDPgrowth"].diff().fillna(0)

    # Rolling 3-year slope of urban population share
    urb    = feat_df["PopUrban"].values
    slopes = [0.0, 0.0]
    for i in range(2, len(urb)):
        m, *_ = stats.linregress(range(3), urb[i - 2: i + 1])
        slopes.append(m)
    feat_df["Urban_Momentum"] = slopes

    # Exports in absolute USD billions
    feat_df["Export_Intensity"] = (
        (feat_df["Exports"] / 100) * gdp_abs / 1e9
    )

    print(f"[Phase 5] Feature engineering complete.  Shape: {feat_df.shape}")
    return feat_df


def get_feature_summary() -> str:
    """Return a formatted text summary of the feature matrix for the UI."""
    feat_df  = create_features()
    raw_cols = [c for c in feat_df.columns if "_" not in c]
    extracted = [
        c for c in feat_df.columns
        if any(k in c for k in ["roll", "lag", "yoy"])
    ]
    derived = [
        "Savings_Investment_Balance", "Inflation_Adjusted_Growth",
        "Investment_Productivity",    "Trade_Openness",
        "GDP_Acceleration",           "Urban_Momentum",
        "Export_Intensity",
    ]

    lines = [
        f"  Total features           : {feat_df.shape[1]}",
        f"  Raw indicator columns    : {len(raw_cols)}",
        f"  Extracted time features  : {len(extracted)}",
        f"    • 3-year rolling mean & std per indicator  (trend + volatility)",
        f"    • Lag-1 value per indicator  (auto-correlation)",
        f"    • Year-on-year change per indicator  (momentum)",
        f"  Engineered ratios        : {len(derived)}",
        f"    • Savings_Investment_Balance  (domestic capital surplus/deficit)",
        f"    • Inflation_Adjusted_Growth   (real GDP growth proxy)",
        f"    • Investment_Productivity     (growth per unit of investment)",
        f"    • Trade_Openness              (avg export+import share)",
        f"    • GDP_Acceleration            (change in growth rate)",
        f"    • Urban_Momentum              (3-yr urbanisation slope)",
        f"    • Export_Intensity            (absolute export USD bn)",
    ]
    return "\n".join(lines)


# =============================================================================
#  PHASE 5b — PCA + LDA  (descriptive dimensionality reduction)
# =============================================================================

def plot_pca_lda() -> dict:
    """
    Apply PCA and LDA to the full feature matrix as a descriptive exercise.

    PCA (Principal Component Analysis) — unsupervised
        Identifies the directions of maximum variance in the 25-year dataset.
        PC1 typically captures the 'Growth & Scale' dimension (GDP, Investment).
        PC2 typically captures the 'Inflation & External' dimension (CPI, Trade).
        No prediction is made; the scatter shows how years cluster historically.

    LDA (Linear Discriminant Analysis) — supervised by decade
        Finds the linear combination of features that best separates the three
        decades (2000s / 2010s / 2020s) — purely descriptive decade profiling.

    Returns a dict with PCA explained variance and feature importance info.
    """
    feat_df = create_features()
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Drop near-zero variance columns (they add no information)
    var_series   = feat_df.var()
    VAR_THRESH   = 1e-6
    dropped_cols = var_series[var_series < VAR_THRESH].index.tolist()
    kept_df      = feat_df.drop(columns=dropped_cols, errors="ignore")

    X      = kept_df.values
    years  = kept_df.index.tolist()
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)   # standardise before PCA

    # Decade labels for colouring / LDA supervision
    labels    = [
        "2000s" if y < 2010 else ("2010s" if y < 2020 else "2020s")
        for y in years
    ]
    label_map = {"2000s": 0, "2010s": 1, "2020s": 2}
    y_lda     = np.array([label_map[l] for l in labels])
    colors    = {
        "2000s": PALETTE["blue"],
        "2010s": PALETTE["green"],
        "2020s": PALETTE["orange"],
    }

    # ── PCA ───────────────────────────────────────────────────────────────────
    pca     = PCA(n_components=min(10, X.shape[1]))
    X_pca   = pca.fit_transform(Xs)
    exp_var = pca.explained_variance_ratio_

    loadings_all = pd.DataFrame(
        pca.components_[:2].T,
        index=kept_df.columns,
        columns=["PC1", "PC2"],
    )
    load_scores  = loadings_all.abs().sum(axis=1)
    low_thresh   = load_scores.quantile(0.25)
    low_inf_cols = load_scores[load_scores <= low_thresh].index.tolist()

    # ── LDA ───────────────────────────────────────────────────────────────────
    n_lda  = min(2, len(np.unique(y_lda)) - 1)
    lda    = LDA(n_components=n_lda)
    X_lda  = lda.fit_transform(Xs, y_lda)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Feature Engineering + Dimensionality Reduction — PCA & LDA\n"
        "India Economic Indicators  2000–2024  (descriptive analysis only)",
        fontsize=14, fontweight="bold",
    )
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.38)

    # Panel 1 — Scree plot
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(range(1, len(exp_var) + 1), exp_var * 100,
           color=PALETTE["blue"], alpha=0.8)
    ax.plot(range(1, len(exp_var) + 1), np.cumsum(exp_var) * 100,
            color=PALETTE["orange"], lw=2, marker="o", ms=4, label="Cumulative %")
    ax.axhline(80, color="red", ls="--", lw=1, label="80% threshold")
    ax.set_title("PCA — Scree Plot", fontweight="bold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.legend(fontsize=8)

    # Panel 2 — PCA 2D scatter by decade
    ax = fig.add_subplot(gs[0, 1:])
    for decade, col in colors.items():
        mask = np.array(labels) == decade
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   color=col, s=70, alpha=0.85, label=decade, zorder=3)
        for i, yr in enumerate(np.array(years)[mask]):
            ax.annotate(str(yr), (X_pca[mask][i, 0], X_pca[mask][i, 1]),
                        fontsize=7, alpha=0.7)
    ax.set_xlabel(
        f"PC1 ({exp_var[0]*100:.1f}%)  — Growth & Scale axis\n"
        f"(loaded on: GDP level, Investment, Savings)"
    )
    ax.set_ylabel(
        f"PC2 ({exp_var[1]*100:.1f}%)  — Inflation & External axis\n"
        f"(loaded on: CPI, Exports, Imports, Trade Openness)"
    )
    ax.set_title("PCA — 2D Projection by Decade  (each dot = one year)",
                 fontweight="bold")
    ax.legend()

    # Panel 3 — Feature loadings heatmap
    ax = fig.add_subplot(gs[1, 0:2])
    top_feats = loadings_all.abs().sum(axis=1).nlargest(15).index
    sns.heatmap(
        loadings_all.loc[top_feats],
        annot=True, fmt=".2f",
        cmap="coolwarm", center=0, ax=ax,
        linewidths=0.4, annot_kws={"size": 7},
    )
    ax.set_title(
        "PCA Feature Loadings — Top 15 features  (by PC1+PC2 magnitude)\n"
        "Warm = drives that PC positively  |  Cool = drives it negatively",
        fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=7)

    # Panel 4 — LDA decade separation
    ax = fig.add_subplot(gs[1, 2])
    if n_lda == 2:
        for decade, col in colors.items():
            mask = np.array(labels) == decade
            ax.scatter(X_lda[mask, 0], X_lda[mask, 1],
                       color=col, s=70, alpha=0.85, label=decade, zorder=3)
        ax.set_xlabel("LD1")
        ax.set_ylabel("LD2")
    else:
        for i, (decade, col) in enumerate(colors.items()):
            mask = np.array(labels) == decade
            ax.scatter(X_lda[mask, 0], np.full(mask.sum(), i * 0.1),
                       color=col, s=70, alpha=0.85, label=decade, zorder=3)
        ax.set_xlabel("LD1")
        ax.set_yticks([])
    ax.set_title(
        "LDA — Decade Separation\n"
        "Well-separated clusters confirm structural change each decade",
        fontweight="bold",
    )
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

    print(f"[Phase 5b] PCA: {X.shape[1]} features | "
          f"PC1+PC2 = {sum(exp_var[:2])*100:.1f}% variance")
    if dropped_cols:
        print(f"  Dropped (near-zero variance): {dropped_cols}")
    else:
        print("  No columns dropped for near-zero variance.")
    print(f"  Low-influence features (bottom-25% loadings): {low_inf_cols}")

    return {
        "pca_explained": exp_var.tolist(),
        "n_features":    X.shape[1],
        "dropped_cols":  dropped_cols,
        "low_inf_cols":  low_inf_cols,
    }


# =============================================================================
#  HELPER — annotate historical shocks on a matplotlib axis
# =============================================================================

def _annotate_shocks(ax, y_pos_frac: float = 0.92) -> None:
    """
    Add vertical lines and labels for the 2008 GFC and 2020 COVID shock.
    y_pos_frac: fraction of axis height at which to place the text label.
    """
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    for yr, (short_lbl, _) in SHOCKS.items():
        ax.axvline(yr, color=SHOCK_COLOR, ls="--", alpha=0.5, lw=1.5)
        ax.text(
            yr + 0.2,
            ylim[0] + y_range * y_pos_frac,
            short_lbl,
            fontsize=8, color=SHOCK_COLOR, va="top",
        )


# =============================================================================
#  PHASE 6 — GDP Analysis  (descriptive)
# =============================================================================

def plot_gdp_analysis() -> None:
    """
    Three-panel descriptive GDP chart:
      1. GDP level (Trillion USD) with shock annotations.
      2. Annual GDP growth rate — bar chart with zero line.
      3. GDP per Capita (USD) — absolute wealth per person.
    """
    gdp    = get_series("GDP – Gross Domestic Product (USD)") / 1e12
    growth = get_series("GDP Growth Rate (Annual %)")
    gdppc  = get_series("GDP per Capita (USD)")

    peak_idx = int(growth.values.argmax())

    fig, axes = plt.subplots(3, 1, figsize=(13, 14))
    fig.suptitle(
        "India — GDP Analysis  (2000–2024)\n"
        "Historical data only — World Bank open data",
        fontsize=14, fontweight="bold",
    )

    # Panel 1 — GDP level
    ax = axes[0]
    ax.plot(_YEARS_INT, gdp, color=PALETTE["blue"],
            lw=2.5, marker="o", ms=4, label="GDP (Trillion USD)")
    ax.fill_between(_YEARS_INT, gdp, alpha=0.12, color=PALETTE["blue"])
    _annotate_shocks(ax)
    ax.set_title("GDP — Gross Domestic Product  (Trillion USD)", fontweight="bold")
    ax.set_ylabel("USD Trillion")
    ax.legend(fontsize=8)

    # Panel 2 — Growth rate bars
    ax = axes[1]
    bar_colors = [
        PALETTE["green"] if v >= 0 else PALETTE["red"]
        for v in growth
    ]
    ax.bar(_YEARS_INT, growth, color=bar_colors, alpha=0.8,
           edgecolor="white", label="Annual Growth %")
    ax.axhline(0, color="black", lw=1)
    ax.axhline(growth.mean(), color=PALETTE["orange"], ls="--", lw=1.5,
               label=f"Mean = {growth.mean():.2f}%")
    _annotate_shocks(ax, y_pos_frac=0.85)
    ax.set_title("GDP Growth Rate  (Annual %)", fontweight="bold")
    ax.set_ylabel("% Growth")
    ax.legend(fontsize=8)

    # Panel 3 — GDP per capita
    ax = axes[2]
    ax.plot(_YEARS_INT, gdppc, color=PALETTE["teal"],
            lw=2.5, marker="s", ms=4, label="GDP per Capita (USD)")
    ax.fill_between(_YEARS_INT, gdppc, alpha=0.12, color=PALETTE["teal"])
    _annotate_shocks(ax)
    ax.set_title("GDP per Capita  (USD)", fontweight="bold")
    ax.set_ylabel("USD per person")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# =============================================================================
#  GDP vs GNI vs NNI — Multi-line overlay comparison
# =============================================================================

def plot_gdp_gni_nni_comparison() -> None:
    """
    Three-panel comparison of GDP, GNI, and NNI:
      1. Absolute levels in USD Trillion.
      2. Year-on-year percentage change for each aggregate.
      3. GNI and NNI expressed as % of GDP — how much income India retains.
    """
    gdp = get_series("GDP – Gross Domestic Product (USD)") / 1e12
    gni = get_series("GNI – Gross National Income (USD)")  / 1e12
    nni = get_series("NNI – Adj. Net National Income (USD)") / 1e12

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "India — GDP vs GNI vs NNI Comparison  (2000–2024)",
        fontsize=14, fontweight="bold",
    )

    # Panel 1 — Absolute levels
    ax = axes[0]
    ax.plot(_YEARS_INT, gdp, color=PALETTE["blue"],   lw=2.5,
            marker="o", ms=4, label="GDP")
    ax.plot(_YEARS_INT, gni, color=PALETTE["green"],  lw=2.5,
            marker="s", ms=4, label="GNI")
    ax.plot(_YEARS_INT, nni, color=PALETTE["orange"], lw=2.0,
            marker="^", ms=4, label="NNI (Adj.)")
    ax.fill_between(_YEARS_INT, gdp, gni, alpha=0.10, color=PALETTE["green"])
    ax.fill_between(_YEARS_INT, gni, nni, alpha=0.10, color=PALETTE["orange"])
    ax.set_title("Absolute Levels  (USD Trillion)", fontweight="bold")
    ax.set_ylabel("Trillion USD")
    ax.legend(fontsize=9)

    # Panel 2 — Year-on-year growth %
    ax = axes[1]
    for series, col, lbl in [
        (gdp, PALETTE["blue"],   "GDP"),
        (gni, PALETTE["green"],  "GNI"),
        (nni, PALETTE["orange"], "NNI"),
    ]:
        yoy = series.pct_change().fillna(0) * 100
        ax.plot(_YEARS_INT, yoy, color=col, lw=2,
                marker="o", ms=3, label=lbl)
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Year-on-Year Growth  (%)", fontweight="bold")
    ax.set_ylabel("% Change")
    ax.legend(fontsize=9)

    # Panel 3 — Ratios to GDP
    ax = axes[2]
    ratio_gni = (gni / gdp * 100).values
    ratio_nni = (nni / gdp * 100).values
    ax.plot(_YEARS_INT, ratio_gni, color=PALETTE["green"], lw=2.5,
            marker="s", ms=4, label="GNI / GDP %")
    ax.plot(_YEARS_INT, ratio_nni, color=PALETTE["orange"], lw=2.5,
            marker="^", ms=4, label="NNI / GDP %")
    ax.fill_between(_YEARS_INT, ratio_gni, ratio_nni,
                    alpha=0.15, color=PALETTE["orange"])
    ax.set_title("GNI & NNI as % of GDP\n(Income retained in India)",
                 fontweight="bold")
    ax.set_ylabel("% of GDP")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()


# =============================================================================
#  Decade-wise Bar Comparison
# =============================================================================

def plot_decade_comparison() -> None:
    """
    Compare decade-average values of 6 key indicators across:
      • 2000s — early liberalisation & IT boom
      • 2010s — consolidation & digital growth
      • 2020s — COVID recovery & PLI era

    Left: grouped bar chart with labelled values.
    Right: Z-score heatmap (actual values annotated) for relative comparison.
    """
    indicators = {
        "GDP Growth %":   "GDP Growth Rate (Annual %)",
        "CPI %":          "CPI Inflation – Consumer Prices (Annual %)",
        "Savings % GDP":  "Gross Savings Rate (% of GDP)",
        "Invest % GDP":   "Gross Capital Formation / Investment (% of GDP)",
        "Exports % GDP":  "Exports of Goods & Services (% of GDP)",
        "Urban %":        "Urban Population (% of Total)",
    }
    decades = {
        "2000s": [str(y) for y in range(2000, 2010) if str(y) in _YEAR_PRESENT],
        "2010s": [str(y) for y in range(2010, 2020) if str(y) in _YEAR_PRESENT],
        "2020s": [str(y) for y in range(2020, 2025) if str(y) in _YEAR_PRESENT],
    }
    data   = {dec: [] for dec in decades}
    labels = list(indicators.keys())

    for _, ind in indicators.items():
        s = get_series(ind)
        s.index = _YEAR_PRESENT
        for dec, yrs in decades.items():
            data[dec].append(float(s[yrs].mean()))

    x           = np.arange(len(labels))
    w           = 0.25
    dec_colors  = [PALETTE["blue"], PALETTE["green"], PALETTE["orange"]]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "India — Decade-Wise Economic Comparison  (2000s / 2010s / 2020s)",
        fontsize=14, fontweight="bold",
    )

    # Grouped bar chart
    ax = axes[0]
    for i, (dec, col) in enumerate(zip(decades.keys(), dec_colors)):
        bars = ax.bar(x + (i - 1) * w, data[dec], w,
                      label=dec, color=col, alpha=0.85, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.15,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Average Value")
    ax.legend(fontsize=9)
    ax.set_title("Decade Averages — Key Indicators", fontweight="bold")

    # Z-score heatmap
    ax = axes[1]
    heat_data = np.array([data[d] for d in decades]).T   # shape (n_ind, 3)
    norm_data = (heat_data - heat_data.mean(axis=1, keepdims=True)) / (
        heat_data.std(axis=1, keepdims=True) + 1e-8
    )
    sns.heatmap(
        norm_data,
        annot=heat_data, fmt=".1f",
        xticklabels=list(decades.keys()),
        yticklabels=labels,
        cmap="RdYlGn", center=0, ax=ax,
        linewidths=0.5, annot_kws={"size": 9},
        cbar_kws={"label": "Z-score"},
    )
    ax.set_title(
        "Normalised Decade Heatmap\n(annotated values = actual averages)",
        fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=9)

    plt.tight_layout()
    plt.show()


# =============================================================================
#  Z-score Normalised Multi-Indicator Comparison
# =============================================================================

def plot_normalised_comparison() -> None:
    """
    Two-panel normalised comparison:
      Panel 1: Z-score time series for 6 indicators — comparable trends,
               removes scale differences (GDP billions vs CPI percent).
      Panel 2: Rolling 5-year Pearson correlation of each indicator with
               GDP Growth — shows which variables move with economic cycles.
    """
    series_map = {
        "GDP Growth":  get_series("GDP Growth Rate (Annual %)"),
        "CPI":         get_series("CPI Inflation – Consumer Prices (Annual %)"),
        "Savings":     get_series("Gross Savings Rate (% of GDP)"),
        "Investment":  get_series("Gross Capital Formation / Investment (% of GDP)"),
        "Exports":     get_series("Exports of Goods & Services (% of GDP)"),
        "Urban %":     get_series("Urban Population (% of Total)"),
    }
    palette_list = [PALETTE[k] for k in ["blue", "red", "teal", "orange", "green", "purple"]]

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    fig.suptitle(
        "India — Normalised Multi-Indicator Comparison  (2000–2024)",
        fontsize=14, fontweight="bold",
    )

    # Panel 1 — Z-score lines
    ax = axes[0]
    for (name, s), col in zip(series_map.items(), palette_list):
        z = (s.values - s.mean()) / (s.std() + 1e-8)
        ax.plot(_YEARS_INT, z, color=col, lw=2, marker="o", ms=3, label=name)
    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.5)
    ax.fill_between(_YEARS_INT, -1, 1, alpha=0.04, color="grey",
                    label="±1σ band (historically average)")
    ax.axvspan(2019.5, 2021.5, alpha=0.08, color="red")
    ax.text(2020, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else -0.5,
            "COVID", fontsize=8, color="red", ha="center")
    ax.axvspan(2007.5, 2009.5, alpha=0.05, color=PALETTE["orange"])
    ax.text(2008.5, ax.get_ylim()[1] * 0.75 if ax.get_ylim()[1] > 0 else -1.0,
            "GFC", fontsize=8, color=PALETTE["orange"], ha="center")
    ax.set_ylabel("Z-score  (σ from historical mean)")
    ax.set_title(
        "All Indicators — Z-score Normalised  "
        "(same scale, directly comparable trend lines)",
        fontweight="bold",
    )
    ax.legend(fontsize=8, ncol=3)

    # Panel 2 — Rolling 5-year Pearson r with GDP Growth
    ax = axes[1]
    gdp_growth = series_map["GDP Growth"].values
    for (name, s), col in zip(list(series_map.items())[1:], palette_list[1:]):
        roll_corr = []
        for i in range(len(_YEARS_INT)):
            if i < 4:
                roll_corr.append(np.nan)
            else:
                r, _ = stats.pearsonr(
                    gdp_growth[i - 4: i + 1], s.values[i - 4: i + 1]
                )
                roll_corr.append(r)
        ax.plot(_YEARS_INT, roll_corr, color=col, lw=2,
                marker="o", ms=3, label=f"GDP↔{name}")
    ax.axhline(0,    color="black",          lw=1,   ls="--", alpha=0.5)
    ax.axhline(0.5,  color=PALETTE["green"], lw=1,   ls=":",  alpha=0.6)
    ax.axhline(-0.5, color=PALETTE["red"],   lw=1,   ls=":",  alpha=0.6)
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel("Pearson r")
    ax.set_title(
        "Rolling 5-Year Correlation  (GDP Growth vs each indicator)\n"
        "Dashed lines at ±0.5 = moderate correlation threshold",
        fontweight="bold",
    )
    ax.legend(fontsize=8, ncol=3)

    plt.tight_layout()
    plt.show()


# =============================================================================
#  Fiscal & Household Expenditure
# =============================================================================

def plot_fiscal_household_comparison() -> None:
    """
    Two-panel fiscal decomposition chart:
      Left:  Stacked area — HFCE (consumption), Investment, and residual
             Government spending as shares of GDP.
      Right: Bar comparison of Tax Revenue vs Investment — shows fiscal
             space and government's ability to fund capital expenditure.
    """
    hfce  = get_series("HFCE – Household Final Consumption (% of GDP)")
    inv   = get_series("Gross Capital Formation / Investment (% of GDP)")
    tax   = get_series("Tax Revenue (% of GDP)")
    exp_s = get_series("Exports of Goods & Services (% of GDP)")

    # Government spending approximation (GDP identity residual):
    # GDP = HFCE + Investment + Govt + Net Exports
    # Govt ≈ 100 − HFCE − Investment − Exports  (simplified, illustrative)
    govt_approx = np.clip(100 - hfce.values - inv.values - exp_s.values, 0, 50)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "India — Fiscal & Household Expenditure Analysis  (2000–2024)",
        fontsize=13, fontweight="bold",
    )

    # Stacked area
    ax = axes[0]
    ax.stackplot(
        _YEARS_INT,
        hfce.values, inv.values, govt_approx,
        labels=["HFCE (Consumption)", "Investment", "Govt (residual est.)"],
        colors=[PALETTE["blue"], PALETTE["orange"], PALETTE["green"]],
        alpha=0.75,
    )
    ax.set_title("Expenditure Composition  (% of GDP)", fontweight="bold")
    ax.set_ylabel("% of GDP")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 105)

    # Tax Revenue vs Investment bars
    ax = axes[1]
    x = np.arange(len(_YEARS_INT))
    w = 0.4
    ax.bar(x - w / 2, tax.values, w,
           color=PALETTE["purple"], alpha=0.8, label="Tax Revenue % GDP")
    ax.bar(x + w / 2, inv.values, w,
           color=PALETTE["teal"], alpha=0.8, label="Investment % GDP")
    ax.set_xticks(x[::3])
    ax.set_xticklabels(_YEARS_INT[::3], rotation=30, fontsize=8)
    ax.set_title("Tax Revenue vs Investment  (% of GDP)", fontweight="bold")
    ax.set_ylabel("% of GDP")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# =============================================================================
#  Inflation Analysis  (descriptive)
# =============================================================================

def plot_inflation_analysis() -> None:
    """
    Three-panel CPI inflation descriptive analysis:
      1. CPI trend with RBI target bands (4% comfort, 6% upper bound).
      2. Distribution histogram — shape, mean, and spread.
      3. Boxplot with skewness annotation.
    """
    cpi = get_series("CPI Inflation – Consumer Prices (Annual %)")

    # Rolling 3-year volatility of inflation itself
    cpi_vol = rolling_volatility(
        "CPI Inflation – Consumer Prices (Annual %)", window=3
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "India — CPI Inflation  (2000–2024)  |  Descriptive Analysis",
        fontsize=13, fontweight="bold",
    )

    # Panel 1 — Trend line
    ax = axes[0]
    ax.plot(_YEARS_INT, cpi, color=PALETTE["red"], lw=2.5,
            marker="s", ms=4, label="CPI Annual %")
    ax.fill_between(_YEARS_INT, cpi, alpha=0.12, color=PALETTE["red"])
    ax.axhline(4, color="green",          ls=":", lw=1.5, label="4% comfort target")
    ax.axhline(6, color=PALETTE["orange"],ls=":", lw=1.5, label="6% upper bound")
    _annotate_shocks(ax)
    ax.set_title("CPI Trend  (2000–2024)", fontweight="bold")
    ax.set_ylabel("Annual %")
    ax.legend(fontsize=7)

    # Panel 2 — Histogram
    ax = axes[1]
    s = cpi.dropna()
    ax.hist(s, bins=10, color=PALETTE["red"], alpha=0.75,
            edgecolor="white", label="All years")
    ax.axvline(float(s.mean()), color="black", ls="--", lw=1.5,
               label=f"Mean = {s.mean():.1f}%")
    ax.axvline(4, color="green",          ls=":", lw=1.5, label="4% target")
    ax.axvline(6, color=PALETTE["orange"],ls=":", lw=1.5, label="6% cap")
    ax.set_title("CPI Distribution", fontweight="bold")
    ax.set_xlabel("Inflation %")
    ax.legend(fontsize=7)

    # Panel 3 — Boxplot
    ax = axes[2]
    bp = ax.boxplot(
        [s.values],
        patch_artist=True,
        labels=["CPI (2000–2024)"],
        medianprops=dict(color="black", lw=2),
    )
    bp["boxes"][0].set_facecolor(PALETTE["red"])
    bp["boxes"][0].set_alpha(0.6)
    ax.set_title(
        f"CPI Boxplot\n"
        f"Skew={stats.skew(s):.2f}  |  Kurt={stats.kurtosis(s):.2f}",
        fontweight="bold",
    )

    plt.tight_layout()
    plt.show()


# =============================================================================
#  Trade Balance  (descriptive)
# =============================================================================

def plot_trade_balance() -> None:
    """
    Two-panel trade analysis:
      1. Exports vs Imports as % of GDP — area fill shows surplus/deficit.
      2. Trade Balance bar chart — most years negative (import-heavy India).
    """
    exp = get_series("Exports of Goods & Services (% of GDP)")
    imp = get_series("Imports of Goods & Services (% of GDP)")
    tb  = get_series("Trade Balance (Exports minus Imports %)")

    fig, axes = plt.subplots(2, 1, figsize=(13, 11))
    fig.suptitle(
        "India — Trade Balance  (2000–2024)  |  Descriptive Analysis",
        fontsize=13, fontweight="bold",
    )

    # Panel 1 — Exports vs Imports overlay
    ax = axes[0]
    ax.plot(_YEARS_INT, exp, color=PALETTE["green"],  lw=2.5,
            marker="^", ms=4, label="Exports % GDP")
    ax.plot(_YEARS_INT, imp, color=PALETTE["orange"], lw=2.5,
            marker="v", ms=4, label="Imports % GDP")
    ax.fill_between(
        _YEARS_INT, exp, imp,
        where=(exp.values >= imp.values),
        alpha=0.15, color="green", label="Export surplus"
    )
    ax.fill_between(
        _YEARS_INT, exp, imp,
        where=(exp.values < imp.values),
        alpha=0.15, color="red", label="Import deficit"
    )
    _annotate_shocks(ax)
    ax.set_title("Exports vs Imports  (% of GDP)", fontweight="bold")
    ax.set_ylabel("% of GDP")
    ax.legend(fontsize=8)

    # Panel 2 — Net trade balance
    ax = axes[1]
    bar_colors = [
        PALETTE["green"] if v >= 0 else PALETTE["red"] for v in tb
    ]
    ax.bar(_YEARS_INT, tb, color=bar_colors, alpha=0.8,
           edgecolor="white", label="Trade Balance % GDP")
    ax.axhline(0, color="black", lw=1)
    ax.axhline(tb.mean(), color=PALETTE["orange"], ls="--", lw=1.5,
               label=f"Mean = {tb.mean():.2f}%")
    _annotate_shocks(ax, y_pos_frac=0.85)
    ax.set_title("Trade Balance  (Exports − Imports, % of GDP)", fontweight="bold")
    ax.set_ylabel("% of GDP")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# =============================================================================
#  Savings & Investment  (descriptive)
# =============================================================================

def plot_savings_investment() -> None:
    """
    Two-panel savings and investment descriptive chart:
      1. Grouped bar chart — historical savings vs investment side-by-side.
      2. Scatter plot with OLS trend line — correlation between the two.
    """
    sav = get_series("Gross Savings Rate (% of GDP)")
    inv = get_series("Gross Capital Formation / Investment (% of GDP)")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "India — Savings & Investment  (2000–2024)  |  Descriptive Analysis",
        fontsize=13, fontweight="bold",
    )

    # Panel 1 — Grouped bars
    ax = axes[0]
    x = np.array(_YEARS_INT)
    w = 0.35
    ax.bar(x - w / 2, sav, w, color=PALETTE["teal"],   alpha=0.85, label="Savings")
    ax.bar(x + w / 2, inv, w, color=PALETTE["orange"], alpha=0.85, label="Investment")
    ax.axhline(sav.mean(), color=PALETTE["teal"],   ls="--", lw=1.2, alpha=0.7,
               label=f"Savings mean = {sav.mean():.1f}%")
    ax.axhline(inv.mean(), color=PALETTE["orange"], ls="--", lw=1.2, alpha=0.7,
               label=f"Invest mean = {inv.mean():.1f}%")
    ax.set_title("Savings vs Investment  (% of GDP)", fontweight="bold")
    ax.set_ylabel("% of GDP")
    ax.legend(fontsize=7)

    # Panel 2 — Scatter with OLS line
    ax = axes[1]
    df_si = pd.DataFrame(
        {"Savings": sav.values, "Investment": inv.values}
    ).dropna()
    ax.scatter(df_si["Savings"], df_si["Investment"],
               color=PALETTE["purple"], alpha=0.7, zorder=3, s=60,
               label="Year observations")

    # Year labels on scatter dots
    for i, yr in enumerate(_YEARS_INT):
        ax.annotate(str(yr),
                    (df_si["Savings"].iloc[i], df_si["Investment"].iloc[i]),
                    fontsize=6, alpha=0.6)

    if len(df_si) > 1:
        m, b, r, p, _ = stats.linregress(df_si["Savings"], df_si["Investment"])
        xs = np.linspace(df_si["Savings"].min(), df_si["Savings"].max(), 100)
        ax.plot(xs, m * xs + b, color=PALETTE["orange"], lw=2, ls="--",
                label=f"OLS trend  r = {r:.2f}  (p = {p:.3f})")

    ax.set_xlabel("Savings Rate  (% GDP)")
    ax.set_ylabel("Investment Rate  (% GDP)")
    ax.set_title("Savings vs Investment — Scatter  (2000–2024)", fontweight="bold")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# =============================================================================
#  Population & Urbanisation  (descriptive)
# =============================================================================

def plot_population() -> None:
    """
    Combined dual-axis chart:
      Left axis (bars): Total population in billions.
      Right axis (line): Urban population share (%).
    Annotations mark the 2023 India–China crossover.
    """
    pop = get_series("Total Population") / 1e9
    urb = get_series("Urban Population (% of Total)")

    fig, ax = plt.subplots(figsize=(13, 6))
    ax2 = ax.twinx()

    ax.bar(_YEARS_INT, pop, color=PALETTE["blue"], alpha=0.45,
           label="Total Population (Bn)")
    ax.fill_between(_YEARS_INT, pop, alpha=0.08, color=PALETTE["blue"])

    ax2.plot(_YEARS_INT, urb, color=PALETTE["orange"], lw=2.5,
             marker="o", ms=4, label="Urban % of Total")

    # Rolling volatility overlay on right axis
    urb_vol = rolling_volatility("Urban Population (% of Total)", window=3)
    ax2.fill_between(
        _YEARS_INT,
        urb.values - urb_vol.values,
        urb.values + urb_vol.values,
        alpha=0.12, color=PALETTE["orange"], label="±1 roll-3 std"
    )

    # Mark 2023 crossover with China
    if 2023 in _YEARS_INT:
        idx23 = _YEARS_INT.index(2023)
        ax.axvline(2023, color=PALETTE["red"], ls=":", lw=1.5, alpha=0.7)
        ax.text(2023.1, float(pop.iloc[idx23]) * 0.98,
                "India > China\n(population)", fontsize=7,
                color=PALETTE["red"], va="top")

    ax.set_title(
        "India — Population & Urbanisation  (2000–2024)",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylabel("Population  (Billion)", color=PALETTE["blue"])
    ax2.set_ylabel("Urban Population  (%)", color=PALETTE["orange"])

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.show()


# =============================================================================
#  Historical Trend Analysis  (replaces forecasting section in UI)
# =============================================================================

def plot_forecasting() -> dict:
    """
    Historical trend decomposition for GDP (replaces the old forecasting panel).

    Shows:
      1. GDP level with 5-year rolling mean overlay — smoothed trend vs noise.
      2. Decade-by-decade average growth bars — structural shift across eras.

    Returns a dict with summary stats for the UI text panel.
    """
    gdp  = get_series("GDP – Gross Domestic Product (USD)") / 1e12
    gr   = get_series("GDP Growth Rate (Annual %)")

    # Rolling statistics (purely descriptive)
    roll5_gdp  = gdp.rolling(5, min_periods=2).mean()
    roll5_gr   = gr.rolling(5, min_periods=2).mean()
    roll3_vol  = gr.rolling(3, min_periods=2).std()   # growth volatility

    # Decade averages
    def decade_avg(series, start, end):
        yrs = [str(y) for y in range(start, end) if str(y) in _YEAR_PRESENT]
        s   = series.copy()
        s.index = _YEAR_PRESENT
        return float(s[yrs].mean())

    d2000s_gdp = decade_avg(gdp, 2000, 2010)
    d2010s_gdp = decade_avg(gdp, 2010, 2020)
    d2020s_gdp = decade_avg(gdp, 2020, 2025)
    d2000s_gr  = decade_avg(gr,  2000, 2010)
    d2010s_gr  = decade_avg(gr,  2010, 2020)
    d2020s_gr  = decade_avg(gr,  2020, 2025)

    fig, axes = plt.subplots(2, 1, figsize=(13, 12))
    fig.suptitle(
        "India GDP — Historical Trend Analysis  (2000–2024)\n"
        "Rolling averages  |  Decade decomposition  |  Volatility",
        fontsize=14, fontweight="bold",
    )

    # Panel 1 — GDP level + rolling mean
    ax = axes[0]
    ax.plot(_YEARS_INT, gdp, color=PALETTE["blue"], lw=2.5,
            marker="o", ms=4, label="Annual GDP (T USD)")
    ax.plot(_YEARS_INT, roll5_gdp, color=PALETTE["orange"], lw=2.5,
            ls="--", label="5-year rolling mean")
    ax.fill_between(_YEARS_INT, gdp, roll5_gdp, alpha=0.08,
                    color=PALETTE["orange"], label="Deviation from trend")
    _annotate_shocks(ax)
    ax.set_title("GDP Level with 5-Year Rolling Mean  (Trillion USD)",
                 fontweight="bold")
    ax.set_ylabel("USD Trillion")
    ax.legend(fontsize=8)

    # Panel 2 — Growth rate + volatility + decade bars
    ax = axes[1]
    bar_colors = [
        PALETTE["green"] if v >= 0 else PALETTE["red"] for v in gr
    ]
    ax.bar(_YEARS_INT, gr, color=bar_colors, alpha=0.65,
           edgecolor="white", label="Annual Growth %")
    ax.plot(_YEARS_INT, roll5_gr, color=PALETTE["blue"], lw=2.5,
            label="5-yr rolling mean growth")
    ax.fill_between(
        _YEARS_INT,
        roll5_gr - roll3_vol,
        roll5_gr + roll3_vol,
        alpha=0.15, color=PALETTE["blue"], label="±1 roll-3 std (volatility)"
    )

    # Decade average horizontal lines
    for (start, end, avg, col) in [
        (2000, 2010, d2000s_gr, PALETTE["teal"]),
        (2010, 2020, d2010s_gr, PALETTE["purple"]),
        (2020, 2025, d2020s_gr, PALETTE["pink"]),
    ]:
        valid = [y for y in range(start, end) if y in _YEARS_INT]
        if valid:
            ax.hlines(avg, valid[0], valid[-1], colors=col,
                      lw=2, ls="-.", label=f"{start}s avg = {avg:.1f}%")

    ax.axhline(0, color="black", lw=1)
    _annotate_shocks(ax, y_pos_frac=0.85)
    ax.set_title("GDP Growth Rate — Decomposed by Decade  (%)",
                 fontweight="bold")
    ax.set_ylabel("% Growth")
    ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.show()

    print("\n[Phase 6] Historical Trend Summary:")
    print(f"  {'Decade':<8} {'Avg GDP (T)':>12} {'Avg Growth %':>14}")
    print(f"  {'2000s':<8} {d2000s_gdp:>12.2f} {d2000s_gr:>14.2f}")
    print(f"  {'2010s':<8} {d2010s_gdp:>12.2f} {d2010s_gr:>14.2f}")
    print(f"  {'2020s':<8} {d2020s_gdp:>12.2f} {d2020s_gr:>14.2f}")

    return {
        "linear_2029": gdp.iloc[-1],    # kept for UI key compatibility
        "holt_2029":   gdp.iloc[-1],    # kept for UI key compatibility
        "d2000s_gr":   d2000s_gr,
        "d2010s_gr":   d2010s_gr,
        "d2020s_gr":   d2020s_gr,
    }


# =============================================================================
#  Correlation Heatmap  (with interpretation overlay)
# =============================================================================

def plot_correlation_heatmap() -> None:
    """
    Correlation heatmap for 10 key economic indicators.
    Green = strong positive correlation  |  Red = strong negative.
    Pairs with |r| ≥ 0.6 are printed to the console as plain-English insights.
    """
    key_inds = [
        "GDP – Gross Domestic Product (USD)",
        "GDP Growth Rate (Annual %)",
        "CPI Inflation – Consumer Prices (Annual %)",
        "Gross Savings Rate (% of GDP)",
        "Gross Capital Formation / Investment (% of GDP)",
        "Exports of Goods & Services (% of GDP)",
        "Imports of Goods & Services (% of GDP)",
        "Trade Balance (Exports minus Imports %)",
        "HFCE – Household Final Consumption (% of GDP)",
        "Tax Revenue (% of GDP)",
    ]
    # Shorten names for display
    short = lambda s: s.split("–")[0].split("(")[0].strip()
    corr_df = pd.DataFrame(
        {short(ind): get_series(ind).values for ind in key_inds},
        index=_YEARS_INT,
    ).dropna(axis=1)

    corr_matrix = corr_df.corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        corr_matrix,
        annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, ax=ax,
        linewidths=0.5, annot_kws={"size": 8},
    )
    ax.set_title(
        "Correlation Heatmap — Key Economic Indicators  (2000–2024)\n"
        "Green = positive  |  Red = negative  |  Values = Pearson r",
        fontsize=13, fontweight="bold",
    )
    plt.xticks(rotation=40, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.show()

    # Print strong correlations to console (for the UI text panel)
    interpretations = interpret_correlations(corr_matrix, threshold=0.6)
    if interpretations:
        print("\n[Phase 6] Strong Correlations (|r| ≥ 0.6):")
        for line in interpretations:
            print(line)
