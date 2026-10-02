import unicodedata
import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from google import genai

# 1. CRITICAL: Add the parent directory to Python's path FIRST
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 2. Now Python can safely find and import from the src folder
from src.similarity_engine import PlayerSimilarityEngine
from src.system_fit import SystemFitEngine
from src.common import DATA_DIR, canon_club, name_key, read_json
from src.database import SCHEMA as DB_SCHEMA, read_table

st.set_page_config(
    page_title="StatTrick | Scouting Intelligence",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded"
)

# --- 0. DESIGN TOKENS (shared by CSS + Plotly) ---
MINT = "#35E08C"          
MINT_DIM = "#1F8F59"
CYAN = "#4CC9F0"          
CORAL = "#FF5C7A"         
AMBER = "#FFB020"
INK = "#06100D"           
PANEL = "#0D1A16"
PANEL_HI = "#13241F"
LINE = "#1D332C"
TEXT = "#E9F2ED"
TEXT_DIM = "#93AAA1"
GRID = "#16281F"

POS_ACCENT = {
    "GK": AMBER,
    "CB": CYAN, "FULLBACK": CYAN,
    "CDM": MINT, "CM": MINT, "CAM": MINT,
    "WINGER": CORAL, "ST": CORAL,
}

# Club-name canonicalisation now lives in one place: src/common.py (canon_club), so the
# ingestion pipeline and the app can never disagree on what a club is called (audit #10).

# --- 1. METRICS & LABELS DICTIONARY ---
METRIC_LABELS = {
    "gls_per90": "Goals / 90",
    "ast_per90": "Assists / 90",
    "xg_per90": "Expected Goals (xG) / 90",
    "xag_per90": "Expected Assists (xA) / 90",
    "sh_per90": "Shots / 90",
    "sot_per90": "Shots on Target / 90",
    "prgc_per90": "Progressive Carries / 90",
    "prgp_per90": "Progressive Passes / 90",
    "tkl_per90": "Tackles / 90",
    "int_per90": "Interceptions / 90",
    "clr_per90": "Clearances / 90",
    "blk_per90": "Blocks / 90",
    "aer_won_per90": "Aerials Won / 90",
    "saves_per90": "Saves / 90",
    "psxg_net_per90": "PSxG Net / 90",
    "min": "Career Minutes",
    "actual_value_m": "Market Value (€M)",
    "predicted_value_m": "Performance Value (€M)",
    "status_premium_m": "Status & Contract Premium (€M)",
    "forecast_value_m": "Market Forecast, 12 mo (€M)",
    "forecast_change_pct": "Forecast Change (%)",
    "market_profile_value_m": "Market-Profile Value (€M)",
    "surplus_value_m": "Surplus Value (€M)",
    "surplus_pct": "Surplus (%)",
    "contract_years_left": "Contract (Years)",
    "age_clean": "Age",
    "pos_clean": "Position",
    "squad": "Club",
    "league": "League",
    "has_advanced": "Advanced Data Available",
    "edge_z": "Edge (band widths)",
    "club_goals_pm": "Club Goals / Match",
    "club_caps": "Team-mates' Avg Caps",
    "international_caps": "International Caps",
    "pred_value_low_m": "Fair Band Floor (€M)",
    "predicted_mean_m": "Model Mean Value (€M)",
    "match_method": "TM Match Method",
}

def label(col: str) -> str:
    return METRIC_LABELS.get(col, col.replace("_", " ").title())

def fmt_big(value: float) -> str:
    """Totals: switch to billions above EUR 1,000M so the figure fits a card."""
    if pd.isna(value):
        return "—"
    return f"€{value / 1000:.2f}bn" if abs(value) >= 1000 else fmt_eur_m(value)


def fmt_eur_m(value: float) -> str:
    if pd.isna(value):
        return "—"
    sign = "-" if value < 0 else ""
    return f"{sign}€{abs(value):.1f}M"

def format_value(val, col_name: str) -> str:
    if isinstance(val, (pd.Series, np.ndarray, list)):
        val = val.iloc[0] if hasattr(val, "iloc") and len(val) > 0 else (val[0] if len(val) > 0 else np.nan)

    if pd.isna(val):
        return "—"
    if col_name in ["actual_value_m", "predicted_value_m", "surplus_value_m"]:
        return fmt_eur_m(float(val))
    if col_name == "surplus_pct":
        try:
            return f"{'+' if float(val) >= 0 else ''}{float(val):.1f}%"
        except (ValueError, TypeError):
            return str(val)
    if col_name == "contract_years_left":
        try:
            return f"{float(val):.1f}"
        except (ValueError, TypeError):
            return str(val)
    if "per90" in col_name:
        try:
            return f"{float(val):.2f}"
        except (ValueError, TypeError):
            return str(val)
    if col_name == "min":
        try:
            return f"{int(float(val)):,}"
        except (ValueError, TypeError):
            return str(val)
    if isinstance(val, (float, np.floating)) and val.is_integer():
        return str(int(val))
    return str(val)

def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    return ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')

def get_percentile(df: pd.DataFrame, col: str, target_val: float, position: str = None):
    """Percentile within the position pool, or None when the stat was never recorded for this
    player. (It used to return 50, drawing a missing stat as 'exactly average'.)"""
    if target_val is None or pd.isna(target_val) or col not in df.columns or df[col].isnull().all():
        return None
    if position and "pos_clean" in df.columns:
        pool = df[df["pos_clean"] == position]
        if len(pool) < 15:
            pool = df
    else:
        pool = df
    series = pd.to_numeric(pool[col], errors="coerce").dropna()
    if series.empty: return None
    return int(np.round((series <= target_val).mean() * 100))

def get_radar_metrics(position: str, available_cols: list) -> list:
    position_templates = {
        "ST": ["gls_per90", "xg_per90", "sot_per90", "sh_per90", "aer_won_per90"],
        "WINGER": ["xg_per90", "xag_per90", "prgc_per90", "prgp_per90", "sh_per90"],
        "CAM": ["ast_per90", "xag_per90", "prgp_per90", "prgc_per90", "sh_per90"],
        "CM": ["prgp_per90", "prgc_per90", "tkl_per90", "int_per90", "xag_per90"],
        "CDM": ["tkl_per90", "int_per90", "blk_per90", "clr_per90", "prgp_per90"],
        "FULLBACK": ["tkl_per90", "int_per90", "prgc_per90", "prgp_per90", "clr_per90"],
        "CB": ["clr_per90", "blk_per90", "aer_won_per90", "int_per90", "tkl_per90"],
        "GK": ["saves_per90", "psxg_net_per90", "clr_per90", "prgp_per90", "tkl_per90"],
    }

    selected = position_templates.get(position, position_templates["CM"])
    valid = [m for m in selected if m in available_cols]

    if len(valid) < 5:
        fallbacks = ["xg_per90", "xag_per90", "tkl_per90", "int_per90", "prgc_per90", "prgp_per90", "gls_per90"]
        for f in fallbacks:
            if f in available_cols and f not in valid:
                valid.append(f)
            if len(valid) == 5:
                break
    return valid[:5]

def get_position_form_metric(pos_tag: str, df_p: pd.DataFrame, actuals: bool = False):
    pos_tag = str(pos_tag).upper()

    def get_series(col):
        if col not in df_p.columns:
            return pd.Series(np.nan, index=df_p.index, dtype=float)
        return pd.to_numeric(df_p[col], errors="coerce")

    def total(*cols):
        # A composite is only valid for a season in which EVERY component was tracked.
        # (.add(fill_value=0) summed whatever existed, so Tkl+Int+Clr became Int alone in
        # 2025-26 and every centre-back's line collapsed.)
        return pd.concat([get_series(c) for c in cols], axis=1).sum(axis=1, min_count=len(cols))

    if actuals:
        return total("gls_per90", "ast_per90"), "Goal Contribution / 90 (Gls + Ast)"

    # 1. Determine the primary tracking metric based on position
    if "GK" in pos_tag:
        psxg = get_series("psxg_net_per90")
        if psxg.count() >= 2:
            metric, title = psxg, "Net PSxG / 90 (Shot Stopping Delta)"
        else:
            metric, title = get_series("saves_per90"), "Saves / 90"

    elif "CB" in pos_tag:
        metric = total("tkl_per90", "int_per90", "clr_per90")
        title = "Defensive Interventions / 90 (Tkl+Int+Clr)"

    elif "CDM" in pos_tag:
        metric = total("tkl_per90", "int_per90")
        title = "Ball-Winning Actions / 90 (Tackles + Interceptions)"

    elif "FULLBACK" in pos_tag or pos_tag == "CM":
        metric = total("prgp_per90", "prgc_per90")
        title = "Progression Volume / 90 (PrgP + PrgC)"

    elif "CAM" in pos_tag:
        # xA exists for a single season in the data, so it cannot carry a trend line
        metric = total("prgp_per90", "prgc_per90")
        title = "Progression Volume / 90 (PrgP + PrgC)"

    else:  
        metric = get_series("xg_per90")
        title = "Expected Goals / 90 (xG)"
        
    # 2. UNIVERSAL FALLBACK
    # Missing seasons are already NaN (never 0), so a real 0.00 stays a real 0.00.
    metric_clean = metric

    # If the player has fewer than 2 valid seasons of tracking data, default to Actuals
    if metric_clean.count() < 2:
        return total("gls_per90", "ast_per90"), "Actual Goal Contribution / 90 (Gls + Ast)"
        
    return metric_clean, title

# --- 2. PRESENTATION HELPERS ---
def initials(name: str) -> str:
    parts = [p for p in str(name).split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()

def safe_float(val, fallback=0.0) -> float:
    try:
        f = float(val)
        return fallback if pd.isna(f) else f
    except (TypeError, ValueError):
        return fallback

def section_heading(text: str, hint: str = "") -> None:
    hint_html = f'<span class="sec-hint">{hint}</span>' if hint else ""
    st.markdown(
        f'<div class="sec-head"><span class="sec-tick"></span>'
        f'<span class="sec-title">{text}</span>{hint_html}</div>',
        unsafe_allow_html=True,
    )

def render_identity(name: str, club: str, position: str, league: str, accent: str) -> None:
    st.markdown(
        '<div class="identity">'
        f'<div class="identity-crest" style="border-color: {accent}44; color: {accent};">{initials(name)}</div>'
        '<div class="identity-body">'
        f'<div class="identity-name">{name}</div>'
        '<div class="identity-meta">'
        f'<span class="chip" style="background: {accent}1F; color: {accent}; border-color: {accent}3D;">{position}</span>'
        f'<span class="chip">{club}</span>'
        f'<span class="chip chip-ghost">{league}</span>'
        "</div></div>"
        f'<div class="identity-stripe" style="background: linear-gradient(180deg, {accent}, transparent);"></div>'
        "</div>",
        unsafe_allow_html=True,
    )

def render_valuation_bar(actual: float, predicted: float, low: float, high: float) -> None:
    a, p = safe_float(actual), safe_float(predicted)
    lo, hi = safe_float(low, p * 0.85), safe_float(high, p * 1.15)
    if hi < lo:
        lo, hi = hi, lo
    scale = max(a, p, hi) * 1.15
    if scale <= 0:
        return
    pct = lambda v: max(0.0, min(100.0, v / scale * 100.0))
    band_l, band_w = pct(lo), max(pct(hi) - pct(lo), 1.2)
    a_pos, p_pos = pct(a), pct(p)
    undervalued = p >= a
    verdict_color = MINT if undervalued else CORAL
    verdict = "Model rates him above the market" if undervalued else "Market is paying above the model"

    st.markdown(
        '<div class="valbar-wrap">'
        '<div class="valbar-head">'
        f'<span class="valbar-verdict" style="color: {verdict_color};">{verdict}</span>'
        f'<span class="valbar-scale">0 — {fmt_eur_m(scale)}</span>'
        "</div>"
        '<div class="valbar-track">'
        f'<div class="valbar-band" style="left: {band_l}%; width: {band_w}%;"></div>'
        f'<div class="valbar-fill" style="width: {a_pos}%;"></div>'
        f'<div class="valbar-pin valbar-pin-market" style="left: {a_pos}%;"></div>'
        f'<div class="valbar-pin valbar-pin-model" style="left: {p_pos}%;"></div>'
        "</div>"
        '<div class="valbar-key">'
        f'<span><i class="key-dot" style="background: {CYAN};"></i>Market {fmt_eur_m(a)}</span>'
        f'<span><i class="key-dot" style="background: {MINT};"></i>Model {fmt_eur_m(p)}</span>'
        f'<span><i class="key-dot key-band"></i>Fair band {fmt_eur_m(lo)} – {fmt_eur_m(hi)}</span>'
        "</div></div>",
        unsafe_allow_html=True,
    )

def radar_axes(stats: list, pct_lists: list):
    """Keep only the axes every plotted player has data for."""
    keep = [i for i in range(len(stats)) if all(p[i] is not None for p in pct_lists)]
    dropped = [stats[i] for i in range(len(stats)) if i not in keep]
    return [stats[i] for i in keep], [[p[i] for i in keep] for p in pct_lists], dropped


def render_fit_gauge(club_line: str, archetype: str, tier: str, score) -> None:
    no_score = score is None or pd.isna(score)
    val = max(0.0, min(100.0, safe_float(score)))
    color = TEXT_DIM if no_score else (MINT if val >= 80 else (CYAN if val >= 65 else (AMBER if val >= 45 else CORAL)))
    score = "—" if no_score else score
    sweep = val * 3.6
    st.markdown(
        '<div class="fit-shell">'
        '<div class="fit-copy">'
        f'<div class="fit-club">{club_line}</div>'
        f'<div class="fit-archetype">{archetype}</div>'
        f'<div class="fit-tier" style="color: {color};">{tier}</div>'
        "</div>"
        f'<div class="fit-gauge" style="background: conic-gradient({color} 0deg {sweep}deg, rgba(255,255,255,0.07) {sweep}deg 360deg);">'
        '<div class="fit-gauge-core">'
        f'<div class="fit-gauge-num" style="color: {color};">{score}</div>'
        '<div class="fit-gauge-unit">fit index</div>'
        "</div></div></div>",
        unsafe_allow_html=True,
    )

# --- 3. CSS DESIGN SYSTEM (Floodlit Theme) ---
def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --ink: #06100D;
            --panel: #0D1A16;
            --panel-hi: #13241F;
            --line: #1D332C;
            --mint: #35E08C;
            --cyan: #4CC9F0;
            --coral: #FF5C7A;
            --amber: #FFB020;
            --text: #E9F2ED;
            --text-dim: #93AAA1;
            --text-faint: #6B8279;
            --display: 'Barlow Condensed', 'Inter', sans-serif;
            --body: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            --radius: 14px;
            --shadow: 0 1px 0 rgba(255,255,255,0.03) inset, 0 18px 40px -26px rgba(0,0,0,0.9);
        }

        html, body, [class*="css"] {
            font-family: var(--body) !important;
            color: var(--text) !important;
            -webkit-font-smoothing: antialiased;
        }

        .stApp {
            background:
                radial-gradient(1100px 520px at 8% -12%, rgba(53, 224, 140, 0.10), transparent 62%),
                radial-gradient(900px 460px at 96% -8%, rgba(76, 201, 240, 0.08), transparent 60%),
                var(--ink);
        }
        .block-container {
            padding-top: 1.6rem !important;
            padding-bottom: 4rem !important;
            max-width: 1560px;
        }
        h1, h2, h3, h4, h5 { color: var(--text) !important; letter-spacing: -0.01em; }
        ::selection { background: rgba(53, 224, 140, 0.28); }

        ::-webkit-scrollbar { width: 10px; height: 10px; }
        ::-webkit-scrollbar-track { background: var(--ink); }
        ::-webkit-scrollbar-thumb { background: #22392F; border-radius: 6px; border: 2px solid var(--ink); }
        ::-webkit-scrollbar-thumb:hover { background: #2E4B3E; }

        .hero {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 2.1rem 2.2rem;
            margin-bottom: 1.6rem;
            background: linear-gradient(115deg, #10231C 0%, #0A1714 46%, #081311 100%);
            box-shadow: var(--shadow);
        }
        .hero::before {
            content: "";
            position: absolute; inset: 0;
            background: repeating-linear-gradient(
                100deg,
                rgba(255,255,255,0.022) 0 78px,
                rgba(255,255,255,0) 78px 156px
            );
            pointer-events: none;
        }
        .hero::after {
            content: "";
            position: absolute;
            right: -110px; top: 50%;
            width: 360px; height: 360px;
            transform: translateY(-50%);
            border: 1px solid rgba(53, 224, 140, 0.16);
            border-radius: 50%;
            pointer-events: none;
        }
        .hero-inner {
            position: relative;
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 2rem;
            flex-wrap: wrap;
        }
        .hero-kicker {
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--mint);
            letter-spacing: 0.04em;
            margin-bottom: 0.45rem;
        }
        .hero-title {
            font-family: var(--display);
            font-size: 4.1rem;
            font-weight: 700;
            line-height: 0.88;
            letter-spacing: -0.015em;
            color: #FFFFFF;
            text-transform: uppercase;
        }
        .hero-title em { font-style: normal; color: var(--mint); }
        .hero-sub {
            color: var(--text-dim);
            font-size: 0.97rem;
            max-width: 46ch;
            margin-top: 0.7rem;
            line-height: 1.5;
        }
        .hero-stat {
            text-align: right;
            border-left: 1px solid var(--line);
            padding-left: 1.6rem;
        }
        .hero-stat-num {
            font-family: var(--display);
            font-size: 3.3rem;
            font-weight: 700;
            line-height: 1;
            color: #FFFFFF;
            font-variant-numeric: tabular-nums;
        }
        .hero-stat-cap {
            color: var(--text-dim);
            font-size: 0.82rem;
            font-weight: 500;
            margin-top: 0.25rem;
        }
        .hero-live {
            display: inline-flex; align-items: center; gap: 7px;
            margin-top: 0.7rem;
            font-size: 0.74rem; font-weight: 600; color: var(--mint);
        }
        .hero-live i {
            width: 6px; height: 6px; border-radius: 50%;
            background: var(--mint); box-shadow: 0 0 0 3px rgba(53,224,140,0.16);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0C1815, #081311) !important;
            border-right: 1px solid var(--line) !important;
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.7rem; }
        [data-testid="stSidebar"] * { color: var(--text-dim) !important; }
        [data-testid="stSidebar"] label p { color: var(--text) !important; font-weight: 600 !important; font-size: 0.85rem !important; }
        [data-testid="stSidebar"] .stMarkdown h3 {
            font-family: var(--display) !important;
            font-size: 1.15rem !important;
            text-transform: uppercase;
            letter-spacing: 0.02em;
            color: var(--text) !important;
        }
        [data-testid="stSidebar"] a {
            color: var(--mint) !important;
            text-decoration: none !important;
            border-bottom: 1px solid rgba(53,224,140,0.3);
        }
        [data-testid="stSidebar"] hr { border-color: var(--line) !important; margin: 1.5rem 0 !important; }

        div[data-baseweb="select"] > div {
            background-color: #0F1F1A !important;
            border: 1px solid var(--line) !important;
            border-radius: 10px !important;
            min-height: 44px;
            color: var(--text) !important;
            transition: border-color .16s ease, box-shadow .16s ease;
        }
        div[data-baseweb="select"] > div:hover { border-color: #2A4A3E !important; }
        div[data-baseweb="select"] > div:focus-within {
            border-color: var(--mint) !important;
            box-shadow: 0 0 0 3px rgba(53,224,140,0.15) !important;
        }
        div[data-baseweb="select"] span { color: var(--text) !important; }
        div[data-baseweb="select"] svg { fill: var(--text-dim) !important; }
        div[data-baseweb="popover"] li { background-color: #0F1F1A !important; color: var(--text) !important; }
        div[data-baseweb="popover"] li:hover { background-color: #16302A !important; }
        div[data-baseweb="select"] span[data-baseweb="tag"] {
            background-color: rgba(53,224,140,0.14) !important;
            border: 1px solid rgba(53,224,140,0.3) !important;
            border-radius: 7px !important;
        }
        div[data-baseweb="select"] span[data-baseweb="tag"] span { color: var(--mint) !important; font-weight: 600 !important; }
        [data-testid="stWidgetLabel"] p {
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            color: var(--text-dim) !important;
        }
        [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
            background-color: var(--mint) !important;
            border: 2px solid #0B1714 !important;
            box-shadow: 0 0 0 4px rgba(53,224,140,0.16) !important;
        }
        [data-testid="stSlider"] [data-testid="stThumbValue"] { color: var(--mint) !important; font-weight: 700 !important; }
        [data-testid="stSlider"] [data-testid="stTickBarMin"],
        [data-testid="stSlider"] [data-testid="stTickBarMax"] { color: var(--text-faint) !important; font-size: .72rem !important; }
        [data-testid="stCheckbox"] [data-baseweb="checkbox"] span[aria-checked="true"] {
            background-color: var(--mint) !important; border-color: var(--mint) !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: rgba(13, 26, 22, 0.75);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 5px;
            flex-wrap: nowrap;
            overflow-x: auto;
            scrollbar-width: none;
        }
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
        .stTabs [data-baseweb="tab"] {
            position: relative;
            flex: 0 0 auto !important;
            width: auto !important;
            white-space: nowrap;
            background: transparent !important;
            color: var(--text-dim) !important;
            padding: 11px 20px 13px !important;
            border-radius: 9px !important;
            height: auto !important;
            transition: background-color .16s ease, color .16s ease;
        }
        .stTabs [data-baseweb="tab"] p {
            font-size: 0.9rem !important;
            font-weight: 600 !important;
            color: inherit !important;
            margin: 0 !important;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: var(--text) !important;
            background: rgba(255,255,255,0.035) !important;
        }
        .stTabs [aria-selected="true"] {
            color: #FFFFFF !important;
            background: linear-gradient(180deg, #17332B, #101F1B) !important;
            box-shadow: inset 0 0 0 1px rgba(53,224,140,0.26);
        }
        .stTabs [aria-selected="true"]::after {
            content: "";
            position: absolute;
            left: 20px; right: 20px; bottom: 6px;
            height: 2px; border-radius: 2px;
            background: var(--mint);
            box-shadow: 0 0 10px rgba(53,224,140,0.55);
        }
        .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }
        .stTabs [data-baseweb="tab-panel"] { padding-top: 1.7rem; }

        div[data-testid="stAlert"] {
            background: linear-gradient(180deg, rgba(53,224,140,0.07), rgba(53,224,140,0.03)) !important;
            border: 1px solid rgba(53,224,140,0.22) !important;
            border-left: 3px solid var(--mint) !important;
            border-radius: 11px !important;
            padding: .9rem 1.1rem !important;
        }
        div[data-testid="stAlert"] p, div[data-testid="stAlert"] li {
            color: #CDE6DA !important; font-size: .88rem !important; line-height: 1.55 !important;
        }
        div[data-testid="stAlert"] strong { color: var(--mint) !important; }
        div[data-testid="stAlert"] svg { fill: var(--mint) !important; }
        .stCaption, [data-testid="stCaptionContainer"] p {
            color: var(--text-faint) !important; font-size: .82rem !important; line-height: 1.55 !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--panel) !important;
            border: 1px solid var(--line) !important;
            border-radius: var(--radius) !important;
            box-shadow: var(--shadow);
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--line) !important;
            border-radius: 12px !important;
            overflow: hidden;
        }

        .metric-card {
            position: relative;
            height: 100%;
            min-height: 116px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding: 1.15rem 1.3rem;
            border: 1px solid var(--line);
            border-radius: var(--radius);
            background: linear-gradient(160deg, var(--panel-hi), var(--panel) 62%);
            overflow: hidden;
            transition: border-color .18s ease, transform .18s ease, box-shadow .18s ease;
        }
        .metric-card::after {
            content: "";
            position: absolute; inset: 0;
            background: repeating-linear-gradient(100deg, rgba(255,255,255,0.016) 0 40px, rgba(255,255,255,0) 40px 80px);
            pointer-events: none;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: #2B4A3D;
            box-shadow: 0 20px 36px -26px rgba(0,0,0,0.95);
        }
        .metric-top-bar { position: absolute; top: 0; left: 0; right: 0; height: 2px; }
        .metric-top-bar.positive { background: linear-gradient(90deg, var(--mint), rgba(53,224,140,0)); }
        .metric-top-bar.negative { background: linear-gradient(90deg, var(--coral), rgba(255,92,122,0)); }
        .metric-top-bar.neutral  { background: linear-gradient(90deg, var(--cyan), rgba(76,201,240,0)); }
        .metric-label {
            color: var(--text-faint);
            font-size: .76rem;
            font-weight: 500;
            position: relative;
        }
        .metric-value {
            font-family: var(--display);
            font-size: 2.15rem;
            font-weight: 700;
            line-height: 1.05;
            margin-top: .3rem;
            color: #FFFFFF;
            font-variant-numeric: tabular-nums;
            position: relative;
        }
        .metric-delta {
            font-size: .79rem; font-weight: 600; margin-top: .4rem; position: relative;
            font-variant-numeric: tabular-nums;
        }
        .metric-delta.positive { color: var(--mint); }
        .metric-delta.negative { color: var(--coral); }
        .metric-delta.neutral  { color: var(--text-faint); font-weight: 500; }

        .identity {
            position: relative;
            display: flex; align-items: center; gap: 1.1rem;
            padding: 1.15rem 1.4rem;
            border: 1px solid var(--line);
            border-radius: var(--radius);
            background: linear-gradient(120deg, var(--panel-hi), var(--panel) 70%);
            margin-bottom: 1rem;
            overflow: hidden;
        }
        .identity-stripe { position: absolute; left: 0; top: 0; bottom: 0; width: 3px; }
        .identity-crest {
            width: 56px; height: 56px; flex: none;
            display: grid; place-items: center;
            border: 1px solid; border-radius: 12px;
            background: rgba(255,255,255,0.03);
            font-family: var(--display); font-size: 1.5rem; font-weight: 700;
        }
        .identity-name {
            font-family: var(--display);
            font-size: 2.05rem; font-weight: 700; line-height: 1;
            color: #FFFFFF; text-transform: uppercase; letter-spacing: -0.01em;
        }
        .identity-meta { display: flex; gap: 8px; margin-top: .55rem; flex-wrap: wrap; }
        .chip {
            font-size: .76rem; font-weight: 600;
            padding: 3px 10px; border-radius: 999px;
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.035);
            color: var(--text-dim);
        }
        .chip-ghost { color: var(--text-faint); background: transparent; }

        .valbar-wrap { padding: .25rem .1rem .1rem; }
        .valbar-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: .7rem; }
        .valbar-verdict { font-size: .88rem; font-weight: 600; }
        .valbar-scale { font-size: .74rem; color: var(--text-faint); font-variant-numeric: tabular-nums; }
        .valbar-track {
            position: relative; height: 14px; border-radius: 7px;
            background: #0B1714; border: 1px solid var(--line); overflow: hidden;
        }
        .valbar-band { position: absolute; top: 0; bottom: 0; background: rgba(53,224,140,0.16); }
        .valbar-fill {
            position: absolute; top: 0; bottom: 0; left: 0;
            background: linear-gradient(90deg, rgba(76,201,240,0.12), rgba(76,201,240,0.42));
        }
        .valbar-pin { position: absolute; top: -3px; bottom: -3px; width: 2px; transform: translateX(-1px); }
        .valbar-pin-market { background: var(--cyan); box-shadow: 0 0 8px rgba(76,201,240,0.7); }
        .valbar-pin-model  { background: var(--mint);  box-shadow: 0 0 8px rgba(53,224,140,0.7); }
        .valbar-key { display: flex; gap: 1.3rem; margin-top: .7rem; flex-wrap: wrap; }
        .valbar-key span { font-size: .78rem; color: var(--text-dim); display: inline-flex; align-items: center; gap: 7px; }
        .key-dot { width: 8px; height: 8px; border-radius: 2px; display: inline-block; }
        .key-band { background: rgba(53,224,140,0.3); border: 1px solid rgba(53,224,140,0.5); }

        .fit-shell {
            display: flex; align-items: center; justify-content: space-between; gap: 1.4rem;
            padding: 1rem 1.25rem;
            border: 1px solid var(--line); border-radius: var(--radius);
            background: linear-gradient(120deg, var(--panel-hi), var(--panel) 70%);
            margin-top: 0.35rem;
        }
        .fit-club { font-size: .78rem; color: var(--text-faint); font-weight: 500; }
        .fit-archetype {
            font-family: var(--display); font-size: 1.65rem; font-weight: 700;
            color: #FFFFFF; line-height: 1.05; margin-top: .2rem; text-transform: uppercase;
        }
        .fit-tier { font-size: .84rem; font-weight: 600; margin-top: .3rem; }
        .fit-gauge {
            width: 96px; height: 96px; flex: none; border-radius: 50%;
            display: grid; place-items: center;
        }
        .fit-gauge-core {
            width: 76px; height: 76px; border-radius: 50%;
            background: #0B1714; display: grid; place-items: center; text-align: center;
        }
        .fit-gauge-num {
            font-family: var(--display); font-size: 1.95rem; font-weight: 700; line-height: 1;
            font-variant-numeric: tabular-nums;
        }
        .fit-gauge-unit { font-size: .60rem; color: var(--text-faint); margin-top: 2px; }

        .sec-head { display: flex; align-items: baseline; gap: 11px; margin: 2.2rem 0 .8rem; }
        .sec-tick { width: 3px; height: 17px; border-radius: 2px; background: var(--mint); transform: translateY(3px); }
        .sec-title {
            font-family: var(--display); font-size: 1.5rem; font-weight: 700;
            text-transform: uppercase; color: #FFFFFF;
        }
        .sec-hint { font-size: .84rem; color: var(--text-faint); }
        .panel-title {
            font-size: .82rem; font-weight: 600; color: var(--text-dim); margin-bottom: .5rem;
        }
        .result-count {
            font-size: .82rem; color: var(--text-dim); margin: .9rem 0 .5rem;
            font-variant-numeric: tabular-nums;
        }
        .result-count b { color: var(--mint); font-weight: 700; }

        .stDownloadButton button, .stButton button {
            background: rgba(53,224,140,0.10) !important;
            border: 1px solid rgba(53,224,140,0.32) !important;
            color: var(--mint) !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: .84rem !important;
            padding: .5rem 1.1rem !important;
            transition: background-color .16s ease, border-color .16s ease;
        }
        .stDownloadButton button:hover, .stButton button:hover {
            background: rgba(53,224,140,0.18) !important;
            border-color: var(--mint) !important;
            color: var(--mint) !important;
        }
        .stDownloadButton button:focus, .stButton button:focus {
            box-shadow: 0 0 0 3px rgba(53,224,140,0.18) !important;
        }

        .side-brand {
            display: flex; align-items: center; gap: 10px;
            padding-bottom: 1.1rem; margin-bottom: 1.2rem;
            border-bottom: 1px solid var(--line);
        }
        .side-brand-mark {
            width: 34px; height: 34px; border-radius: 9px; flex: none;
            display: grid; place-items: center;
            background: rgba(53,224,140,0.12);
            border: 1px solid rgba(53,224,140,0.3);
            color: var(--mint);
            font-family: var(--display); font-weight: 700; font-size: 1.1rem;
        }
        .side-brand-name {
            font-family: var(--display); font-size: 1.3rem; font-weight: 700;
            color: #FFFFFF; text-transform: uppercase; line-height: 1;
        }
        .side-brand-role { font-size: .72rem; color: var(--text-faint); margin-top: 2px; }

        @media (prefers-reduced-motion: reduce) {
            .metric-card, .stTabs [data-baseweb="tab"] { transition: none !important; }
            .metric-card:hover { transform: none; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_metrics_html(cards_data: list):
    cols = st.columns(len(cards_data))
    for col, (label_text, value_text, delta_text, sentiment) in zip(cols, cards_data):
        delta_tag = f'<div class="metric-delta {sentiment}">{delta_text}</div>' if delta_text else ""
        card = (
            f'<div class="metric-card">'
            f'<div class="metric-top-bar {sentiment}"></div>'
            f'<div class="metric-label">{label_text}</div>'
            f'<div class="metric-value">{value_text}</div>'
            f"{delta_tag}"
            f"</div>"
        )
        col.markdown(card, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def generate_tactical_brief(player_dict, squad_dict, fit_data, feature_stats, _api_key):
    """Generates a constrained AI scouting brief based strictly on vector math.

    _api_key is prefixed with an underscore so Streamlit excludes it from the cache key
    (it isn't part of "what changed"). More importantly, this function now RAISES on
    failure instead of returning an error string: st.cache_data only caches a successful
    return, so a transient API error is never cached and poisoning a future good call
    (the original bug - audit #11). The caller is responsible for catching the exception.
    """
    from src.system_fit import STYLE_FEATURES

    # Compare Z-SCORE deltas, not raw per-90 deltas: large-scale metrics like progressive
    # passes used to always "win" over small-scale ones like xG regardless of which one
    # was actually more unusual for this player, since raw deltas aren't comparable across
    # metrics with very different scales.
    deltas = {}
    for feature in STYLE_FEATURES:
        mean, std = feature_stats.get(feature, (0.0, 1.0))
        p_z = (float(player_dict.get(feature, mean) or mean) - mean) / std
        s_z = (float(squad_dict.get(feature, mean) or mean) - mean) / std
        deltas[feature] = p_z - s_z

    best_feature = max(deltas, key=deltas.get)
    worst_feature = min(deltas, key=deltas.get)

    prompt = f"""
    You are a Lead Tactical Scout for {fit_data['target_squad']}.
    Write a 3-sentence executive summary evaluating a player's fit for our "{fit_data['archetype']}" system.

    MANDATORY CONSTRAINTS:
    - You MUST state that their strongest tactical synergy is {label(best_feature).replace(' / 90', '')} ({deltas[best_feature]:+.2f} standard deviations vs our squad average).
    - You MUST state that their biggest tactical friction is {label(worst_feature).replace(' / 90', '')} ({deltas[worst_feature]:+.2f} standard deviations vs our squad average).
    - Do NOT hallucinate any other statistics. Do not use filler introductions.
    - Write in a professional, analytical front-office tone.
    """

    client = genai.Client(api_key=_api_key)
    response = client.models.generate_content(
        model='gemini-3.8-flash',   # NOTE: confirm this model string against your key with src/test.py
        contents=prompt,
        config={"temperature": 0.2},
    )
    return response.text

def apply_custom_theme(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT, size=12),
        colorway=[MINT, CYAN, CORAL, AMBER, "#A78BFA"],
        margin=dict(t=44, b=40, l=40, r=40),
        hoverlabel=dict(
            bgcolor="#0F1F1A",
            bordercolor=LINE,
            font=dict(family="Inter, sans-serif", color=TEXT, size=12),
        ),
        legend=dict(font=dict(color=TEXT_DIM, size=11)),
    )
    fig.update_xaxes(
        gridcolor=GRID, zerolinecolor="#22392F", linecolor=LINE,
        tickfont=dict(color=TEXT_DIM, size=11), title_font=dict(color=TEXT_DIM, size=12),
    )
    fig.update_yaxes(
        gridcolor=GRID, zerolinecolor="#22392F", linecolor=LINE,
        tickfont=dict(color=TEXT_DIM, size=11), title_font=dict(color=TEXT_DIM, size=12),
    )
    fig.update_polars(
        bgcolor="rgba(255,255,255,0.018)",
        radialaxis=dict(gridcolor=GRID, linecolor=GRID),
        angularaxis=dict(gridcolor=GRID, linecolor="#22392F", tickfont=dict(color=TEXT_DIM, size=11)),
    )
    return fig

# --- 4. DATA LOADER & CACHING ---
# --- 4. DATA LOADER & CACHING ---
@st.cache_data(show_spinner=False, ttl=3600)
def load_data():
    try:
        df = read_table("players_master")
    except ValueError:
        st.error("Database not found in Postgres. Run src/valuation_model.py first.")
        st.stop()
        
    df.columns = [c.lower() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()].copy()
    
    if "pos_clean" not in df.columns:
        pos_cols = [c for c in df.columns if c.startswith("pos_clean_")]
        if pos_cols: df["pos_clean"] = df[pos_cols].idxmax(axis=1).str.replace("pos_clean_", "")
        else: df["pos_clean"] = "MF"

    if "league" not in df.columns:
        league_cols = [c for c in df.columns if c.startswith("league_")]
        if league_cols: df["league"] = df[league_cols].idxmax(axis=1).str.replace("league_", "")
        else: df["league"] = "Unknown"

    df["pos_clean"] = df["pos_clean"].astype(str).str.upper()
    df["squad"] = df["squad"].astype(str).map(canon_club)
    df["league"] = df["league"].astype(str).str.strip()
    if "player" not in df.columns and "fb_name" in df.columns:
        df["player"] = df["fb_name"]
    return df

@st.cache_resource(ttl=3600)
def load_engine():
    return PlayerSimilarityEngine()

@st.cache_resource(ttl=3600)
def load_system_engine():
    engine = SystemFitEngine()
    if hasattr(engine, "club_profiles") and "squad" in engine.club_profiles.columns:
        engine.club_profiles["squad"] = engine.club_profiles["squad"].map(canon_club)
    return engine

def season_rollup(history: pd.DataFrame) -> pd.DataFrame:
    """One row per season. A mid-season move leaves two stints; combine them weighted by
    minutes instead of discarding the smaller one."""
    if history.empty:
        return history
    per90 = [c for c in history.columns if c.endswith("_per90")]
    rows = []
    for season, g in history.groupby("season", sort=True):
        mins = pd.to_numeric(g["min"], errors="coerce").fillna(0.0)
        row = {"season": season, "min": float(mins.sum()),
               "squad": " / ".join(dict.fromkeys(g.sort_values("min", ascending=False)["squad"].astype(str)))}
        for c in per90:
            v = pd.to_numeric(g[c], errors="coerce")
            w = mins[v.notna()]
            row[c] = float((v.dropna() * w).sum() / w.sum()) if w.sum() > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, ttl=3600)
def load_timeline_data():
    try:
        df_time = read_table("player_timeline")
        df_time.columns = [c.lower() for c in df_time.columns]
        if "squad" in df_time.columns:
            df_time["squad"] = df_time["squad"].astype(str).map(canon_club)
        return df_time
    except ValueError:
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=3600)
def load_value_history():
    try:
        h = read_table("player_value_history")
        h["date"] = pd.to_datetime(h["date"], errors="coerce")
        return h.dropna(subset=["date"])
    except Exception:                      # table not built yet (no valuation history downloaded)
        return pd.DataFrame(columns=["pkey", "date", "value_m"])


# --- 5. APP EXECUTION ---
system_engine = load_system_engine()
inject_custom_css()
df_master = load_data()
df_timeline = load_timeline_data()

try:
    engine = load_engine()
except Exception as e:
    st.error(f"Error loading similarity engine: {e}")
    st.stop()

# Hero Header
st.markdown(
    '<div class="hero"><div class="hero-inner">'
    "<div>"
    '<div class="hero-kicker">Scouting &amp; valuation intelligence</div>'
    '<div class="hero-title">Stat<em>Trick</em></div>'
    '<div class="hero-sub">Find the player you already have, somewhere cheaper. '
    "Tactical clones, fair-value modelling and system fit in one place.</div>"
    "</div>"
    '<div class="hero-stat">'
    f'<div class="hero-stat-num">{len(df_master):,}</div>'
    '<div class="hero-stat-cap">players in the database</div>'
    '<div class="hero-live"><i></i>Model synced</div>'
    "</div></div></div>",
    unsafe_allow_html=True,
)

all_players = sorted(df_master["player"].dropna().unique().tolist())


def _find_default_player(players: list, preferred: str) -> int:
    """Exact match first, then accent/casing-insensitive match, then the most expensive
    player in the database - never a silent, unexplained fall-back to row 0."""
    if preferred in players:
        return players.index(preferred)
    target_key = name_key(preferred)
    for i, p in enumerate(players):
        if name_key(p) == target_key:
            return i
    fallback = df_master.loc[df_master["actual_value_m"].idxmax(), "player"] if len(df_master) else players[0]
    return players.index(fallback) if fallback in players else 0


default_idx = _find_default_player(all_players, "Fermín López")

player_search_dict = {}
for p in all_players:
    clean_name = normalize_name(p)
    if clean_name != p:
        player_search_dict[p] = f"{p} ({clean_name})"
    else:
        player_search_dict[p] = p

st.sidebar.markdown(
    '<div class="side-brand">'
    '<div class="side-brand-mark">ST</div>'
    '<div><div class="side-brand-name">StatTrick</div>'
    '<div class="side-brand-role">Recruitment console</div></div>'
    "</div>",
    unsafe_allow_html=True,
)

target = st.sidebar.selectbox(
    "Select Target Player", 
    all_players, 
    index=default_idx,
    format_func=lambda x: player_search_dict[x]
)
top_k = st.sidebar.slider("Number of Clones", 3, 10, 5)
filter_pos = st.sidebar.checkbox("Enforce Same Position Group", False)

# A player stays in the database for a season after his last top-5-league minutes, so someone who
# has left (retired, moved to MLS / Saudi Arabia, relegated) would still be offered as a target.
CURRENT_SEASON = str(df_master["last_season"].max()) if "last_season" in df_master.columns else None
active_only = st.sidebar.checkbox(
    "Only players active this season", True,
    help=f"Hides anyone with no top-5-league minutes in {CURRENT_SEASON}. They may have left for "
         "another league, retired, or not yet played 90 minutes this season.")
if CURRENT_SEASON:
    df_master["active_now"] = df_master["last_season"].astype(str) == CURRENT_SEASON
else:
    df_master["active_now"] = True
RECRUITABLE = set(df_master.loc[df_master["active_now"], "player"]) if active_only else set(df_master["player"])

st.sidebar.markdown("---")
st.sidebar.markdown("### DATA SOURCES")
st.sidebar.caption(
    "• **Tactical data:** [FBref](https://fbref.com/)\n\n"
    "• **xG & xA since 2025-26:** [Understat](https://understat.com/)\n\n"
    "• **Financials:** [Transfermarkt](https://www.transfermarkt.com/)"
)

if DB_SCHEMA:
    st.sidebar.warning(f"Reading the **{DB_SCHEMA}** schema, not production.")

metrics = read_json(DATA_DIR / "model_metrics.json", default={})
last_modified = metrics.get("built_on")
if not last_modified:
    db_path = DATA_DIR / "master_scouting_db.csv"
    last_modified = (datetime.fromtimestamp(db_path.stat().st_mtime).strftime("%b %d, %Y")
                     if db_path.exists() else "Weekly")

sync_detail = ""
if metrics:
    sync_detail = (f" · OOF R² {metrics.get('oof_r2_eur', '—')} · "
                   f"{metrics.get('players_without_advanced_stats', 0)} players missing xG/xA data")

st.sidebar.caption(
    f"• **Sync Cadence:** Automated weekly (Mondays 04:00 UTC)\n\n"
    f"*Last Model Sync: {last_modified} | XGBoost + Exp Decay (out-of-fold){sync_detail}*"
)

# Shortlist lives in this browser session only (it is not saved to the database)
st.session_state.setdefault("shortlist", [])


def _shortlist_add(names):
    for n_ in names:
        if n_ not in st.session_state["shortlist"]:
            st.session_state["shortlist"].append(n_)


def _shortlist_remove(names):
    st.session_state["shortlist"] = [n_ for n_ in st.session_state["shortlist"] if n_ not in set(names)]


tab1, tab_replace, tab2, tab3, tab4, tab_team, tab_short, tab5 = st.tabs(
    ["Player", "Replace a Player", "Arbitrage Screener", "Head-to-Head Sandbox", "Gap Analysis",
     "Team View", "Shortlist", "Methodology"])      # label must stay constant: a changing label resets the active tab
st.sidebar.caption(f"**Shortlist:** {len(st.session_state['shortlist'])} player(s) saved this session")

WHY_LABELS = {
    "why_age": "Age", "why_experience": "Career minutes", "why_output": "On-pitch output",
    "why_league": "League", "why_position": "Position", "why_coverage": "Data coverage",
}


def render_why(row: pd.Series) -> None:
    """Factor-by-factor explanation of the model's estimate. Each factor multiplies the value of a
    typical player; the effects come from the same out-of-fold model that priced him."""
    items = [(lab, safe_float(row.get(col), np.nan)) for col, lab in WHY_LABELS.items() if col in row.index]
    items = [(lab, c) for lab, c in items if pd.notna(c)]
    if not items:
        return
    items.sort(key=lambda x: -abs(x[1]))
    biggest = max(abs(c) for _, c in items) or 1.0
    rows = []
    for lab, c in items:
        pct = (np.exp(c) - 1) * 100
        color = MINT if c >= 0 else CORAL
        width = max(abs(c) / biggest * 50, 0.6)
        side = f"left: 50%; width: {width}%;" if c >= 0 else f"right: 50%; width: {width}%;"
        rows.append(
            '<div style="display:grid; grid-template-columns: 150px 1fr 70px; align-items:center; gap:12px; margin:6px 0;">'
            f'<span style="font-size:.8rem; color:var(--text-dim);">{lab}</span>'
            '<span style="position:relative; height:10px; background:#0B1714; border:1px solid var(--line); border-radius:5px;">'
            '<i style="position:absolute; left:50%; top:-2px; bottom:-2px; width:1px; background:var(--line);"></i>'
            f'<i style="position:absolute; top:0; bottom:0; {side} background:{color}; opacity:.75; border-radius:4px;"></i></span>'
            f'<span style="font-size:.8rem; font-weight:600; text-align:right; color:{color}; font-variant-numeric:tabular-nums;">{pct:+.0f}%</span>'
            '</div>')
    base = np.expm1(safe_float(row.get("why_base"), 0.0))
    st.markdown('<div style="padding:.2rem .1rem;">' + "".join(rows) + "</div>", unsafe_allow_html=True)
    st.caption(f"Starting from a typical player in the database ({fmt_eur_m(base)}), each factor raises or lowers "
               "the estimate by the percentage shown; together they give the performance value. "
               "Club, international status and contract are deliberately left out.")

# --- TAB 1: TACTICAL CLONING ---
with tab1:
    target_row = df_master[df_master["player"] == target].iloc[0]
    pos = target_row.get("pos_clean", "MF")
    accent = POS_ACCENT.get(str(pos).upper(), MINT)

    render_identity(
        target,
        str(target_row.get("squad", "N/A")),
        str(pos),
        str(target_row.get("league", "—")),
        accent,
    )
    if target in st.session_state["shortlist"]:
        st.button("✓ On shortlist · remove", key="sl_toggle", on_click=_shortlist_remove, args=([target],))
    else:
        st.button("＋ Add to shortlist", key="sl_toggle", on_click=_shortlist_add, args=([target],))
    if CURRENT_SEASON and str(target_row.get("last_season")) != CURRENT_SEASON:
        st.warning(f"**No top-5-league minutes in {CURRENT_SEASON}.** His last recorded season is "
                   f"{target_row.get('last_season')} at {target_row.get('squad')}. He may have left for another "
                   "league, retired, or not yet played this season; club, value and contract may be out of date.")

    core_cards = [
        (label("squad"), str(target_row.get("squad", "N/A")), None, "neutral"),
        (label("pos_clean"), str(pos), None, "neutral"),
        (label("age_clean"), int(target_row.get("age_clean", 25)), None, "neutral"),
        (label("min"), f"{int(target_row.get('min', 0)):,}", None, "neutral"),
        (label("contract_years_left"), format_value(target_row.get("contract_years_left", 2), "contract_years_left"), None, "neutral"),
    ]
    render_metrics_html(core_cards)
    st.write("")

    # The page is long, so it is split into three views of the same player.
    sub_value, sub_style, sub_fit = st.tabs(["Value", "Style & clones", "Fit & form"])
    with sub_value:
        actual_val = target_row.get("actual_value_m", 0.0)
        pred_val = target_row.get("predicted_value_m", 0.0)
        pred_low = target_row.get("pred_value_low_m", pred_val * 0.85)
        pred_high = target_row.get("pred_value_high_m", pred_val * 1.15)
        surplus_val = target_row.get("surplus_value_m", 0.0)

        sentiment = "positive" if surplus_val > 0 else "negative"
        delta_str = f"{'+' if surplus_val > 0 else ''}€{surplus_val:.1f}M ({'Undervalued' if surplus_val > 0 else 'Market premium'})"
        band_str = f"Fair band {fmt_eur_m(pred_low)} – {fmt_eur_m(pred_high)}"

        financial_cards = [
            ("Market value · Transfermarkt", fmt_eur_m(actual_val), "Public consensus price", "neutral"),
            ("Performance value · model", fmt_eur_m(pred_val), band_str, "neutral"),
        ]
        premium = target_row.get("status_premium_m")
        if premium is not None and pd.notna(premium):
            financial_cards.append((
                "Status & contract premium", f"{'+' if premium > 0 else ''}{fmt_eur_m(premium)}",
                "What club level, caps and contract add" if premium >= 0 else "Club level, caps and contract pull it down",
                "neutral"))
        financial_cards.append(("Market vs performance", fmt_eur_m(surplus_val), delta_str, sentiment))
        render_metrics_html(financial_cards)

        # Market forecast: a separate model that openly uses Transfermarkt's own value history
        fc_val = target_row.get("forecast_value_m") if "forecast_value_m" in target_row.index else None
        if fc_val is not None and pd.notna(fc_val):
            fc_chg = safe_float(target_row.get("forecast_change_pct"))
            fc_sent = "positive" if fc_chg > 2 else ("negative" if fc_chg < -2 else "neutral")
            st.write("")
            render_metrics_html([
                ("Market forecast · 12 months", fmt_eur_m(fc_val),
                 f"{fc_chg:+.0f}% from {fmt_eur_m(target_row.get('forecast_base_value_m'))} "
                 f"(valued {target_row.get('forecast_base_date')})", fc_sent),
                ("Likely range", f"{fmt_eur_m(target_row.get('forecast_low_m'))} – {fmt_eur_m(target_row.get('forecast_high_m'))}",
                 "70% of back-test misses fell inside this", "neutral"),
                ("What this is", "Price outlook",
                 "Predicts Transfermarkt's next valuation from its history, age, club and output", "neutral"),
            ])

        # Market value over time, with the 12-month forecast as a dashed continuation
        vh = load_value_history()
        vh = vh[vh["pkey"] == target_row.get("pkey")].sort_values("date") if len(vh) else vh
        if len(vh) >= 2:
            st.write("")
            with st.container(border=True):
                st.markdown('<div class="panel-title">Transfermarkt value over time · and where the forecast puts it next</div>',
                            unsafe_allow_html=True)
                fig_val = go.Figure()
                fig_val.add_trace(go.Scatter(
                    x=vh["date"], y=vh["value_m"], mode="lines+markers", name="Transfermarkt value",
                    line=dict(color=CYAN, width=2.5), marker=dict(size=6, color=INK, line=dict(color=CYAN, width=2)),
                    hovertemplate="%{x|%b %Y}<br>€%{y:.1f}M<extra></extra>"))
                if fc_val is not None and pd.notna(fc_val):
                    base_date = pd.to_datetime(target_row.get("forecast_base_date"), errors="coerce")
                    if pd.notna(base_date):
                        fc_date = base_date + pd.DateOffset(years=1)
                        lo_, hi_ = safe_float(target_row.get("forecast_low_m"), fc_val), safe_float(target_row.get("forecast_high_m"), fc_val)
                        fig_val.add_trace(go.Scatter(
                            x=[base_date, fc_date], y=[safe_float(target_row.get("forecast_base_value_m")), fc_val],
                            mode="lines", line=dict(color=MINT, width=2, dash="dash"), hoverinfo="skip", showlegend=False))
                        fig_val.add_trace(go.Scatter(
                            x=[fc_date], y=[fc_val], mode="markers", name="Forecast (12 months)",
                            marker=dict(size=11, color=MINT, symbol="diamond", line=dict(color=INK, width=2)),
                            error_y=dict(type="data", symmetric=False, array=[max(hi_ - fc_val, 0)],
                                         arrayminus=[max(fc_val - lo_, 0)], color=MINT, thickness=1.5, width=6),
                            hovertemplate="Forecast %{x|%b %Y}<br>€%{y:.1f}M"
                                          f"<br>likely range €{lo_:.1f}M – €{hi_:.1f}M<extra></extra>"))
                perf_ = safe_float(pred_val, np.nan)
                if pd.notna(perf_):
                    fig_val.add_hline(y=perf_, line=dict(color=AMBER, width=1.2, dash="dot"),
                                      annotation_text=f"Performance value €{perf_:.1f}M",
                                      annotation_position="bottom left", annotation_font=dict(color=AMBER, size=11))
                fig_val.update_layout(height=280, margin=dict(t=20, b=30, l=45, r=20),
                                      legend=dict(orientation="h", y=1.12, x=0), yaxis_title="€M",
                                      yaxis=dict(rangemode="tozero"))
                st.plotly_chart(apply_custom_theme(fig_val), use_container_width=True, config={"displayModeBar": False})
                st.caption("Solid line: Transfermarkt's published valuations. Dashed line and diamond: the forecast, with "
                           "the range that held 70% of back-test misses. Dotted line: what his output alone is worth today.")

        st.write("")
        with st.container(border=True):
            render_valuation_bar(actual_val, pred_val, pred_low, pred_high)

        st.caption(
            "Performance value is what his on-pitch output, minutes, age, position and league are usually "
            "worth (median estimate), with a fair band the market price falls inside about 70% of the time. "
            "It ignores his club, international status and contract; those are shown separately as the premium."
        )
        notes = []
        if safe_float(target_row.get("tm_club_mismatch")) == 1:
            notes.append(f"**Market value pre-dates his move.** Transfermarkt still lists him at "
                         f"{target_row.get('tm_club', 'his previous club')}; the value shown and the missing "
                         "contract length describe that spell, not his current one.")
        if str(target_row.get("match_method", "exact")) != "exact":
            notes.append(f"**Name matched approximately** to Transfermarkt's “{target_row.get('tm_match_name', '?')}” "
                         f"({target_row.get('match_method')}). Check it is the same person.")
        n_comp = target_row.get("n_comparables")
        if (pd.notna(n_comp) and n_comp < 10 and safe_float(target_row.get("outside_band")) == -1
                and safe_float(actual_val) >= 20):
            notes.append(f"**Few comparable players.** Only {int(n_comp)} others near his age carry a market value "
                         "anywhere close to his, so the model has little to learn from. Treat the performance value as "
                         "what age and output alone justify, not as a price.")
        if notes:
            st.warning("\n\n".join(notes))

        # Data coverage is routine information, not a warning
        last_season = target_row.get("last_season")
        adv_last = target_row.get("adv_last_season")
        xg_last = target_row.get("xg_last_season") if "xg_last_season" in target_row.index else adv_last
        has = lambda v: v is not None and not pd.isna(v) and bool(v)
        parts = [f"Goals, assists, shots and minutes: through {last_season} (updated weekly)"]
        parts.append(f"xG and xA: through {xg_last}" if has(xg_last) else "xG and xA: not available for him")
        parts.append(f"progression and defensive actions: through {adv_last}, the last season published"
                     if has(adv_last) else "progression and defensive actions: not published for his seasons")
        coverage = "**Data coverage** · " + ". ".join(p if p.startswith("xG") else p[0].upper() + p[1:] for p in parts) + "."
        st.caption(coverage)

        if "why_age" in df_master.columns:
            section_heading("Why this value", "What drives his performance value")
            with st.container(border=True):
                render_why(target_row)

    with sub_style:
        section_heading("Statistical clones", "Closest tactical output to your target")
        st.info("**What is a clone?** A player who shares a highly similar statistical profile and on-pitch playstyle to your target. Calculated via cosine similarity across multi-season tactical metrics.")

        # over-fetch, then keep the closest matches that are still recruitable
        results = engine.find_similar_players(target, top_n=top_k * 12 if active_only else top_k, same_position=filter_pos)
        if isinstance(results, pd.DataFrame) and active_only:
            results = results[results["Player"].isin(RECRUITABLE)].head(top_k).reset_index(drop=True)
            if results.empty:
                results = None

        if isinstance(results, pd.DataFrame):
            display_results = results.copy()
            if "actual_value_m" in df_master.columns:
                display_results = display_results.merge(
                    df_master[["player", "actual_value_m", "predicted_value_m", "surplus_value_m"]],
                    left_on="Player", right_on="player", how="left"
                ).drop(columns=["player"])
                for val_col in ["actual_value_m", "predicted_value_m", "surplus_value_m"]:
                    display_results[label(val_col)] = display_results[val_col].apply(fmt_eur_m)
                display_results.drop(columns=["actual_value_m", "predicted_value_m", "surplus_value_m"], inplace=True)
        
            clone_col_config = {}
            for c in display_results.columns:
                if "similar" in str(c).lower() or "score" in str(c).lower() or "confidence" in str(c).lower():
                    # Values arrive as "95.5%" display strings, which ProgressColumn cannot draw a
                    # bar from directly. The original code tried pd.to_numeric() on the raw string,
                    # got NaN because of the trailing '%', and silently skipped the bar for every
                    # row - fixed by stripping '%' and converting the column itself to numeric.
                    numeric = pd.to_numeric(display_results[c].astype(str).str.rstrip("%"), errors="coerce")
                    if numeric.isna().all():
                        continue
                    display_results[c] = numeric
                    col_max = float(numeric.max())
                    scale = 1.0 if col_max <= 1.0 else 100.0
                    clone_col_config[c] = st.column_config.ProgressColumn(
                        str(c), min_value=0.0, max_value=scale,
                        format="%.3f" if scale == 1.0 else "%.1f%%",
                    )
            st.dataframe(display_results, hide_index=True, use_container_width=True, column_config=clone_col_config)
            st.download_button(
                "Download clone list (CSV)",
                display_results.to_csv(index=False).encode("utf-8"),
                file_name=f"stattrick_clones_{str(target).replace(' ', '_')}.csv",
                mime="text/csv",
            )
            top_clone = display_results.iloc[0]["Player"]

            if hasattr(engine, "explain"):
                with st.expander("Why are they similar? Compare the target with one clone"):
                    pick = st.selectbox("Clone", display_results["Player"].tolist(), key="explain_clone")
                    why = engine.explain(target, pick)
                    if why and (why["alike"] or why["differ"]):
                        ex_l, ex_r = st.columns(2)
                        line = lambda c, a, b: f"- **{label(c)}**: {a:.2f} vs {b:.2f}"      # noqa: E731
                        ex_l.markdown("**Most alike**\n\n" + ("\n".join(line(*r) for r in why["alike"]) or "—"))
                        ex_r.markdown("**Biggest differences**\n\n" + ("\n".join(line(*r) for r in why["differ"])
                                                                          or "No stat differs by much."))
                        st.caption(f"Each line reads {target.split()[-1]} vs {pick.split()[-1]}, per 90 minutes. Only stats "
                                   "recorded for both players are compared, weighted for the target's position.")
                    else:
                        st.caption("Too few stats recorded for both players to explain this pair.")
        else:
            top_clone = None
            st.warning("Could not compute clones.")

    with sub_fit:
        # --- CONTEXTUAL SYSTEM FIT & FORM VISUALIZER ---
        section_heading("Tactical portability & form trajectory", "System compatibility and multi-season output")

        with st.container(border=True):
            col_fit, col_form = st.columns([1, 1.15])

            with col_fit:
                available_squads = sorted(system_engine.club_profiles["squad"].unique().tolist())
                default_club_idx = available_squads.index("Arsenal") if "Arsenal" in available_squads else 0
                selected_squad = st.selectbox("Simulate transfer to acquiring club", available_squads, index=default_club_idx)

                fit_data = system_engine.calculate_system_fit(target, selected_squad)
                render_fit_gauge(
                    f"{selected_squad} tactical archetype",
                    fit_data["archetype"],
                    fit_data["tier"],
                    fit_data["fit_score"],
                )

                try:
                    api_key = st.secrets["GEMINI_API_KEY"]
                except (KeyError, FileNotFoundError):
                    # st.secrets raises rather than behaving like a normal dict when no
                    # secrets.toml exists at all, so `"X" in st.secrets` used to crash the app
                    # on a machine with no secrets file configured (audit #11).
                    api_key = st.text_input("Enter free Gemini API Key to unlock AI Insights:", type="password")

                if api_key:
                    if st.button("Generate AI Tactical Brief", key="ai_insight"):
                        with st.spinner("Analyzing vector synergy..."):
                            p_dict = df_master[df_master["player"] == target].iloc[0].to_dict()
                            club_rows = system_engine.club_profiles[
                                system_engine.club_profiles["squad"].str.lower() == selected_squad.lower()
                            ]
                            # SAFE EXTRACT: Fallback to club average if 'pos_group' is missing from remote data
                            if "pos_group" in club_rows.columns:
                                group_rows = club_rows[club_rows["pos_group"] == fit_data.get("position_group")]
                            else:
                                group_rows = club_rows
                            
                            s_dict = (group_rows.iloc[0] if not group_rows.empty else club_rows.iloc[0]).to_dict()

                            try:
                                brief = generate_tactical_brief(
                                    p_dict, s_dict, fit_data,
                                    getattr(system_engine, "group_stats", {}).get(
                                        fit_data.get("position_group"), system_engine.feature_stats),
                                    api_key)
                            except Exception as exc:
                                st.error(f"Scouting AI unavailable: {exc}")
                                brief = None

                        if brief:
                            st.markdown(
                                f'<div style="padding: 1rem; border-left: 3px solid var(--mint); background: rgba(53,224,140,0.05); margin-top: 1rem; font-size: 0.88rem; border-radius: 8px;">'
                                f'<b style="color: var(--mint);">🤖 AI Tactical Brief</b><br><br>{brief}'
                                f'</div>', 
                                unsafe_allow_html=True
                            )

            with col_form:
                target_pos = str(target_row.get("pos_clean", "MF")).upper()
                if not df_timeline.empty and "season" in df_timeline.columns:
                    # Join on identity (accent-free name + birth year), not on the display name:
                    # names collide (two 'Rodri's) and vary by source ('Fermin' / 'Fermín').
                    target_pkey = target_row.get("pkey")
                    if "pkey" in df_timeline.columns and pd.notna(target_pkey):
                        player_history = df_timeline[df_timeline["pkey"] == target_pkey].copy()
                    else:
                        player_history = df_timeline[df_timeline["player"].str.lower() == str(target).lower()].copy()

                    # 1. One row per season, stints combined by minutes
                    seasons_played = sorted(player_history["season"].dropna().unique().tolist())
                    player_history = season_rollup(player_history)

                    # 2. Calculate the specific positional metric
                    form_view = st.radio(
                        "Trend", ["Role metric", "Goals + assists · every season"], horizontal=True,
                        label_visibility="collapsed", key="form_view",
                        help="Role metrics use event data (xG, progression, defensive actions), which the "
                             "provider publishes up to 2024-25. Goals and assists are current to this week.")
                    series_values, metric_title = get_position_form_metric(
                        target_pos, player_history, actuals=form_view.startswith("Goals"))
                    player_history["form_metric"] = series_values
                
                    # 3. Filter out ONLY the seasons where the form metric is completely missing
                    player_history = player_history.dropna(subset=["form_metric"])

                    # 4. Allow plotting even if there is only 1 valid historical season
                    if not player_history.empty:
                        st.markdown(f'<div class="panel-title" style="margin-bottom: 2px;">Form vs. Baseline · Multi-Season {metric_title}</div>', unsafe_allow_html=True)

                        fig_timeline = go.Figure()
                        fig_timeline.add_trace(go.Scatter(
                            x=player_history["season"],
                            y=player_history["form_metric"],
                            customdata=player_history["squad"],
                            mode="lines+markers",
                            line=dict(color=MINT, width=2.5, shape="spline", smoothing=0.3),
                            marker=dict(size=8, color=INK, line=dict(color=MINT, width=2)),
                            fill="tozeroy",
                            fillcolor="rgba(53, 224, 140, 0.12)",
                            hovertemplate=f"<b>%{{x}}</b> · %{{customdata}}<br>{metric_title}: %{{y:.2f}}<extra></extra>"
                        ))

                        fig_timeline.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="Inter, sans-serif", color=TEXT, size=11),
                            margin=dict(t=15, b=25, l=35, r=20),
                            height=185,
                            xaxis=dict(showgrid=False, linecolor=LINE, tickfont=dict(color=TEXT_DIM, size=10)),
                            yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=LINE, tickfont=dict(color=TEXT_DIM, size=10))
                        )
                        st.plotly_chart(fig_timeline, use_container_width=True, config={'displayModeBar': False})
                        untracked = [s for s in seasons_played if s not in set(player_history["season"])]
                        if untracked:
                            st.caption(f"This metric is published up to {player_history['season'].max()}. "
                                       "Switch to “Goals + assists” above for his output through the current season.")
                    else:
                        st.markdown(f'<div class="panel-title" style="margin-bottom: 2px;">Form vs. Baseline · Multi-Season {metric_title}</div>', unsafe_allow_html=True)
                        st.markdown(
                            '<div style="height: 160px; display: grid; place-items: center; border: 1px dashed var(--line); border-radius: 10px; color: var(--text-faint); font-size: 0.85rem; margin-top: 8px;">'
                            'No valid tactical data available for this metric'
                            '</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown('<div class="panel-title" style="margin-bottom: 2px;">Form vs. Baseline · Multi-Season Output</div>', unsafe_allow_html=True)
                    st.markdown(
                        '<div style="height: 160px; display: grid; place-items: center; border: 1px dashed var(--line); border-radius: 10px; color: var(--text-faint); font-size: 0.85rem; margin-top: 8px;">'
                        'Run valuation_model.py to initialize player_timeline_db.csv'
                        '</div>',
                        unsafe_allow_html=True
                    )

    with sub_style:
        # --- RADAR AND CHART STUDIO ---
        section_heading("Tactical overlay", "Percentile footprint and a free-form scatter")
        with st.container(border=True):
            col_radar, col_scatter = st.columns(2)
            with col_radar:
                st.markdown(f'<div class="panel-title">{pos} tactical footprint · percentile vs position</div>', unsafe_allow_html=True)
                if top_clone:
                    radar_stats = get_radar_metrics(pos, df_master.columns)
                    target_pcts = [get_percentile(df_master, s, target_row.get(s), pos) for s in radar_stats]
                    clone_row = df_master[df_master["player"] == top_clone].iloc[0]
                    clone_pcts = [get_percentile(df_master, s, clone_row.get(s), pos) for s in radar_stats]
                    radar_stats, (target_pcts, clone_pcts), radar_dropped = radar_axes(radar_stats, [target_pcts, clone_pcts])
                    if radar_dropped:
                        st.caption("Not recorded for one of these players, so left off the radar: "
                                   + ", ".join(label(s) for s in radar_dropped))
                if top_clone and len(radar_stats) < 3:
                    st.markdown(
                        '<div style="height: 220px; display: grid; place-items: center; border: 1px dashed var(--line); '
                        'border-radius: 10px; color: var(--text-faint); font-size: 0.85rem; margin-top: 8px;">'
                        'Not enough shared event data to draw a radar for this pair</div>', unsafe_allow_html=True)
                if top_clone and len(radar_stats) >= 3:
                    categories = [label(s) for s in radar_stats] + [label(radar_stats[0])]
                    target_pcts.append(target_pcts[0]); clone_pcts.append(clone_pcts[0])

                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(r=target_pcts, theta=categories, fill="toself", name=target, line=dict(color=MINT, width=2), fillcolor="rgba(53, 224, 140, 0.20)"))
                    fig_radar.add_trace(go.Scatterpolar(r=clone_pcts, theta=categories, fill="toself", name=top_clone, line=dict(color=CYAN, width=2), fillcolor="rgba(76, 201, 240, 0.16)"))
                    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 100])), legend=dict(orientation="h", y=1.14, xanchor="center", x=0.5))
                    st.plotly_chart(apply_custom_theme(fig_radar), use_container_width=True)

            with col_scatter:
                st.markdown('<div class="panel-title">Chart studio · plot any two metrics</div>', unsafe_allow_html=True)
                available_axes = [c for c in ["actual_value_m", "predicted_value_m", "surplus_value_m"] + engine.feature_cols if c in df_master.columns]
                drop1, drop2 = st.columns(2)
                mx = drop1.selectbox("X-Axis", available_axes, format_func=label, index=0)
                my = drop2.selectbox("Y-Axis", available_axes, format_func=label, index=3 if len(available_axes) > 3 else 0)
                scatter_df = df_master.dropna(subset=[mx, my]).copy()
                fig_scatter = go.Figure()

                fig_scatter.add_trace(go.Scatter(x=scatter_df[mx], y=scatter_df[my], mode="markers", marker=dict(color="#4B6B5F", opacity=0.55, size=6, line=dict(width=0)), customdata=np.stack((scatter_df["player"], scatter_df[mx], scatter_df[my]), axis=-1), hovertemplate="<b>%{customdata[0]}</b><br>"+label(mx)+": %{customdata[1]:.2f}<br>"+label(my)+": %{customdata[2]:.2f}<extra></extra>", name="Population"))
                if len(scatter_df) > 1:
                    slope, intercept = np.polyfit(scatter_df[mx], scatter_df[my], 1)
                    x_trend = np.linspace(scatter_df[mx].min(), scatter_df[mx].max(), 100)
                    fig_scatter.add_trace(go.Scatter(x=x_trend, y=slope * x_trend + intercept, mode="lines", line=dict(color=CYAN, width=1.6, dash="dot"), opacity=0.7, hoverinfo="skip", name="Trend"))

                fig_scatter.add_trace(go.Scatter(x=[target_row.get(mx)], y=[target_row.get(my)], mode="markers+text", marker=dict(color=MINT, size=15, symbol="diamond", line=dict(color="#06100D", width=2)), text=[target.split()[-1]], textposition="top center", textfont=dict(color=MINT, size=12), name=target))
                if top_clone:
                    clone_val_x = df_master[df_master["player"] == top_clone].iloc[0].get(mx)
                    clone_val_y = df_master[df_master["player"] == top_clone].iloc[0].get(my)
                    fig_scatter.add_trace(go.Scatter(x=[clone_val_x], y=[clone_val_y], mode="markers+text", marker=dict(color=CYAN, size=15, symbol="star", line=dict(color="#06100D", width=2)), text=[top_clone.split()[-1]], textposition="top center", textfont=dict(color=CYAN, size=12), name=top_clone))
                fig_scatter.update_layout(xaxis_title=label(mx), yaxis_title=label(my), showlegend=False)
                st.plotly_chart(apply_custom_theme(fig_scatter), use_container_width=True)

# --- REPLACE A PLAYER ---
with tab_replace:
    section_heading("Replace a player", "The same profile, cheaper, and suited to your club")
    st.info("**How this works:** choose the player you need to replace in the sidebar, and the club doing the buying here. StatTrick finds "
            "the closest statistical matches, then keeps only those you could plausibly sign: active this season, "
            "within budget and age, and not already at your club. Each is scored for fit to your club's style.")
    rp_squads = sorted(system_engine.club_profiles["squad"].unique().tolist())
    rp1, rp2 = st.columns(2)
    rp_player = target                      # the player chosen in the sidebar
    rp1.markdown(f'<div class="panel-title">Player to replace</div>'
                 f'<div class="fit-archetype" style="font-size:1.4rem;">{rp_player}</div>'
                 f'<div class="fit-club">Change him with “Select Target Player” in the sidebar</div>',
                 unsafe_allow_html=True)
    rp_row = df_master[df_master["player"] == rp_player].iloc[0]
    rp_home = str(rp_row.get("squad", ""))
    rp_club = rp2.selectbox("Buying club", rp_squads,
                            index=rp_squads.index(rp_home) if rp_home in rp_squads else 0, key=f"rp_club_{rp_player}")
    rp_val = safe_float(rp_row.get("actual_value_m"), 20.0)
    rp3, rp4, rp5, rp6 = st.columns(4)
    rp_budget = rp3.slider("Budget: max market value (€M)", 1, 200, int(min(200, max(1, round(rp_val)))), key=f"rp_budget_{rp_player}",
                           help="Transfermarkt value as a stand-in for the fee. Defaults to the value of the player being replaced.")
    rp_age = rp4.slider("Max age", 17, 36, 28, key="rp_age")
    rp_same_pos = rp5.checkbox("Same position only", True, key="rp_same_pos")
    rp_rank = rp6.selectbox("Rank by", ["Blend", "Similarity", "Value edge", "Club fit"], key="rp_rank")

    pool = engine.find_similar_players(rp_player, top_n=400, same_position=rp_same_pos)
    if not isinstance(pool, pd.DataFrame) or pool.empty:
        st.warning("Could not compute matches for this player.")
    else:
        pool = pool.rename(columns={"Player": "player"})
        pool["similarity"] = pd.to_numeric(pool["Similarity Score"].astype(str).str.rstrip("%"), errors="coerce")
        keep_cols = [c for c in ["player", "squad", "league", "pos_clean", "age_clean", "contract_years_left",
                                 "actual_value_m", "predicted_value_m", "edge_z", "outside_band",
                                 "forecast_change_pct", "match_method"] if c in df_master.columns]
        cand = pool[["player", "similarity"]].merge(df_master[keep_cols], on="player", how="inner")
        cand = cand[cand["player"].isin(RECRUITABLE)
                    & (cand["squad"].str.lower() != rp_club.lower())
                    & (cand["actual_value_m"] <= rp_budget)
                    & (pd.to_numeric(cand["age_clean"], errors="coerce") <= rp_age)]
        if "match_method" in cand.columns:
            cand = cand[cand["match_method"] == "exact"]
        cand = cand.head(40).copy()                       # the 40 closest that pass the filters
        if cand.empty:
            st.warning("No one fits. Raise the budget or the age limit, or untick 'Same position only'.")
        else:
            cand["fit"] = [system_engine.calculate_system_fit(p, rp_club).get("fit_score") for p in cand["player"]]
            z = lambda s: (s - s.mean()) / (s.std() if s.std() and s.std() > 0 else 1.0)      # noqa: E731
            fit_num = pd.to_numeric(cand["fit"], errors="coerce")
            edge_num = pd.to_numeric(cand.get("edge_z"), errors="coerce") if "edge_z" in cand.columns else pd.Series(0.0, index=cand.index)
            cand["blend"] = (0.5 * z(cand["similarity"]) + 0.25 * z(edge_num.fillna(edge_num.median())).clip(-2.5, 2.5)
                             + 0.25 * z(fit_num.fillna(fit_num.median() if fit_num.notna().any() else 50.0)))
            sort_col = {"Blend": "blend", "Similarity": "similarity", "Value edge": "edge_z", "Club fit": "fit"}[rp_rank]
            if sort_col not in cand.columns:
                sort_col = "similarity"
            cand = cand.sort_values(sort_col, ascending=False).head(15)

            cheaper = cand["actual_value_m"].median()
            render_metrics_html([
                ("Replacing", str(rp_player).split()[-1], f"{rp_row.get('pos_clean', '')} · {fmt_eur_m(rp_val)} · age {safe_float(rp_row.get('age_clean')):.0f}", "neutral"),
                ("Candidates shown", f"{len(cand)}", f"Ranked by {rp_rank.lower()}", "neutral"),
                ("Median market value", fmt_eur_m(cheaper),
                 f"{(1 - cheaper / rp_val) * 100:.0f}% below the player replaced" if rp_val > 0 and cheaper < rp_val else "Of the candidates", "positive" if cheaper < rp_val else "neutral"),
                ("Best similarity", f"{cand['similarity'].max():.0f}%", "Closest statistical match", "neutral"),
            ])
            st.write("")
            show = cand[[c for c in ["player", "squad", "pos_clean", "age_clean", "similarity", "fit", "actual_value_m",
                                     "predicted_value_m", "edge_z", "forecast_change_pct", "contract_years_left"] if c in cand.columns]].copy()
            for c_ in ["actual_value_m", "predicted_value_m"]:
                if c_ in show.columns:
                    show[c_] = show[c_].apply(fmt_eur_m)
            if "forecast_change_pct" in show.columns:
                show["forecast_change_pct"] = show["forecast_change_pct"].apply(lambda v: format_value(v, "surplus_pct"))
            show = show.rename(columns={"similarity": "Similarity (%)", "fit": f"Fit to {rp_club}",
                                        **{c_: label(c_) for c_ in show.columns if c_ not in ("similarity", "fit")}})
            st.dataframe(show, hide_index=True, use_container_width=True, column_config={
                "Similarity (%)": st.column_config.ProgressColumn("Similarity (%)", min_value=0.0, max_value=100.0, format="%.1f%%"),
                f"Fit to {rp_club}": st.column_config.NumberColumn(f"Fit to {rp_club}", format="%.1f"),
            })
            rb1, rb2 = st.columns(2)
            rb1.download_button("Download candidates (CSV)", show.to_csv(index=False).encode("utf-8"),
                                file_name=f"stattrick_replace_{str(rp_player).replace(' ', '_')}.csv", mime="text/csv")
            _rp_top = cand["player"].head(5).tolist()
            rb2.button(f"＋ Add top {len(_rp_top)} to shortlist", key="rp_add", on_click=_shortlist_add, args=(_rp_top,))
            st.caption("Similarity describes statistical style, not quality. Value edge is how far the market price sits "
                       "below the performance value, in units of the player's own fair band; a blank fit means the club "
                       "has too few players in that position group to profile. Blend = half similarity, a quarter value "
                       "edge, a quarter club fit.")

# --- TAB 2: ARBITRAGE SCREENER ---
with tab2:
    section_heading("Arbitrage screener", "Output that outruns the price tag")
    st.info("**What is the arbitrage screener?** This tool highlights players whose underlying tactical outputs significantly outperform their current public market valuation. Set your criteria below to discover hidden gems.")

    with st.container(border=True):
        has_edge = "edge_z" in df_master.columns
        modes = (["Edge vs fair band  (recommended)"] if has_edge else []) + ["Surplus %", "Surplus €M"]
        rank_mode = st.radio(
            "Rank by", modes, horizontal=True,
            help="Raw surplus is dominated by regression to the mean: cheap players always look "
                 "undervalued and expensive ones overvalued. Edge measures the gap in units of the "
                 "player's own fair-band half-width, and only counts when the market price sits "
                 "outside that band.",
        )
        rank_col = ("edge_z" if rank_mode.startswith("Edge") else
                    "surplus_pct" if rank_mode.startswith("Surplus %") else "surplus_value_m")
        opt1, opt2 = st.columns(2)
        only_outside = opt1.checkbox("Only players priced below their fair band", value=has_edge,
                                     disabled=not has_edge)
        only_exact = opt2.checkbox("Only exact Transfermarkt name matches", value=True)
        has_fc = "forecast_change_pct" in df_master.columns
        only_rising = st.checkbox("Only players whose market value is forecast to rise", value=False,
                                  disabled=not has_fc,
                                  help="Uses the separate 12-month market forecast. Combined with the fair-band "
                                       "filter this finds players who are cheap for their output AND expected "
                                       "to get more expensive.")

        col_f1, col_f2, col_f3 = st.columns(3)
        max_age = col_f1.slider("Max Age", 17, 36, 24)
        if rank_col == "edge_z":
            min_surplus = col_f2.slider("Min Edge (band half-widths)", 0.0, 4.0, 1.0, 0.1)
        elif rank_col == "surplus_pct":
            min_surplus = col_f2.slider("Min Surplus (%)", 0.0, 200.0, 20.0)
        else:
            min_surplus = col_f2.slider("Min Surplus Value (€M)", 0.0, 30.0, 5.0)
        min_minutes = col_f3.slider("Minimum Career Minutes", 1500, 6000, 1500, 100)

        # Budget search: narrow to what the club can actually sign
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        pos_pick = col_b1.multiselect("Position", sorted(df_master["pos_clean"].dropna().unique().tolist()),
                                      placeholder="Any position")
        budget = col_b2.slider("Max market value (€M)", 1, 150, 150,
                               help="Transfermarkt value, as a proxy for the fee. 150 = no limit.")
        min_contract = col_b3.slider("Min contract years left", 0.0, 5.0, 0.0, 0.5,
                                     help="Players with an unknown contract are kept when this is 0.")
        league_pick = col_b4.multiselect("League", sorted(df_master["league"].dropna().unique().tolist()),
                                         placeholder="Any league")

        screener_df = df_master[(pd.to_numeric(df_master["age_clean"], errors="coerce") <= max_age) & (df_master[rank_col] >= min_surplus) & (df_master["min"] >= min_minutes)].copy()
        if only_outside and "outside_band" in screener_df.columns:
            screener_df = screener_df[screener_df["outside_band"] == 1]
        if only_exact and "match_method" in screener_df.columns:
            screener_df = screener_df[screener_df["match_method"] == "exact"]
        screener_df = screener_df[screener_df["player"].isin(RECRUITABLE)]
        if pos_pick:
            screener_df = screener_df[screener_df["pos_clean"].isin(pos_pick)]
        if league_pick:
            screener_df = screener_df[screener_df["league"].isin(league_pick)]
        if budget < 150:
            screener_df = screener_df[screener_df["actual_value_m"] <= budget]
        if min_contract > 0:
            screener_df = screener_df[screener_df["contract_years_left"] >= min_contract]
        if only_rising and has_fc:
            screener_df = screener_df[screener_df["forecast_change_pct"] > 0]
        if not screener_df.empty:
            screener_df = screener_df.sort_values(by=rank_col, ascending=False)
            display_columns = [c for c in ["player", "squad", "league", "pos_clean", "age_clean", "contract_years_left", "actual_value_m", "pred_value_low_m", "predicted_value_m", "edge_z", "surplus_value_m", "surplus_pct", "forecast_change_pct"] if c in screener_df.columns]
            output_df = screener_df[display_columns].copy()
            for curr_col in ["actual_value_m", "pred_value_low_m", "predicted_value_m", "surplus_value_m"]:
                if curr_col in output_df.columns: output_df[curr_col] = output_df[curr_col].apply(fmt_eur_m)
            for pct_col in ["surplus_pct", "forecast_change_pct"]:
                if pct_col in output_df.columns:
                    output_df[pct_col] = output_df[pct_col].apply(lambda v: format_value(v, "surplus_pct"))
            output_df.rename(columns={c: label(c) for c in output_df.columns}, inplace=True)
            # euro totals use the mean estimate; summing medians understates the pool
            if "predicted_mean_m" in screener_df.columns:
                pool_surplus = (pd.to_numeric(screener_df["predicted_mean_m"], errors="coerce")
                                - pd.to_numeric(screener_df["actual_value_m"], errors="coerce"))
            else:
                pool_surplus = pd.to_numeric(screener_df["surplus_value_m"], errors="coerce")
            pool_age = pd.to_numeric(screener_df["age_clean"], errors="coerce")
            best_row = screener_df.iloc[0]
            st.write("")
            render_metrics_html([
                ("Players matched", f"{len(screener_df):,}", f"Sorted by {label(rank_col)}", "neutral"),
                ("Combined model edge", fmt_eur_m(pool_surplus.sum()), "Total value the market is missing", "positive"),
                ("Median age of pool", f"{pool_age.median():.0f}" if pool_age.notna().any() else "—", "Resale runway", "neutral"),
                ("Biggest single edge", str(best_row.get("player", "—")).split()[-1],
                 f"{fmt_eur_m(best_row.get('surplus_value_m', 0))} · {best_row.get('squad', '—')}", "positive"),
            ])
            st.write("")
            st.dataframe(output_df, hide_index=True, use_container_width=True)
            dl_col, add_col = st.columns([1, 1])
            dl_col.download_button(
                "Download these results (CSV)",
                output_df.to_csv(index=False).encode("utf-8"),
                file_name="stattrick_screener.csv",
                mime="text/csv",
            )
            _top = screener_df["player"].head(25).tolist()
            add_col.button(f"＋ Add top {len(_top)} to shortlist", key="sl_add_screen",
                           on_click=_shortlist_add, args=(_top,))
        else:
            st.warning("No players match the chosen filter parameters. Widen the age range or lower the surplus threshold.")

# --- TAB 3: HEAD-TO-HEAD SANDBOX ---
with tab3:
    section_heading("Head-to-head sandbox", "Up to three profiles, side by side")
    st.info("**How to use the sandbox:** Select up to three players to directly compare their physical, financial, and tactical profiles. The radar overlay normalizes their statistics against all players in the primary selected player's position.")

    default_h2h = [p for p in ["Bukayo Saka", "Phil Foden", "Cole Palmer"] if p in all_players][:2]
    selected_players = st.multiselect("Select Players (Max 3)", all_players, default=default_h2h, max_selections=3)

    if selected_players:
        h2h_df = df_master[df_master["player"].isin(selected_players)].copy()

        chip_colors = [MINT, CYAN, CORAL]
        chips = "".join(
            f'<span class="chip" style="background: {chip_colors[i % 3]}1F; color: {chip_colors[i % 3]}; '
            f'border-color: {chip_colors[i % 3]}3D;">{p}</span>'
            for i, p in enumerate(selected_players)
        )
        st.markdown(f'<div class="identity-meta" style="margin: .2rem 0 1rem;">{chips}</div>', unsafe_allow_html=True)

        with st.container(border=True):
            col_table, col_radar_h2h = st.columns([1, 1.25])
            with col_table:
                st.markdown('<div class="panel-title">Core metrics comparison</div>', unsafe_allow_html=True)

                tactical_metrics = [c for c in engine.feature_cols if "per90" in c][:5]
                raw_fields = ["squad", "league", "pos_clean", "age_clean", "min"] + tactical_metrics + ["actual_value_m", "surplus_value_m"]

                core_fields = list(dict.fromkeys([c for c in raw_fields if c in h2h_df.columns]))

                table_df = h2h_df[["player"] + core_fields].loc[:, ~h2h_df[["player"] + core_fields].columns.duplicated()].copy()
                for c in core_fields:
                    table_df[c] = table_df[c].apply(lambda x: format_value(x, c))

                table_df.rename(columns={c: label(c) for c in table_df.columns}, inplace=True)
                st.dataframe(table_df.set_index(label("player")).T, use_container_width=True)

            with col_radar_h2h:
                if len(selected_players) >= 2:
                    first_pos = h2h_df[h2h_df["player"] == selected_players[0]].iloc[0].get("pos_clean", "MF")
                    st.markdown(f'<div class="panel-title">Tactical overlay · {first_pos} basis</div>', unsafe_allow_html=True)
                    h2h_radar_stats = get_radar_metrics(first_pos, df_master.columns)
                    h2h_pcts = [[get_percentile(df_master, s, h2h_df[h2h_df["player"] == p_name].iloc[0].get(s), first_pos)
                                 for s in h2h_radar_stats] for p_name in selected_players]
                    h2h_radar_stats, h2h_pcts, h2h_dropped = radar_axes(h2h_radar_stats, h2h_pcts)
                    if h2h_dropped:
                        st.caption("Not recorded for every selected player, so left off the radar: "
                                   + ", ".join(label(s) for s in h2h_dropped))
                    if len(h2h_radar_stats) < 3:
                        st.info("Too few shared metrics to draw a radar for these players.")
                    else:
                        radar_cats = [label(s) for s in h2h_radar_stats] + [label(h2h_radar_stats[0])]
                        fig_h2h = go.Figure()

                        trace_colors = [MINT, CYAN, CORAL]
                        fill_colors = ["rgba(53, 224, 140, 0.20)", "rgba(76, 201, 240, 0.16)", "rgba(255, 92, 122, 0.14)"]

                        for idx, p_name in enumerate(selected_players):
                            p_pcts = list(h2h_pcts[idx])
                            p_pcts.append(p_pcts[0])
                            fig_h2h.add_trace(go.Scatterpolar(r=p_pcts, theta=radar_cats, fill="toself", name=p_name, line=dict(color=trace_colors[idx], width=2), fillcolor=fill_colors[idx]))

                        fig_h2h.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 100])), legend=dict(orientation="h", y=1.14, xanchor="center", x=0.5))
                        st.plotly_chart(apply_custom_theme(fig_h2h), use_container_width=True)

# --- TAB 4: GAP ANALYSIS ---
with tab4:
    section_heading("Gap analysis (Archetype-calibrated)", "Find players who plug specific system deficits")
    st.info(
        "**How this works:** We evaluate the club's metrics against their specific **Tactical Archetype's DNA** rather than generic European totals. "
        "This ensures possession-heavy teams aren't erroneously penalized for low clearance volumes."
    )

    col_g1, col_g2 = st.columns(2)
    gap_squads = sorted(system_engine.club_profiles["squad"].unique().tolist())
    gap_default = gap_squads.index("Arsenal") if "Arsenal" in gap_squads else 0
    gap_target_club = col_g1.selectbox("Select Club to Analyze", gap_squads, index=gap_default, key="gap_club")
    target_unit = col_g2.selectbox("Target Recruitment Department", ["All Positions", "Midfield (CM/CDM/CAM)", "Defense (CB/Fullback)", "Attack (ST/Winger)"])

    UNIT_TO_GROUP = {"Midfield (CM/CDM/CAM)": "MID", "Defense (CB/Fullback)": "DEF", "Attack (ST/Winger)": "ATT"}
    GROUP_TO_POS = {"GK": ["GK"], "DEF": ["CB", "FULLBACK"], "MID": ["CM", "CDM", "CAM"], "ATT": ["ST", "WINGER"]}

    if st.button("Run Inverse Recruitment Engine", type="primary"):
        with st.spinner(f"Analyzing {gap_target_club}'s tactical deficiencies..."):
            from src.system_fit import STYLE_FEATURES, ARCHETYPE_PROFILES
            club_df = system_engine.club_profiles.copy()

            # club_profiles now has ONE ROW PER POSITION GROUP per club (audit #10), so a
            # club's "weakest metric" has to be found within the position group(s) actually
            # being searched - comparing a defensive metric to the club's attacking archetype
            # (or vice versa) was the original bug.
            club_rows = club_df[club_df["squad"].str.lower() == gap_target_club.lower()]
            
            # SAFE EXTRACT: Ensure pos_group exists before subsetting
            if "pos_group" in club_rows.columns:
                search_groups = [UNIT_TO_GROUP.get(target_unit, "ALL")] if target_unit in UNIT_TO_GROUP \
                    else club_rows["pos_group"].unique().tolist()
                club_rows = club_rows[club_rows["pos_group"].isin(search_groups)]

            best = None  # (gap, feature, pos_group, archetype)
            for _, row in club_rows.iterrows():
                archetype = row.get("tactical_archetype", "Balanced Mid-Block & Pragmatic")
                ideal = ARCHETYPE_PROFILES.get(archetype, ARCHETYPE_PROFILES["Balanced Mid-Block & Pragmatic"])
                
                # Default to "Squad" if the pos_group column is missing
                current_group = row.get("pos_group", "Squad")
                
                for feat in STYLE_FEATURES:
                    if feat not in row.index or feat not in ideal or pd.isna(row[feat]):
                        continue
                    # compare this unit to the SAME unit at other clubs; pooling all groups made
                    # every defence "lack shots" and every attack "lack interceptions"
                    peers = club_df[club_df["pos_group"] == current_group][feat] if "pos_group" in club_df.columns else club_df[feat]
                    feat_std = peers.std()
                    club_z = (row[feat] - peers.mean()) / (feat_std if feat_std and feat_std > 0 else 1.0)
                    gap = ideal[feat] - club_z
                    if best is None or gap > best[0]:
                        best = (gap, feat, current_group, archetype)
            if best is None:
                st.warning(f"No profiled cohort for {gap_target_club} in this department "
                          "(too few qualifying minutes to build a reliable position profile).")
                st.stop()

            gap_magnitude, weakness_metric, weak_group, club_archetype = best

            if gap_magnitude > 0:
                verdict = (f'their squad output in <b>{label(weakness_metric)}</b> is '
                          f'<b>{gap_magnitude:.2f} standard deviations</b> below their archetype\u2019s ideal blueprint')
            else:
                verdict = (f'their <b>{label(weakness_metric)}</b> output already meets or exceeds the archetype\u2019s '
                          f'blueprint - this is their smallest surplus, not a deficit, shown as the closest thing to a gap')

            st.markdown(
                f'<div style="padding: 1.2rem; border-left: 3px solid var(--coral); background: rgba(255,92,122,0.05); margin: 1rem 0; border-radius: 8px;">'
                f'<b style="color: var(--coral);">🚨 Archetype Vulnerability ({weak_group}): {label(weakness_metric)}</b><br><br>'
                f'{gap_target_club}\u2019s {weak_group.lower()} unit plays a <b>{club_archetype}</b> system, but {verdict}. '
                'Surfacing targeted arbitrage solutions below:'
                f'</div>', 
                unsafe_allow_html=True
            )

            candidates = df_master[
                (df_master["squad"].str.lower() != gap_target_club.lower()) &
                (df_master["surplus_pct"] > 0) &
                (df_master["min"] >= 900) &
                (df_master["age_clean"] <= 28) &
                (df_master["pos_clean"].isin(GROUP_TO_POS[weak_group]))
            ].copy()
            # Same safeguards as the screener: market price below the fair band, exact name
            # match, and rank on band-normalised edge rather than raw surplus %.
            use_edge = "edge_z" in candidates.columns
            if "outside_band" in candidates.columns:
                candidates = candidates[candidates["outside_band"] == 1]
            if "match_method" in candidates.columns:
                candidates = candidates[candidates["match_method"] == "exact"]
            candidates = candidates[candidates["player"].isin(RECRUITABLE)]
            candidates = candidates.dropna(subset=[weakness_metric]) if weakness_metric in candidates.columns else candidates
            value_col = "edge_z" if use_edge else "surplus_pct"

            if not candidates.empty and weakness_metric in candidates.columns:
                m_mean = candidates[weakness_metric].mean()
                m_std = candidates[weakness_metric].std() if candidates[weakness_metric].std() > 0 else 1.0
                s_mean = candidates[value_col].mean()
                s_std = candidates[value_col].std() if candidates[value_col].std() > 0 else 1.0

                # cap at +/-2.5 SD so one small-sample outlier cannot carry the ranking
                candidates["metric_z"] = ((candidates[weakness_metric] - m_mean) / m_std).clip(-2.5, 2.5)
                candidates["surplus_z"] = ((candidates[value_col] - s_mean) / s_std).clip(-2.5, 2.5)
                candidates["gap_score"] = (candidates["metric_z"] * 1.5) + candidates["surplus_z"]

                top_targets = candidates.sort_values(by="gap_score", ascending=False).head(5)

                display_cols = [c for c in ["player", "squad", "pos_clean", "age_clean", weakness_metric, "actual_value_m",
                                            "pred_value_low_m", "predicted_value_m", "edge_z", "surplus_pct"]
                                if c in top_targets.columns]
                display_df = top_targets[display_cols].copy()

                for c in [c for c in ["actual_value_m", "pred_value_low_m", "predicted_value_m"] if c in display_df.columns]:
                    display_df[c] = display_df[c].apply(fmt_eur_m)
                display_df["surplus_pct"] = display_df["surplus_pct"].apply(lambda v: format_value(v, "surplus_pct"))
                display_df[weakness_metric] = display_df[weakness_metric].apply(lambda x: f"{float(x):.2f}")

                display_df.rename(columns={c: label(c) for c in display_df.columns}, inplace=True)
                st.dataframe(display_df, hide_index=True, use_container_width=True)
            else:
                st.warning("No players matched the specific position filter and surplus criteria.")

# --- TEAM VIEW ---
with tab_team:
    section_heading("Team view", "One club's squad, priced three ways")
    tv_squads = sorted(df_master["squad"].dropna().unique().tolist())
    tv_default = str(target_row.get("squad", ""))
    tv_club = st.selectbox("Club", tv_squads, index=tv_squads.index(tv_default) if tv_default in tv_squads else 0,
                           key=f"tv_club_{tv_default}", help="Starts on the club of the player chosen in the sidebar.")
    tv_all = df_master[df_master["squad"] == tv_club].copy()
    tv_only_active = st.checkbox("Only players with minutes this season", True, key="tv_active")
    tv = tv_all[tv_all["active_now"]] if tv_only_active else tv_all
    if tv.empty:
        st.info("No players in the database for this club with the current filter.")
    else:
        mv, pv = pd.to_numeric(tv["actual_value_m"], errors="coerce"), pd.to_numeric(tv["predicted_value_m"], errors="coerce")
        cyl = pd.to_numeric(tv["contract_years_left"], errors="coerce")
        over = int((tv["outside_band"] == -1).sum()) if "outside_band" in tv.columns else 0
        under = int((tv["outside_band"] == 1).sum()) if "outside_band" in tv.columns else 0
        render_metrics_html([
            ("Players", f"{len(tv)}", f"of {len(tv_all)} in the database", "neutral"),
            ("Squad market value", fmt_big(mv.sum()), "Transfermarkt", "neutral"),
            ("Squad performance value", fmt_big(pv.sum()), f"{(pv.sum() / mv.sum() - 1) * 100:+.0f}% vs market" if mv.sum() > 0 else "", "neutral"),
            ("Median age", f"{pd.to_numeric(tv['age_clean'], errors='coerce').median():.1f}", "", "neutral"),
            ("Contracts ending within a year", f"{int((cyl <= 1.0).sum())}", f"{int(cyl.isna().sum())} unknown", "negative" if (cyl <= 1.0).sum() else "neutral"),
        ])
        st.write("")
        with st.container(border=True):
            st.markdown('<div class="panel-title">Market value against performance value · each dot is a player</div>', unsafe_allow_html=True)
            band = tv["outside_band"] if "outside_band" in tv.columns else pd.Series(0, index=tv.index)
            fig_tv = go.Figure()
            top_ = float(np.nanmax([mv.max(), pv.max()])) * 1.08 if len(tv) else 1.0
            fig_tv.add_trace(go.Scatter(x=[0, top_], y=[0, top_], mode="lines", line=dict(color=LINE, width=1.2, dash="dot"),
                                        hoverinfo="skip", showlegend=False))
            for code, name_, color in [(-1, "Priced above performance band", CORAL), (0, "Inside band", TEXT_DIM), (1, "Priced below performance band", MINT)]:
                part = tv[band == code]
                if part.empty:
                    continue
                fig_tv.add_trace(go.Scatter(
                    x=part["predicted_value_m"], y=part["actual_value_m"], mode="markers", name=name_,
                    marker=dict(size=10, color=color, line=dict(color=INK, width=1.5)),
                    customdata=np.stack((part["player"], part["pos_clean"], part["age_clean"]), axis=-1),
                    hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]} · %{customdata[2]:.0f}<br>"
                                  "Performance €%{x:.1f}M<br>Market €%{y:.1f}M<extra></extra>"))
            fig_tv.update_layout(height=380, margin=dict(t=20, b=40, l=50, r=20), xaxis_title="Performance value (€M)",
                                 yaxis_title="Market value (€M)", legend=dict(orientation="h", y=1.1, x=0))
            st.plotly_chart(apply_custom_theme(fig_tv), use_container_width=True, config={"displayModeBar": False})
            st.caption(f"Above the dotted line the market pays more than the output justifies; below it, less. "
                       f"{over} player(s) sit above their fair band and {under} below it.")

        tv_cols = [c for c in ["player", "pos_clean", "age_clean", "min", "contract_years_left", "actual_value_m",
                               "predicted_value_m", "status_premium_m", "edge_z", "forecast_value_m", "forecast_change_pct"] if c in tv.columns]
        tv_show = tv.sort_values("actual_value_m", ascending=False)[tv_cols].copy()
        tv_export = tv_show.rename(columns={c: label(c) for c in tv_cols})
        for c_ in ["actual_value_m", "predicted_value_m", "status_premium_m", "forecast_value_m"]:
            if c_ in tv_show.columns:
                tv_show[c_] = tv_show[c_].apply(fmt_eur_m)
        if "forecast_change_pct" in tv_show.columns:
            tv_show["forecast_change_pct"] = tv_show["forecast_change_pct"].apply(lambda v: format_value(v, "surplus_pct"))
        if "min" in tv_show.columns:
            tv_show["min"] = tv_show["min"].apply(lambda v: format_value(v, "min"))
        if "contract_years_left" in tv_show.columns:
            tv_show["contract_years_left"] = tv_show["contract_years_left"].apply(lambda v: format_value(v, "contract_years_left"))
        st.dataframe(tv_show.rename(columns={c: label(c) for c in tv_cols}), hide_index=True, use_container_width=True)
        st.download_button("Download squad (CSV)", tv_export.to_csv(index=False).encode("utf-8"),
                           file_name=f"stattrick_team_{tv_club.replace(' ', '_')}.csv", mime="text/csv")
        st.caption("Clubs are as FBref lists them this season. Market values and contracts come from the Transfermarkt "
                   "snapshot, so a player who moved here recently carries his previous club's valuation and a blank contract. "
                   "Only players with 1,500+ career minutes in the top five leagues are in the database.")

# --- SHORTLIST ---
with tab_short:
    section_heading("Shortlist", "Players you have saved in this session")
    sl = [p for p in st.session_state["shortlist"] if p in set(df_master["player"])]
    if not sl:
        st.info("Nothing saved yet. Use **＋ Add to shortlist** on a player's page, or add the top results "
                "from the Arbitrage Screener.")
    else:
        sl_df = df_master.set_index("player").loc[sl].reset_index()
        sl_cols = [c for c in ["player", "squad", "league", "pos_clean", "age_clean", "contract_years_left",
                               "actual_value_m", "predicted_value_m", "pred_value_low_m", "pred_value_high_m",
                               "status_premium_m", "edge_z", "forecast_value_m", "forecast_change_pct",
                               "min", "match_method", "last_season"] if c in sl_df.columns]
        export_df = sl_df[sl_cols].rename(columns={c: label(c) for c in sl_cols})
        show_df = sl_df[sl_cols].copy()
        for c in ["actual_value_m", "predicted_value_m", "pred_value_low_m", "pred_value_high_m",
                  "status_premium_m", "forecast_value_m"]:
            if c in show_df.columns:
                show_df[c] = show_df[c].apply(fmt_eur_m)
        if "forecast_change_pct" in show_df.columns:
            show_df["forecast_change_pct"] = show_df["forecast_change_pct"].apply(lambda v: format_value(v, "surplus_pct"))
        show_df = show_df.rename(columns={c: label(c) for c in sl_cols})
        render_metrics_html([
            ("Players saved", f"{len(sl):,}", "This session only", "neutral"),
            ("Combined market value", fmt_eur_m(pd.to_numeric(sl_df["actual_value_m"], errors="coerce").sum()),
             "Transfermarkt", "neutral"),
            ("Combined performance value", fmt_eur_m(pd.to_numeric(sl_df["predicted_value_m"], errors="coerce").sum()),
             "Model", "neutral"),
            ("Median age", f"{pd.to_numeric(sl_df['age_clean'], errors='coerce').median():.0f}", "", "neutral"),
        ])
        st.write("")
        st.dataframe(show_df, hide_index=True, use_container_width=True)
        c_dl, c_rm, c_clr = st.columns([1, 2, 1])
        c_dl.download_button("Download shortlist (CSV)", export_df.to_csv(index=False).encode("utf-8"),
                             file_name="stattrick_shortlist.csv", mime="text/csv")
        to_remove = c_rm.multiselect("Remove players", sl, label_visibility="collapsed",
                                     placeholder="Select players to remove", key="sl_remove_pick")
        if to_remove:
            c_rm.button("Remove selected", key="sl_remove_btn", on_click=_shortlist_remove, args=(to_remove,))
        c_clr.button("Clear shortlist", key="sl_clear", on_click=_shortlist_remove, args=(sl,))
        st.caption("The shortlist is kept in this browser session and is lost when the tab is closed. "
                   "Download the CSV to keep it.")

# --- TAB 5: METHODOLOGY ---
with tab5:
    section_heading("Methodology", "What the numbers mean, where they come from, and where they stop")
    _n = len(df_master)
    _cur = CURRENT_SEASON or "the current season"
    _adv = df_master["adv_last_season"].dropna().max() if "adv_last_season" in df_master.columns else None
    _xg = df_master["xg_last_season"].dropna().max() if "xg_last_season" in df_master.columns else None
    _active = int(df_master["active_now"].sum())
    _ylog, _plog = np.log1p(df_master["actual_value_m"]), np.log1p(df_master["predicted_value_m"])
    _r2 = 1 - ((_ylog - _plog) ** 2).sum() / ((_ylog - _ylog.mean()) ** 2).sum()
    _cov = (df_master["outside_band"] == 0).mean() if "outside_band" in df_master.columns else np.nan
    _stale = int(df_master["tm_club_mismatch"].sum()) if "tm_club_mismatch" in df_master.columns else 0
    _approx = int((df_master["match_method"] != "exact").sum()) if "match_method" in df_master.columns else 0

    _fc = metrics.get("forecast", {}) if isinstance(metrics, dict) else {}
    if _fc.get("available"):
        _fc_text = (
            f"A third number, separate from the two above: where Transfermarkt is likely to put his value in "
            f"12 months (forecast for {_fc.get('forecast_for')}, from values as of {_fc.get('value_as_of')}). "
            f"Unlike the performance value it openly uses Transfermarkt's own history: today's value, its recent "
            f"momentum, his peak, age, club level and output.\n\n"
            f"Back-test on a year the model never saw ({_fc.get('backtest')}, {_fc.get('backtest_players'):,} players): "
            f"the forecast landed within 25% of the real value for {_fc.get('within_25pct', 0):.0%} of players and "
            f"within 50% for {_fc.get('within_50pct', 0):.0%}; the typical miss was {_fc.get('median_miss_pct', 0):.0f}%. "
            f"It called the direction (up or down) correctly {_fc.get('direction_right', 0):.0%} of the time. "
            f"Assuming no change would have been within 25% for only {_fc.get('no_change_within_25pct', 0):.0%}. "
            f"It explained {_fc.get('r2_of_change', 0):.0%} of the variation in one-year value changes, and missed by "
            f"€{_fc.get('mae_eur_m')}M on average against €{_fc.get('mae_eur_m_if_no_change')}M for assuming no change. "
            f"The fifth of players it rated highest were forecast {_fc.get('top_fifth_predicted_pct'):+.0f}% and actually "
            f"moved {_fc.get('top_fifth_actual_pct'):+.0f}% ({_fc.get('top_fifth_share_rose', 0):.0%} of them rose); the "
            f"lowest fifth were forecast {_fc.get('bottom_fifth_predicted_pct'):+.0f}% and moved "
            f"{_fc.get('bottom_fifth_actual_pct'):+.0f}%. The typical established player lost "
            f"{abs(_fc.get('median_actual_change_pct', 0)):.0f}% that year, so a flat forecast is a good one.")
    else:
        _fc_text = "Not available in this build (the dated valuation history has not been downloaded)."

    render_metrics_html([
        ("Players", f"{_n:,}", f"{_active:,} with minutes in {_cur}", "neutral"),
        ("Model fit (log R²)", f"{_r2:.2f}", "Out-of-fold: every player priced by a model that never saw him", "neutral"),
        ("Fair-band coverage", f"{_cov:.0%}" if pd.notna(_cov) else "—", "Target 70%", "neutral"),
        ("xG and xA through", str(_xg or _adv or "—"), f"Progression and defending through {_adv or '—'}", "neutral"),
    ])
    st.write("")
    m_left, m_right = st.columns(2)
    with m_left:
        with st.container(border=True):
            st.markdown(f"""
**What StatTrick is**

Two separate numbers, kept apart on purpose.

**Performance value** answers "what is this output worth?". It is built only from:

- on-pitch output per 90 minutes, weighted towards recent seasons
- career minutes
- age, position and league

It knows nothing about which club he plays for, how famous he is, or how long his contract runs.
The screener, the edge and "undervalued" all refer to this number.

**Status & contract premium** is what gets added (or taken away) once club level, international caps and
contract length are allowed in. These move a price without saying how good the player is, so they are shown
as a separate line rather than mixed into the performance value.

No Transfermarkt price is ever a model input. Transfermarkt's value is used only as the answer the models
are trained to approximate and the benchmark they are compared against.

**Data sources and coverage**

| Data | Source | Through |
|---|---|---|
| Goals, assists, shots, minutes | FBref | {_cur}, refreshed weekly |
| xG and xA | FBref to {_adv or "—"}, then Understat | {_xg or _cur} |
| Progression, defensive actions | FBref | {_adv or "—"} (last season published) |
| Market value, contract, position | Transfermarkt | latest snapshot |

Leagues covered: Premier League, La Liga, Serie A, Bundesliga, Ligue 1. A player needs 1,500 career minutes
and an appearance in one of the last two seasons to be included.
""")
    with m_right:
        with st.container(border=True):
            st.markdown(f"""
**How to read the numbers**

- **Performance value** is the median estimate for a player with this output, age, position and league.
- **Status & contract premium** is the difference once club level, caps and contract are added.
- **Fair band** is the range the market price falls inside about 70% of the time for comparable players.
  A price inside the band is normal disagreement, not a signal.
- **Edge** is how far the market price sits from the model, in units of that player's own band.
  The screener only lists players priced below their performance band.
- **Why this value** breaks one player's performance value into factors. Each factor's percentage is its effect
  relative to a typical player.
- **Similarity** compares per-90 statistical profiles, weighted by position. It describes style, not quality.
- **Fit index** compares a player to the target club's profile for his position group. 50 means no relationship.

**Market forecast**

{_fc_text}

**Known limits**

- Progression and defensive stats stop at {_adv or "—"}; later seasons are judged on goals, assists, shots,
  minutes, xG and xA. xG comes from two providers with slightly different models, so small steps between
  {_adv or "—"} and the following season are not meaningful.
- {_stale:,} players have changed club since their Transfermarkt record was last updated; their market value
  and contract describe the previous club and are flagged on their page.
- {_approx:,} players were matched to Transfermarkt by an approximate name match and are flagged.
- Very old or very young players with exceptional market values have few comparables; the model tends to
  price them low and says so on their page.
- A model edge is a reason to look closer, not a valuation to transact on.
""")

# --- Application Footer ---
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; padding: 25px 0 10px 0; font-family: -apple-system, BlinkMacSystemFont, sans-serif;">
        <span style="color: #6b7280; font-size: 0.85rem; letter-spacing: 0.02em;">
            Engineered & Maintained by 
            <a href="https://www.linkedin.com/in/sujalsharmaa/" target="_blank" style="color: #00ff87; text-decoration: none; font-weight: 600; border-bottom: 1px dotted #00ff87;">
                Sujal Sharma
            </a>
            &nbsp;•&nbsp; Automated MLOps & System Fit Architecture
        </span>
    </div>
    """,
    unsafe_allow_html=True
)