import unicodedata
import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.similarity_engine import PlayerSimilarityEngine
from src.system_fit import SystemFitEngine

st.set_page_config(
    page_title="StatTrick | Scouting Intelligence",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded"
)

# --- 0. DESIGN TOKENS (shared by CSS + Plotly) ---
MINT = "#35E08C"          # primary accent — floodlit pitch green
MINT_DIM = "#1F8F59"
CYAN = "#4CC9F0"          # secondary accent
CORAL = "#FF5C7A"         # premium / negative
AMBER = "#FFB020"
INK = "#06100D"           # app base
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
    "predicted_value_m": "Model Value (€M)",
    "surplus_value_m": "Surplus Value (€M)",
    "contract_years_left": "Contract (Years)",
    "age_clean": "Age",
    "pos_clean": "Position",
    "squad": "Club",
    "league": "League",
}

def label(col: str) -> str:
    return METRIC_LABELS.get(col, col.replace("_", " ").title())

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
    """Strips accents and special characters for keyboard searchability."""
    if not isinstance(name, str):
        return str(name)
    return ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')

def get_percentile(df: pd.DataFrame, col: str, target_val: float, position: str = None) -> int:
    if pd.isna(target_val) or col not in df.columns or df[col].isnull().all():
        return 50
    if position and "pos_clean" in df.columns:
        pool = df[df["pos_clean"] == position]
        if len(pool) < 15:
            pool = df
    else:
        pool = df
    series = pd.to_numeric(pool[col], errors="coerce").dropna()
    if series.empty: return 50
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

def render_fit_gauge(club_line: str, archetype: str, tier: str, score) -> None:
    val = max(0.0, min(100.0, safe_float(score)))
    color = MINT if val >= 80 else (CYAN if val >= 70 else CORAL)
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
@st.cache_data
def load_data():
    db_path = "data/master_scouting_db.csv"
    if not os.path.exists(db_path):
        st.error(f"Database not found at {db_path}.")
        st.stop()
    df = pd.read_csv(db_path)
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
    df["squad"] = df["squad"].astype(str).str.title()
    df["league"] = df["league"].astype(str).str.title()
    return df

@st.cache_resource
def load_engine():
    return PlayerSimilarityEngine(data_path="data/master_scouting_db.csv")

@st.cache_resource
def load_system_engine():
    return SystemFitEngine()

@st.cache_data
def load_timeline_data():
    timeline_path = "data/player_timeline_db.csv"
    if not os.path.exists(timeline_path):
        return pd.DataFrame()
    df_time = pd.read_csv(timeline_path)
    df_time.columns = [c.lower() for c in df_time.columns]
    return df_time

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
default_idx = all_players.index("Fermín López") if "Fermín López" in all_players else 0

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

st.sidebar.markdown("---")
st.sidebar.markdown("### DATA SOURCES")
st.sidebar.caption(
    "• **Tactical data:** [FBref](https://fbref.com/)\n\n"
    "• **Financials:** [Transfermarkt](https://www.transfermarkt.com/)"
)

# Dynamically calculate last sync from database file modification time
db_path = "data/master_scouting_db.csv"
if os.path.exists(db_path):
    last_modified = datetime.fromtimestamp(os.path.getmtime(db_path)).strftime("%b %d, %Y")
else:
    last_modified = "Weekly"

st.sidebar.caption(
    f"• **Sync Cadence:** Automated weekly (Mondays 04:00 UTC)\n\n"
    f"*Last Model Sync: {last_modified} | XGBoost + Exp Decay*"
)

tab1, tab2, tab3 = st.tabs(["Tactical Cloning", "Arbitrage Screener", "Head-to-Head Sandbox"])

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

    core_cards = [
        (label("squad"), str(target_row.get("squad", "N/A")), None, "neutral"),
        (label("pos_clean"), str(pos), None, "neutral"),
        (label("age_clean"), int(target_row.get("age_clean", 25)), None, "neutral"),
        (label("min"), f"{int(target_row.get('min', 0)):,}", None, "neutral"),
        (label("contract_years_left"), format_value(target_row.get("contract_years_left", 2), "contract_years_left"), None, "neutral"),
    ]
    render_metrics_html(core_cards)
    st.write("")

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
        ("Model fair value · XGBoost", fmt_eur_m(pred_val), band_str, "neutral"),
        ("Market inefficiency", fmt_eur_m(surplus_val), delta_str, sentiment),
    ]
    render_metrics_html(financial_cards)

    st.write("")
    with st.container(border=True):
        render_valuation_bar(actual_val, pred_val, pred_low, pred_high)

    st.caption(
        "The fair band is the 15th-to-85th percentile prediction interval from quantile regression. "
        "A wide band means higher performance variance, or more room to negotiate."
    )

    section_heading("Statistical clones", "Closest tactical output to your target")
    st.info("**What is a clone?** A player who shares a highly similar statistical profile and on-pitch playstyle to your target. Calculated via cosine similarity across multi-season tactical metrics.")

    results = engine.find_similar_players(target, top_n=top_k, same_position=filter_pos)

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
            if "similar" in str(c).lower() or "score" in str(c).lower():
                try:
                    col_max = float(pd.to_numeric(display_results[c], errors="coerce").max())
                except (TypeError, ValueError):
                    continue
                if pd.isna(col_max):
                    continue
                scale = 1.0 if col_max <= 1.0 else 100.0
                clone_col_config[c] = st.column_config.ProgressColumn(
                    str(c), min_value=0.0, max_value=scale,
                    format="%.3f" if scale == 1.0 else "%.1f",
                )
        st.dataframe(display_results, hide_index=True, use_container_width=True, column_config=clone_col_config)
        st.download_button(
            "Download clone list (CSV)",
            display_results.to_csv(index=False).encode("utf-8"),
            file_name=f"stattrick_clones_{str(target).replace(' ', '_')}.csv",
            mime="text/csv",
        )
        top_clone = display_results.iloc[0]["Player"]
    else:
        top_clone = None
        st.warning("Could not compute clones.")

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

        with col_form:
            st.markdown('<div class="panel-title" style="margin-bottom: 2px;">Form vs. Baseline · Multi-Season xG + xA / 90</div>', unsafe_allow_html=True)
            if not df_timeline.empty and "season" in df_timeline.columns:
                player_history = df_timeline[df_timeline["player"].str.lower() == str(target).lower()].copy()

                if not player_history.empty and len(player_history) > 1:
                    player_history = player_history.sort_values(by="season")
                    xg_series = pd.to_numeric(player_history.get("xg_per90", 0), errors="coerce").fillna(0)
                    xa_series = pd.to_numeric(player_history.get("xag_per90", 0), errors="coerce").fillna(0)
                    player_history["xg_xa"] = xg_series + xa_series

                    fig_timeline = go.Figure()
                    fig_timeline.add_trace(go.Scatter(
                        x=player_history["season"],
                        y=player_history["xg_xa"],
                        mode="lines+markers",
                        line=dict(color=MINT, width=2.5, shape="spline", smoothing=0.3),
                        marker=dict(size=8, color=INK, line=dict(color=MINT, width=2)),
                        fill="tozeroy",
                        fillcolor="rgba(53, 224, 140, 0.12)",
                        hovertemplate="<b>%{x}</b><br>xG + xA / 90: %{y:.2f}<extra></extra>"
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
                else:
                    st.markdown(
                        '<div style="height: 160px; display: grid; place-items: center; border: 1px dashed var(--line); border-radius: 10px; color: var(--text-faint); font-size: 0.85rem; margin-top: 8px;">'
                        'Insufficient multi-season history to plot trend'
                        '</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.markdown(
                    '<div style="height: 160px; display: grid; place-items: center; border: 1px dashed var(--line); border-radius: 10px; color: var(--text-faint); font-size: 0.85rem; margin-top: 8px;">'
                    'Run valuation_model.py to initialize player_timeline_db.csv'
                    '</div>',
                    unsafe_allow_html=True
                )

    # --- RADAR AND CHART STUDIO ---
    section_heading("Tactical overlay", "Percentile footprint and a free-form scatter")
    with st.container(border=True):
        col_radar, col_scatter = st.columns(2)
        with col_radar:
            st.markdown(f'<div class="panel-title">{pos} tactical footprint · percentile vs position</div>', unsafe_allow_html=True)
            if top_clone:
                radar_stats = get_radar_metrics(pos, df_master.columns)
                categories = [label(s) for s in radar_stats] + [label(radar_stats[0])]
                target_pcts = [get_percentile(df_master, s, target_row.get(s, 0), pos) for s in radar_stats]
                clone_row = df_master[df_master["player"] == top_clone].iloc[0]
                clone_pcts = [get_percentile(df_master, s, clone_row.get(s, 0), pos) for s in radar_stats]
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

# --- TAB 2: ARBITRAGE SCREENER ---
with tab2:
    section_heading("Arbitrage screener", "Output that outruns the price tag")
    st.info("**What is the arbitrage screener?** This tool highlights players whose underlying tactical outputs significantly outperform their current public market valuation. Set your criteria below to discover hidden gems.")

    with st.container(border=True):
        col_f1, col_f2, col_f3 = st.columns(3)
        max_age = col_f1.slider("Max Age", 17, 36, 24)
        min_surplus = col_f2.slider("Min Surplus Value (€M)", 0.0, 30.0, 5.0)
        min_minutes = col_f3.slider("Minimum Career Minutes", 500, 6000, 1500)

        screener_df = df_master[(pd.to_numeric(df_master["age_clean"], errors="coerce") <= max_age) & (df_master["surplus_value_m"] >= min_surplus) & (df_master["min"] >= min_minutes)].copy()
        if not screener_df.empty:
            screener_df = screener_df.sort_values(by="surplus_value_m", ascending=False)
            display_columns = [c for c in ["player", "squad", "league", "pos_clean", "age_clean", "actual_value_m", "predicted_value_m", "surplus_value_m"] if c in screener_df.columns]
            output_df = screener_df[display_columns].copy()
            for curr_col in ["actual_value_m", "predicted_value_m", "surplus_value_m"]:
                if curr_col in output_df.columns: output_df[curr_col] = output_df[curr_col].apply(fmt_eur_m)
            output_df.rename(columns={c: label(c) for c in output_df.columns}, inplace=True)
            pool_surplus = pd.to_numeric(screener_df["surplus_value_m"], errors="coerce")
            pool_age = pd.to_numeric(screener_df["age_clean"], errors="coerce")
            best_row = screener_df.iloc[0]
            st.write("")
            render_metrics_html([
                ("Players matched", f"{len(screener_df):,}", "Sorted by surplus value", "neutral"),
                ("Combined model edge", fmt_eur_m(pool_surplus.sum()), "Total value the market is missing", "positive"),
                ("Median age of pool", f"{pool_age.median():.0f}" if pool_age.notna().any() else "—", "Resale runway", "neutral"),
                ("Biggest single edge", str(best_row.get("player", "—")).split()[-1],
                 f"{fmt_eur_m(best_row.get('surplus_value_m', 0))} · {best_row.get('squad', '—')}", "positive"),
            ])
            st.write("")
            st.dataframe(output_df, hide_index=True, use_container_width=True)
            st.download_button(
                "Download shortlist (CSV)",
                output_df.to_csv(index=False).encode("utf-8"),
                file_name="stattrick_shortlist.csv",
                mime="text/csv",
            )
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
                    first_pos = h2h_df.iloc[0].get("pos_clean", "MF")
                    st.markdown(f'<div class="panel-title">Tactical overlay · {first_pos} basis</div>', unsafe_allow_html=True)
                    h2h_radar_stats = get_radar_metrics(first_pos, df_master.columns)
                    radar_cats = [label(s) for s in h2h_radar_stats] + [label(h2h_radar_stats[0])]
                    fig_h2h = go.Figure()

                    trace_colors = [MINT, CYAN, CORAL]
                    fill_colors = ["rgba(53, 224, 140, 0.20)", "rgba(76, 201, 240, 0.16)", "rgba(255, 92, 122, 0.14)"]

                    for idx, p_name in enumerate(selected_players):
                        p_record = h2h_df[h2h_df["player"] == p_name].iloc[0]
                        p_pcts = [get_percentile(df_master, s, p_record.get(s, 0), first_pos) for s in h2h_radar_stats]
                        p_pcts.append(p_pcts[0])
                        fig_h2h.add_trace(go.Scatterpolar(r=p_pcts, theta=radar_cats, fill="toself", name=p_name, line=dict(color=trace_colors[idx], width=2), fillcolor=fill_colors[idx]))

                    fig_h2h.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 100])), legend=dict(orientation="h", y=1.14, xanchor="center", x=0.5))
                    st.plotly_chart(apply_custom_theme(fig_h2h), use_container_width=True)