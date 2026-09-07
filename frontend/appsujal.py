import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.similarity_engine import PlayerSimilarityEngine

st.set_page_config(
    page_title="StatTrick | Scouting Intelligence",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded"
)

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
    if pd.isna(val):
        return "—"
    if col_name in ["actual_value_m", "predicted_value_m", "surplus_value_m"]:
        return fmt_eur_m(val)
    if "per90" in col_name:
        return f"{val:.2f}"
    if col_name == "min":
        return f"{int(val):,}"
    if isinstance(val, (float, np.floating)) and val.is_integer():
        return str(int(val))
    return str(val)


# --- 2. POSITIONAL PERCENTILE SCALING ---
def get_percentile(df: pd.DataFrame, col: str, target_val: float, position: str = None) -> int:
    if pd.isna(target_val) or col not in df.columns or df[col].isnull().all():
        return 50

    # Filter comparison pool by position group if available
    if position and "pos_clean" in df.columns:
        pool = df[df["pos_clean"] == position]
        if len(pool) < 15:
            pool = df
    else:
        pool = df

    series = pd.to_numeric(pool[col], errors="coerce").dropna()
    if series.empty:
        return 50
    return int(np.round((series <= target_val).mean() * 100))


def get_radar_metrics(position: str, available_cols: list) -> list:
    """Returns a tailored 5-stat metric cluster based on player position."""
    position_templates = {
        "FW": ["gls_per90", "xg_per90", "xag_per90", "sh_per90", "sot_per90"],
        "MF": ["ast_per90", "xag_per90", "prgp_per90", "prgc_per90", "tkl_per90"],
        "DF": ["tkl_per90", "int_per90", "clr_per90", "blk_per90", "aer_won_per90"],
        "GK": ["saves_per90", "psxg_net_per90", "clr_per90", "prgp_per90", "tkl_per90"],
    }

    selected = position_templates.get(position, position_templates["MF"])
    valid = [m for m in selected if m in available_cols]

    if len(valid) < 5:
        fallbacks = ["xg_per90", "xag_per90", "tkl_per90", "int_per90", "prgc_per90", "prgp_per90", "gls_per90"]
        for f in fallbacks:
            if f in available_cols and f not in valid:
                valid.append(f)
            if len(valid) == 5:
                break

    return valid[:5]


# --- 3. CSS DESIGN SYSTEM ---
def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg-primary: #05070A;
            --bg-secondary: #0D121C;
            --bg-panel: rgba(16, 22, 34, 0.75);
            --border-subtle: rgba(255, 255, 255, 0.08);
            --accent-grass: #39FF88;
            --accent-cyan: #22D3EE;
            --accent-red: #FF4D6D;
            --text-primary: #F8FAFC;
            --text-secondary: #94A3B8;
            --text-muted: #64748B;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
        }

        .stApp {
            background-color: var(--bg-primary);
            background-image: radial-gradient(circle at 50% 0%, rgba(34, 211, 238, 0.06) 0%, transparent 55%);
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: var(--bg-panel) !important;
            border-color: var(--border-subtle) !important;
            border-radius: 14px !important;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .financial-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .metric-card {
            background: var(--bg-panel);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 1.15rem 1.25rem;
            position: relative;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        .metric-card:hover {
            transform: translateY(-2px);
            border-color: rgba(34, 211, 238, 0.35);
        }

        .metric-top-bar {
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
        }
        .metric-top-bar.positive { background-color: var(--accent-grass); }
        .metric-top-bar.negative { background-color: var(--accent-red); }
        .metric-top-bar.neutral { background: linear-gradient(90deg, var(--accent-grass), var(--accent-cyan)); }

        .metric-label {
            color: var(--text-muted);
            font-size: 0.72rem;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.5px;
        }

        .metric-value {
            color: var(--text-primary);
            font-size: 1.45rem;
            font-weight: 800;
            margin-top: 0.25rem;
            font-variant-numeric: tabular-nums;
        }

        .metric-delta {
            font-size: 0.82rem;
            font-weight: 600;
            margin-top: 0.3rem;
        }
        .metric-delta.positive { color: var(--accent-grass); }
        .metric-delta.negative { color: var(--accent-red); }

        .brand-text {
            font-size: 2.4rem;
            font-weight: 800;
            letter-spacing: -1px;
            background: linear-gradient(135deg, var(--accent-grass), var(--accent-cyan));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            border-bottom: 1px solid var(--border-subtle);
        }

        .stTabs [data-baseweb="tab"] {
            color: var(--text-secondary) !important;
            padding: 10px 0;
            font-weight: 600;
        }

        .stTabs [aria-selected="true"] {
            color: var(--text-primary) !important;
            border-bottom-color: var(--accent-grass) !important;
        }

        .stButton button, .stDownloadButton button {
            background-color: var(--accent-grass) !important;
            color: #05070A !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
            border: none !important;
            transition: all 0.2s ease !important;
        }

        .stButton button:hover, .stDownloadButton button:hover {
            background-color: var(--accent-cyan) !important;
            box-shadow: 0 0 12px rgba(34, 211, 238, 0.4) !important;
        }

        @media (max-width: 768px) {
            .block-container { padding: 1rem; }
            .brand-text { font-size: 1.9rem; }
            .metric-value { font-size: 1.25rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metrics_html(cards_data: list, grid_class: str = "metric-grid"):
    cards_html = []
    for label_text, value_text, delta_text, sentiment in cards_data:
        delta_tag = f'<div class="metric-delta {sentiment}">{delta_text}</div>' if delta_text else ""
        card = (
            f'<div class="metric-card">'
            f'<div class="metric-top-bar {sentiment}"></div>'
            f'<div class="metric-label">{label_text}</div>'
            f'<div class="metric-value">{value_text}</div>'
            f'{delta_tag}'
            f"</div>"
        )
        cards_html.append(card)
    st.markdown(f'<div class="{grid_class}">{"".join(cards_html)}</div>', unsafe_allow_html=True)


def apply_dark_theme(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#F8FAFC"),
        margin=dict(t=35, b=35, l=35, r=35),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    return fig


# --- 4. DATA LOADER & CACHING ---
@st.cache_data
def load_data():
    db_path = "data/master_scouting_db.csv"
    if not os.path.exists(db_path):
        st.error(f"Database not found at {db_path}. Please execute `python src/valuation_model.py`.")
        st.stop()
    df = pd.read_csv(db_path)
    df.columns = [c.lower() for c in df.columns]

    # Reconstruct categorical fields if missing
    if "pos_clean" not in df.columns:
        pos_cols = [c for c in df.columns if c.startswith("pos_clean_")]
        if pos_cols:
            df["pos_clean"] = df[pos_cols].idxmax(axis=1).str.replace("pos_clean_", "")
        else:
            df["pos_clean"] = "MF"

    if "league" not in df.columns:
        league_cols = [c for c in df.columns if c.startswith("league_")]
        if league_cols:
            df["league"] = df[league_cols].idxmax(axis=1).str.replace("league_", "")
        else:
            df["league"] = "Unknown"

    df["pos_clean"] = df["pos_clean"].astype(str).str.upper()
    df["squad"] = df["squad"].astype(str).str.title()
    df["league"] = df["league"].astype(str).str.title()
    return df


@st.cache_resource
def load_engine():
    return PlayerSimilarityEngine(data_path="data/master_scouting_db.csv")


# --- 5. APP EXECUTION ---
inject_custom_css()
df_master = load_data()

try:
    engine = load_engine()
except Exception as e:
    st.error(f"Error loading similarity engine: {e}")
    st.stop()

# Header Banner
header_col1, header_col2 = st.columns([1, 1])
with header_col1:
    st.markdown(
        '<div class="brand-text">STATTRICK</div>'
        '<div style="color: #64748B; font-weight: 500; font-size: 0.92rem; margin-bottom: 1.5rem;">'
        "Scouting & Valuation Intelligence Engine"
        "</div>",
        unsafe_allow_html=True,
    )
with header_col2:
    st.markdown(
        f'<div style="text-align: right; margin-top: 1rem;">'
        f'<span style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); '
        f'padding: 6px 16px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; color: #F8FAFC;">'
        f"⚽ {len(df_master):,} Players Tracked</span></div>",
        unsafe_allow_html=True,
    )

tab1, tab2, tab3 = st.tabs(["🔍 Tactical Cloning", "💎 Arbitrage Screener", "⚔️ Head-to-Head Sandbox"])


# --- TAB 1: TACTICAL CLONING ---
with tab1:
    all_players = sorted(df_master["player"].dropna().unique().tolist())
    default_idx = all_players.index("Fermín López") if "Fermín López" in all_players else 0

    target = st.sidebar.selectbox("Select Target Player", all_players, index=default_idx)
    top_k = st.sidebar.slider("Number of Clones", 3, 10, 5)
    filter_pos = st.sidebar.checkbox("Enforce Same Position Group", False)

    target_row = df_master[df_master["player"] == target].iloc[0]
    pos = target_row.get("pos_clean", "MF")

    # Core Attribute Cards
    core_cards = [
        (label("squad"), str(target_row.get("squad", "N/A")), None, "neutral"),
        (label("pos_clean"), str(pos), None, "neutral"),
        (label("age_clean"), int(target_row.get("age_clean", 25)), None, "neutral"),
        (label("min"), f"{int(target_row.get('min', 0)):,}", None, "neutral"),
        (label("contract_years_left"), format_value(target_row.get("contract_years_left", 2), "contract_years_left"), None, "neutral"),
    ]
    render_metrics_html(core_cards, "metric-grid")

    # Financial Cards
    actual_val = target_row.get("actual_value_m", 0.0)
    pred_val = target_row.get("predicted_value_m", 0.0)
    surplus_val = target_row.get("surplus_value_m", 0.0)

    sentiment = "positive" if surplus_val > 0 else "negative"
    delta_str = f"{'+' if surplus_val > 0 else ''}€{surplus_val:.1f}M ({'Undervalued' if surplus_val > 0 else 'Market Premium'})"

    financial_cards = [
        ("Market Value (Transfermarkt)", fmt_eur_m(actual_val), None, "neutral"),
        ("Model Value (XGBoost)", fmt_eur_m(pred_val), None, "neutral"),
        ("Surplus Value", fmt_eur_m(surplus_val), delta_str, sentiment),
    ]
    render_metrics_html(financial_cards, "financial-grid")

    st.markdown(
        "<h3 style='font-size: 15px; text-transform: uppercase; color: var(--text-muted); margin-top: 1rem; margin-bottom: 0.75rem;'>"
        "Statistical Clones</h3>",
        unsafe_allow_html=True,
    )

    results = engine.find_similar_players(target, top_n=top_k, same_position=filter_pos)

    if isinstance(results, pd.DataFrame):
        display_results = results.copy()
        if "actual_value_m" in df_master.columns:
            display_results = display_results.merge(
                df_master[["player", "actual_value_m", "predicted_value_m", "surplus_value_m"]],
                left_on="Player",
                right_on="player",
                how="left",
            ).drop(columns=["player"])

            for val_col in ["actual_value_m", "predicted_value_m", "surplus_value_m"]:
                display_results[label(val_col)] = display_results[val_col].apply(fmt_eur_m)
            display_results.drop(columns=["actual_value_m", "predicted_value_m", "surplus_value_m"], inplace=True)

        st.dataframe(display_results, hide_index=True, use_container_width=True)
        top_clone = display_results.iloc[0]["Player"]

        st.download_button(
            label="📥 Export Dossier",
            data=display_results.to_csv(index=False).encode("utf-8"),
            file_name=f"{target.replace(' ', '_')}_Dossier.csv",
            mime="text/csv",
        )
    else:
        top_clone = None
        st.warning("Could not compute clones under the specified constraints.")

    # Charts
    with st.container(border=True):
        col_radar, col_scatter = st.columns(2)

        # Radar Footprint
        with col_radar:
            st.markdown(
                f"<h3 style='font-size: 14px; text-transform: uppercase; color: var(--text-muted);'>"
                f"{pos} Tactical Footprint</h3>",
                unsafe_allow_html=True,
            )
            if top_clone:
                radar_stats = get_radar_metrics(pos, df_master.columns)
                categories = [label(s) for s in radar_stats] + [label(radar_stats[0])]

                target_pcts = [get_percentile(df_master, s, target_row.get(s, 0), pos) for s in radar_stats]
                clone_row = df_master[df_master["player"] == top_clone].iloc[0]
                clone_pcts = [get_percentile(df_master, s, clone_row.get(s, 0), pos) for s in radar_stats]

                target_pcts.append(target_pcts[0])
                clone_pcts.append(clone_pcts[0])

                fig_radar = go.Figure()
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=target_pcts,
                        theta=categories,
                        fill="toself",
                        name=target,
                        line_color="#39FF88",
                        fillcolor="rgba(57,255,136,0.22)",
                    )
                )
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=clone_pcts,
                        theta=categories,
                        fill="toself",
                        name=top_clone,
                        line_color="#22D3EE",
                        fillcolor="rgba(34,211,238,0.15)",
                    )
                )
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(visible=False, range=[0, 100]),
                        angularaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                    ),
                    legend=dict(orientation="h", y=1.1, xanchor="center", x=0.5),
                )
                st.plotly_chart(apply_dark_theme(fig_radar), use_container_width=True)

        # Dynamic Chart Studio
        with col_scatter:
            st.markdown(
                "<h3 style='font-size: 14px; text-transform: uppercase; color: var(--text-muted);'>"
                "Chart Studio</h3>",
                unsafe_allow_html=True,
            )

            available_axes = [
                c for c in ["actual_value_m", "predicted_value_m", "surplus_value_m"] + engine.feature_cols
                if c in df_master.columns
            ]

            drop1, drop2 = st.columns(2)
            mx = drop1.selectbox("X-Axis", available_axes, format_func=label, index=0)
            my = drop2.selectbox("Y-Axis", available_axes, format_func=label, index=3 if len(available_axes) > 3 else 0)

            scatter_df = df_master.dropna(subset=[mx, my]).copy()
            fig_scatter = go.Figure()

            # Background Population
            fig_scatter.add_trace(
                go.Scatter(
                    x=scatter_df[mx],
                    y=scatter_df[my],
                    mode="markers",
                    marker_color="rgba(255,255,255,0.12)",
                    customdata=np.stack((scatter_df["player"], scatter_df[mx], scatter_df[my]), axis=-1),
                    hovertemplate="<b>%{customdata[0]}</b><br>"
                    + label(mx) + ": %{customdata[1]:.2f}<br>"
                    + label(my) + ": %{customdata[2]:.2f}<extra></extra>",
                    name="Population",
                )
            )

            # Trendline
            if len(scatter_df) > 1:
                slope, intercept = np.polyfit(scatter_df[mx], scatter_df[my], 1)
                x_trend = np.linspace(scatter_df[mx].min(), scatter_df[mx].max(), 100)
                fig_scatter.add_trace(
                    go.Scatter(
                        x=x_trend,
                        y=slope * x_trend + intercept,
                        mode="lines",
                        line=dict(color="rgba(34, 211, 238, 0.5)", width=2, dash="dash"),
                        hoverinfo="skip",
                        name="Trend (OLS)",
                    )
                )

            # Highlight Target
            fig_scatter.add_trace(
                go.Scatter(
                    x=[target_row.get(mx)],
                    y=[target_row.get(my)],
                    mode="markers+text",
                    marker=dict(color="#39FF88", size=14, symbol="diamond", line=dict(color="#FFF", width=1)),
                    text=[target.split()[-1]],
                    textposition="top center",
                    name=target,
                )
            )

            # Highlight Top Clone
            if top_clone:
                clone_val_x = df_master[df_master["player"] == top_clone].iloc[0].get(mx)
                clone_val_y = df_master[df_master["player"] == top_clone].iloc[0].get(my)
                fig_scatter.add_trace(
                    go.Scatter(
                        x=[clone_val_x],
                        y=[clone_val_y],
                        mode="markers+text",
                        marker=dict(color="#22D3EE", size=14, symbol="star", line=dict(color="#FFF", width=1)),
                        text=[top_clone.split()[-1]],
                        textposition="top center",
                        name=top_clone,
                    )
                )

            fig_scatter.update_layout(xaxis_title=label(mx), yaxis_title=label(my), showlegend=False)
            st.plotly_chart(apply_dark_theme(fig_scatter), use_container_width=True)


# --- TAB 2: ARBITRAGE SCREENER ---
with tab2:
    st.markdown(
        "<h3 style='font-size: 15px; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.75rem;'>"
        "Undervalued Asset Screener</h3>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        col_f1, col_f2, col_f3 = st.columns(3)
        max_age = col_f1.slider("Max Age", 17, 36, 24)
        min_surplus = col_f2.slider("Min Surplus Value (€M)", 0.0, 30.0, 5.0)
        min_minutes = col_f3.slider("Minimum Career Minutes", 500, 6000, 1500)

        screener_df = df_master[
            (pd.to_numeric(df_master["age_clean"], errors="coerce") <= max_age)
            & (df_master["surplus_value_m"] >= min_surplus)
            & (df_master["min"] >= min_minutes)
        ].copy()

        if not screener_df.empty:
            screener_df = screener_df.sort_values(by="surplus_value_m", ascending=False)
            display_columns = [
                c for c in ["player", "squad", "league", "pos_clean", "age_clean", "actual_value_m", "predicted_value_m", "surplus_value_m"]
                if c in screener_df.columns
            ]

            output_df = screener_df[display_columns].copy()
            for curr_col in ["actual_value_m", "predicted_value_m", "surplus_value_m"]:
                if curr_col in output_df.columns:
                    output_df[curr_col] = output_df[curr_col].apply(fmt_eur_m)

            output_df.rename(columns={c: label(c) for c in output_df.columns}, inplace=True)
            st.dataframe(output_df, hide_index=True, use_container_width=True)
        else:
            st.info("No players match the chosen filter parameters.")


# --- TAB 3: HEAD-TO-HEAD SANDBOX ---
with tab3:
    st.markdown(
        "<h3 style='font-size: 15px; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.75rem;'>"
        "Direct Comparison Sandbox</h3>",
        unsafe_allow_html=True,
    )

    default_h2h = [p for p in ["Bukayo Saka", "Phil Foden", "Cole Palmer"] if p in all_players][:2]
    selected_players = st.multiselect("Select Players (Max 3)", all_players, default=default_h2h, max_selections=3)

    if selected_players:
        h2h_df = df_master[df_master["player"].isin(selected_players)].copy()

        with st.container(border=True):
            col_table, col_radar_h2h = st.columns([1, 1.25])

            with col_table:
                st.markdown(
                    "<h4 style='font-size: 13px; text-transform: uppercase; color: var(--text-muted);'>"
                    "Core Metrics Comparison</h4>",
                    unsafe_allow_html=True,
                )

                core_fields = [
                    c for c in ["squad", "league", "pos_clean", "age_clean", "min"] + engine.feature_cols[:5] + ["actual_value_m", "surplus_value_m"]
                    if c in h2h_df.columns
                ]

                table_df = h2h_df[["player"] + core_fields].copy()
                for c in core_fields:
                    table_df[c] = table_df[c].apply(lambda x: format_value(x, c))

                table_df.rename(columns={c: label(c) for c in table_df.columns}, inplace=True)
                st.dataframe(table_df.set_index(label("player")).T, use_container_width=True)

            with col_radar_h2h:
                if len(selected_players) >= 2:
                    first_pos = h2h_df.iloc[0].get("pos_clean", "MF")
                    st.markdown(
                        f"<h4 style='font-size: 13px; text-transform: uppercase; color: var(--text-muted);'>"
                        f"Tactical Overlay ({first_pos} Basis)</h4>",
                        unsafe_allow_html=True,
                    )

                    h2h_radar_stats = get_radar_metrics(first_pos, df_master.columns)
                    radar_cats = [label(s) for s in h2h_radar_stats] + [label(h2h_radar_stats[0])]

                    fig_h2h = go.Figure()
                    trace_colors = ["#39FF88", "#22D3EE", "#FF4D6D"]
                    fill_colors = ["rgba(57,255,136,0.22)", "rgba(34,211,238,0.15)", "rgba(255,77,109,0.15)"]

                    for idx, p_name in enumerate(selected_players):
                        p_record = h2h_df[h2h_df["player"] == p_name].iloc[0]
                        p_pcts = [get_percentile(df_master, s, p_record.get(s, 0), first_pos) for s in h2h_radar_stats]
                        p_pcts.append(p_pcts[0])

                        fig_h2h.add_trace(
                            go.Scatterpolar(
                                r=p_pcts,
                                theta=radar_cats,
                                fill="toself",
                                name=p_name,
                                line=dict(color=trace_colors[idx], width=2),
                                fillcolor=fill_colors[idx],
                            )
                        )

                    fig_h2h.update_layout(
                        polar=dict(
                            bgcolor="rgba(0,0,0,0)",
                            radialaxis=dict(visible=False, range=[0, 100]),
                            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                        ),
                        legend=dict(orientation="h", y=1.1, xanchor="center", x=0.5),
                    )
                    st.plotly_chart(apply_dark_theme(fig_h2h), use_container_width=True)