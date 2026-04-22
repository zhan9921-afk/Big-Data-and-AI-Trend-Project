from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_adapter_v4_real import load_dashboard_data
from llm_explainer_v5_openai import (
    compare_players,
    explain_chart,
    explain_player,
    explain_query_results,
    explain_smart_query_insights,
    explain_team,
)
from query_planner_v4_llm import TEAM_ALIASES, build_dashboard_plan, merge_plan_with_ui_filters

BASE_DIR = Path(__file__).resolve().parent
PAGES = ["📋 Shortlist", "🔍 Player Analysis", "🏟️ Team Analysis", "⚖️ Compare", "🧠 Smart Query"]
TEAM_METRICS = {
    "Current APY": "current_apy",
    "Actual Next APY": "actual_apy",
    "Predicted Next APY": "predicted_apy",
    "Guaranteed": "curr_guaranteed",
    "Contract Total": "contract_total",
}

st.set_page_config(page_title="GridironIQ Real-Data Dashboard", page_icon="🏈", layout="wide")

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Source+Sans+3:wght@400;600;700&display=swap');
    /* Kill the white header strip Streamlit puts at the very top */
    header[data-testid="stHeader"] { background: #101B30 !important; }
    div[data-testid="stToolbar"] { background: transparent !important; }
    header[data-testid="stHeader"]::before { background: #101B30 !important; }
    .stApp > header { background: #101B30 !important; }

    .stApp { background-color: #101B30; color: #F4F7FB; }
    section[data-testid="stSidebar"] { background-color: #13223E; border-right: 1px solid #2A3F5F; }
    .hero { background: linear-gradient(135deg, #013369 0%, #141C2F 55%, #D50A0A 100%); border-radius: 18px; padding: 28px 36px; margin-bottom: 18px; }
    .hero h1 { font-family: 'Bebas Neue', sans-serif; color: white; font-size: 3rem; letter-spacing: 5px; margin: 0; }
    .hero p { color: #D8E2EF; margin: 6px 0 0 0; font-size: 1.05rem; }
    .player-card { background: linear-gradient(160deg, #131B2E 0%, #1A2540 100%); border: 1px solid #2A3550; border-radius: 16px; padding: 18px 22px; margin-bottom: 12px; }
    .player-card .name { font-family: 'Bebas Neue', sans-serif; color: white; font-size: 1.8rem; letter-spacing: 2px; }
    .player-card .meta { color: #A8BDD9; margin-top: 2px; }
    .metrics-line { color: #DDE8F6; margin-top: 12px; font-size: .95rem; }
    .stat-row { display: flex; gap: 12px; margin-top: 14px; flex-wrap: wrap; }
    .stat-box { background: #0B1120; border: 1px solid #1E2A3A; border-radius: 10px; padding: 10px 14px; min-width: 120px; flex: 1; }
    .stat-label { color: #7F95B2; font-size: .75rem; text-transform: uppercase; letter-spacing: 1px; }
    .stat-value { color: white; font-size: 1.15rem; font-weight: 700; }
    .ai-box { background: linear-gradient(160deg, #0D1A2F 0%, #152238 100%); border: 1px solid #2A3F5F; border-left: 4px solid #4A90D9; border-radius: 12px; padding: 18px 20px; margin: 10px 0 14px 0; color: #E5EEF9; line-height: 1.65; }
    .ai-title { color: #7CC1FF; font-weight: 700; font-size: .82rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
    .kpi-card { background: #131B2E; border: 1px solid #1E2A3A; border-radius: 12px; padding: 14px 18px; text-align: center; }
    .kpi-label { color: #7F95B2; font-size: .75rem; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-value { color: white; font-size: 1.75rem; font-weight: 700; }
    .badge-sign, .badge-avoid, .badge-monitor { padding: 5px 14px; border-radius: 999px; font-weight: 700; font-size: .82rem; display: inline-block; }
    .badge-sign { background: linear-gradient(135deg, #0D6E3B, #28a745); color: white; }
    .badge-avoid { background: linear-gradient(135deg, #8B1A1A, #dc3545); color: white; }
    .badge-monitor { background: linear-gradient(135deg, #8B7500, #ffc107); color: black; }
    .headshot-img { width: 92px; height: 92px; object-fit: cover; border-radius: 14px; border: 1px solid #2A3550; background: #0B1120; }
    .headshot-fallback { width: 92px; height: 92px; border-radius: 14px; border: 1px solid #2A3550; display:flex; align-items:center; justify-content:center; background: linear-gradient(145deg, #0B1120, #18243A); color: #8FB3FF; font-size: 1.8rem; font-weight: 700; }
    .footer { text-align: center; color: #6E7F99; font-size: .8rem; padding: 20px 0 8px 0; border-top: 1px solid #1E2A3A; margin-top: 36px; }
    .query-help { background: #12213A; border: 1px solid #2D4670; border-radius: 12px; padding: 10px 12px; color: #D8E2EF; margin-bottom: 12px; font-size: 0.9rem; }
    .count-banner { background: linear-gradient(160deg, #101E34 0%, #17294A 100%); border: 1px solid #2A3F5F; border-radius: 12px; padding: 12px 16px; color: #DDE8F6; margin: 8px 0 14px 0; }
    div[data-testid="stTextInput"] input { background: #F7FAFF !important; color: #0B1120 !important; border: 2px solid #6CB6FF !important; border-radius: 10px !important; font-weight: 600 !important; }
    /* Center the navigation tabs */
    div[data-testid="stRadio"] > div { display: flex; justify-content: center; }

    /* =========================================================
       NFL DARK-THEME OVERRIDES — tables, selects, containers
       Fixes white backgrounds + low-contrast gray text.
       ========================================================= */

    /* Global text contrast — no more muted grays */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label,
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #F4F7FB !important;
    }
    h1, h2, h3, h4, h5, h6 { color: #FFFFFF !important; }
    .stApp a { color: #7CC1FF !important; }

    /* Streamlit native dataframe / table — force dark
       st.dataframe renders into an inner iframe/canvas via glide-data-grid;
       we set color-scheme:dark + theme vars so it picks up dark tokens. */
    .stDataFrame, div[data-testid="stDataFrame"],
    div[data-testid="stDataFrameResizable"],
    .stTable, div[data-testid="stTable"] {
        border-radius: 12px;
        border: 1px solid #2A3F5F;
        background-color: #17274A !important;
        color-scheme: dark !important;
        --gdg-bg-cell: #17274A;
        --gdg-bg-cell-medium: #1B2E54;
        --gdg-bg-header: #1E3158;
        --gdg-bg-header-hovered: #24396A;
        --gdg-bg-header-has-focus: #24396A;
        --gdg-text-dark: #FFFFFF;
        --gdg-text-medium: #FFFFFF;
        --gdg-text-light: #E8EEF8;
        --gdg-text-header: #FFD166;
        --gdg-text-header-selected: #FFFFFF;
        --gdg-border-color: #2A3F5F;
        --gdg-horizontal-border-color: #2A3F5F;
        --gdg-accent-color: #D50A0A;
        --gdg-accent-light: rgba(213,10,10,0.2);
        --gdg-bg-bubble: #1B2E54;
        --gdg-bg-bubble-selected: #D50A0A;
    }
    /* Inner iframe + canvas — dark */
    div[data-testid="stDataFrame"] iframe {
        background-color: #17274A !important;
        color-scheme: dark !important;
    }
    /* HTML table fallback (also used inside some dataframes) */
    .stDataFrame table, div[data-testid="stDataFrame"] table,
    .stTable table, div[data-testid="stTable"] table {
        background-color: #17274A !important;
        color: #FFFFFF !important;
    }
    .stDataFrame thead tr th, div[data-testid="stDataFrame"] thead tr th,
    .stTable thead tr th, div[data-testid="stTable"] thead tr th {
        background-color: #1E3158 !important;
        color: #FFD166 !important;
        border-bottom: 2px solid #D50A0A !important;
        font-weight: 700 !important;
        font-family: 'Bebas Neue', sans-serif !important;
        letter-spacing: 1.2px !important;
        text-transform: uppercase !important;
        font-size: .95rem !important;
    }
    .stDataFrame tbody tr td, div[data-testid="stDataFrame"] tbody tr td,
    .stTable tbody tr td, div[data-testid="stTable"] tbody tr td {
        background-color: #17274A !important;
        color: #FFFFFF !important;
        border-bottom: 1px solid #2A3F5F !important;
        font-size: .95rem !important;
    }
    .stDataFrame tbody tr:nth-child(even) td,
    div[data-testid="stDataFrame"] tbody tr:nth-child(even) td,
    .stTable tbody tr:nth-child(even) td,
    div[data-testid="stTable"] tbody tr:nth-child(even) td {
        background-color: #1B2E54 !important;
    }
    .stDataFrame tbody tr:hover td,
    div[data-testid="stDataFrame"] tbody tr:hover td {
        background-color: #24396A !important;
    }
    /* Row-number / corner cell */
    div[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] { background: #17274A !important; }

    /* ===== Custom NFL-styled HTML tables (rendered by render_dark_table) ===== */
    .nfl-table-wrap {
        border-radius: 10px; overflow: hidden;
        border: 1px solid #2A3F5F;
        background: #17274A;
        margin: 8px 0 14px 0;
    }
    table.nfl-table {
        width: 100%;
        border-collapse: collapse;
        background: #17274A !important;
        color: #FFFFFF !important;
        font-size: .82rem;
        white-space: nowrap;
    }
    table.nfl-table thead tr { background: #1E3158 !important; }
    table.nfl-table thead th {
        background: #1E3158 !important;
        color: #FFD166 !important;
        font-family: 'Source Sans 3', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: .3px !important;
        text-transform: none !important;
        font-size: .78rem !important;
        padding: 7px 10px !important;
        text-align: left !important;
        border-bottom: 2px solid #D50A0A !important;
        white-space: nowrap !important;
    }
    table.nfl-table tbody td {
        background: #17274A !important;
        color: #FFFFFF !important;
        padding: 6px 10px !important;
        border-bottom: 1px solid #2A3F5F !important;
        white-space: nowrap !important;
    }
    table.nfl-table tbody tr:nth-child(even) td { background: #1B2E54 !important; }
    table.nfl-table tbody tr:hover td { background: #24396A !important; }

    /* HR / separators — visible against navy */
    hr, [data-testid="stMarkdownContainer"] hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, #D50A0A 20%, #D50A0A 80%, transparent) !important;
        opacity: .55 !important;
        margin: 14px 0 !important;
    }

    /* Plotly chart — modebar icons + legend text white */
    .modebar-btn path, .modebar-btn--logo path { fill: #FFFFFF !important; }
    .modebar-container { background: transparent !important; }
    .plot-container .legendtext, .plot-container .legendtitletext {
        fill: #FFFFFF !important;
    }

    /* Selectbox — kill white dropdown */
    .stSelectbox > div > div,
    div[data-testid="stSelectbox"] > div > div {
        background-color: #131B2E !important;
        color: #F4F7FB !important;
        border-color: #2A3F5F !important;
    }
    .stSelectbox div[data-baseweb="select"] > div,
    div[data-baseweb="select"] > div {
        background-color: #131B2E !important;
        color: #F4F7FB !important;
        border-color: #2A3F5F !important;
    }
    div[data-baseweb="select"] input { color: #F4F7FB !important; }
    div[data-baseweb="select"] [class*="ValueContainer"] { color: #F4F7FB !important; }
    div[data-baseweb="select"] span { color: #F4F7FB !important; }
    /* Dropdown menu popover */
    div[data-baseweb="popover"] [role="listbox"],
    div[data-baseweb="menu"] ul {
        background-color: #131B2E !important;
        border: 1px solid #2A3F5F !important;
    }
    div[data-baseweb="popover"] [role="option"],
    div[data-baseweb="menu"] li {
        background-color: #131B2E !important;
        color: #F4F7FB !important;
    }
    div[data-baseweb="popover"] [role="option"]:hover,
    div[data-baseweb="menu"] li:hover {
        background-color: #1A2E52 !important;
        color: #FFFFFF !important;
    }
    div[data-baseweb="popover"] [aria-selected="true"] {
        background-color: #D50A0A !important;
        color: #FFFFFF !important;
    }

    /* Multiselect tags */
    span[data-baseweb="tag"] {
        background-color: #1A2E52 !important;
        color: #F4F7FB !important;
        border: 1px solid #2A3F5F !important;
    }

    /* Number / text input (non-sidebar) */
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextArea"] textarea {
        background-color: #131B2E !important;
        color: #F4F7FB !important;
        border: 1px solid #2A3F5F !important;
    }

    /* Slider labels + track */
    div[data-testid="stSlider"] label,
    div[data-testid="stSlider"] [data-testid="stTickBarMin"],
    div[data-testid="stSlider"] [data-testid="stTickBarMax"],
    div[data-testid="stSlider"] [data-baseweb="slider"] div {
        color: #F4F7FB !important;
    }

    /* Radio / checkbox labels */
    div[data-testid="stRadio"] label,
    div[data-testid="stCheckbox"] label,
    div[data-testid="stRadio"] label p,
    div[data-testid="stCheckbox"] label p {
        color: #F4F7FB !important;
    }

    /* Expander + info/warning boxes — kill white */
    div[data-testid="stExpander"],
    div[data-testid="stExpander"] > details,
    div[data-testid="stExpander"] summary {
        background-color: #131B2E !important;
        color: #F4F7FB !important;
        border: 1px solid #2A3F5F !important;
        border-radius: 10px !important;
    }
    div[data-testid="stExpander"] summary p { color: #F4F7FB !important; }
    div[data-testid="stAlert"],
    div[data-testid="stNotification"] {
        background-color: #131B2E !important;
        color: #F4F7FB !important;
        border: 1px solid #2A3F5F !important;
        border-left: 4px solid #D50A0A !important;
        border-radius: 10px !important;
    }
    div[data-testid="stAlert"] p, div[data-testid="stNotification"] p { color: #F4F7FB !important; }

    /* Metric widget */
    div[data-testid="stMetric"] {
        background-color: #131B2E !important;
        border: 1px solid #2A3F5F !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
    }
    div[data-testid="stMetricLabel"] p { color: #A8BDD9 !important; }
    div[data-testid="stMetricValue"] { color: #FFFFFF !important; }

    /* Generic Streamlit buttons (non-sidebar) */
    .stButton > button {
        background: linear-gradient(135deg, #1A2540 0%, #0F1A2E 100%) !important;
        color: #F4F7FB !important;
        border: 1px solid #2A3F5F !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        border-color: #D50A0A !important;
        background: linear-gradient(135deg, #1E2E50 0%, #152238 100%) !important;
        color: #FFFFFF !important;
    }

    /* Tabs */
    div[data-testid="stTabs"] button[role="tab"] {
        color: #A8BDD9 !important;
        background: transparent !important;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #FFFFFF !important;
        border-bottom: 2px solid #D50A0A !important;
    }

    /* Plotly chart background already transparent — make modebar readable */
    .modebar { background: transparent !important; }
    .modebar-btn path { fill: #A8BDD9 !important; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def get_artifacts(base_dir: str):
    return load_dashboard_data(Path(base_dir))


@st.cache_data
def load_optional_game_logs(base_dir: str) -> pd.DataFrame:
    base = Path(base_dir)
    candidates = [
        base / "output" / "player_game_logs.csv",
        base / "output" / "player_weekly.csv",
        base / "output" / "weekly_stats.csv",
        base / "player_game_logs.csv",
        base / "player_weekly.csv",
        base / "weekly_stats.csv",
    ]
    for path in candidates:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:
                return pd.DataFrame()
    return pd.DataFrame()


artifacts = get_artifacts(str(BASE_DIR))
df = artifacts["scored_players"].copy()
contract_history = artifacts["contract_history"].copy()
team_summary = artifacts["team_summary"].copy()
team_available = artifacts["team_available"]
game_logs = load_optional_game_logs(str(BASE_DIR))

VERDICT_COLOR = {"Undervalued": "#2ecc8a", "Fair": "#5b8ef0", "Overvalued": "#f05c5c"}

for key, value in {
    "selected_player_name": None,
    "selected_team_name": None,
    "active_page": PAGES[0],
    "pending_page": None,
    "last_query_text": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

if st.session_state.get("pending_page"):
    st.session_state["active_page"] = st.session_state["pending_page"]
    st.session_state["pending_page"] = None


def get_badge(gap: float) -> str:
    if gap > 2:
        return '<span class="badge-sign">✅ UNDER MARKET</span>'
    if gap < -2:
        return '<span class="badge-avoid">❌ OVER MARKET</span>'
    return '<span class="badge-monitor">👀 CLOSE TO MARKET</span>'


def safe_int(v) -> str:
    try:
        if pd.isna(v):
            return "—"
        return f"{int(round(float(v))):,}"
    except Exception:
        return "—"


def render_dark_table(dataframe: pd.DataFrame) -> None:
    """Render a pandas DataFrame as an HTML table so our dark NFL CSS applies.
    Replaces st.dataframe (which uses a glide canvas that ignores CSS)."""
    if dataframe is None or dataframe.empty:
        st.info("No rows to display.")
        return
    styler = (
        dataframe.style
        .hide(axis="index")
        .set_table_attributes('class="nfl-table"')
        .format(precision=2, na_rep="—")
    )
    html = f'<div class="nfl-table-wrap">{styler.to_html()}</div>'
    st.markdown(html, unsafe_allow_html=True)


def safe_float(v, digits: int = 1, pct: bool = False) -> str:
    try:
        if pd.isna(v):
            return "—"
        num = float(v)
        return f"{num:.{digits}f}%" if pct else f"{num:.{digits}f}"
    except Exception:
        return "—"


def apply_filters(data: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    out = data.copy()
    if filters.get("position"):
        out = out[out["position"].astype(str).str.upper() == str(filters["position"]).upper()]
    if filters.get("team") and team_available:
        out = out[out["team"].astype(str).str.lower() == str(filters["team"]).lower()]
    if filters.get("objective") == "undervalued":
        out = out[out["value_gap"] >= 0]
    elif filters.get("objective") == "overvalued":
        out = out[out["value_gap"] <= 0]
    if filters.get("max_salary") is not None:
        out = out[out["actual_apy"] <= float(filters["max_salary"]) / 1_000_000]
    if filters.get("season") is not None:
        out = out[out["season"] == filters["season"]]
    if filters.get("min_games") is not None:
        out = out[out["games_played"] >= filters["min_games"]]
    return out.sort_values(["value_gap", "predicted_apy"], ascending=[False, False]).reset_index(drop=True)


def _player_metrics_text(row: pd.Series) -> str:
    pos = str(row.get("position", "")).upper()
    if pos == "QB":
        return f"<b>Pass Yds:</b> {safe_int(row.get('pass_yards'))} · <b>Pass TDs:</b> {safe_int(row.get('pass_tds'))} · <b>Comp%:</b> {safe_float(row.get('completion_pct'), 1, pct=True)} · <b>YPA:</b> {safe_float(row.get('yards_per_attempt'))}"
    if pos == "RB":
        return f"<b>Rush Yds:</b> {safe_int(row.get('rush_yards'))} · <b>Rush TDs:</b> {safe_int(row.get('rush_tds'))} · <b>YPC:</b> {safe_float(row.get('yards_per_carry'))} · <b>Explosive Run%:</b> {safe_float(row.get('explosive_run_rate'), 1, pct=True)}"
    return f"<b>Rec Yds:</b> {safe_int(row.get('rec_yards'))} · <b>Rec TDs:</b> {safe_int(row.get('rec_tds'))} · <b>Catch%:</b> {safe_float(row.get('catch_rate'), 1, pct=True)} · <b>Yards/Rec:</b> {safe_float(row.get('yards_per_rec'))}"


def render_player_card(row: pd.Series) -> None:
    badge = get_badge(float(row.get("value_gap", 0) or 0))
    initials = "".join([x[0] for x in str(row.get("player_name", "NA")).split()[:2]]) or "NA"
    team_text = row.get("team") if pd.notna(row.get("team")) else "Team unavailable"
    meta = f"{row.get('position', '—')} · {team_text} · Season {safe_int(row.get('season'))} · {safe_int(row.get('games_played'))} games"
    headshot_url = row.get("headshot_url")
    headshot_html = (
        f'<img class="headshot-img" src="{headshot_url}" alt="{row.get("player_name", "Player")}"/>'
        if pd.notna(headshot_url) and str(headshot_url).strip()
        else f'<div class="headshot-fallback">{initials}</div>'
    )
    html = f"""
    <div class="player-card">
      <div style="display:flex; gap:18px; align-items:flex-start; justify-content:space-between;">
        <div style="display:flex; gap:16px; align-items:center;">
          {headshot_html}
          <div>
            <div class="name">{row.get('player_name', 'Unknown')}</div>
            <div class="meta">{meta}</div>
            <div style="margin-top:8px;">{badge}</div>
            <div class="metrics-line">{_player_metrics_text(row)}</div>
          </div>
        </div>
      </div>
      <div class="stat-row">
        <div class="stat-box"><div class="stat-label">Current APY</div><div class="stat-value">${safe_float(row.get('current_apy'))}M</div></div>
        <div class="stat-box"><div class="stat-label">Actual Next APY</div><div class="stat-value">${safe_float(row.get('actual_apy'))}M</div></div>
        <div class="stat-box"><div class="stat-label">Predicted Next APY</div><div class="stat-value">${safe_float(row.get('predicted_apy'))}M</div></div>
        <div class="stat-box"><div class="stat-label">Guaranteed</div><div class="stat-value">${safe_float(row.get('curr_guaranteed'))}M</div></div>
        <div class="stat-box"><div class="stat-label">Contract Length</div><div class="stat-value">{safe_float(row.get('curr_years'))}</div></div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def make_prediction_scatter(data: pd.DataFrame):
    fig = px.scatter(
        data, x="actual_apy", y="predicted_apy", color="verdict",
        color_discrete_map=VERDICT_COLOR, hover_name="player_name",
        hover_data={"position": True, "season": True, "games_played": True, "current_apy": ':.1f'},
        labels={"actual_apy": "Actual Next APY ($M)", "predicted_apy": "Predicted Next APY ($M)"},
        opacity=0.78,
    )
    if len(data) > 0:
        max_val = float(max(data["actual_apy"].max(), data["predicted_apy"].max())) + 3
        fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color="#667085", dash="dot"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#C9D1E0")
    return fig


def make_position_gap_chart(data: pd.DataFrame):
    tmp = data.groupby("position", as_index=False)["value_gap"].mean().sort_values("value_gap", ascending=False)
    fig = px.bar(tmp, x="position", y="value_gap", text_auto=".1f", labels={"value_gap": "Avg Prediction Gap ($M)"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#C9D1E0")
    return fig


def make_position_accuracy_box(data: pd.DataFrame):
    fig = px.box(data, x="position", y="contract_accuracy_abs", points="outliers", labels={"contract_accuracy_abs": "Absolute APY Error ($M)"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#C9D1E0")
    return fig


def make_team_gap_chart(summary: pd.DataFrame):
    fig = px.bar(summary.sort_values("avg_gap", ascending=False), x="team", y="avg_gap", hover_data=["players", "n_undervalued"], labels={"avg_gap": "Avg Prediction Gap ($M)"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#C9D1E0", xaxis_tickangle=-30)
    return fig


def make_team_metric_chart(data: pd.DataFrame, team: str, metric_col: str, metric_label: str):
    roster = data[data["team"] == team].copy()
    roster["contract_total"] = pd.to_numeric(roster.get("current_apy"), errors="coerce").fillna(0) * pd.to_numeric(roster.get("curr_years"), errors="coerce").fillna(0)
    roster[metric_col] = pd.to_numeric(roster.get(metric_col), errors="coerce")
    roster = roster.sort_values(metric_col, ascending=False).head(15)
    fig = px.bar(roster, x="player_name", y=metric_col, labels={metric_col: metric_label, "player_name": ""}, title=f"{team} roster by {metric_label}")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#C9D1E0", xaxis_tickangle=-35)
    return fig


def make_expiring_contracts_chart(data: pd.DataFrame, team: str, metric_col: str, metric_label: str):
    roster = data[data["team"] == team].copy()
    roster["contract_total"] = pd.to_numeric(roster.get("current_apy"), errors="coerce").fillna(0) * pd.to_numeric(roster.get("curr_years"), errors="coerce").fillna(0)
    roster[metric_col] = pd.to_numeric(roster.get(metric_col), errors="coerce")
    expiring = roster.sort_values("season", ascending=False).copy()
    if "curr_years" in expiring.columns:
        expiring = expiring[pd.to_numeric(expiring["curr_years"], errors="coerce").fillna(0) <= 2]
    expiring = expiring.sort_values("predicted_apy", ascending=False).head(12)
    if expiring.empty:
        expiring = roster.sort_values("predicted_apy", ascending=False).head(12)
    fig = px.bar(expiring, x="player_name", y=[metric_col, "predicted_apy"], barmode="group", labels={"value": "$M", "variable": ""}, title=f"{team} expiring contracts vs predicted next contract")
    fig.for_each_trace(lambda t: t.update(name=metric_label if t.name == metric_col else "Predicted Next APY"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#C9D1E0", xaxis_tickangle=-35)
    return fig


def make_player_radar(row: pd.Series):
    pos = str(row.get("position", "")).upper()
    if pos == "QB":
        axes = ["Pass Yards", "Pass TDs", "Comp %", "YPA", "EPA/play", "Games"]
        vals = [row.get("pass_yards_pct", 0), row.get("pass_tds_pct", 0), row.get("completion_pct_pct", 0), row.get("yards_per_attempt_pct", 0), row.get("pass_epa_per_play_pct", 0), row.get("games_played_pct", 0)]
    elif pos == "RB":
        axes = ["Rush Yards", "Rush TDs", "YPC", "Rush EPA", "Explosive Run", "Games"]
        vals = [row.get("rush_yards_pct", 0), row.get("rush_tds_pct", 0), row.get("yards_per_carry_pct", 0), row.get("rush_epa_per_play_pct", 0), row.get("explosive_run_rate_pct", 0), row.get("games_played_pct", 0)]
    else:
        axes = ["Rec Yards", "Rec TDs", "Catch Rate", "Yards/Rec", "Rec EPA", "Games"]
        vals = [row.get("rec_yards_pct", 0), row.get("rec_tds_pct", 0), row.get("catch_rate_pct", 0), row.get("yards_per_rec_pct", 0), row.get("rec_epa_per_play_pct", 0), row.get("games_played_pct", 0)]
    vals = vals + [vals[0]]
    axes = axes + [axes[0]]
    fig = go.Figure(go.Scatterpolar(r=vals, theta=axes, fill="toself", name=str(row.get("player_name", "Player"))))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), paper_bgcolor="rgba(0,0,0,0)", font_color="#C9D1E0", showlegend=False)
    return fig


def make_contract_timeline(player_row: pd.Series):
    player = player_row["player"]
    hist = contract_history[contract_history["player"] == player].copy()
    fig = go.Figure()
    if not hist.empty:
        fig.add_trace(go.Scatter(x=hist["year_signed"], y=hist["apy"], mode="lines+markers", name="Historical APY"))
    next_year = player_row.get("next_year_signed") if pd.notna(player_row.get("next_year_signed")) else (player_row.get("season", 2024) + 1)
    fig.add_trace(go.Scatter(x=[next_year], y=[player_row.get("actual_apy")], mode="markers", marker=dict(size=12), name="Actual next APY"))
    fig.add_trace(go.Scatter(x=[next_year], y=[player_row.get("predicted_apy")], mode="markers", marker=dict(size=12, symbol="diamond"), name="Predicted next APY"))
    fig.update_layout(title="Contract timeline", xaxis_title="Year signed", yaxis_title="APY ($M)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#C9D1E0")
    return fig


def _normalize_name(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()


def make_recent_games_chart(player_row: pd.Series) -> Optional[go.Figure]:
    if game_logs.empty:
        return None
    g = game_logs.copy()
    name_col = next((c for c in ["player_name", "player", "display_name", "name"] if c in g.columns), None)
    if name_col is None:
        return None
    g["player_norm"] = _normalize_name(g[name_col])
    target = _normalize_name(pd.Series([player_row.get("player_name")])).iloc[0]
    g = g[g["player_norm"] == target].copy()
    if g.empty:
        return None
    time_col = next((c for c in ["week", "game_date", "date"] if c in g.columns), None)
    if time_col is None:
        g["game_order"] = range(1, len(g) + 1)
        time_col = "game_order"
    g = g.sort_values(time_col).tail(5)
    pos = str(player_row.get("position", "")).upper()
    if pos == "QB":
        metric_cols = [("pass_yards", "Pass Yds"), ("pass_tds", "Pass TDs")]
    elif pos == "RB":
        metric_cols = [("rush_yards", "Rush Yds"), ("rush_tds", "Rush TDs")]
    else:
        metric_cols = [("rec_yards", "Rec Yds"), ("rec_tds", "Rec TDs")]
    usable = [(c, label) for c, label in metric_cols if c in g.columns]
    if not usable:
        return None
    plot_df = g[[time_col] + [c for c, _ in usable]].copy()
    rename = {c: label for c, label in usable}
    plot_df = plot_df.rename(columns=rename).melt(id_vars=[time_col], var_name="Metric", value_name="Value")
    fig = px.line(plot_df, x=time_col, y="Value", color="Metric", markers=True, title=f"{player_row.get('player_name')} recent game trend")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#C9D1E0")
    return fig


def make_player_contract_bar(row: pd.Series):
    labels = ["Current APY", "Actual Next APY", "Predicted Next APY"]
    values = [float(row.get("current_apy") or 0), float(row.get("actual_apy") or 0), float(row.get("predicted_apy") or 0)]
    fig = px.bar(x=labels, y=values, labels={"x": "", "y": "$M"}, title=f"{row.get('player_name')} contract snapshot")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#C9D1E0")
    return fig


def make_compare_contract_chart(a: pd.Series, b: pd.Series):
    cmp_df = pd.DataFrame({
        "player": [a["player_name"], b["player_name"]],
        "actual_apy": [a["actual_apy"], b["actual_apy"]],
        "predicted_apy": [a["predicted_apy"], b["predicted_apy"]],
        "current_apy": [a["current_apy"], b["current_apy"]],
    })
    fig = px.bar(cmp_df, x="player", y=["current_apy", "actual_apy", "predicted_apy"], barmode="group", labels={"value": "$M", "variable": ""})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#C9D1E0")
    return fig


def choose_smart_chart(plan: Dict, smart_results: pd.DataFrame, focus_player_row: Optional[pd.Series], focus_compare_row: Optional[pd.Series]):
    chart_name = plan.get("smart_chart") or "value_scatter"
    if chart_name == "position_gap" and not smart_results.empty:
        return make_position_gap_chart(smart_results), chart_name
    if chart_name == "team_gap" and team_available and not team_summary.empty:
        return make_team_gap_chart(team_summary), chart_name
    if chart_name == "player_contract_bar" and focus_player_row is not None:
        return make_player_contract_bar(focus_player_row), chart_name
    if chart_name == "team_expiring" and team_available and plan.get("focus_team"):
        return make_expiring_contracts_chart(df, plan["focus_team"], "current_apy", "Current APY"), chart_name
    if chart_name == "compare_contracts" and focus_player_row is not None and focus_compare_row is not None:
        return make_compare_contract_chart(focus_player_row, focus_compare_row), chart_name
    return make_prediction_scatter(smart_results), "value_scatter"


def get_focus_player_row(display_data: pd.DataFrame, full_data: pd.DataFrame, focus_name: Optional[str]) -> Optional[pd.Series]:
    candidate = focus_name or st.session_state.get("selected_player_name")
    for source in (display_data, full_data, df):
        if candidate and not source.empty and candidate in source["player_name"].tolist():
            return source[source["player_name"] == candidate].iloc[0]
    if not display_data.empty:
        return display_data.iloc[0]
    if not full_data.empty:
        return full_data.iloc[0]
    return None


def get_similar_peers(base_df: pd.DataFrame, player_row: pd.Series, n: int = 8) -> pd.DataFrame:
    peers = base_df[(base_df["position"] == player_row["position"]) & (base_df["player_name"] != player_row["player_name"])].copy()
    if peers.empty:
        return peers
    peers["apy_distance"] = (pd.to_numeric(peers["current_apy"], errors="coerce") - float(player_row.get("current_apy") or 0)).abs()
    peers["contract_total"] = pd.to_numeric(peers["current_apy"], errors="coerce").fillna(0) * pd.to_numeric(peers["curr_years"], errors="coerce").fillna(0)
    return peers.sort_values(["apy_distance", "contract_accuracy_abs", "value_gap"], ascending=[True, True, False]).head(n)


def pick_default_compare_row(player_row: Optional[pd.Series], results_df: pd.DataFrame) -> Optional[pd.Series]:
    if player_row is None:
        return None
    peers = get_similar_peers(results_df if not results_df.empty else df, player_row, n=1)
    return peers.iloc[0] if not peers.empty else None


st.markdown(
    """
<div class="hero">
    <h1>GRIDIRONIQ</h1>
    <p>NFL Player Contract Valuation System — Powered by XGBoost + LLM Intelligence</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 🏈 GRIDIRONIQ")
    st.markdown("---")
    st.markdown("#### 🔍 Manager Query")
    st.markdown(
        '<div class="query-help"><b>Free-text query is enabled.</b><br>Try: <i>show undervalued WRs under $12M</i>, <i>show Chiefs roster</i>, <i>player profile for Joe Burrow</i>.</div>',
        unsafe_allow_html=True,
    )
    if "user_query_v12" not in st.session_state:
        st.session_state["user_query_v12"] = ""
    user_query = st.text_input("Ask anything about players, contracts, teams, or charts", value=st.session_state["user_query_v12"], placeholder="Try: show undervalued WRs under $12M")
    st.markdown('<p style="color:#FFFFFF; font-size:.82rem; margin:10px 0 4px 0;">⚡ Quick queries</p>', unsafe_allow_html=True)
    for label in ["Show undervalued QBs", "Compare budget WRs", "Plot WR value chart", "Show safe RBs under $5M", "Player profile for Joe Burrow"]:
        if st.button(label, use_container_width=True):
            st.session_state["user_query_v12"] = label
            st.rerun()

    st.markdown("---")
    st.markdown("#### ⚙️ Filters")
    positions = ["All"] + sorted(df["position"].dropna().astype(str).unique().tolist())
    seasons = ["All"] + sorted(df["season"].dropna().astype(int).unique().tolist(), reverse=True)
    teams = ["All"] + (sorted(df["team"].dropna().astype(str).unique().tolist()) if team_available else [])
    ui_position = st.selectbox("Position", positions, index=0)
    ui_team = st.selectbox("Team", teams, index=0, disabled=not team_available)
    ui_max_salary_m = st.slider("Max actual next APY ($M)", 0.0, float(max(10.0, df["actual_apy"].max())), float(min(20.0, df["actual_apy"].max())), 0.5)
    ui_min_games = st.slider("Min games", 0, 17, 8, 1)
    ui_season = st.selectbox("Season", seasons, index=0)
    ui_objective = st.selectbox("Objective", ["balanced", "undervalued", "overvalued"], index=0)
    top_k = st.slider("Show top", 3, 20, 8, 1)
    show_full_results = st.checkbox("Show full filtered results instead of shortlist", value=False)
    llm_planner_on = st.checkbox("Use LLM query planner when available", value=True)

plan = build_dashboard_plan(user_query=user_query, df=df, team_aliases=TEAM_ALIASES, use_llm=llm_planner_on)

normalized_query = " ".join(str(user_query).strip().lower().split())
if st.session_state.get("last_query_text") != normalized_query:
    if plan.get("focus_player"):
        st.session_state["selected_player_name"] = plan["focus_player"]
    if plan.get("focus_team"):
        st.session_state["selected_team_name"] = plan["focus_team"]
    st.session_state["last_query_text"] = normalized_query

    # Auto-navigate to the right tab based on query intent
    import re as _re
    _q = normalized_query
    _target_page = None
    if plan.get("shortlist_mode") == "single_player" and plan.get("focus_player"):
        _target_page = "🔍 Player Analysis"
    elif any(_re.search(p, _q) for p in [r"\bcompare\b", r"\bvs\b", r"\bversus\b"]):
        _target_page = "⚖️ Compare"
    elif any(_re.search(p, _q) for p in [r"\broster\b", r"\bteam\b"]) and plan.get("focus_team"):
        _target_page = "🏟️ Team Analysis"
    elif any(_re.search(p, _q) for p in [r"\bprofile\b", r"\banalyze\b", r"\bstats\b"]) and plan.get("focus_player"):
        _target_page = "🔍 Player Analysis"

    if _target_page and st.session_state.get("active_page") != _target_page:
        st.session_state["pending_page"] = _target_page
        st.rerun()
ui_filters = {
    "position": None if ui_position == "All" else ui_position,
    "team": None if ui_team == "All" else ui_team,
    "max_salary": ui_max_salary_m * 1_000_000,
    "min_games": ui_min_games,
    "season": None if ui_season == "All" else int(ui_season),
    "objective": ui_objective,
}
final_filters = merge_plan_with_ui_filters(plan, ui_filters, query_priority=True)
results = apply_filters(df, final_filters)

if plan.get("shortlist_mode") == "single_player" and plan.get("focus_player"):
    shortlist = df[df["player_name"] == plan["focus_player"]].head(1).copy()
elif plan.get("shortlist_mode") == "player_plus_similar" and plan.get("focus_player"):
    focus_candidates = df[df["player_name"] == plan["focus_player"]].head(1).copy()
    if not focus_candidates.empty:
        focus_row_for_shortlist = focus_candidates.iloc[0]
        peers = get_similar_peers(results if not results.empty else df, focus_row_for_shortlist, n=max(top_k - 1, 0))
        shortlist = pd.concat([focus_candidates, peers], ignore_index=True).drop_duplicates(subset=["player_name"]).head(top_k)
    else:
        shortlist = results.head(top_k).copy()
else:
    shortlist = results.head(top_k).copy()

display_df = results.copy() if show_full_results and plan.get("shortlist_mode") == "default" else shortlist.copy()
focus_player_row = get_focus_player_row(display_df, results if not results.empty else df, plan.get("focus_player"))
if focus_player_row is not None and pd.notna(focus_player_row.get("team")):
    focus_team = str(focus_player_row.get("team"))
else:
    focus_team = plan.get("focus_team")

if plan.get("focus_player") and focus_player_row is not None:
    st.session_state["selected_player_name"] = focus_player_row.get("player_name")
if plan.get("focus_team") and focus_team:
    st.session_state["selected_team_name"] = focus_team

focus_compare_row = pick_default_compare_row(focus_player_row, results if not results.empty else df)

kpis = st.columns(4)
for col, (label, value) in zip(
    kpis,
    [
        ("Showing", len(display_df)),
        ("Avg prediction gap", f"${display_df['value_gap'].mean():+.1f}M" if not display_df.empty else "—"),
        ("Undervalued", int((display_df["value_gap"] > 2).sum()) if not display_df.empty else 0),
        ("Overvalued", int((display_df["value_gap"] < -2).sum()) if not display_df.empty else 0),
    ],
):
    col.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)

st.markdown(
    f'<div class="count-banner"><b>Full dataset:</b> {len(df):,} rows &nbsp;&nbsp;|&nbsp;&nbsp; <b>Filtered results:</b> {len(results):,} rows &nbsp;&nbsp;|&nbsp;&nbsp; <b>Currently displayed:</b> {len(display_df):,} rows</div>',
    unsafe_allow_html=True,
)
page = st.radio("Navigation", PAGES, horizontal=True, label_visibility="collapsed", key="active_page")

if page == "📋 Shortlist":
    if display_df.empty:
        st.warning("No players matched the current query + filters.")
    else:
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.plotly_chart(make_prediction_scatter(display_df), use_container_width=True, key="shortlist_scatter_v12")
            st.markdown(f'<div class="ai-box"><div class="ai-title">Chart explanation</div>{explain_chart("value_scatter", display_df, plan)}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="ai-box"><div class="ai-title">Result summary</div>{explain_query_results(display_df, plan)}</div>', unsafe_allow_html=True)
        st.markdown("---")
        for idx, row in display_df.iterrows():
            render_player_card(row)
            if st.button(f"Open {row.get('player_name')} in Player Analysis", key=f"open_player_v12_{idx}_{row.get('player_name')}"):
                st.session_state["selected_player_name"] = row.get("player_name")
                if pd.notna(row.get("team")):
                    st.session_state["selected_team_name"] = row.get("team")
                st.session_state["pending_page"] = "🔍 Player Analysis"
                st.rerun()

elif page == "🔍 Player Analysis":
    if focus_player_row is None:
        st.info("No players available for profile analysis.")
    else:
        # Use full dataset for player selection so filters don't hide the selected player
        all_options = df["player_name"].tolist()
        default_player = st.session_state.get("selected_player_name") or focus_player_row["player_name"]
        player_name = st.selectbox("Select player", all_options, index=all_options.index(default_player) if default_player in all_options else 0)
        player_row = df[df["player_name"] == player_name].iloc[0]
        st.session_state["selected_player_name"] = player_name
        if pd.notna(player_row.get("team")):
            st.session_state["selected_team_name"] = player_row.get("team")

        render_player_card(player_row)
        st.markdown(f'<div class="ai-box"><div class="ai-title">Player explanation</div>{explain_player(player_row, plan)}</div>', unsafe_allow_html=True)
        top_left, top_right = st.columns(2)
        with top_left:
            st.plotly_chart(make_player_radar(player_row), use_container_width=True, key=f"radar_{player_name}_v12")
        with top_right:
            st.plotly_chart(make_contract_timeline(player_row), use_container_width=True, key=f"timeline_{player_name}_v12")

        recent_fig = make_recent_games_chart(player_row)
        st.subheader("Recent games")
        if recent_fig is None:
            st.info("No game-level file was detected, so recent-game trend is unavailable in this build.")
        else:
            st.plotly_chart(recent_fig, use_container_width=True, key=f"recent_games_{player_name}_v12")

        peers = get_similar_peers(df, player_row, n=8)
        st.subheader("Position peers by current APY")
        if peers.empty:
            st.info("No comparable peers found for this player.")
        else:
            peer_show = peers[["player_name", "team", "season", "current_apy", "curr_guaranteed", "curr_years", "contract_total", "actual_apy", "predicted_apy", "value_gap"]].rename(columns={
                "player_name": "Player", "team": "Team", "season": "Season", "current_apy": "Current APY", "curr_guaranteed": "Guaranteed", "curr_years": "Contract Length", "contract_total": "Contract Total", "actual_apy": "Actual Next APY", "predicted_apy": "Predicted Next APY", "value_gap": "Prediction Gap",
            })
            render_dark_table(peer_show)

elif page == "🏟️ Team Analysis":
    if not team_available:
        st.info("Team-level views are ready, but your current real files still need `output/player_team_map.csv`.")
        st.subheader("Fallback: model accuracy by position")
        st.plotly_chart(make_position_accuracy_box(df), use_container_width=True)
        st.markdown(f'<div class="ai-box"><div class="ai-title">Chart explanation</div>{explain_chart("accuracy_box", df, plan)}</div>', unsafe_allow_html=True)
    else:
        team_options = sorted(df["team"].dropna().unique())
        session_team = st.session_state.get("selected_team_name")
        default_team = session_team if session_team in team_options else (focus_team if focus_team in team_options else team_options[0])
        selected_team = st.selectbox("Select team", team_options, index=team_options.index(default_team), key="team_v12")
        st.session_state["selected_team_name"] = selected_team
        team_df = df[df["team"] == selected_team].copy()
        team_df["contract_total"] = pd.to_numeric(team_df.get("current_apy"), errors="coerce").fillna(0) * pd.to_numeric(team_df.get("curr_years"), errors="coerce").fillna(0)
        st.markdown(f'<div class="ai-box"><div class="ai-title">Team explanation</div>{explain_team(team_df, selected_team, plan)}</div>', unsafe_allow_html=True)

        # Roster detail FIRST (moved up per feedback)
        st.subheader(f"{selected_team} roster detail")
        metric_title_col, metric_control_col = st.columns([3.2, 1.2])
        with metric_title_col:
            st.markdown("##### Select metric")
        with metric_control_col:
            team_metric_label = st.selectbox(
                "Metric",
                list(TEAM_METRICS.keys()),
                index=0,
                key="team_metric_selector_v13",
                label_visibility="collapsed",
            )
        team_metric_col = TEAM_METRICS[team_metric_label]
        detail_cols = [c for c in ["player_name", "position", team_metric_col, "predicted_apy", "value_gap"] if c in team_df.columns]
        detail_df = team_df[detail_cols].rename(columns={team_metric_col: team_metric_label, "player_name": "Player", "position": "Pos", "predicted_apy": "Predicted Next APY", "value_gap": "Prediction Gap"})
        render_dark_table(detail_df)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                make_expiring_contracts_chart(df, selected_team, "current_apy", "Current APY"),
                use_container_width=True,
                key=f"team_expiring_{selected_team}_v13",
            )
            st.markdown(
                f'<div class="ai-box"><div class="ai-title">Expiring contracts explanation</div>{explain_chart("team_expiring", team_df, {"team": selected_team, "metric": "Current APY", **plan})}</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.plotly_chart(
                make_team_metric_chart(df, selected_team, team_metric_col, team_metric_label),
                use_container_width=True,
                key=f"team_metric_{selected_team}_v13",
            )
            st.markdown(
                f'<div class="ai-box"><div class="ai-title">Roster chart explanation</div>{explain_chart("roster", team_df, {"team": selected_team, "metric": team_metric_label, **plan})}</div>',
                unsafe_allow_html=True,
            )

        # League overview BELOW (moved down per feedback)
        st.subheader("League team overview")
        if not team_summary.empty:
            st.plotly_chart(make_team_gap_chart(team_summary), use_container_width=True, key="team_gap_chart_v12")
            render_dark_table(team_summary)

elif page == "⚖️ Compare":
    compare_pool = shortlist if len(shortlist) >= 2 else (results if not results.empty else df).head(10)
    if compare_pool.empty:
        st.info("Need players in scope to compare.")
    else:
        names = compare_pool["player_name"].tolist()
        default_a = focus_player_row["player_name"] if focus_player_row is not None and focus_player_row["player_name"] in names else names[0]
        default_b = focus_compare_row["player_name"] if focus_compare_row is not None and focus_compare_row["player_name"] in names else (names[1] if len(names) > 1 else names[0])
        col1, col2 = st.columns(2)
        with col1:
            a_name = st.selectbox("Player A", names, index=names.index(default_a), key="cmp_a_v12")
        with col2:
            b_name = st.selectbox("Player B", names, index=names.index(default_b), key="cmp_b_v12")
        a = compare_pool[compare_pool["player_name"] == a_name].iloc[0]
        b = compare_pool[compare_pool["player_name"] == b_name].iloc[0]
        c1, c2 = st.columns(2)
        with c1:
            render_player_card(a)
        with c2:
            render_player_card(b)
        st.plotly_chart(make_compare_contract_chart(a, b), use_container_width=True, key="compare_contract_chart_v12")
        st.markdown(f'<div class="ai-box"><div class="ai-title">Comparison explanation</div>{compare_players(a, b, plan)}</div>', unsafe_allow_html=True)

elif page == "🧠 Smart Query":
    smart_results = results.copy()
    if plan.get("shortlist_mode") == "single_player" and focus_player_row is not None:
        smart_results = focus_player_row.to_frame().T
    fig, chosen_name = choose_smart_chart(plan, smart_results, focus_player_row, focus_compare_row)
    st.plotly_chart(fig, use_container_width=True, key="smart_query_chart_v13")
    if chosen_name == "team_gap" and not team_summary.empty:
        chart_data = team_summary
    elif chosen_name == "player_contract_bar" and focus_player_row is not None:
        chart_data = focus_player_row.to_frame().T
    else:
        chart_data = smart_results
    st.markdown(
        f'<div class="ai-box"><div class="ai-title">Manager insights</div>{explain_smart_query_insights(plan, smart_results, focus_player_row, focus_compare_row, user_query=user_query, chart_name=chosen_name)}</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Smart-query table")
    smart_cols = [c for c in ["player_name", "team", "position", "season", "current_apy", "actual_apy", "predicted_apy", "value_gap", "games_played"] if c in smart_results.columns]
    render_dark_table(
        smart_results[smart_cols].rename(columns={"player_name": "Player", "team": "Team", "position": "Pos", "season": "Season", "current_apy": "Current APY", "actual_apy": "Actual Next APY", "predicted_apy": "Predicted Next APY", "value_gap": "Prediction Gap", "games_played": "Games"})
    )

else:
    st.info("Select a tab above to explore.")

st.markdown(
    """
<div class="footer">
    🏈 <b>GRIDIRONIQ</b> — NFL Contract Valuation System by Team 8<br>
    Big Data & AI · Spring 2026 · University of Minnesota
</div>
""",
    unsafe_allow_html=True,
)
