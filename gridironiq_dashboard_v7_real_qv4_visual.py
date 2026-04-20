from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_adapter_v4_real import load_dashboard_data
from explainer_v3 import compare_players, explain_chart, explain_player, explain_query_results, explain_team
from query_parser_v4 import TEAM_ALIASES, merge_query_with_ui_filters, parse_query

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="GridironIQ Real-Data Dashboard", page_icon="🏈", layout="wide")

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Source+Sans+3:wght@400;600;700&display=swap');

    .stApp { background-color: #0B1120; color: #E8E8E8; }
    section[data-testid="stSidebar"] { background-color: #0D1525; border-right: 1px solid #1E2A3A; }

    .hero {
        background: linear-gradient(135deg, #013369 0%, #141C2F 55%, #D50A0A 100%);
        border-radius: 18px; padding: 28px 36px; margin-bottom: 18px;
    }
    .hero h1 { font-family: 'Bebas Neue', sans-serif; color: white; font-size: 3rem; letter-spacing: 5px; margin: 0; }
    .hero p { color: #D8E2EF; margin: 6px 0 0 0; font-size: 1.05rem; }

    .player-card { background: linear-gradient(160deg, #131B2E 0%, #1A2540 100%); border: 1px solid #2A3550; border-radius: 16px; padding: 18px 22px; margin-bottom: 12px; }
    .player-card .name { font-family: 'Bebas Neue', sans-serif; color: white; font-size: 1.8rem; letter-spacing: 2px; }
    .player-card .meta { color: #A8BDD9; margin-top: 2px; }
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

    .query-help {
        background: #12213A;
        border: 1px solid #2D4670;
        border-radius: 12px;
        padding: 10px 12px;
        color: #D8E2EF;
        margin-bottom: 12px;
        font-size: 0.9rem;
    }

    /* Make inputs more visible */
    div[data-testid="stTextInput"] input {
        background: #F7FAFF !important;
        color: #0B1120 !important;
        border: 2px solid #6CB6FF !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    div[data-testid="stTextInput"] label p {
        color: #E8F1FF !important;
        font-weight: 700 !important;
    }
    div[data-testid="stTextInput"] input::placeholder {
        color: #5A7090 !important;
        opacity: 1 !important;
    }
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
    div[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
        background: #F7FAFF !important;
        color: #0B1120 !important;
        border-radius: 10px !important;
    }
    div[data-testid="stSlider"] * {
        color: #E8F1FF !important;
    }
    .count-banner {
        background: linear-gradient(160deg, #101E34 0%, #17294A 100%);
        border: 1px solid #2A3F5F;
        border-radius: 12px;
        padding: 12px 16px;
        color: #DDE8F6;
        margin: 8px 0 14px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def get_artifacts(base_dir: str):
    return load_dashboard_data(Path(base_dir))


artifacts = get_artifacts(str(BASE_DIR))
df = artifacts["scored_players"].copy()
contract_history = artifacts["contract_history"].copy()
team_summary = artifacts["team_summary"].copy()
team_available = artifacts["team_available"]

VERDICT_COLOR = {"Undervalued": "#2ecc8a", "Fair": "#5b8ef0", "Overvalued": "#f05c5c"}


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


def safe_float(v, digits: int = 1) -> str:
    try:
        if pd.isna(v):
            return "—"
        return f"{float(v):.{digits}f}"
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
          </div>
        </div>
      </div>
      <div class="stat-row">
        <div class="stat-box"><div class="stat-label">Current APY</div><div class="stat-value">${safe_float(row.get('current_apy'))}M</div></div>
        <div class="stat-box"><div class="stat-label">Actual Next APY</div><div class="stat-value">${safe_float(row.get('actual_apy'))}M</div></div>
        <div class="stat-box"><div class="stat-label">Predicted Next APY</div><div class="stat-value">${safe_float(row.get('predicted_apy'))}M</div></div>
        <div class="stat-box"><div class="stat-label">Prediction Gap</div><div class="stat-value">${safe_float(row.get('value_gap'))}M</div></div>
        <div class="stat-box"><div class="stat-label">Confidence</div><div class="stat-value">{safe_float(row.get('confidence_pct'), 0)}%</div></div>
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


def make_team_mix_chart(data: pd.DataFrame, team: str):
    tmp = data[data["team"] == team]["position"].value_counts().reset_index()
    tmp.columns = ["position", "count"]
    fig = px.pie(tmp, names="position", values="count", hole=0.4)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#C9D1E0")
    return fig


def make_team_roster_chart(data: pd.DataFrame, team: str):
    roster = data[data["team"] == team].sort_values("actual_apy", ascending=False).head(15)
    fig = px.bar(roster, x="player_name", y=["actual_apy", "predicted_apy"], barmode="group", labels={"value": "$M", "variable": ""})
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


def choose_chart(parsed: Dict, data: pd.DataFrame):
    chart_name = parsed.get("target_chart")
    if chart_name == "position_gap":
        return make_position_gap_chart(data), chart_name
    if chart_name == "team_gap" and team_available and not team_summary.empty:
        return make_team_gap_chart(team_summary), chart_name
    return make_prediction_scatter(data), "value_scatter"


st.markdown(
    """
<div class="hero">
    <h1>GRIDIRONIQ</h1>
    <p>Refreshed real-data dashboard aligned to your current files and optional nflreadpy team/headshot outputs.</p>
</div>
""",
    unsafe_allow_html=True,
)
st.caption(artifacts["data_note"])
st.caption("Files used")
st.json(artifacts["data_source"], expanded=False)
st.caption(artifacts["unused_files_note"])

with st.sidebar:
    st.markdown("### 🏈 GRIDIRONIQ")
    st.markdown("---")
    st.markdown("#### 🔍 Manager Query")
    st.markdown(
        '<div class="query-help"><b>Free-text query is enabled.</b><br>Try: <i>show undervalued WRs under $12M</i>, <i>show Chiefs roster</i>, <i>player profile for Joe Burrow</i>.</div>',
        unsafe_allow_html=True,
    )
    if "user_query_v7" not in st.session_state:
        st.session_state["user_query_v7"] = "Show undervalued RB under $10M"
    user_query = st.text_input(
        "Ask anything about players, contracts, teams, or charts",
        value=st.session_state["user_query_v7"],
        placeholder="e.g. show undervalued WRs under $12M",
    )
    for label in [
        "Show undervalued QBs",
        "Compare budget WRs",
        "Plot WR value chart",
        "Show safe RBs under $15M",
        "Player profile for Joe Burrow",
    ]:
        if st.button(label, use_container_width=True):
            st.session_state["user_query_v7"] = label
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

parsed = parse_query(
    user_query,
    team_aliases=TEAM_ALIASES,
    available_player_names=df["player_name"].tolist(),
)

ui_filters = {
    "position": None if ui_position == "All" else ui_position,
    "team": None if ui_team == "All" else ui_team,
    "max_salary": ui_max_salary_m * 1_000_000,
    "min_games": ui_min_games,
    "season": None if ui_season == "All" else int(ui_season),
    "objective": ui_objective,
}
final_filters = merge_query_with_ui_filters(parsed, ui_filters, query_priority=True)
results = apply_filters(df, final_filters)
shortlist = results.head(top_k).copy()
display_df = results.copy() if show_full_results else shortlist.copy()

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
    f'<div class="count-banner"><b>Full dataset:</b> {len(df):,} rows &nbsp;&nbsp;|&nbsp;&nbsp; '
    f'<b>Filtered results:</b> {len(results):,} rows &nbsp;&nbsp;|&nbsp;&nbsp; '
    f'<b>Currently displayed:</b> {len(display_df):,} rows '
    f'({"full filtered results" if show_full_results else f"top {top_k}"})</div>',
    unsafe_allow_html=True,
)

st.markdown(f'<div class="ai-box"><div class="ai-title">Result summary</div>{explain_query_results(display_df, parsed)}</div>', unsafe_allow_html=True)

shortlist_tab, player_tab, team_tab, compare_tab, smart_tab, debug_tab = st.tabs([
    "📋 Shortlist", "🔍 Player Analysis", "🏟️ Team Analysis", "⚖️ Compare", "🧠 Smart Query", "🛠️ Query Debug"
])

with shortlist_tab:
    if display_df.empty:
        st.warning("No players matched the current query + filters.")
    else:
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.plotly_chart(make_prediction_scatter(shortlist), use_container_width=True, key="shortlist_scatter")
            st.markdown(
                f'<div class="ai-box"><div class="ai-title">Chart explanation</div>{explain_chart("value_scatter", display_df, final_filters)}</div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.plotly_chart(make_position_gap_chart(shortlist), use_container_width=True, key="shortlist_position_gap")
            st.markdown(
                f'<div class="ai-box"><div class="ai-title">Position chart explanation</div>{explain_chart("position_gap", display_df, final_filters)}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("---")
        for _, row in display_df.iterrows():
            render_player_card(row)

with player_tab:
    if display_df.empty:
        st.info("No players available for profile analysis.")
    else:
        options = display_df["player_name"].tolist()
        default_player = parsed.get("player_name") if parsed.get("player_name") in options else options[0]
        player_name = st.selectbox("Select player", options, index=options.index(default_player) if default_player in options else 0)
        player_row = display_df[display_df["player_name"] == player_name].iloc[0]
        render_player_card(player_row)
        st.markdown(f'<div class="ai-box"><div class="ai-title">Player explanation</div>{explain_player(player_row)}</div>', unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            st.plotly_chart(make_player_radar(player_row), use_container_width=True, key=f"radar_{player_name}")
        with right:
            st.plotly_chart(make_contract_timeline(player_row), use_container_width=True, key=f"timeline_{player_name}")
        peers = df[(df["position"] == player_row["position"]) & (df["player_name"] != player_name)].sort_values("contract_accuracy_abs").head(8)
        st.subheader("Position peers")
        st.dataframe(
            peers[["player_name", "season", "current_apy", "actual_apy", "predicted_apy", "value_gap"]].rename(columns={
                "player_name": "Player", "season": "Season", "current_apy": "Current APY", "actual_apy": "Actual Next APY",
                "predicted_apy": "Predicted Next APY", "value_gap": "Prediction Gap",
            }),
            use_container_width=True, hide_index=True,
        )

with team_tab:
    if not team_available:
        st.info("Team-level views are ready, but your current real files still need `output/player_team_map.csv`. Run `python build_team_overview_dataset.py --season 2024` after installing nflreadpy.")
        st.subheader("Fallback: model accuracy by position")
        st.plotly_chart(make_position_accuracy_box(df), use_container_width=True)
        st.markdown(f'<div class="ai-box"><div class="ai-title">Chart explanation</div>{explain_chart("accuracy_box", df, final_filters)}</div>', unsafe_allow_html=True)
    else:
        selected_team = st.selectbox("Select team", sorted(df["team"].dropna().unique()), key="team_v7")
        team_df = df[df["team"] == selected_team].copy()
        st.markdown(f'<div class="ai-box"><div class="ai-title">Team explanation</div>{explain_team(team_df, selected_team)}</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(make_team_roster_chart(df, selected_team), use_container_width=True, key=f"team_roster_{selected_team}")
            st.markdown(f'<div class="ai-box"><div class="ai-title">Roster chart explanation</div>{explain_chart("roster", team_df, {"team": selected_team})}</div>', unsafe_allow_html=True)
        with c2:
            st.plotly_chart(make_team_mix_chart(df, selected_team), use_container_width=True, key=f"team_mix_{selected_team}")
            st.markdown(f'<div class="ai-box"><div class="ai-title">Mix chart explanation</div>{explain_chart("team_mix", team_df, {"team": selected_team})}</div>', unsafe_allow_html=True)

        if not team_summary.empty:
            st.subheader("League team overview")
            st.plotly_chart(make_team_gap_chart(team_summary), use_container_width=True, key="team_gap_chart")
            st.dataframe(team_summary, use_container_width=True, hide_index=True)

        st.subheader(f"{selected_team} roster detail")
        st.dataframe(team_df[["player_name", "position", "actual_apy", "predicted_apy", "value_gap"]], use_container_width=True, hide_index=True)

with compare_tab:
    compare_pool = shortlist if len(shortlist) >= 2 else display_df.head(2)
    if len(compare_pool) < 2:
        st.info("Need at least two players in the displayed results to compare.")
    else:
        names = compare_pool["player_name"].tolist()
        col1, col2 = st.columns(2)
        with col1:
            a_name = st.selectbox("Player A", names, index=0, key="cmp_a_v7")
        with col2:
            b_name = st.selectbox("Player B", names, index=1 if len(names) > 1 else 0, key="cmp_b_v7")
        a = compare_pool[compare_pool["player_name"] == a_name].iloc[0]
        b = compare_pool[compare_pool["player_name"] == b_name].iloc[0]
        c1, c2 = st.columns(2)
        with c1:
            render_player_card(a)
        with c2:
            render_player_card(b)
        st.markdown(f'<div class="ai-box"><div class="ai-title">Comparison explanation</div>{compare_players(a, b)}</div>', unsafe_allow_html=True)

with smart_tab:
    smart_results = results.copy()
    fig, chosen_name = choose_chart(parsed, smart_results)
    st.plotly_chart(fig, use_container_width=True, key="smart_query_chart")
    chart_data = team_summary if chosen_name == "team_gap" and not team_summary.empty else smart_results
    st.markdown(
        f'<div class="ai-box"><div class="ai-title">Chart explanation</div>{explain_chart(chosen_name, chart_data, final_filters)}</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Smart-query table")
    st.dataframe(
        smart_results[["player_name", "position", "season", "current_apy", "actual_apy", "predicted_apy", "value_gap", "games_played"]].rename(columns={
            "player_name": "Player", "position": "Pos", "season": "Season", "current_apy": "Current APY",
            "actual_apy": "Actual Next APY", "predicted_apy": "Predicted Next APY", "value_gap": "Prediction Gap", "games_played": "Games",
        }),
        use_container_width=True, hide_index=True,
    )

with debug_tab:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Parsed Query")
        st.json(parsed)
    with c2:
        st.markdown("#### Final Filters")
        st.json(final_filters)
    st.markdown("#### Dataset counts")
    st.json({
        "full_dataset_rows": int(len(df)),
        "filtered_rows": int(len(results)),
        "displayed_rows": int(len(display_df)),
        "unique_players_full": int(df["player_name"].nunique()),
    })
    st.markdown("#### Where to swap in future files")
    st.markdown(
        """
- **Core scored output:** replace `output/dashboard_model_results.csv`
- **Season-level player stats + current contract fields:** replace `output/dataset(3).csv`
- **Contract history timeline:** replace `output/contracts_clean(3).csv`
- **Team + headshot layer:** generate `output/player_team_map.csv`, `output/team_roster_detail.csv`, `output/team_summary.csv`
"""
    )

st.markdown(
    """
<div class="footer">
    🏈 GRIDIRONIQ refreshed for your current real-data workflow.<br>
    Core inputs: dashboard_model_results.csv, dataset(3).csv, contracts_clean(3).csv. Optional team/headshot files live in output/.
</div>
""",
    unsafe_allow_html=True,
)
