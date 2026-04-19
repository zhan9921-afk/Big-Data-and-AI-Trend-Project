"""
app_test_v4.py

GridironIQ - NFL Contract Recommender
Smart Query Page — Championship Edition v4
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st

from query_parser_v2 import parse_query, merge_query_with_ui_filters
from recommendation_engine import get_ranked_table
from explainer import explain_player, compare_players

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "shortlist_base.csv"

@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)

st.set_page_config(
    page_title="GridironIQ",
    page_icon="🏈",
    layout="wide"
)

# ─────────────────────────────────────────
# NFL-THEMED CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Source+Sans+3:wght@400;600;700&display=swap');

    .stApp {
        background-color: #0B1120;
        color: #E8E8E8;
    }

    .hero {
        background: linear-gradient(135deg, #013369 0%, #1a1a2e 50%, #D50A0A 100%);
        border-radius: 16px;
        padding: 30px 40px;
        margin-bottom: 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: "🏈";
        position: absolute;
        font-size: 120px;
        opacity: 0.08;
        top: -20px;
        right: 20px;
    }
    .hero h1 {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 3.2rem;
        color: white;
        letter-spacing: 6px;
        margin: 0;
    }
    .hero p {
        font-family: 'Source Sans 3', sans-serif;
        color: #B0B8C8;
        font-size: 1.1rem;
        margin-top: 8px;
    }

    section[data-testid="stSidebar"] {
        background-color: #0D1525;
        border-right: 1px solid #1E2A3A;
    }

    .player-card {
        background: linear-gradient(160deg, #131B2E 0%, #1A2540 100%);
        border: 1px solid #2A3550;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 12px;
        transition: transform 0.2s, border-color 0.2s;
    }
    .player-card:hover {
        transform: translateY(-2px);
        border-color: #4A6FA5;
    }
    .player-card .name {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.6rem;
        color: white;
        letter-spacing: 2px;
        margin: 0;
    }
    .player-card .meta {
        color: #7B8DA8;
        font-size: 0.9rem;
        margin: 4px 0 12px 0;
    }
    .player-card .stats-row {
        display: flex;
        gap: 20px;
        margin-top: 8px;
    }
    .player-card .stat-box {
        background: #0B1120;
        border-radius: 8px;
        padding: 10px 16px;
        text-align: center;
        flex: 1;
    }
    .stat-box .stat-label {
        color: #5A7090;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stat-box .stat-value {
        color: white;
        font-size: 1.3rem;
        font-weight: 700;
    }

    .badge-sign {
        background: linear-gradient(135deg, #0D6E3B, #28a745);
        color: white;
        padding: 5px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-avoid {
        background: linear-gradient(135deg, #8B1A1A, #dc3545);
        color: white;
        padding: 5px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-monitor {
        background: linear-gradient(135deg, #8B7500, #ffc107);
        color: black;
        padding: 5px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }

    .ai-box {
        background: linear-gradient(160deg, #0D1A2F 0%, #152238 100%);
        border: 1px solid #2A3F5F;
        border-left: 4px solid #4A90D9;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 12px 0;
        color: #C8D4E4;
        line-height: 1.7;
    }
    .ai-box .ai-title {
        color: #4A90D9;
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }

    .vs-badge {
        background: #D50A0A;
        color: white;
        padding: 6px 14px;
        border-radius: 50%;
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.2rem;
        text-align: center;
        display: inline-block;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #131B2E;
        border-radius: 8px;
        color: #8A9BB8;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #013369;
        color: white;
    }

    .footer {
        text-align: center;
        color: #4A5568;
        font-size: 0.8rem;
        padding: 20px 0;
        border-top: 1px solid #1E2A3A;
        margin-top: 40px;
    }

    .kpi-row {
        display: flex;
        gap: 12px;
        margin-bottom: 20px;
    }
    .kpi-card {
        background: #131B2E;
        border: 1px solid #1E2A3A;
        border-radius: 10px;
        padding: 16px 20px;
        flex: 1;
        text-align: center;
    }
    .kpi-card .kpi-label {
        color: #5A7090;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .kpi-card .kpi-value {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>GRIDIRONIQ</h1>
    <p>AI-Powered NFL Contract Intelligence — From Data to Action</p>
</div>
""", unsafe_allow_html=True)

if not DATA_PATH.exists():
    st.error("shortlist_base.csv not found")
    st.stop()

df = load_data()

# ─────────────────────────────────────────
# SIDEBAR WITH CLICKABLE EXAMPLES
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏈 GRIDIRONIQ")
    st.markdown("---")
    st.markdown("#### 🔍 Manager Query")

    # Initialize session state for query
    if "user_query" not in st.session_state:
        st.session_state["user_query"] = "Find undervalued RB under $6M"

    user_query = st.text_input(
        "What are you looking for?",
        value=st.session_state["user_query"],
        label_visibility="collapsed",
        placeholder="e.g. Find undervalued RB under $6M"
    )

    # Clickable example queries
    st.markdown("#### 💡 Try these:")

    example_queries = [
        ("🏟️", "Show undervalued QBs"),
        ("💰", "Cheap RB under $5M"),
        ("📊", "Compare budget WRs"),
        ("🔍", "Safe TE under $10M"),
        ("⭐", "Find best value players"),
        ("🏈", "Show overvalued QBs"),
    ]

    for icon, eq in example_queries:
        if st.button(f"{icon} {eq}", key=f"eq_{eq}", use_container_width=True):
            st.session_state["user_query"] = eq
            st.rerun()

    st.markdown("---")
    st.markdown("#### ⚙️ Filters")

    positions = ["All"] + sorted(df["position"].dropna().astype(str).unique().tolist())
    seasons = ["All"] + sorted(df["season"].dropna().astype(int).unique().tolist(), reverse=True)

    ui_position = st.selectbox("Position", positions, index=0)
    ui_max_salary_m = st.slider("Max salary ($M)", 0.0, 60.0, 10.0, 0.5)
    ui_min_games = st.slider("Min games", 0, 17, 8, 1)
    ui_season = st.selectbox("Season", seasons, index=0)
    ui_objective = st.selectbox("Objective", ["balanced", "undervalued", "overvalued"], index=1)
    top_k = st.slider("Show top", 3, 20, 5, 1)

# ─────────────────────────────────────────
# PARSE QUERY
# ─────────────────────────────────────────
parsed = parse_query(user_query)

ui_filters = {
    "position": None if ui_position == "All" else ui_position,
    "max_salary": ui_max_salary_m * 1_000_000,
    "min_games": ui_min_games,
    "season": None if ui_season == "All" else int(ui_season),
    "objective": ui_objective,
}

final_filters = merge_query_with_ui_filters(
    parsed_query=parsed,
    ui_filters=ui_filters,
    query_priority=True
)

ranked = get_ranked_table(df, filters=final_filters)

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def get_badge(gap):
    if gap > 2:
        return '<span class="badge-sign">✅ SIGN</span>'
    elif gap < -2:
        return '<span class="badge-avoid">❌ AVOID</span>'
    else:
        return '<span class="badge-monitor">👀 MONITOR</span>'

def get_position_stats_html(row):
    pos = row.get('position', '')
    if pos == 'QB':
        return f"""
        <div class="stats-row">
            <div class="stat-box"><div class="stat-label">Pass Yards</div><div class="stat-value">{int(row.get('pass_yards',0)):,}</div></div>
            <div class="stat-box"><div class="stat-label">Pass TDs</div><div class="stat-value">{int(row.get('pass_tds',0))}</div></div>
            <div class="stat-box"><div class="stat-label">Comp %</div><div class="stat-value">{row.get('completion_pct',0):.1f}%</div></div>
        </div>"""
    elif pos == 'RB':
        return f"""
        <div class="stats-row">
            <div class="stat-box"><div class="stat-label">Rush Yards</div><div class="stat-value">{int(row.get('rush_yards',0)):,}</div></div>
            <div class="stat-box"><div class="stat-label">Rush TDs</div><div class="stat-value">{int(row.get('rush_tds',0))}</div></div>
            <div class="stat-box"><div class="stat-label">YPC</div><div class="stat-value">{row.get('yards_per_carry',0):.1f}</div></div>
        </div>"""
    else:
        return f"""
        <div class="stats-row">
            <div class="stat-box"><div class="stat-label">Rec Yards</div><div class="stat-value">{int(row.get('rec_yards',0)):,}</div></div>
            <div class="stat-box"><div class="stat-label">Rec TDs</div><div class="stat-value">{int(row.get('rec_tds',0))}</div></div>
            <div class="stat-box"><div class="stat-label">Catch %</div><div class="stat-value">{row.get('catch_rate',0):.1f}%</div></div>
        </div>"""

# ─────────────────────────────────────────
# KPI SUMMARY
# ─────────────────────────────────────────
if not ranked.empty:
    shortlist = ranked.head(top_k).copy()
    n_under = len(shortlist[shortlist['value_gap'] > 2])
    n_over = len(shortlist[shortlist['value_gap'] < -2])
    avg_gap = shortlist['value_gap'].mean()

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card"><div class="kpi-label">Players Found</div><div class="kpi-value">{len(shortlist)}</div></div>
        <div class="kpi-card"><div class="kpi-label">Undervalued</div><div class="kpi-value" style="color:#28a745">{n_under}</div></div>
        <div class="kpi-card"><div class="kpi-label">Overvalued</div><div class="kpi-value" style="color:#dc3545">{n_over}</div></div>
        <div class="kpi-card"><div class="kpi-label">Avg Value Gap</div><div class="kpi-value">${avg_gap:.1f}M</div></div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📋 Shortlist", "🔍 Player Analysis", "⚖️ Compare", "🧠 Query Debug"])

# TAB 1: SHORTLIST
with tab1:
    if ranked.empty:
        st.warning("No players matched. Adjust your query or filters.")
    else:
        shortlist = ranked.head(top_k).copy()
        for _, player in shortlist.iterrows():
            gap = player.get('value_gap', 0)
            badge = get_badge(gap)
            stats_html = get_position_stats_html(player)
            st.markdown(f"""
<div class="player-card">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <div class="name">{player['player_name']}</div>
            <div class="meta">{player['position']} · Season {int(player['season'])} · {int(player['games_played'])} games</div>
        </div>
        <div>{badge}</div>
    </div>
    <div class="stats-row">
        <div class="stat-box"><div class="stat-label">Current APY</div><div class="stat-value">${player['actual_apy']:.1f}M</div></div>
        <div class="stat-box"><div class="stat-label">Predicted APY</div><div class="stat-value">${player['predicted_apy']:.1f}M</div></div>
        <div class="stat-box"><div class="stat-label">Value Gap</div><div class="stat-value" style="color:{'#28a745' if gap > 0 else '#dc3545'}">${gap:+.1f}M</div></div>
    </div>
    {stats_html}
</div>
""", unsafe_allow_html=True)

# TAB 2: PLAYER ANALYSIS
with tab2:
    if ranked.empty:
        st.info("No players to analyze.")
    else:
        shortlist = ranked.head(top_k).copy()
        selected_name = st.selectbox("Select a player", shortlist["player_name"].tolist(), key="analysis")
        selected_row = shortlist[shortlist["player_name"] == selected_name].iloc[0]
        gap = selected_row.get('value_gap', 0)
        badge = get_badge(gap)
        stats_html = get_position_stats_html(selected_row)

        st.markdown(f"""
<div class="player-card">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <div class="name">{selected_row['player_name']}</div>
            <div class="meta">{selected_row['position']} · Season {int(selected_row['season'])} · {int(selected_row['games_played'])} games</div>
        </div>
        <div>{badge}</div>
    </div>
    <div class="stats-row">
        <div class="stat-box"><div class="stat-label">Current APY</div><div class="stat-value">${selected_row['actual_apy']:.1f}M</div></div>
        <div class="stat-box"><div class="stat-label">Predicted APY</div><div class="stat-value">${selected_row['predicted_apy']:.1f}M</div></div>
        <div class="stat-box"><div class="stat-label">Value Gap</div><div class="stat-value" style="color:{'#28a745' if gap > 0 else '#dc3545'}">${gap:+.1f}M</div></div>
    </div>
    {stats_html}
</div>
""", unsafe_allow_html=True)

        explanation = explain_player(selected_row)
        st.markdown(f"""
<div class="ai-box">
    <div class="ai-title">🤖 AI Contract Analysis</div>
    {explanation}
</div>
""", unsafe_allow_html=True)

# TAB 3: COMPARE
with tab3:
    if ranked.empty:
        st.info("No players to compare.")
    else:
        shortlist = ranked.head(top_k).copy()
        names = shortlist["player_name"].tolist()

        if len(names) >= 2:
            col1, col_vs, col2 = st.columns([2, 0.5, 2])
            with col1:
                player_a_name = st.selectbox("Player A", names, index=0, key="cmp_a")
            with col_vs:
                st.markdown("<div style='text-align:center; padding-top:28px'><span class='vs-badge'>VS</span></div>", unsafe_allow_html=True)
            with col2:
                player_b_name = st.selectbox("Player B", names, index=1, key="cmp_b")

            if player_a_name == player_b_name:
                st.warning("Choose two different players.")
            else:
                player_a = shortlist[shortlist["player_name"] == player_a_name].iloc[0]
                player_b = shortlist[shortlist["player_name"] == player_b_name].iloc[0]

                col1, col2 = st.columns(2)
                with col1:
                    gap_a = player_a.get('value_gap', 0)
                    badge_a = get_badge(gap_a)
                    stats_a = get_position_stats_html(player_a)
                    st.markdown(f"""
<div class="player-card">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div><div class="name">{player_a['player_name']}</div><div class="meta">{player_a['position']} · {int(player_a['games_played'])} games</div></div>
        <div>{badge_a}</div>
    </div>
    <div class="stats-row">
        <div class="stat-box"><div class="stat-label">APY</div><div class="stat-value">${player_a['actual_apy']:.1f}M</div></div>
        <div class="stat-box"><div class="stat-label">Predicted</div><div class="stat-value">${player_a['predicted_apy']:.1f}M</div></div>
        <div class="stat-box"><div class="stat-label">Gap</div><div class="stat-value" style="color:{'#28a745' if gap_a > 0 else '#dc3545'}">${gap_a:+.1f}M</div></div>
    </div>
    {stats_a}
</div>
""", unsafe_allow_html=True)

                with col2:
                    gap_b = player_b.get('value_gap', 0)
                    badge_b = get_badge(gap_b)
                    stats_b = get_position_stats_html(player_b)
                    st.markdown(f"""
<div class="player-card">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div><div class="name">{player_b['player_name']}</div><div class="meta">{player_b['position']} · {int(player_b['games_played'])} games</div></div>
        <div>{badge_b}</div>
    </div>
    <div class="stats-row">
        <div class="stat-box"><div class="stat-label">APY</div><div class="stat-value">${player_b['actual_apy']:.1f}M</div></div>
        <div class="stat-box"><div class="stat-label">Predicted</div><div class="stat-value">${player_b['predicted_apy']:.1f}M</div></div>
        <div class="stat-box"><div class="stat-label">Gap</div><div class="stat-value" style="color:{'#28a745' if gap_b > 0 else '#dc3545'}">${gap_b:+.1f}M</div></div>
    </div>
    {stats_b}
</div>
""", unsafe_allow_html=True)

                comparison = compare_players(player_a, player_b)
                st.markdown(f"""
<div class="ai-box">
    <div class="ai-title">🤖 AI Head-to-Head Analysis</div>
    {comparison}
</div>
""", unsafe_allow_html=True)

# TAB 4: QUERY DEBUG
with tab4:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Parsed Query")
        st.json(parsed)
    with col2:
        st.markdown("#### Final Filters")
        st.json(final_filters)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div class="footer">
    🏈 GRIDIRONIQ — Team 8 | Big Data & AI Trends Market | Spring 2026<br>
    Adam Getzkin · Mallika Kommera · Jay Pederson · Ariel Zhan · Zhen Zhang
</div>
""", unsafe_allow_html=True)
