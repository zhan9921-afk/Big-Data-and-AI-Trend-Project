"""
app_test.py

Minimal non-LLM Streamlit test app for the NFL contract recommendation demo.

Expected files in the same folder:
- shortlist_base.csv
- query_parser_v2.py
- recommendation_engine.py
- explainer.py

Run:
    streamlit run app_test.py
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


st.set_page_config(page_title="NFL Contract Recommender Test", layout="wide")
st.title("NFL Contract Recommender")
st.caption("Test version without LLM")

if not DATA_PATH.exists():
    st.error("shortlist_base.csv not found in the same folder as app_test.py")
    st.stop()

df = load_data()

with st.sidebar:
    st.header("Query Box")
    user_query = st.text_input(
        "Type a manager request",
        value="Find undervalued RB under $6M"
    )

    st.header("Structured Filters")
    positions = ["All"] + sorted(df["position"].dropna().astype(str).unique().tolist())
    seasons = ["All"] + sorted(df["season"].dropna().astype(int).unique().tolist(), reverse=True)

    ui_position = st.selectbox("Position", positions, index=0)
    ui_max_salary_m = st.slider("Max salary ($M)", min_value=0.0, max_value=60.0, value=10.0, step=0.5)
    ui_min_games = st.slider("Min games", min_value=0, max_value=17, value=8, step=1)
    ui_season = st.selectbox("Season", seasons, index=0)
    ui_objective = st.selectbox("Objective", ["balanced", "undervalued", "overvalued"], index=1)
    top_k = st.slider("Top K", min_value=3, max_value=20, value=5, step=1)

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

left, right = st.columns([1.4, 1])

with left:
    st.subheader("Parsed Query")
    st.json(parsed)

    st.subheader("Final Filters Used")
    st.json(final_filters)

    if ranked.empty:
        st.warning("No players matched the current query or filters.")
    else:
        shortlist = ranked.head(top_k).copy()

        display_cols = [
            "player_name",
            "position",
            "season",
            "games_played",
            "actual_apy",
            "predicted_apy",
            "value_gap",
            "recommendation_score",
            "budget_status",
            "availability_band",
            "risk_flag",
        ]
        display_cols = [c for c in display_cols if c in shortlist.columns]

        st.subheader("Shortlist")
        st.dataframe(
            shortlist[display_cols].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

        chart_df = shortlist[["player_name", "actual_apy", "predicted_apy"]].copy()
        chart_df = chart_df.set_index("player_name")
        st.subheader("Actual vs Predicted APY")
        st.bar_chart(chart_df)

with right:
    if ranked.empty:
        st.info("Adjust the query or filters to generate a shortlist.")
    else:
        shortlist = ranked.head(top_k).copy()

        st.subheader("Single Player Explanation")
        selected_name = st.selectbox("Choose a player", shortlist["player_name"].tolist())
        selected_row = shortlist[shortlist["player_name"] == selected_name].iloc[0]
        st.write(explain_player(selected_row))

        st.subheader("Compare Two Players")
        names = shortlist["player_name"].tolist()

        if len(names) >= 2:
            player_a_name = st.selectbox("Player A", names, index=0)
            player_b_name = st.selectbox("Player B", names, index=1 if len(names) > 1 else 0)

            if player_a_name == player_b_name:
                st.warning("Choose two different players to compare.")
            else:
                player_a = shortlist[shortlist["player_name"] == player_a_name].iloc[0]
                player_b = shortlist[shortlist["player_name"] == player_b_name].iloc[0]
                st.write(compare_players(player_a, player_b))

st.markdown("---")
st.markdown("This test app uses the rule-based query parser and template explainer only.")
