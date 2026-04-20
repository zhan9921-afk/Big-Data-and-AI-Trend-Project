
from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    try:
        if value is None or pd.isna(value):
            return default
    except Exception:
        pass
    return str(value)


def explain_player(player_row: pd.Series | Dict[str, Any]) -> str:
    row = pd.Series(player_row) if isinstance(player_row, dict) else player_row
    name = _safe_str(row.get("player_name"), "This player")
    pos = _safe_str(row.get("position"), "player")
    team = _safe_str(row.get("team"), "team")
    actual = _safe_float(row.get("actual_apy"))
    predicted = _safe_float(row.get("predicted_apy"))
    gap = _safe_float(row.get("value_gap"))
    games = int(_safe_float(row.get("games_played")))

    parts = []
    if _safe_float(row.get("pass_yards")) > 0:
        parts.append(f"{int(_safe_float(row.get('pass_yards'))):,} pass yards")
    if _safe_float(row.get("rush_yards")) > 0:
        parts.append(f"{int(_safe_float(row.get('rush_yards'))):,} rush yards")
    if _safe_float(row.get("rec_yards")) > 0:
        parts.append(f"{int(_safe_float(row.get('rec_yards'))):,} receiving yards")
    total_tds = int(_safe_float(row.get("total_tds")))
    if total_tds > 0:
        parts.append(f"{total_tds} total TDs")
    perf = ", ".join(parts[:3]) if parts else "limited counting stats in the current filtered view"
    verdict = "undervalued" if gap > 2 else "overvalued" if gap < -2 else "close to market"

    return (
        f"<b>Profile:</b> {name} is a {pos} on {team} with {games} games in scope. "
        f"<b>Production snapshot:</b> The profile currently shows {perf}. "
        f"<b>Contract lens:</b> Actual next APY is ${actual:.1f}M and the model predicts ${predicted:.1f}M, "
        f"creating a ${gap:+.1f}M prediction gap. That screens as <b>{verdict}</b>. "
        f"<b>LLM insertion point:</b> later replace this template with your API-generated explanation."
    )


def compare_players(player_a: pd.Series | Dict[str, Any], player_b: pd.Series | Dict[str, Any]) -> str:
    a = pd.Series(player_a) if isinstance(player_a, dict) else player_a
    b = pd.Series(player_b) if isinstance(player_b, dict) else player_b
    a_name = _safe_str(a.get("player_name"), "Player A")
    b_name = _safe_str(b.get("player_name"), "Player B")
    a_gap = _safe_float(a.get("value_gap"))
    b_gap = _safe_float(b.get("value_gap"))
    a_actual = _safe_float(a.get("actual_apy"))
    b_actual = _safe_float(b.get("actual_apy"))
    better_value = a_name if a_gap > b_gap else b_name
    cheaper = a_name if a_actual < b_actual else b_name

    return (
        f"<b>Value comparison:</b> {better_value} shows the stronger contract surplus "
        f"({a_gap:+.1f}M for {a_name} vs {b_gap:+.1f}M for {b_name}). "
        f"<b>Cost angle:</b> {cheaper} is the cheaper next contract. "
        f"<b>LLM insertion point:</b> this is where your future head-to-head explanation should go."
    )


def explain_query_results(results: pd.DataFrame, filters: Dict[str, Any]) -> str:
    if results.empty:
        return "No players matched the current query, so the dashboard is showing an empty shortlist state."
    avg_gap = results["value_gap"].mean()
    best = results.iloc[0]["player_name"]
    scope = []
    if filters.get("position"):
        scope.append(filters["position"])
    if filters.get("team"):
        scope.append(filters["team"])
    scope_text = " / ".join(scope) if scope else "league-wide"
    return (
        f"The current shortlist is <b>{scope_text}</b> and returns <b>{len(results)}</b> players. "
        f"Average prediction gap is <b>${avg_gap:+.1f}M</b>, and <b>{best}</b> currently leads the ranking. "
        f"This box is the natural insertion point for an LLM-generated query summary."
    )


def explain_team(team_df: pd.DataFrame, team_name: str) -> str:
    if team_df.empty:
        return f"No roster rows are currently available for {team_name}."
    avg_gap = team_df["value_gap"].mean()
    top_name = team_df.sort_values("value_gap", ascending=False).iloc[0]["player_name"]
    under = int((team_df["value_gap"] > 2).sum())
    return (
        f"{team_name} has <b>{len(team_df)}</b> players in scope, with an average prediction gap of "
        f"<b>${avg_gap:+.1f}M</b>. The strongest value signal currently belongs to <b>{top_name}</b>, "
        f"and <b>{under}</b> players screen as clearly undervalued. "
        f"This is where a team-level LLM synopsis should later be inserted."
    )


def explain_chart(chart_name: str, data: pd.DataFrame, filters: Dict[str, Any] | None = None) -> str:
    filters = filters or {}

    if data.empty:
        return "No data is available for the selected chart."

    if chart_name == "team_gap" and "team" in data.columns:
        best_team = data.groupby("team")["value_gap"].mean().sort_values(ascending=False).index[0]
        return (
            f"This chart compares average prediction gap by team. "
            f"<b>{best_team}</b> currently leads on average surplus, suggesting the strongest contract efficiency in this view."
        )

    if chart_name == "position_gap" and "position" in data.columns:
        best_pos = data.groupby("position")["value_gap"].mean().sort_values(ascending=False).index[0]
        return (
            f"This chart summarizes average prediction gap by position. "
            f"<b>{best_pos}</b> appears to be the strongest value pocket in the current filtered sample."
        )

    if chart_name == "roster":
        team = filters.get("team", "the selected team")
        return (
            f"This roster chart compares actual next APY versus modeled next APY within <b>{team}</b>. "
            f"It is useful for spotting which players screen as efficient future deals and which players look expensive relative to model expectations."
        )

    if chart_name == "team_mix":
        team = filters.get("team", "the selected team")
        return (
            f"This composition chart shows how <b>{team}</b> is distributed across position groups in the currently loaded roster detail file."
        )

    if chart_name == "accuracy_box":
        return (
            f"This box plot compares absolute prediction error by position. "
            f"It helps you see which position groups are easiest or hardest for the current model to price accurately."
        )

    if chart_name == "value_scatter":
        n_under = int((data["value_gap"] > 2).sum()) if "value_gap" in data.columns else 0
        n_over = int((data["value_gap"] < -2).sum()) if "value_gap" in data.columns else 0
        return (
            f"This scatter compares actual next APY to predicted next APY. "
            f"Points above the parity line screen as better value. "
            f"In this filtered sample, <b>{n_under}</b> players look clearly undervalued and <b>{n_over}</b> look overvalued."
        )

    return "This chart helps summarize player value, contract efficiency, and where the strongest opportunities appear in the current filtered view."
