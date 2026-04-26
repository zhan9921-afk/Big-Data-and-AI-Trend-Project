
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from llm_client_openai import ask_openai_text, is_llm_configured


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


def _llm_or_fallback(prompt: str, fallback: str, prefer_fallback: bool = False) -> str:
    if prefer_fallback or not is_llm_configured():
        return fallback
    return ask_openai_text(prompt, fallback=fallback)


def explain_player(player_row: pd.Series | Dict[str, Any], plan: Optional[Dict[str, Any]] = None) -> str:
    row = pd.Series(player_row) if isinstance(player_row, dict) else player_row
    name = _safe_str(row.get("player_name"), "This player")
    pos = _safe_str(row.get("position"), "player")
    team = _safe_str(row.get("team"), "team")
    actual = _safe_float(row.get("actual_apy"))
    predicted = _safe_float(row.get("predicted_apy"))
    gap = _safe_float(row.get("value_gap"))
    games = int(_safe_float(row.get("games_played")))
    current = _safe_float(row.get("current_apy"))
    fallback = (
        f"<b>Profile:</b> {name} is a {pos} on {team} with {games} games in scope. "
        f"<b>Contract lens:</b> Current APY is ${current:.1f}M, actual next APY is ${actual:.1f}M, "
        f"and the model predicts ${predicted:.1f}M, creating a ${gap:+.1f}M gap."
    )
    prompt = f"""
You are writing a concise NFL contract dashboard explanation.
Only use the player named below and do not introduce any other player names.
Player: {name}
Position: {pos}
Team: {team}
Games: {games}
Current APY: {current}
Actual next APY: {actual}
Predicted next APY: {predicted}
Prediction gap: {gap}
Query plan: {plan or {}}

Write 4 short sentences in business-friendly language. Use HTML <b> tags for key phrases.
"""
    return _llm_or_fallback(prompt, fallback)


def compare_players(player_a: pd.Series | Dict[str, Any], player_b: pd.Series | Dict[str, Any], plan: Optional[Dict[str, Any]] = None) -> str:
    a = pd.Series(player_a) if isinstance(player_a, dict) else player_a
    b = pd.Series(player_b) if isinstance(player_b, dict) else player_b
    a_name = _safe_str(a.get("player_name"), "Player A")
    b_name = _safe_str(b.get("player_name"), "Player B")
    a_gap = _safe_float(a.get("value_gap"))
    b_gap = _safe_float(b.get("value_gap"))
    a_actual = _safe_float(a.get("actual_apy"))
    b_actual = _safe_float(b.get("actual_apy"))
    fallback = (
        f"<b>Value comparison:</b> {a_name} shows ${a_gap:+.1f}M versus {b_name} at ${b_gap:+.1f}M. "
        f"<b>Cost angle:</b> {a_name if a_actual < b_actual else b_name} is the cheaper next contract."
    )
    prompt = f"""
Write a concise head-to-head comparison for an NFL dashboard.
Only mention these two players: {a_name} and {b_name}.
Player A: actual next APY {a_actual}, prediction gap {a_gap}
Player B: actual next APY {b_actual}, prediction gap {b_gap}
Query plan: {plan or {}}

Write 3-4 short sentences with HTML <b> tags on key conclusions.
"""
    return _llm_or_fallback(prompt, fallback)


def explain_query_results(results: pd.DataFrame, plan: Dict[str, Any]) -> str:
    if results.empty:
        fallback = "No players matched the current query, so the dashboard is showing an empty shortlist state."
    else:
        avg_gap = results["value_gap"].mean()
        best = results.iloc[0]["player_name"]
        scope = []
        if plan.get("focus_position"):
            scope.append(plan["focus_position"])
        if plan.get("focus_team"):
            scope.append(plan["focus_team"])
        if plan.get("focus_player"):
            scope.append(plan["focus_player"])
        scope_text = " / ".join(scope) if scope else "league-wide"
        fallback = (
            f"The current shortlist is <b>{scope_text}</b> and returns <b>{len(results)}</b> players. "
            f"Average prediction gap is <b>${avg_gap:+.1f}M</b>, and <b>{best}</b> currently leads the ranking."
        )
    prompt = f"""
Write a concise dashboard result summary.
Query plan: {plan}
Rows returned: {len(results)}
Fallback summary: {fallback}
If a focus_player is present, anchor the explanation around that player and do not introduce unrelated names.
Use HTML <b> tags where helpful.
"""
    return _llm_or_fallback(prompt, fallback)


def explain_team(team_df: pd.DataFrame, team_name: str, plan: Optional[Dict[str, Any]] = None) -> str:
    if team_df.empty:
        fallback = f"No roster rows are currently available for {team_name}."
    else:
        avg_gap = team_df["value_gap"].mean()
        top_name = team_df.sort_values("value_gap", ascending=False).iloc[0]["player_name"]
        under = int((team_df["valuation_status"] == "Undervalued").sum()) if "valuation_status" in team_df.columns else int((team_df["value_gap"] > 2).sum())
        fallback = (
            f"{team_name} has <b>{len(team_df)}</b> players in scope, with an average prediction gap of "
            f"<b>${avg_gap:+.1f}M</b>. The strongest value signal currently belongs to <b>{top_name}</b>, "
            f"and <b>{under}</b> players screen as clearly undervalued."
        )
    prompt = f"""
Write a concise team-level explanation for an NFL contract dashboard.
Only discuss team context for {team_name}.
Query plan: {plan or {}}
Rows: {len(team_df)}
Fallback summary: {fallback}
Use HTML <b> tags and keep it to 4 short sentences.
"""
    return _llm_or_fallback(prompt, fallback)


def explain_chart(chart_name: str, data: pd.DataFrame, filters: Dict[str, Any] | None = None) -> str:
    filters = filters or {}
    focus_player = filters.get("focus_player")
    focus_team = filters.get("focus_team") or filters.get("team")
    metric = filters.get("metric")
    if data.empty:
        fallback = "No data is available for the selected chart."
    elif chart_name == "team_gap" and "team" in data.columns:
        best_team = data.groupby("team")["value_gap"].mean().sort_values(ascending=False).index[0]
        fallback = f"This chart compares average prediction gap by team. <b>{best_team}</b> currently leads on average surplus."
    elif chart_name == "position_gap" and "position" in data.columns:
        best_pos = data.groupby("position")["value_gap"].mean().sort_values(ascending=False).index[0]
        fallback = f"This chart summarizes average prediction gap by position. <b>{best_pos}</b> appears strongest in the current filtered sample."
    elif chart_name == "roster":
        fallback = f"This roster chart displays <b>{metric or 'the selected metric'}</b> for players on <b>{focus_team or 'the selected team'}</b>."
    elif chart_name == "team_expiring":
        fallback = f"This chart highlights <b>{focus_team or 'the selected team'}</b> players whose deals are expiring soon and compares current APY with predicted next APY."
    elif chart_name == "accuracy_box":
        fallback = "This box plot compares absolute prediction error by position."
    elif chart_name == "player_contract_bar":
        fallback = f"This chart gives a quick snapshot of current APY, actual next APY, and predicted next APY for <b>{focus_player or 'the focused player'}</b>."
    elif chart_name == "compare_contracts":
        fallback = "This comparison chart lines up current, actual next, and predicted APY for the two comparison players."
    else:
        if "valuation_status" in data.columns:
            n_under = int((data["valuation_status"] == "Undervalued").sum())
            n_over = int((data["valuation_status"] == "Overvalued").sum())
        else:
            n_under = int((data["value_gap"] > 2).sum()) if "value_gap" in data.columns else 0
            n_over = int((data["value_gap"] < -2).sum()) if "value_gap" in data.columns else 0
        fallback = f"This scatter compares actual next APY to predicted next APY. <b>{n_under}</b> players look undervalued and <b>{n_over}</b> look overvalued."
    prompt = f"""
Write a concise explanation for this dashboard chart.
Chart name: {chart_name}
Filters / plan context: {filters}
Row count: {len(data)}
Fallback explanation: {fallback}
If focus_player is present, stay anchored to that player and do not introduce unrelated player names.
Use 2-4 short sentences with HTML <b> tags when useful.
"""
    prefer_fallback = chart_name in {"player_contract_bar", "compare_contracts"} and bool(focus_player)
    return _llm_or_fallback(prompt, fallback, prefer_fallback=prefer_fallback)


def explain_smart_query_insights(
    plan: Dict[str, Any],
    results: pd.DataFrame,
    focus_player_row: Optional[pd.Series],
    focus_compare_row: Optional[pd.Series],
    user_query: str = "",
    chart_name: str = "",
) -> str:
    top_rows = []
    if not results.empty:
        cols = [c for c in ["player_name", "team", "position", "current_apy", "actual_apy", "predicted_apy", "pred_years", "pred_guar", "predicted_total_value", "valuation_pct", "valuation_status", "value_gap", "games_played"] if c in results.columns]
        top_rows = results[cols].head(5).to_dict("records")

    if focus_player_row is not None:
        fallback = (
            f"<b>What matters:</b> The query is centered on <b>{focus_player_row.get('player_name')}</b> of the <b>{focus_player_row.get('team')}</b>. "
            f"The model shows current APY at <b>${_safe_float(focus_player_row.get('current_apy')):.1f}M</b>, actual next APY at "
            f"<b>${_safe_float(focus_player_row.get('actual_apy')):.1f}M</b>, and predicted next APY at "
            f"<b>${_safe_float(focus_player_row.get('predicted_apy')):.1f}M</b>. "
            f"<b>Manager takeaway:</b> Use this page to judge whether the club should treat this player as a value retain, a monitor case, or a contract-risk discussion."
        )
        if focus_compare_row is not None:
            fallback += f" <b>Useful benchmark:</b> A nearby contract peer is <b>{focus_compare_row.get('player_name')}</b>."
    elif results.empty:
        fallback = "<b>No actionable read yet:</b> the current smart query returned no rows, so broaden the request or relax filters before making a contract decision."
    else:
        top_line = results.iloc[0]
        fallback = (
            f"<b>What matters:</b> The current smart-query view is led by <b>{top_line.get('player_name')}</b> at "
            f"<b>${_safe_float(top_line.get('value_gap')):+.1f}M</b> prediction gap. "
            f"<b>Manager takeaway:</b> Use the top rows to separate immediate value opportunities from players who require more contract scrutiny."
        )

    prompt = f"""
You are an NFL front-office analytics assistant writing for a team manager.
Your job is to answer the user's query with decision-oriented insights, not generic chart narration.

User query: {user_query}
Smart-query chart shown: {chart_name}
Dashboard plan: {plan}
Focused player row: {None if focus_player_row is None else dict(focus_player_row)}
Focused comparison peer: {None if focus_compare_row is None else dict(focus_compare_row)}
Top result rows: {top_rows}

Write one compact HTML block with 2 short paragraphs.
Paragraph 1 should explain the most decision-relevant takeaway for a team manager.
Paragraph 2 should translate that takeaway into action-oriented insight such as retain / compare / monitor / negotiate / re-check supporting evidence.
Ground every statement in the provided context. Do not invent player names, teams, metrics, or claims not present in the data.
If a focused player exists, keep the answer anchored to that player.
Use HTML <b> tags for the most important conclusions.
Do not say 'chart', 'dashboard', 'visual', or 'this box'.
"""
    return _llm_or_fallback(prompt, fallback, prefer_fallback=False)
