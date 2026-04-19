"""
explainer.py

LLM-powered explanation layer for the NFL contract recommendation app.

This module calls Databricks-hosted Llama for natural language explanations.
Falls back to template-based explanations if LLM is unavailable.

Main functions (same interface as before):
- explain_player(player_row)
- compare_players(player_a, player_b)
"""

from __future__ import annotations
from typing import Any
import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# CONFIG: Set USE_LLM = True to use Llama
# Set to False to use template fallback
# ─────────────────────────────────────────
USE_LLM = True

# ─────────────────────────────────────────
# LLM CLIENT SETUP
# ─────────────────────────────────────────
llm_client = None

if USE_LLM:
    try:
        import mlflow.deployments
        llm_client = mlflow.deployments.get_deploy_client("databricks")
        print("✅ LLM client connected!")
    except Exception as e:
        print(f"⚠️ LLM not available, using template fallback: {e}")
        USE_LLM = False


def call_llm(prompt: str) -> str:
    """Call LLM and return text response"""
    if not USE_LLM or llm_client is None:
        return ""
    try:
        response = llm_client.predict(
            endpoint="databricks-meta-llama-3-1-8b-instruct",
            inputs={"messages": [{"role": "user", "content": prompt}]}
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"LLM call failed: {e}")
        return ""


# ─────────────────────────────────────────
# SAFE HELPERS (same as original)
# ─────────────────────────────────────────
def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return str(value)


def _format_money(value: Any) -> str:
    x = _safe_float(value, default=np.nan)
    if pd.isna(x):
        return "N/A"
    return f"${x:.1f}M"


def _get(row: pd.Series, key: str, default: Any = 0) -> Any:
    return row[key] if key in row.index else default


def _position(row: pd.Series) -> str:
    return _safe_str(_get(row, "position", "Unknown")).upper()


def _player_name(row: pd.Series) -> str:
    return _safe_str(_get(row, "player_name", "This player"), "This player")


def _value_label(gap: float) -> str:
    if gap >= 3:
        return "strongly undervalued"
    if gap > 0:
        return "slightly undervalued"
    if gap <= -3:
        return "strongly overpriced"
    if gap < 0:
        return "slightly overpriced"
    return "fairly priced"


def _recommendation(gap: float) -> str:
    if gap > 2:
        return "Sign"
    elif gap < -2:
        return "Avoid"
    else:
        return "Monitor"


# ─────────────────────────────────────────
# BUILD STATS SUMMARY BY POSITION
# ─────────────────────────────────────────
def _build_stats_text(row: pd.Series) -> str:
    pos = _position(row)

    if pos == "QB":
        return (
            f"Pass yards: {int(_safe_float(_get(row, 'pass_yards', 0))):,}, "
            f"Pass TDs: {int(_safe_float(_get(row, 'pass_tds', 0)))}, "
            f"Interceptions: {int(_safe_float(_get(row, 'interceptions', 0)))}, "
            f"Completion %: {_safe_float(_get(row, 'completion_pct', 0)):.1f}%, "
            f"EPA/play: {_safe_float(_get(row, 'pass_epa_per_play', 0)):.2f}"
        )
    elif pos == "RB":
        return (
            f"Rush yards: {int(_safe_float(_get(row, 'rush_yards', 0))):,}, "
            f"Rush TDs: {int(_safe_float(_get(row, 'rush_tds', 0)))}, "
            f"Carries: {int(_safe_float(_get(row, 'carries', 0)))}, "
            f"Yards/carry: {_safe_float(_get(row, 'yards_per_carry', 0)):.1f}, "
            f"EPA/play: {_safe_float(_get(row, 'rush_epa_per_play', 0)):.2f}"
        )
    else:  # WR / TE
        return (
            f"Receiving yards: {int(_safe_float(_get(row, 'rec_yards', 0))):,}, "
            f"Receiving TDs: {int(_safe_float(_get(row, 'rec_tds', 0)))}, "
            f"Receptions: {int(_safe_float(_get(row, 'receptions', 0)))}, "
            f"Catch rate: {_safe_float(_get(row, 'catch_rate', 0)):.1f}%, "
            f"EPA/play: {_safe_float(_get(row, 'rec_epa_per_play', 0)):.2f}"
        )


# ─────────────────────────────────────────
# TEMPLATE FALLBACK (original logic)
# ─────────────────────────────────────────
def _template_explain(row: pd.Series) -> str:
    name = _player_name(row)
    predicted = _safe_float(_get(row, "predicted_apy", 0))
    actual = _safe_float(_get(row, "actual_apy", 0))
    gap = _safe_float(_get(row, "value_gap", predicted - actual))
    games = _safe_float(_get(row, "games_played", 0))
    stats = _build_stats_text(row)
    rec = _recommendation(gap)

    return (
        f"PERFORMANCE: {name} posted {stats} across {int(games)} games this season. "
        f"CONTRACT: The model estimates a fair value of {_format_money(predicted)} "
        f"versus an actual APY of {_format_money(actual)}, "
        f"creating a value gap of {_format_money(gap)} — suggesting the player is {_value_label(gap)}. "
        f"RECOMMENDATION: {rec} — "
        f"{'This player delivers strong production relative to cost and represents excellent value.' if gap > 2 else ''}"
        f"{'This player is being paid above their predicted market value based on recent performance.' if gap < -2 else ''}"
        f"{'This player is fairly compensated relative to production. Continue to monitor performance trends.' if -2 <= gap <= 2 else ''}"
    )


def _template_compare(a: pd.Series, b: pd.Series) -> str:
    a_name = _player_name(a)
    b_name = _player_name(b)
    a_gap = _safe_float(_get(a, "value_gap", 0))
    b_gap = _safe_float(_get(b, "value_gap", 0))
    a_actual = _safe_float(_get(a, "actual_apy", 0))
    b_actual = _safe_float(_get(b, "actual_apy", 0))
    a_games = _safe_float(_get(a, "games_played", 0))
    b_games = _safe_float(_get(b, "games_played", 0))

    cheaper = a_name if a_actual < b_actual else b_name
    better_value = a_name if a_gap > b_gap else b_name
    more_available = a_name if a_games > b_games else b_name

    return (
        f"{cheaper} is the cheaper option. "
        f"{better_value} shows the larger value gap, suggesting stronger upside. "
        f"{more_available} has been more available with more games played. "
        f"Overall, {better_value} appears to be the better value signing."
    )


# ─────────────────────────────────────────
# LLM-POWERED EXPLANATIONS
# ─────────────────────────────────────────
def _llm_explain(row: pd.Series) -> str:
    name = _player_name(row)
    pos = _position(row)
    predicted = _safe_float(_get(row, "predicted_apy", 0))
    actual = _safe_float(_get(row, "actual_apy", 0))
    gap = _safe_float(_get(row, "value_gap", predicted - actual))
    games = _safe_float(_get(row, "games_played", 0))
    stats = _build_stats_text(row)
    rec = _recommendation(gap)

    prompt = f"""You are an NFL contract analyst writing for a team general manager.
Use ONLY the data provided below. Do not guess or add information not given.
Write in a confident, professional tone. Be specific with numbers.

Player: {name}
Position: {pos}
Games played: {int(games)}
Stats: {stats}
Current APY: {_format_money(actual)}
Model predicted fair APY: {_format_money(predicted)}
Value gap: {_format_money(gap)}
Assessment: {_value_label(gap)}
Recommendation: {rec}

Write exactly 3 sections in this format:

PERFORMANCE: [1-2 sentences analyzing their on-field production using the stats above]
CONTRACT: [1-2 sentences comparing predicted vs actual value and what the gap means for the team]
RECOMMENDATION: {rec} — [1 sentence with a specific, actionable suggestion for the GM]
"""

    result = call_llm(prompt)
    if result and "PERFORMANCE:" in result:
        return result
    else:
        return _template_explain(row)


def _llm_compare(a: pd.Series, b: pd.Series) -> str:
    a_name = _player_name(a)
    b_name = _player_name(b)
    a_stats = _build_stats_text(a)
    b_stats = _build_stats_text(b)

    prompt = f"""You are an NFL contract analyst helping a GM decide between two players.
Use ONLY the data provided. Do not guess. Be specific with numbers.

PLAYER A: {a_name}
Position: {_position(a)}
Games: {int(_safe_float(_get(a, 'games_played', 0)))}
Stats: {a_stats}
Current APY: {_format_money(_safe_float(_get(a, 'actual_apy', 0)))}
Predicted APY: {_format_money(_safe_float(_get(a, 'predicted_apy', 0)))}
Value gap: {_format_money(_safe_float(_get(a, 'value_gap', 0)))}

PLAYER B: {b_name}
Position: {_position(b)}
Games: {int(_safe_float(_get(b, 'games_played', 0)))}
Stats: {b_stats}
Current APY: {_format_money(_safe_float(_get(b, 'actual_apy', 0)))}
Predicted APY: {_format_money(_safe_float(_get(b, 'predicted_apy', 0)))}
Value gap: {_format_money(_safe_float(_get(b, 'value_gap', 0)))}

In 3-4 sentences, compare these two players:
1. Who is the better value and why (use specific numbers)
2. Who has better production
3. Who would you recommend signing and why
"""

    result = call_llm(prompt)
    if result and len(result) > 50:
        return result
    else:
        return _template_compare(a, b)


# ─────────────────────────────────────────
# PUBLIC API (same interface as original)
# ─────────────────────────────────────────
def explain_player(player_row: pd.Series | dict) -> str:
    """Generate explanation for a single player"""
    row = pd.Series(player_row) if isinstance(player_row, dict) else player_row

    if USE_LLM:
        return _llm_explain(row)
    else:
        return _template_explain(row)


def compare_players(player_a: pd.Series | dict, player_b: pd.Series | dict) -> str:
    """Generate comparison between two players"""
    a = pd.Series(player_a) if isinstance(player_a, dict) else player_a
    b = pd.Series(player_b) if isinstance(player_b, dict) else player_b

    if USE_LLM:
        return _llm_compare(a, b)
    else:
        return _template_compare(a, b)


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "player_name": "Saquon Barkley",
        "position": "RB",
        "actual_apy": 20.6,
        "predicted_apy": 25.0,
        "value_gap": 4.4,
        "games_played": 16,
        "rush_yards": 2005,
        "rush_tds": 13,
        "carries": 345,
        "yards_per_carry": 5.8,
        "rush_epa_per_play": 0.14,
    }

    sample2 = {
        "player_name": "Josh Jacobs",
        "position": "RB",
        "actual_apy": 12.0,
        "predicted_apy": 16.5,
        "value_gap": 4.5,
        "games_played": 17,
        "rush_yards": 1329,
        "rush_tds": 15,
        "carries": 289,
        "yards_per_carry": 4.6,
        "rush_epa_per_play": 0.09,
    }

    print("=== Single Player ===")
    print(explain_player(sample))
    print()
    print("=== Compare ===")
    print(compare_players(sample, sample2))
