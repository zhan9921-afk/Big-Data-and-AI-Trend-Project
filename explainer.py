"""
explainer.py

Template-based explanation layer for the NFL contract recommendation app.

This module intentionally does NOT call any LLM.
It is designed so you can later replace or augment these functions
with an LLM-based explanation service without changing the rest of
the product pipeline too much.

Main functions
--------------
- explain_player(player_row)
- compare_players(player_a, player_b)
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np


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


def _format_money_m(value: Any) -> str:
    x = _safe_float(value, default=np.nan)
    if pd.isna(x):
        return "N/A"
    if abs(x) >= 100000:
        return f"${x / 1_000_000:.1f}M"
    return f"${x:.1f}M"


def _format_num(value: Any, digits: int = 0) -> str:
    x = _safe_float(value, default=np.nan)
    if pd.isna(x):
        return "N/A"
    return f"{x:.{digits}f}"


def _get(row: pd.Series, key: str, default: Any = 0) -> Any:
    return row[key] if key in row.index else default


def _position(row: pd.Series) -> str:
    return _safe_str(_get(row, "position", "Unknown")).upper()


def _player_name(row: pd.Series) -> str:
    return _safe_str(_get(row, "player_name", "This player"), "This player")


def _availability_band(games_played: float) -> str:
    if games_played >= 14:
        return "strong"
    if games_played >= 10:
        return "solid"
    if games_played >= 6:
        return "moderate"
    return "limited"


def _value_label(value_gap: float) -> str:
    if value_gap >= 3:
        return "strongly undervalued"
    if value_gap > 0:
        return "slightly undervalued"
    if value_gap <= -3:
        return "strongly overpriced"
    if value_gap < 0:
        return "slightly overpriced"
    return "fairly priced"


def _pick_best_metric_phrase(row: pd.Series) -> str:
    pos = _position(row)

    if pos == "WR":
        rec_yards = _safe_float(_get(row, "rec_yards", 0))
        rec_tds = _safe_float(_get(row, "rec_tds", 0))
        targets = _safe_float(_get(row, "targets", 0))
        return (
            f"{_format_num(rec_yards)} receiving yards, "
            f"{_format_num(rec_tds)} receiving TDs, and "
            f"{_format_num(targets)} targets"
        )

    if pos == "TE":
        rec_yards = _safe_float(_get(row, "rec_yards", 0))
        rec_tds = _safe_float(_get(row, "rec_tds", 0))
        targets = _safe_float(_get(row, "targets", 0))
        return (
            f"{_format_num(rec_yards)} receiving yards, "
            f"{_format_num(rec_tds)} receiving TDs, and "
            f"{_format_num(targets)} targets"
        )

    if pos == "RB":
        rush_yards = _safe_float(_get(row, "rush_yards", 0))
        rush_tds = _safe_float(_get(row, "rush_tds", 0))
        ypc = _safe_float(_get(row, "yards_per_carry", np.nan))
        if pd.isna(ypc):
            return (
                f"{_format_num(rush_yards)} rushing yards and "
                f"{_format_num(rush_tds)} rushing TDs"
            )
        return (
            f"{_format_num(rush_yards)} rushing yards, "
            f"{_format_num(rush_tds)} rushing TDs, and "
            f"{_format_num(ypc, 1)} yards per carry"
        )

    if pos == "QB":
        pass_yards = _safe_float(_get(row, "pass_yards", 0))
        pass_tds = _safe_float(_get(row, "pass_tds", 0))
        completion_pct = _safe_float(_get(row, "completion_pct", np.nan))
        if pd.isna(completion_pct):
            return (
                f"{_format_num(pass_yards)} passing yards and "
                f"{_format_num(pass_tds)} passing TDs"
            )
        return (
            f"{_format_num(pass_yards)} passing yards, "
            f"{_format_num(pass_tds)} passing TDs, and "
            f"{_format_num(completion_pct, 1)}% completion"
        )

    rec_yards = _safe_float(_get(row, "rec_yards", 0))
    rush_yards = _safe_float(_get(row, "rush_yards", 0))
    pass_yards = _safe_float(_get(row, "pass_yards", 0))
    return (
        f"{_format_num(pass_yards)} passing yards, "
        f"{_format_num(rush_yards)} rushing yards, and "
        f"{_format_num(rec_yards)} receiving yards"
    )


def _performance_summary(row: pd.Series) -> str:
    pos = _position(row)
    metrics_text = _pick_best_metric_phrase(row)

    if pos in {"WR", "TE"}:
        return f"As a {pos}, the key production indicators are {metrics_text}."
    if pos == "RB":
        return f"As a running back, the key production indicators are {metrics_text}."
    if pos == "QB":
        return f"As a quarterback, the key production indicators are {metrics_text}."
    return f"Key production indicators include {metrics_text}."


def _value_summary(row: pd.Series) -> str:
    predicted = _safe_float(_get(row, "predicted_apy", np.nan))
    actual = _safe_float(_get(row, "actual_apy", np.nan))
    gap = _safe_float(_get(row, "value_gap", predicted - actual))

    return (
        f"The model estimates this player at {_format_money_m(predicted)} "
        f"versus an actual APY of {_format_money_m(actual)}, "
        f"creating a value gap of {_format_money_m(gap)} and suggesting the player is "
        f"{_value_label(gap)}."
    )


def _availability_summary(row: pd.Series) -> str:
    games_played = _safe_float(_get(row, "games_played", 0))
    return (
        f"He played {_format_num(games_played)} games, which indicates "
        f"{_availability_band(games_played)} availability."
    )


def explain_player(player_row: pd.Series | dict) -> str:
    row = pd.Series(player_row) if isinstance(player_row, dict) else player_row

    name = _player_name(row)
    intro = f"{name} is a recommended target."

    parts = [
        intro,
        _value_summary(row),
        _availability_summary(row),
        _performance_summary(row),
    ]
    return " ".join(parts)


def compare_players(player_a: pd.Series | dict, player_b: pd.Series | dict) -> str:
    a = pd.Series(player_a) if isinstance(player_a, dict) else player_a
    b = pd.Series(player_b) if isinstance(player_b, dict) else player_b

    a_name = _player_name(a)
    b_name = _player_name(b)

    a_actual = _safe_float(_get(a, "actual_apy", np.nan))
    b_actual = _safe_float(_get(b, "actual_apy", np.nan))

    a_gap = _safe_float(_get(a, "value_gap", np.nan))
    b_gap = _safe_float(_get(b, "value_gap", np.nan))

    a_games = _safe_float(_get(a, "games_played", 0))
    b_games = _safe_float(_get(b, "games_played", 0))

    a_rec_score = _safe_float(_get(a, "recommendation_score", np.nan))
    b_rec_score = _safe_float(_get(b, "recommendation_score", np.nan))

    if pd.isna(a_actual) or pd.isna(b_actual):
        cheaper_text = "The pricing comparison is incomplete because one player is missing APY data."
    elif a_actual < b_actual:
        cheaper_text = f"{a_name} is cheaper at {_format_money_m(a_actual)} versus {_format_money_m(b_actual)} for {b_name}."
    elif a_actual > b_actual:
        cheaper_text = f"{b_name} is cheaper at {_format_money_m(b_actual)} versus {_format_money_m(a_actual)} for {a_name}."
    else:
        cheaper_text = f"Both players are priced similarly at {_format_money_m(a_actual)}."

    if pd.isna(a_gap) or pd.isna(b_gap):
        gap_text = "The value-gap comparison is incomplete because one player is missing modeled value information."
    elif a_gap > b_gap:
        gap_text = f"{a_name} shows the larger value gap ({_format_money_m(a_gap)} vs {_format_money_m(b_gap)}), suggesting stronger upside relative to price."
    elif a_gap < b_gap:
        gap_text = f"{b_name} shows the larger value gap ({_format_money_m(b_gap)} vs {_format_money_m(a_gap)}), suggesting stronger upside relative to price."
    else:
        gap_text = f"Both players have a similar value gap of around {_format_money_m(a_gap)}."

    if a_games > b_games:
        availability_text = f"{a_name} has been more available, playing {_format_num(a_games)} games versus {_format_num(b_games)} for {b_name}."
    elif a_games < b_games:
        availability_text = f"{b_name} has been more available, playing {_format_num(b_games)} games versus {_format_num(a_games)} for {a_name}."
    else:
        availability_text = f"Both players show similar availability at {_format_num(a_games)} games played."

    if not pd.isna(a_rec_score) and not pd.isna(b_rec_score):
        if a_rec_score > b_rec_score:
            performance_text = f"Overall, {a_name} grades out better in the recommendation model."
        elif a_rec_score < b_rec_score:
            performance_text = f"Overall, {b_name} grades out better in the recommendation model."
        else:
            performance_text = "Overall, the recommendation model views the two players similarly."
    else:
        performance_text = (
            f"In terms of raw production, {a_name} posted {_pick_best_metric_phrase(a)}, "
            f"while {b_name} posted {_pick_best_metric_phrase(b)}."
        )

    return " ".join([
        cheaper_text,
        gap_text,
        availability_text,
        performance_text,
    ])


if __name__ == "__main__":
    sample_wr = {
        "player_name": "Player A",
        "position": "WR",
        "actual_apy": 6,
        "predicted_apy": 8.2,
        "value_gap": 2.2,
        "games_played": 15,
        "rec_yards": 980,
        "rec_tds": 7,
        "targets": 118,
        "recommendation_score": 84,
    }

    sample_wr_2 = {
        "player_name": "Player B",
        "position": "WR",
        "actual_apy": 7.5,
        "predicted_apy": 8.0,
        "value_gap": 0.5,
        "games_played": 12,
        "rec_yards": 870,
        "rec_tds": 5,
        "targets": 101,
        "recommendation_score": 76,
    }

    print(explain_player(sample_wr))
    print()
    print(compare_players(sample_wr, sample_wr_2))
