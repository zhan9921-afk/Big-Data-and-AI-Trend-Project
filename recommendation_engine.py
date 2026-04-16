"""
recommendation_engine.py

Manager-facing recommendation engine for NFL contract value analysis.

This module is intentionally focused on:
1. filtering candidate players
2. scoring/ranking players
3. returning shortlist outputs

It does NOT include LLM explanation logic.
That should live in a separate app/service layer.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Core validation / helpers
# -------------------------------------------------------------------

REQUIRED_BASE_COLUMNS = {
    "player_id",
    "player_name",
    "position",
    "season",
    "games_played",
    "actual_apy",
    "predicted_apy",
    "value_gap",
}

OPTIONAL_STAT_COLUMNS = {
    "pass_yards",
    "pass_tds",
    "rush_yards",
    "rush_tds",
    "rec_yards",
    "rec_tds",
    "targets",
    "receptions",
}


def validate_input_df(df: pd.DataFrame) -> None:
    """
    Validate that the input dataframe contains the minimum columns needed
    by the recommendation engine.
    """
    missing = REQUIRED_BASE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Input dataframe is missing required columns: "
            + ", ".join(sorted(missing))
        )


def copy_df(df: pd.DataFrame) -> pd.DataFrame:
    """Safe shallow wrapper for consistency."""
    return df.copy()


def to_numeric_if_exists(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """
    Convert listed columns to numeric when they exist.
    Non-convertible values become NaN.
    """
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def min_max_scale(
    series: pd.Series,
    fill_value: float = 50.0,
    clip_range: tuple[float, float] = (0.0, 100.0),
) -> pd.Series:
    """
    Scale a numeric series into a 0-100 range.
    If the series is constant or empty, return a constant fill_value.
    """
    s = pd.to_numeric(series, errors="coerce")

    if s.notna().sum() == 0:
        return pd.Series(fill_value, index=series.index, dtype=float)

    s_min = s.min()
    s_max = s.max()

    if pd.isna(s_min) or pd.isna(s_max) or s_min == s_max:
        return pd.Series(fill_value, index=series.index, dtype=float)

    scaled = 100 * (s - s_min) / (s_max - s_min)
    return scaled.clip(lower=clip_range[0], upper=clip_range[1])


# -------------------------------------------------------------------
# Filtering
# -------------------------------------------------------------------

def filter_players(
    df: pd.DataFrame,
    position: Optional[Union[str, list[str], tuple[str, ...], set[str]]] = None,
    max_salary: Optional[float] = None,
    min_games: Optional[int] = None,
    season: Optional[int] = None,
    objective: str = "undervalued",
    min_value_gap: Optional[float] = None,
    max_value_gap: Optional[float] = None,
) -> pd.DataFrame:
    """
    Filter player pool based on manager constraints.

    Parameters
    ----------
    df : pd.DataFrame
        Input shortlist base table.
    position : str or list-like, optional
        Example: "RB" or ["WR", "TE"].
    max_salary : float, optional
        Maximum actual APY the manager is willing to pay.
    min_games : int, optional
        Minimum games played.
    season : int, optional
        Restrict to a specific season.
    objective : str
        One of:
        - "undervalued": keep only positive value_gap
        - "overvalued": keep only negative value_gap
        - "balanced": do not filter on value_gap sign
    min_value_gap : float, optional
        Additional lower bound on value_gap.
    max_value_gap : float, optional
        Additional upper bound on value_gap.

    Returns
    -------
    pd.DataFrame
    """
    validate_input_df(df)
    out = copy_df(df)

    # normalize data types for safe filtering
    out = to_numeric_if_exists(
        out,
        [
            "season",
            "games_played",
            "actual_apy",
            "predicted_apy",
            "value_gap",
        ],
    )

    if position is not None:
        if isinstance(position, (list, tuple, set)):
            valid_positions = set(position)
            out = out[out["position"].isin(valid_positions)]
        else:
            out = out[out["position"] == position]

    if max_salary is not None:
        out = out[out["actual_apy"].fillna(np.inf) <= float(max_salary)]

    if min_games is not None:
        out = out[out["games_played"].fillna(0) >= int(min_games)]

    if season is not None:
        out = out[out["season"] == int(season)]

    objective = (objective or "balanced").lower()

    if objective == "undervalued":
        out = out[out["value_gap"] > 0]
    elif objective == "overvalued":
        out = out[out["value_gap"] < 0]
    elif objective == "balanced":
        pass
    else:
        raise ValueError(
            "objective must be one of: 'undervalued', 'overvalued', 'balanced'"
        )

    if min_value_gap is not None:
        out = out[out["value_gap"].fillna(-np.inf) >= float(min_value_gap)]

    if max_value_gap is not None:
        out = out[out["value_gap"].fillna(np.inf) <= float(max_value_gap)]

    return out.reset_index(drop=True)


# -------------------------------------------------------------------
# Scoring logic
# -------------------------------------------------------------------

def _compute_qb_raw_score(df: pd.DataFrame) -> pd.Series:
    return (
        0.04 * df.get("pass_yards", 0).fillna(0)
        + 6.0 * df.get("pass_tds", 0).fillna(0)
        + 0.01 * df.get("rush_yards", 0).fillna(0)
        + 6.0 * df.get("rush_tds", 0).fillna(0)
    )


def _compute_rb_raw_score(df: pd.DataFrame) -> pd.Series:
    return (
        0.08 * df.get("rush_yards", 0).fillna(0)
        + 6.0 * df.get("rush_tds", 0).fillna(0)
        + 0.05 * df.get("rec_yards", 0).fillna(0)
        + 0.5 * df.get("receptions", 0).fillna(0)
        + 0.15 * df.get("targets", 0).fillna(0)
    )


def _compute_wr_raw_score(df: pd.DataFrame) -> pd.Series:
    return (
        0.08 * df.get("rec_yards", 0).fillna(0)
        + 6.0 * df.get("rec_tds", 0).fillna(0)
        + 0.6 * df.get("receptions", 0).fillna(0)
        + 0.2 * df.get("targets", 0).fillna(0)
    )


def _compute_te_raw_score(df: pd.DataFrame) -> pd.Series:
    return (
        0.08 * df.get("rec_yards", 0).fillna(0)
        + 6.0 * df.get("rec_tds", 0).fillna(0)
        + 0.7 * df.get("receptions", 0).fillna(0)
        + 0.2 * df.get("targets", 0).fillna(0)
    )


def _compute_fallback_raw_score(df: pd.DataFrame) -> pd.Series:
    return (
        0.03 * df.get("pass_yards", 0).fillna(0)
        + 0.05 * df.get("rush_yards", 0).fillna(0)
        + 0.05 * df.get("rec_yards", 0).fillna(0)
    )


def compute_performance_score(df: pd.DataFrame) -> pd.Series:
    """
    Position-aware performance score on a 0-100 scale.

    This is intentionally heuristic and product-facing:
    - QB emphasizes passing
    - RB emphasizes rushing + some receiving
    - WR/TE emphasize receiving volume/output
    """
    out = pd.Series(index=df.index, dtype=float)

    qb_mask = df["position"] == "QB"
    rb_mask = df["position"] == "RB"
    wr_mask = df["position"] == "WR"
    te_mask = df["position"] == "TE"
    other_mask = ~(qb_mask | rb_mask | wr_mask | te_mask)

    if qb_mask.any():
        out.loc[qb_mask] = min_max_scale(_compute_qb_raw_score(df.loc[qb_mask]))

    if rb_mask.any():
        out.loc[rb_mask] = min_max_scale(_compute_rb_raw_score(df.loc[rb_mask]))

    if wr_mask.any():
        out.loc[wr_mask] = min_max_scale(_compute_wr_raw_score(df.loc[wr_mask]))

    if te_mask.any():
        out.loc[te_mask] = min_max_scale(_compute_te_raw_score(df.loc[te_mask]))

    if other_mask.any():
        out.loc[other_mask] = min_max_scale(
            _compute_fallback_raw_score(df.loc[other_mask])
        )

    return out.fillna(0.0)


def compute_availability_score(df: pd.DataFrame, max_games: int = 17) -> pd.Series:
    """
    Availability score on a 0-100 scale based on games played.
    """
    gp = pd.to_numeric(df["games_played"], errors="coerce").fillna(0).clip(0, max_games)
    return 100 * gp / max_games


def compute_value_gap_score(
    df: pd.DataFrame,
    objective: str = "undervalued",
) -> pd.Series:
    """
    Convert value_gap into a 0-100 score.

    For undervalued search:
        larger positive value_gap => higher score

    For overvalued search:
        more negative value_gap => higher score
        (we invert sign so the most overpriced gets highest score)

    For balanced search:
        still use raw value_gap, favoring undervalued players by default.
    """
    objective = (objective or "balanced").lower()

    if objective == "overvalued":
        return min_max_scale(-df["value_gap"])
    return min_max_scale(df["value_gap"])


def add_scoring_columns(
    df: pd.DataFrame,
    objective: str = "undervalued",
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Add scoring columns used for ranking.

    Default recommendation score:
        0.60 * value_gap_score
      + 0.25 * performance_score
      + 0.15 * availability_score

    Parameters
    ----------
    df : pd.DataFrame
    objective : str
        'undervalued', 'overvalued', or 'balanced'
    weights : dict, optional
        Override default weights.
        Example:
        {
            "value_gap": 0.5,
            "performance": 0.3,
            "availability": 0.2
        }
    """
    validate_input_df(df)
    out = copy_df(df)

    out = to_numeric_if_exists(
        out,
        [
            "season",
            "games_played",
            "actual_apy",
            "predicted_apy",
            "value_gap",
            *OPTIONAL_STAT_COLUMNS,
        ],
    )

    if "value_gap" not in out.columns:
        out["value_gap"] = out["predicted_apy"] - out["actual_apy"]

    weights = weights or {
        "value_gap": 0.60,
        "performance": 0.25,
        "availability": 0.15,
    }

    total_weight = (
        weights.get("value_gap", 0)
        + weights.get("performance", 0)
        + weights.get("availability", 0)
    )
    if total_weight <= 0:
        raise ValueError("At least one scoring weight must be positive.")

    out["value_gap_score"] = compute_value_gap_score(out, objective=objective)
    out["performance_score"] = compute_performance_score(out)
    out["availability_score"] = compute_availability_score(out)

    out["recommendation_score"] = (
        weights.get("value_gap", 0) * out["value_gap_score"]
        + weights.get("performance", 0) * out["performance_score"]
        + weights.get("availability", 0) * out["availability_score"]
    ) / total_weight

    # Useful downstream business flags
    out["budget_status"] = np.where(
        out["value_gap"] > 0, "Below modeled value",
        np.where(out["value_gap"] < 0, "Above modeled value", "At modeled value")
    )

    out["availability_band"] = np.select(
        [
            out["games_played"].fillna(0) >= 14,
            out["games_played"].fillna(0) >= 10,
        ],
        [
            "High",
            "Medium",
        ],
        default="Low",
    )

    out["risk_flag"] = np.select(
        [
            out["games_played"].fillna(0) < 8,
            out["value_gap"] < 0,
        ],
        [
            "Availability risk",
            "Pricing risk",
        ],
        default="Low",
    )

    return out


# -------------------------------------------------------------------
# Ranking / shortlist
# -------------------------------------------------------------------

def rank_players(
    df: pd.DataFrame,
    objective: str = "undervalued",
) -> pd.DataFrame:
    """
    Sort candidate players for manager-facing output.
    """
    out = copy_df(df)
    objective = (objective or "balanced").lower()

    if objective == "undervalued":
        out = out.sort_values(
            by=["recommendation_score", "value_gap", "performance_score", "games_played"],
            ascending=[False, False, False, False],
        )
    elif objective == "overvalued":
        out = out.sort_values(
            by=["recommendation_score", "value_gap", "actual_apy"],
            ascending=[False, True, False],
        )
    elif objective == "balanced":
        out = out.sort_values(
            by=["recommendation_score", "performance_score", "games_played"],
            ascending=[False, False, False],
        )
    else:
        raise ValueError(
            "objective must be one of: 'undervalued', 'overvalued', 'balanced'"
        )

    return out.reset_index(drop=True)


def get_shortlist(
    df: pd.DataFrame,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    End-to-end shortlist pipeline.

    Example
    -------
    filters = {
        "position": "RB",
        "max_salary": 6,
        "min_games": 8,
        "season": 2023,
        "objective": "undervalued"
    }
    """
    if filters is None:
        filters = {}

    objective = filters.get("objective", "undervalued")

    filtered = filter_players(
        df=df,
        position=filters.get("position"),
        max_salary=filters.get("max_salary"),
        min_games=filters.get("min_games"),
        season=filters.get("season"),
        objective=objective,
        min_value_gap=filters.get("min_value_gap"),
        max_value_gap=filters.get("max_value_gap"),
    )

    if filtered.empty:
        return filtered

    scored = add_scoring_columns(
        filtered,
        objective=objective,
        weights=weights,
    )

    ranked = rank_players(scored, objective=objective)
    shortlist = ranked.head(int(top_k)).copy()
    shortlist.reset_index(drop=True, inplace=True)
    return shortlist


# -------------------------------------------------------------------
# Optional utility outputs for product / dashboard use
# -------------------------------------------------------------------

def summarize_candidate_pool(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return a compact summary of the candidate pool for dashboard/header use.
    """
    validate_input_df(df)

    summary = {
        "n_players": int(len(df)),
        "positions": sorted(df["position"].dropna().astype(str).unique().tolist()),
        "seasons": sorted(pd.to_numeric(df["season"], errors="coerce").dropna().astype(int).unique().tolist()),
    }

    if "actual_apy" in df.columns:
        actual = pd.to_numeric(df["actual_apy"], errors="coerce")
        summary["actual_apy_min"] = float(actual.min()) if actual.notna().any() else None
        summary["actual_apy_median"] = float(actual.median()) if actual.notna().any() else None
        summary["actual_apy_max"] = float(actual.max()) if actual.notna().any() else None

    if "value_gap" in df.columns:
        vg = pd.to_numeric(df["value_gap"], errors="coerce")
        summary["value_gap_min"] = float(vg.min()) if vg.notna().any() else None
        summary["value_gap_median"] = float(vg.median()) if vg.notna().any() else None
        summary["value_gap_max"] = float(vg.max()) if vg.notna().any() else None

    return summary


def get_ranked_table(
    df: pd.DataFrame,
    filters: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Return the full ranked table instead of only top_k shortlist.
    Useful for Streamlit tables and downloadable CSVs.
    """
    if filters is None:
        filters = {}

    objective = filters.get("objective", "undervalued")

    filtered = filter_players(
        df=df,
        position=filters.get("position"),
        max_salary=filters.get("max_salary"),
        min_games=filters.get("min_games"),
        season=filters.get("season"),
        objective=objective,
        min_value_gap=filters.get("min_value_gap"),
        max_value_gap=filters.get("max_value_gap"),
    )

    if filtered.empty:
        return filtered

    scored = add_scoring_columns(
        filtered,
        objective=objective,
        weights=weights,
    )

    ranked = rank_players(scored, objective=objective)
    return ranked.reset_index(drop=True)


if __name__ == "__main__":
    # Minimal example for quick local testing.
    example_path = "shortlist_base.csv"
    try:
        demo_df = pd.read_csv(example_path)

        filters = {
            "position": "RB",
            "max_salary": 10,
            "min_games": 8,
            "objective": "undervalued",
        }

        shortlist = get_shortlist(demo_df, filters=filters, top_k=5)
        print(shortlist.head())
    except FileNotFoundError:
        print(
            "Test file not found. Place shortlist_base.csv in the current directory "
            "to run the local example."
        )
