
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

PRIMARY_POSITIONS = {"QB", "RB", "WR", "TE"}

TEAM_ALIASES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LAC": "Los Angeles Chargers", "LAR": "Los Angeles Rams",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}

def _find_file(base_dir: Path, candidates: Iterable[str]) -> Optional[Path]:
    search_dirs = [base_dir / "output", base_dir, Path("/mnt/data")]
    for directory in search_dirs:
        for name in candidates:
            path = directory / name
            if path.exists():
                return path
    return None

def _read_csv(path: Optional[Path]) -> pd.DataFrame:
    if path and path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

def _norm_name(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9 ]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

def _ensure_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out

def _first_existing(df: pd.DataFrame, names: list[str], default=np.nan) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(default, index=df.index)

def _add_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    feature_cols = [
        "games_played", "pass_yards", "pass_tds", "completion_pct", "yards_per_attempt", "pass_epa_per_play",
        "rush_yards", "rush_tds", "yards_per_carry", "rush_epa_per_play", "explosive_run_rate",
        "rec_yards", "rec_tds", "catch_rate", "yards_per_rec", "rec_epa_per_play",
    ]
    out = _ensure_numeric(out, feature_cols)
    for col in feature_cols:
        out[f"{col}_pct"] = out.groupby("position")[col].rank(pct=True).fillna(0.0)
    return out

def _normalize_model_results(model_df: pd.DataFrame) -> pd.DataFrame:
    if model_df.empty:
        return pd.DataFrame()
    out = model_df.copy().rename(columns={
        "Actual_APY": "model_actual_apy",
        "Pred_APY": "model_predicted_apy",
        "Actual_Years": "model_actual_years",
        "Pred_Years": "model_pred_years",
        "Actual_Guar": "model_actual_guar",
        "Pred_Guar": "model_pred_guar",
        "APY_Error": "model_value_gap",
    })
    out["player_norm"] = _norm_name(out["player"])
    out["position"] = out["position"].astype(str).str.upper()
    out["next_year_signed"] = pd.to_numeric(out.get("next_year_signed"), errors="coerce")
    return out

def _prepare_season_rows(dataset_df: pd.DataFrame, model_df: pd.DataFrame) -> pd.DataFrame:
    if dataset_df.empty:
        return pd.DataFrame()

    out = dataset_df.copy()
    out["player"] = _first_existing(out, ["player", "player_name", "display_name"]).astype(str)
    out["player_name"] = out["player"]
    out["player_norm"] = _norm_name(out["player"])
    out["position"] = out["position"].astype(str).str.upper()
    out["season"] = pd.to_numeric(out.get("season"), errors="coerce")
    out["next_year_signed"] = pd.to_numeric(out.get("next_year_signed"), errors="coerce")

    # Standard season-level stats and contract fields.
    out["current_apy"] = pd.to_numeric(_first_existing(out, ["curr_apy", "curr_inflated_apy"]), errors="coerce")
    out["actual_apy"] = pd.to_numeric(_first_existing(out, ["next_inflated_apy", "next_apy", "Actual_APY"]), errors="coerce")
    out["predicted_apy"] = pd.to_numeric(_first_existing(out, ["Pred_APY"]), errors="coerce")
    out["actual_years"] = pd.to_numeric(_first_existing(out, ["next_years", "Actual_Years"]), errors="coerce")
    out["pred_years"] = pd.to_numeric(_first_existing(out, ["Pred_Years"]), errors="coerce")
    out["actual_guar"] = pd.to_numeric(_first_existing(out, ["next_inflated_guaranteed", "next_guaranteed", "Actual_Guar"]), errors="coerce")
    out["pred_guar"] = pd.to_numeric(_first_existing(out, ["Pred_Guaranteed", "Pred_Guar"]), errors="coerce")
    out["curr_years"] = pd.to_numeric(_first_existing(out, ["curr_years"]), errors="coerce")
    out["curr_guaranteed"] = pd.to_numeric(_first_existing(out, ["curr_guaranteed", "curr_inflated_guaranteed"]), errors="coerce")

    # Merge model rows to fill predictions where dataset has the same player/signing record.
    model_norm = _normalize_model_results(model_df)
    if not model_norm.empty:
        mcols = [
            "player_norm", "next_year_signed", "model_actual_apy", "model_predicted_apy",
            "model_actual_years", "model_pred_years", "model_actual_guar", "model_pred_guar",
        ]
        mcols = [c for c in mcols if c in model_norm.columns]
        out = out.merge(model_norm[mcols].drop_duplicates(["player_norm", "next_year_signed"]),
                        on=["player_norm", "next_year_signed"], how="left")
        out["actual_apy"] = out["actual_apy"].fillna(out.get("model_actual_apy"))
        out["predicted_apy"] = out["predicted_apy"].fillna(out.get("model_predicted_apy"))
        out["actual_years"] = out["actual_years"].fillna(out.get("model_actual_years"))
        out["pred_years"] = out["pred_years"].fillna(out.get("model_pred_years"))
        out["actual_guar"] = out["actual_guar"].fillna(out.get("model_actual_guar"))
        out["pred_guar"] = out["pred_guar"].fillna(out.get("model_pred_guar"))

    # Some current-season rows do not have a fresh model prediction but do carry the signed next
    # contract fields. Keep those rows visible for "making over/at least $X in YEAR" queries.
    # Do NOT overwrite model predictions; only provide display fallback columns.
    out["actual_total_value"] = pd.to_numeric(_first_existing(out, ["Actual_Total_Val", "next_inflated_value", "next_value"]), errors="coerce")
    out["predicted_total_value"] = pd.to_numeric(_first_existing(out, ["Pred_Total_Val"]), errors="coerce")
    out["current_total_value"] = out["current_apy"] * out["curr_years"]
    out["value_gap"] = out["predicted_apy"] - out["current_apy"]

    # Valuation is now based on current APY vs predicted APY so current-season queries work:
    # (Current APY - Predicted APY) / Current APY. Negative => under market.
    denom = out["current_apy"].replace(0, np.nan)
    out["valuation_pct"] = (out["current_apy"] - out["predicted_apy"]) / denom
    out["valuation_status"] = "Market Value"
    out.loc[out["valuation_pct"] > 0.15, "valuation_status"] = "Overvalued"
    out.loc[out["valuation_pct"] < -0.15, "valuation_status"] = "Undervalued"
    out.loc[out["valuation_pct"].isna(), "valuation_status"] = "Unscored"
    out["verdict"] = out["valuation_status"].replace({"Market Value": "Fair", "Unscored": "Fair"})
    out["contract_accuracy_abs"] = out["value_gap"].abs()
    out["confidence_pct"] = np.clip(
        55 + 25 * pd.to_numeric(out.get("games_played"), errors="coerce").fillna(0).rank(pct=True).fillna(0)
        + 16 * (1 - out["contract_accuracy_abs"].rank(pct=True).fillna(0)),
        55, 96,
    )

    stat_cols = [
        "games_played", "pass_yards", "pass_tds", "completion_pct", "yards_per_attempt", "pass_epa_per_play",
        "rush_yards", "rush_tds", "yards_per_carry", "rush_epa_per_play", "explosive_run_rate",
        "rec_yards", "rec_tds", "catch_rate", "yards_per_rec", "rec_epa_per_play",
    ]
    out = _ensure_numeric(out, stat_cols)
    out["total_tds"] = out[[c for c in ["pass_tds", "rush_tds", "rec_tds"] if c in out.columns]].fillna(0).sum(axis=1)

    if "team" in out.columns:
        out["team"] = out["team"].replace(TEAM_ALIASES)
    else:
        out["team"] = np.nan
    out["headshot_url"] = np.nan
    return out

def _load_optional_maps(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    team_map = _read_csv(_find_file(base_dir, ["player_team_map.csv", "player_team_map(1).csv"]))
    roster_detail = _read_csv(_find_file(base_dir, ["team_roster_detail.csv", "team_roster_detail(1).csv"]))
    team_summary = _read_csv(_find_file(base_dir, ["team_summary.csv", "team_summary(1).csv"]))

    for frame in [team_map, roster_detail, team_summary]:
        if not frame.empty and "team" in frame.columns:
            frame["team"] = frame["team"].replace(TEAM_ALIASES)

    if not team_map.empty:
        if "player_name" in team_map.columns and "player" not in team_map.columns:
            team_map = team_map.rename(columns={"player_name": "player"})
        if "player" in team_map.columns:
            team_map["player_norm"] = _norm_name(team_map["player"])
        if "season" in team_map.columns:
            team_map["season"] = pd.to_numeric(team_map["season"], errors="coerce")

    if not roster_detail.empty:
        if "player_norm" not in roster_detail.columns:
            src = "player_name" if "player_name" in roster_detail.columns else "player"
            if src in roster_detail.columns:
                roster_detail["player_norm"] = _norm_name(roster_detail[src])
        if "season" in roster_detail.columns:
            roster_detail["season"] = pd.to_numeric(roster_detail["season"], errors="coerce")

    return team_map, roster_detail, team_summary

def _merge_team_and_headshots(scored: pd.DataFrame, team_map: pd.DataFrame, roster_detail: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()

    if not team_map.empty and "player_norm" in team_map.columns:
        cols = [c for c in ["player_norm", "team", "season", "player_id"] if c in team_map.columns]
        tm = team_map[cols].drop_duplicates()
        if "season" in tm.columns and "season" in out.columns:
            out = out.merge(tm, on=["player_norm", "season"], how="left", suffixes=("", "_map"))
        else:
            out = out.merge(tm.drop(columns=[c for c in ["season"] if c in tm.columns]), on="player_norm", how="left", suffixes=("", "_map"))
        if "team_map" in out.columns:
            out["team"] = out["team"].fillna(out["team_map"])
            out = out.drop(columns=["team_map"])
        if "player_id_map" in out.columns:
            out["player_id"] = out.get("player_id", pd.Series(np.nan, index=out.index)).fillna(out["player_id_map"])
            out = out.drop(columns=["player_id_map"])

    if not roster_detail.empty and "player_norm" in roster_detail.columns:
        cols = [c for c in ["player_norm", "team", "season", "headshot_url"] if c in roster_detail.columns]
        rd = roster_detail[cols].drop_duplicates()
        if "season" in rd.columns and "season" in out.columns:
            out = out.merge(rd, on=["player_norm", "season"], how="left", suffixes=("", "_rd"))
        else:
            out = out.merge(rd.drop(columns=[c for c in ["season"] if c in rd.columns]), on="player_norm", how="left", suffixes=("", "_rd"))
        if "team_rd" in out.columns:
            out["team"] = out["team"].fillna(out["team_rd"])
            out = out.drop(columns=["team_rd"])
        if "headshot_url_rd" in out.columns:
            out["headshot_url"] = out["headshot_url"].fillna(out["headshot_url_rd"])
            out = out.drop(columns=["headshot_url_rd"])

    out["team"] = out["team"].replace(TEAM_ALIASES)
    return out

def _build_team_summary(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty or "team" not in scored.columns:
        return pd.DataFrame()
    out = scored.dropna(subset=["team"]).copy()
    return (
        out.groupby("team", as_index=False)
        .agg(
            players=("player_name", "count"),
            avg_gap=("value_gap", "mean"),
            total_gap=("value_gap", "sum"),
            avg_actual_apy=("actual_apy", "mean"),
            avg_predicted_apy=("predicted_apy", "mean"),
            avg_valuation_pct=("valuation_pct", "mean"),
            n_undervalued=("valuation_status", lambda s: int((s == "Undervalued").sum())),
            n_overvalued=("valuation_status", lambda s: int((s == "Overvalued").sum())),
        )
        .sort_values("avg_valuation_pct", ascending=True)
    )

def load_dashboard_data(base_dir: Path) -> Dict[str, object]:
    model_path = _find_file(base_dir, ["dashboard_model_results.csv", "dashboard_model_results(2).csv", "shortlist_result.csv"])
    dataset_path = _find_file(base_dir, ["dataset.csv", "dataset(6).csv", "dataset(3).csv", "season_features.csv", "season_features(3).csv"])
    contracts_path = _find_file(base_dir, ["contracts_clean.csv", "contracts_clean(5).csv", "contracts_clean(3).csv"])

    model_raw = _read_csv(model_path)
    dataset_raw = _read_csv(dataset_path)
    contracts_raw = _read_csv(contracts_path)
    team_map, roster_detail, team_summary_file = _load_optional_maps(base_dir)

    scored = _prepare_season_rows(dataset_raw, model_raw)
    if scored.empty:
        raise FileNotFoundError("No usable season-level dataset was found. Expected dataset.csv or dataset(6).csv.")

    scored = _merge_team_and_headshots(scored, team_map, roster_detail)
    scored = scored[scored["position"].isin(PRIMARY_POSITIONS)].copy()
    scored = _add_percentiles(scored)

    keep_cols = [
        "player", "player_name", "player_id", "player_norm", "position", "season", "next_year_signed",
        "games_played", "current_apy", "actual_apy", "predicted_apy", "value_gap", "contract_accuracy_abs",
        "actual_years", "pred_years", "actual_guar", "pred_guar", "curr_years", "curr_guaranteed", "confidence_pct", "verdict",
        "valuation_pct", "valuation_status", "predicted_total_value", "actual_total_value", "current_total_value",
        "pass_yards", "pass_tds", "completion_pct", "yards_per_attempt", "pass_epa_per_play",
        "rush_yards", "rush_tds", "yards_per_carry", "rush_epa_per_play", "explosive_run_rate",
        "rec_yards", "rec_tds", "catch_rate", "yards_per_rec", "rec_epa_per_play", "total_tds",
        "pass_yards_pct", "pass_tds_pct", "completion_pct_pct", "yards_per_attempt_pct", "pass_epa_per_play_pct",
        "rush_yards_pct", "rush_tds_pct", "yards_per_carry_pct", "rush_epa_per_play_pct", "explosive_run_rate_pct",
        "rec_yards_pct", "rec_tds_pct", "catch_rate_pct", "yards_per_rec_pct", "rec_epa_per_play_pct",
        "games_played_pct", "team", "headshot_url",
    ]
    for col in keep_cols:
        if col not in scored.columns:
            scored[col] = np.nan
    scored = scored[keep_cols].copy()
    scored = scored.sort_values(["season", "valuation_pct", "current_apy"], ascending=[False, True, False]).reset_index(drop=True)

    contract_history = contracts_raw.copy()
    if not contract_history.empty and "player" in contract_history.columns:
        contract_history["player_norm"] = _norm_name(contract_history["player"])
        for col in ["apy", "year_signed"]:
            if col in contract_history.columns:
                contract_history[col] = pd.to_numeric(contract_history[col], errors="coerce")

    team_summary = team_summary_file if not team_summary_file.empty else _build_team_summary(scored)
    team_available = "team" in scored.columns and scored["team"].notna().any()

    data_sources = {
        "dashboard_model_results": str(model_path) if model_path else None,
        "dataset": str(dataset_path) if dataset_path else None,
        "contracts_clean": str(contracts_path) if contracts_path else None,
        "player_team_map": str(_find_file(base_dir, ["player_team_map.csv", "player_team_map(1).csv"])) if team_available else None,
        "team_roster_detail": str(_find_file(base_dir, ["team_roster_detail.csv", "team_roster_detail(1).csv"])) if not roster_detail.empty else None,
    }

    return {
        "scored_players": scored,
        "contract_history": contract_history,
        "team_summary": team_summary,
        "team_available": team_available,
        "data_source": data_sources,
        "data_note": "Using full season-level dataset plus model predictions where available.",
        "unused_files_note": "Season-level dataset is now the primary dashboard table so current-year salary queries can return all relevant players.",
    }
