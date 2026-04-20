
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


PRIMARY_POSITIONS = {"QB", "RB", "WR", "TE"}
TEAM_ALIASES = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams",
    "LV": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}


def _find_file(base_dir: Path, candidates: Iterable[str]) -> Optional[Path]:
    output_dir = base_dir / "output"
    search_dirs = [output_dir, base_dir, Path("/mnt/data")]
    for directory in search_dirs:
        for name in candidates:
            path = directory / name
            if path.exists():
                return path
    return None


def _read_csv(path: Optional[Path]) -> pd.DataFrame:
    return pd.read_csv(path) if path and path.exists() else pd.DataFrame()


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
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out


def _add_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    feature_cols = [
        "games_played", "pass_yards", "pass_tds", "completion_pct", "yards_per_attempt", "pass_epa_per_play",
        "rush_yards", "rush_tds", "yards_per_carry", "rush_epa_per_play", "explosive_run_rate",
        "rec_yards", "rec_tds", "catch_rate", "yards_per_rec", "rec_epa_per_play"
    ]
    out = _ensure_numeric(out, feature_cols)
    for col in feature_cols:
        out[f"{col}_pct"] = out.groupby("position")[col].rank(pct=True).fillna(0.0)
    return out


def _normalize_model_results(model_df: pd.DataFrame) -> pd.DataFrame:
    if model_df.empty:
        return pd.DataFrame(columns=[
            "player", "position", "next_year_signed", "actual_apy", "predicted_apy",
            "actual_years", "pred_years", "actual_guar", "pred_guar", "value_gap"
        ])
    out = model_df.copy()
    rename_map = {
        "Actual_APY": "actual_apy",
        "Pred_APY": "predicted_apy",
        "Actual_Years": "actual_years",
        "Pred_Years": "pred_years",
        "Actual_Guar": "actual_guar",
        "Pred_Guar": "pred_guar",
        "APY_Error": "value_gap",
    }
    out = out.rename(columns=rename_map)
    keep = ["player", "position", "next_year_signed", "actual_apy", "predicted_apy",
            "actual_years", "pred_years", "actual_guar", "pred_guar", "value_gap"]
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan
    out = _ensure_numeric(out, [c for c in keep if c not in {"player", "position"}])
    out["player_norm"] = _norm_name(out["player"])
    out["position"] = out["position"].astype(str).str.upper()
    out["next_year_signed"] = pd.to_numeric(out["next_year_signed"], errors="coerce")
    out["value_gap"] = out["predicted_apy"] - out["actual_apy"]
    return out[keep + ["player_norm"]]


def _prepare_market_rows(dataset_df: pd.DataFrame) -> pd.DataFrame:
    if dataset_df.empty:
        return pd.DataFrame()
    out = dataset_df.copy()
    out["player_norm"] = _norm_name(out["player"])
    out["position"] = out["position"].astype(str).str.upper()
    out["next_year_signed"] = pd.to_numeric(out["next_year_signed"], errors="coerce")
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out = _ensure_numeric(out, [
        "curr_apy", "next_apy", "games_played", "pass_yards", "pass_tds", "completion_pct", "yards_per_attempt", "pass_epa_per_play",
        "rush_yards", "rush_tds", "yards_per_carry", "rush_epa_per_play", "explosive_run_rate",
        "rec_yards", "rec_tds", "catch_rate", "yards_per_rec", "rec_epa_per_play", "curr_years", "curr_guaranteed"
    ])
    out = out.sort_values(["player_norm", "next_year_signed", "season"], ascending=[True, True, False])
    # keep the latest observed season before the next contract for each player/signing
    out = out.drop_duplicates(subset=["player_norm", "next_year_signed"], keep="first")
    return out


def _load_optional_maps(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    team_map = _read_csv(_find_file(base_dir, ["player_team_map.csv"]))
    roster_detail = _read_csv(_find_file(base_dir, ["team_roster_detail.csv"]))
    team_summary = _read_csv(_find_file(base_dir, ["team_summary.csv"]))

    # normalize optional map columns
    if not team_map.empty:
        rename = {}
        if "player_name" in team_map.columns and "player" not in team_map.columns:
            rename["player_name"] = "player"
        if "headshot" in team_map.columns and "headshot_url" not in team_map.columns:
            rename["headshot"] = "headshot_url"
        team_map = team_map.rename(columns=rename)
        if "player" in team_map.columns:
            team_map["player_norm"] = _norm_name(team_map["player"])
        if "team" in team_map.columns:
            team_map["team"] = team_map["team"].replace(TEAM_ALIASES)
        if "season" in team_map.columns:
            team_map["season"] = pd.to_numeric(team_map["season"], errors="coerce")
        if "player_id" not in team_map.columns:
            team_map["player_id"] = np.nan

    if not roster_detail.empty:
        if "player" in roster_detail.columns:
            roster_detail["player_norm"] = _norm_name(roster_detail["player"])
        elif "player_name" in roster_detail.columns:
            roster_detail["player_norm"] = _norm_name(roster_detail["player_name"])
        if "team" in roster_detail.columns:
            roster_detail["team"] = roster_detail["team"].replace(TEAM_ALIASES)

    if not team_summary.empty and "team" in team_summary.columns:
        team_summary["team"] = team_summary["team"].replace(TEAM_ALIASES)

    return team_map, roster_detail, team_summary


def _merge_team_and_headshots(scored: pd.DataFrame, team_map: pd.DataFrame, roster_detail: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    if not team_map.empty:
        map_cols = [c for c in ["player_norm", "player_id", "team", "season", "headshot_url"] if c in team_map.columns]
        tm = team_map[map_cols].copy()
        # prefer season-aware join when available
        if "season" in tm.columns and "season" in out.columns:
            out = out.merge(tm, on=["player_norm", "season"], how="left", suffixes=("", "_map"))
        else:
            out = out.merge(tm.drop(columns=[c for c in ["season"] if c in tm.columns]), on=["player_norm"], how="left", suffixes=("", "_map"))
        if "team_map" in out.columns:
            out["team"] = out["team"].fillna(out["team_map"])
            out = out.drop(columns=["team_map"])
        if "player_id_map" in out.columns:
            out["player_id"] = out["player_id"].fillna(out["player_id_map"])
            out = out.drop(columns=["player_id_map"])
        if "headshot_url_map" in out.columns:
            out["headshot_url"] = out["headshot_url"].fillna(out["headshot_url_map"])
            out = out.drop(columns=["headshot_url_map"])

    if not roster_detail.empty:
        rd_cols = [c for c in ["player_norm", "team", "headshot_url"] if c in roster_detail.columns]
        rd = roster_detail[rd_cols].drop_duplicates(subset=["player_norm"])
        out = out.merge(rd, on="player_norm", how="left", suffixes=("", "_rd"))
        if "team_rd" in out.columns:
            out["team"] = out["team"].fillna(out["team_rd"])
            out = out.drop(columns=["team_rd"])
        if "headshot_url_rd" in out.columns:
            out["headshot_url"] = out["headshot_url"].fillna(out["headshot_url_rd"])
            out = out.drop(columns=["headshot_url_rd"])

    return out


def _build_team_summary(scored: pd.DataFrame) -> pd.DataFrame:
    if "team" not in scored.columns or scored["team"].isna().all():
        return pd.DataFrame()
    out = scored.copy()
    out["is_undervalued"] = (out["value_gap"] > 2).astype(int)
    out["is_overvalued"] = (out["value_gap"] < -2).astype(int)
    summary = (
        out.groupby("team", as_index=False)
        .agg(
            players=("player_name", "count"),
            avg_gap=("value_gap", "mean"),
            total_gap=("value_gap", "sum"),
            avg_actual_apy=("actual_apy", "mean"),
            avg_predicted_apy=("predicted_apy", "mean"),
            n_undervalued=("is_undervalued", "sum"),
            n_overvalued=("is_overvalued", "sum"),
        )
        .sort_values("avg_gap", ascending=False)
    )
    return summary


def load_dashboard_data(base_dir: Path) -> Dict[str, object]:
    model_path = _find_file(base_dir, ["dashboard_model_results.csv", "shortlist_result.csv"])
    dataset_path = _find_file(base_dir, ["dataset(3).csv", "dataset.csv", "season_features(3).csv", "season_features.csv"])
    contracts_path = _find_file(base_dir, ["contracts_clean(3).csv", "contracts_clean.csv"])

    model_raw = _read_csv(model_path)
    dataset_raw = _read_csv(dataset_path)
    contracts_raw = _read_csv(contracts_path)

    team_map, roster_detail, team_summary_file = _load_optional_maps(base_dir)

    model_df = _normalize_model_results(model_raw)
    market_rows = _prepare_market_rows(dataset_raw)

    if model_df.empty:
        raise FileNotFoundError("dashboard_model_results.csv could not be found or was empty.")

    scored = model_df.merge(
        market_rows,
        on=["player_norm", "next_year_signed"],
        how="left",
        suffixes=("", "_market"),
    )

    # tidy fields for dashboard
    if "player_name" not in scored.columns:
        scored["player_name"] = np.nan
    scored["player_name"] = scored["player_name"].fillna(scored["player"])
    scored["player"] = scored["player"].fillna(scored["player_name"])
    scored["season"] = pd.to_numeric(scored.get("season"), errors="coerce")
    scored["current_apy"] = pd.to_numeric(scored.get("curr_apy"), errors="coerce")
    scored["curr_guaranteed"] = pd.to_numeric(scored.get("curr_guaranteed"), errors="coerce")
    scored["curr_years"] = pd.to_numeric(scored.get("curr_years"), errors="coerce")
    scored["actual_apy"] = pd.to_numeric(scored.get("actual_apy"), errors="coerce")
    scored["predicted_apy"] = pd.to_numeric(scored.get("predicted_apy"), errors="coerce")
    scored["value_gap"] = scored["predicted_apy"] - scored["actual_apy"]
    scored["contract_accuracy_abs"] = scored["value_gap"].abs()
    scored["confidence_pct"] = np.clip(55 + 25 * scored["games_played"].fillna(0).rank(pct=True).fillna(0) + 16 * (1 - scored["contract_accuracy_abs"].rank(pct=True).fillna(0)), 55, 96)
    scored["verdict"] = np.where(scored["value_gap"] > 2, "Undervalued", np.where(scored["value_gap"] < -2, "Overvalued", "Fair"))
    scored["total_tds"] = scored[[c for c in ["pass_tds", "rush_tds", "rec_tds"] if c in scored.columns]].fillna(0).sum(axis=1)
    scored = _add_percentiles(scored)

    scored["team"] = np.nan
    scored["headshot_url"] = np.nan
    scored = _merge_team_and_headshots(scored, team_map, roster_detail)

    keep_cols = [
        "player", "player_name", "player_id", "player_norm", "position", "season", "next_year_signed",
        "games_played", "current_apy", "actual_apy", "predicted_apy", "value_gap", "contract_accuracy_abs",
        "actual_years", "pred_years", "actual_guar", "pred_guar", "curr_years", "curr_guaranteed", "confidence_pct", "verdict",
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
    scored["position"] = scored["position"].astype(str).str.upper()

    # focus dashboard on skill positions by default
    scored = scored[scored["position"].isin(PRIMARY_POSITIONS)].copy()
    scored = scored.sort_values(["value_gap", "predicted_apy"], ascending=[False, False]).reset_index(drop=True)

    contract_history = contracts_raw.copy()
    if not contract_history.empty:
        contract_history["player_norm"] = _norm_name(contract_history["player"])
        contract_history["apy"] = pd.to_numeric(contract_history["apy"], errors="coerce")
        contract_history["year_signed"] = pd.to_numeric(contract_history["year_signed"], errors="coerce")
        contract_history = contract_history[contract_history["player_norm"].isin(scored["player_norm"])].copy()

    team_summary = team_summary_file if not team_summary_file.empty else _build_team_summary(scored)
    team_available = "team" in scored.columns and scored["team"].notna().any()

    data_sources = {
        "dashboard_model_results": str(model_path) if model_path else None,
        "dataset": str(dataset_path) if dataset_path else None,
        "contracts_clean": str(contracts_path) if contracts_path else None,
        "player_team_map": str(_find_file(base_dir, ["player_team_map.csv"])) if team_available else None,
        "team_summary": str(_find_file(base_dir, ["team_summary.csv"])) if not team_summary.empty else None,
        "team_roster_detail": str(_find_file(base_dir, ["team_roster_detail.csv"])) if not roster_detail.empty else None,
    }

    note = "Using real model output + season-level features"
    if team_available:
        note += " + team/headshot mapping"
    unused = (
        "Unused right now: passing(3).csv, rushing(3).csv, receiving(3).csv, player_season(3).csv, next_contract_labels(3).csv. "
        "They are not required because dataset(3).csv already contains a merged season-level wide table."
    )

    return {
        "scored_players": scored,
        "contract_history": contract_history,
        "latest_market": market_rows,
        "team_summary": team_summary,
        "latest_roster_detail": roster_detail,
        "team_available": bool(team_available),
        "data_note": note,
        "data_source": data_sources,
        "unused_files_note": unused,
    }
