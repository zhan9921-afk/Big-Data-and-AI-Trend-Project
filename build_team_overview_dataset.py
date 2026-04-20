
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_adapter_v4_real import TEAM_ALIASES, load_dashboard_data


def build_team_overview_dataset(output_dir: Path, season: int | None = None) -> dict[str, Path]:
    try:
        import nflreadpy as nfl
    except ImportError as e:
        raise RuntimeError(
            "nflreadpy is not installed. Run `pip install nflreadpy` first."
        ) from e

    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_dashboard_data(output_dir.parent)
    scored = artifacts["scored_players"].copy()
    if scored.empty:
        raise RuntimeError("No scored player dataset was found. Make sure dashboard_model_results.csv and dataset(3).csv exist.")

    inferred_season = int(pd.to_numeric(scored["season"], errors="coerce").dropna().max()) if scored["season"].notna().any() else 2024
    season = season or inferred_season

    # nflreadpy returns Polars; convert to pandas for consistency
    players = nfl.load_players()
    rosters_weekly = nfl.load_rosters_weekly([season])

    players = players.to_pandas() if hasattr(players, "to_pandas") else players
    rosters_weekly = rosters_weekly.to_pandas() if hasattr(rosters_weekly, "to_pandas") else rosters_weekly

    # standardize likely id and headshot columns
    rename_players = {}
    if "display_name" in players.columns and "player" not in players.columns:
        rename_players["display_name"] = "player"
    if "headshot" in players.columns and "headshot_url" not in players.columns:
        rename_players["headshot"] = "headshot_url"
    players = players.rename(columns=rename_players)

    # latest team per player for the requested season
    rw = rosters_weekly.copy()

    player_key = "gsis_id"

    if player_key not in rw.columns:
        raise KeyError(f"{player_key} not found in rosters_weekly columns")

    if "week" in rw.columns:
        rw["week"] = pd.to_numeric(rw["week"], errors="coerce")
        rw = rw.sort_values([player_key, "week"])
        latest = rw.drop_duplicates(subset=[player_key], keep="last")
    else:
        latest = rw.drop_duplicates(subset=[player_key], keep="last")

    latest = latest.rename(columns={
        "gsis_id": "player_id",         
        "full_name": "player_name"
    })

    # team alias
    if "team" in latest.columns:
        latest["team"] = latest["team"].replace(TEAM_ALIASES)


    
    headshot_candidates = [c for c in ["headshot_url", "headshot", "headshot_image"] if c in players.columns]
    id_candidates = [c for c in ["gsis_id", "player_id"] if c in players.columns]
    if not id_candidates:
        raise RuntimeError("Could not find a player id column in nflreadpy players data.")
    player_id_col = id_candidates[0]
    headshot_col = headshot_candidates[0] if headshot_candidates else None

    player_meta = players[[c for c in [player_id_col, "player", "display_name", headshot_col] if c and c in players.columns]].copy()
    if "display_name" in player_meta.columns and "player" not in player_meta.columns:
        player_meta = player_meta.rename(columns={"display_name": "player"})
    if headshot_col and headshot_col != "headshot_url":
        player_meta = player_meta.rename(columns={headshot_col: "headshot_url"})
    if player_id_col != "player_id":
        player_meta = player_meta.rename(columns={player_id_col: "player_id"})

    team_map = latest[["player_id", "player_name", "team"]].copy()
    team_map["player"] = team_map["player_name"]
    if "player_name" in team_map.columns and "player" not in team_map.columns:
        team_map = team_map.rename(columns={"player_name": "player"})
    team_map["season"] = season
    team_map = team_map.merge(player_meta, on="player_id", how="left", suffixes=("", "_meta"))
    if "player_meta" in team_map.columns:
        team_map["player"] = team_map["player"].fillna(team_map["player_meta"])
        team_map = team_map.drop(columns=["player_meta"])

    # merge back to scored dashboard rows
    scored_use = scored.copy()
    scored_use = scored_use.merge(
        team_map[["player_id", "player", "team", "season", "headshot_url"]].drop_duplicates(),
        on=["player_id", "season"],
        how="left",
        suffixes=("", "_map"),
    )
    for col in ["team", "headshot_url"]:
        map_col = f"{col}_map"
        if map_col in scored_use.columns:
            scored_use[col] = scored_use[col].fillna(scored_use[map_col])
            scored_use = scored_use.drop(columns=[map_col])

    team_roster_detail = scored_use.sort_values(["team", "value_gap"], ascending=[True, False]).copy()
    team_summary = (
        team_roster_detail.dropna(subset=["team"])
        .groupby("team", as_index=False)
        .agg(
            players=("player_name", "count"),
            avg_gap=("value_gap", "mean"),
            total_gap=("value_gap", "sum"),
            avg_actual_apy=("actual_apy", "mean"),
            avg_predicted_apy=("predicted_apy", "mean"),
            n_undervalued=("value_gap", lambda s: int((s > 2).sum())),
            n_overvalued=("value_gap", lambda s: int((s < -2).sum())),
        )
        .sort_values("avg_gap", ascending=False)
    )

    player_team_map_path = output_dir / "player_team_map.csv"
    team_roster_path = output_dir / "team_roster_detail.csv"
    team_summary_path = output_dir / "team_summary.csv"

    team_map.to_csv(player_team_map_path, index=False)
    team_roster_detail.to_csv(team_roster_path, index=False)
    team_summary.to_csv(team_summary_path, index=False)

    return {
        "player_team_map": player_team_map_path,
        "team_roster_detail": team_roster_path,
        "team_summary": team_summary_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build team-level overview files for the GridironIQ dashboard.")
    parser.add_argument("--output-dir", default="output", help="Where to save team overview CSV files.")
    parser.add_argument("--season", type=int, default=None, help="Season to pull current roster/headshots for.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    files = build_team_overview_dataset(output_dir=output_dir, season=args.season)
    print("Created files:")
    for label, path in files.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
