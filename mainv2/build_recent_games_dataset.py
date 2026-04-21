
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _normalize_name(s):
    import pandas as pd

    # 如果误传了 DataFrame，自动取第一列
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]

    return (
        s.astype(str)
        .str.lower()
        .str.replace(r"[^a-z\s]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _rename_weekly_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {
        "player_display_name": "player_name",
        "display_name": "player_name",
        "player": "player_name",
        "recent_team": "team",
        "season_type": "season_phase",
        "passing_yards": "pass_yards",
        "passing_tds": "pass_tds",
        "rushing_yards": "rush_yards",
        "rushing_tds": "rush_tds",
        "receiving_yards": "rec_yards",
        "receiving_tds": "rec_tds",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})
    return out


def load_weekly_from_nflreadpy(season: int) -> pd.DataFrame:
    import nflreadpy as nfl

    attempts = [
        lambda: nfl.load_player_stats(season, summary_level="week"),
        lambda: nfl.load_player_stats([season], summary_level="week"),
        lambda: nfl.load_player_stats(season=season, summary_level="week"),
        lambda: nfl.load_player_stats(seasons=[season], summary_level="week"),
        lambda: nfl.load_player_stats(season, weekly=True),
        lambda: nfl.load_player_stats([season], weekly=True),
    ]

    last_error = None
    for fn in attempts:
        try:
            out = fn()
            if hasattr(out, "to_pandas"):
                out = out.to_pandas()
            if isinstance(out, pd.DataFrame) and not out.empty:
                return _rename_weekly_columns(out)
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Could not load weekly player stats from nflreadpy for season {season}. Last error: {last_error}")


def build_recent_games_dataset(output_dir: Path, season: int = 2024) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    weekly = load_weekly_from_nflreadpy(season)

    required_candidates = {
        "player_name": ["player_name"],
        "week": ["week"],
        "season": ["season"],
    }
    for std, candidates in required_candidates.items():
        if not any(c in weekly.columns for c in candidates):
            raise KeyError(f"Missing required column for {std}. Available columns: {list(weekly.columns)[:30]}")

    keep_cols = [
        c for c in [
            "player_name", "team", "season", "week",
            "pass_yards", "pass_tds", "rush_yards", "rush_tds", "rec_yards", "rec_tds",
            "completions", "attempts", "passing_air_yards", "receiving_receptions",
            "fantasy_points", "fantasy_points_ppr",
        ] if c in weekly.columns
    ]

    recent = weekly[keep_cols].copy()
    recent["player_name"] = recent["player_name"].astype(str)
    recent["player_norm"] = _normalize_name(recent["player_name"])
    recent["season"] = pd.to_numeric(recent["season"], errors="coerce")
    recent["week"] = pd.to_numeric(recent["week"], errors="coerce")
    recent = recent.sort_values(["player_norm", "season", "week"]).reset_index(drop=True)

    out_path = output_dir / "player_weekly.csv"
    recent.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build recent game logs file for dashboard player analysis.")
    parser.add_argument("--output-dir", default="output", help="Directory where player_weekly.csv will be saved.")
    parser.add_argument("--season", type=int, default=2024, help="Season to download weekly player stats for.")
    args = parser.parse_args()

    out_path = build_recent_games_dataset(Path(args.output_dir), season=args.season)
    print(f"Created: {out_path}")


if __name__ == "__main__":
    main()
