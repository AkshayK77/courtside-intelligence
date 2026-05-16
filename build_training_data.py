"""
build_training_data.py — Combine historical team stats with current season.
Reads: historical_teams.csv  +  master.csv (current 2025-26 season)
Saves: training_data.csv
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

CURRENT_SEASON = 2026


def aggregate_current_season() -> pd.DataFrame:
    """Aggregate master.csv player rows into one row per team."""
    master = pd.read_csv("master.csv")

    for col in ("vorp", "ws", "bpm", "mp", "age"):
        master[col] = pd.to_numeric(master[col], errors="coerce")

    # Exclude multi-team TOT rows if present
    master = master[master["team"] != "TOT"]

    rows = []
    for team, grp in master.groupby("team"):
        total_vorp  = grp["vorp"].sum()
        top_vorp    = grp["vorp"].max()
        wins        = grp["ws"].sum()           # WS ≈ wins
        avg_age     = grp["age"].mean()
        total_mp    = grp["mp"].sum()
        avg_bpm     = (grp["bpm"] * grp["mp"]).sum() / total_mp if total_mp > 0 else 0.0
        vorp_conc   = top_vorp / max(total_vorp, 1.0)   # clamp; avoids explosion for negative-VORP teams

        rows.append({
            "season":            CURRENT_SEASON,
            "team_abbr":         team,
            "wins":              round(wins, 2),
            "total_vorp":        round(total_vorp, 2),
            "top_player_vorp":   round(top_vorp, 2),
            "vorp_concentration":round(float(vorp_conc), 4),
            "avg_age":           round(avg_age, 2),
            "avg_bpm":           round(avg_bpm, 4),
        })

    return pd.DataFrame(rows)


def main():
    # Load historical
    try:
        hist = pd.read_csv("historical_teams.csv")
        print(f"Loaded historical_teams.csv — {len(hist)} rows, "
              f"seasons {hist['season'].min()}–{hist['season'].max()}")
    except FileNotFoundError:
        print("historical_teams.csv not found — run fetch_historical.py first")
        hist = pd.DataFrame()

    # Build current season
    print("Aggregating current season from master.csv …")
    current = aggregate_current_season()
    print(f"  {len(current)} teams")

    # Combine
    combined = pd.concat([hist, current], ignore_index=True)
    combined = combined.dropna(subset=["wins", "total_vorp", "avg_age", "avg_bpm"])
    combined = combined.sort_values(["season", "team_abbr"]).reset_index(drop=True)

    print(f"\nTraining dataset: {len(combined)} rows across "
          f"{combined['season'].nunique()} seasons")
    print(combined.tail(5).to_string(index=False))

    combined.to_csv("training_data.csv", index=False)
    print("\nSaved → training_data.csv")


if __name__ == "__main__":
    main()
