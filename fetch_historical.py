"""
fetch_historical.py — Build historical team stats from local Kaggle CSV.
Source: stats_since_1950/Seasons_Stats.csv
Aggregates per-player stats into per-team-season rows.
Saves: historical_teams.csv
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

SOURCE = "stats_since_1950/Seasons_Stats.csv"
START_YEAR = 2000   # inclusive
END_YEAR   = 2017   # inclusive (dataset ends here)


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(SOURCE, low_memory=False)

    # Numeric coercion
    for col in ("Year", "VORP", "WS", "BPM", "MP", "Age"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.rename(columns={"Year": "year", "Tm": "team_abbr", "Age": "age",
                             "VORP": "vorp", "WS": "ws", "BPM": "bpm", "MP": "mp"})

    # Keep relevant years and drop TOT rows (multi-team aggregates)
    df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)]
    df = df[df["team_abbr"] != "TOT"]
    df = df.dropna(subset=["vorp", "ws", "team_abbr"])

    return df


def aggregate_to_teams(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (year, team), grp in df.groupby(["year", "team_abbr"]):
        total_vorp    = grp["vorp"].sum()
        top_vorp      = grp["vorp"].max()
        wins          = grp["ws"].sum()           # WS sums ≈ team wins
        avg_age       = grp["age"].mean()

        # Minutes-weighted average BPM
        total_mp = grp["mp"].sum()
        avg_bpm  = (grp["bpm"] * grp["mp"]).sum() / total_mp if total_mp > 0 else 0.0

        vorp_conc = top_vorp / max(total_vorp, 1.0)   # clamp denominator; avoids explosion for negative-VORP teams

        rows.append({
            "season":            int(year),
            "team_abbr":         team,
            "wins":              round(wins, 2),
            "total_vorp":        round(total_vorp, 2),
            "top_player_vorp":   round(top_vorp, 2),
            "vorp_concentration":round(vorp_conc, 4),
            "avg_age":           round(avg_age, 2),
            "avg_bpm":           round(avg_bpm, 4),
        })

    return pd.DataFrame(rows)


def main():
    print(f"Loading {SOURCE} …")
    df = load_and_clean()
    print(f"  {len(df)} player-season rows after filtering ({START_YEAR}–{END_YEAR})")

    teams = aggregate_to_teams(df)
    print(f"  {len(teams)} team-season rows across {teams['season'].nunique()} seasons")

    # Quick sanity check — 2016-17 GSW should be ≈ 67 wins
    gsw17 = teams[(teams["season"] == 2017) & (teams["team_abbr"] == "GSW")]
    if not gsw17.empty:
        print(f"  Sanity check GSW 2017: wins={gsw17['wins'].iloc[0]:.1f} (actual 67)")

    teams.to_csv("historical_teams.csv", index=False)
    print("Saved → historical_teams.csv")


if __name__ == "__main__":
    main()
