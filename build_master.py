"""
Merges Player Per Game, Advanced, and Salary CSVs into master.csv.
One row per player (using the "TOT" row for traded players), with
salary and all performance metrics side by side.
"""

import pandas as pd
import re


def clean_salary(val) -> float | None:
    if pd.isna(val):
        return None
    cleaned = re.sub(r"[\$,]", "", str(val)).strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def load_stats() -> tuple[pd.DataFrame, pd.DataFrame]:
    ppg = pd.read_csv("Player Per Game.csv")
    adv = pd.read_csv("Advanced.csv")

    # Keep only 2025-26 season rows
    ppg = ppg[ppg["season"] == 2026].copy()
    adv = adv[adv["season"] == 2026].copy()

    MULTI_TEAM_CODES = {"2TM", "3TM", "4TM", "5TM"}

    def build_final_team_map(df: pd.DataFrame) -> dict:
        """For traded players, return the last team they played for (final destination).
        Basketball Reference orders rows chronologically, so the last non-aggregate
        team row is the most recent team."""
        traded_ids = df[df["team"].isin(MULTI_TEAM_CODES)]["player_id"].unique()
        individual_rows = df[
            df["player_id"].isin(traded_ids) & ~df["team"].isin(MULTI_TEAM_CODES)
        ]
        # Last row per player = most recent team (file is in chronological order)
        last_team = (
            individual_rows.groupby("player_id")["team"].last().to_dict()
        )
        return last_team

    def keep_primary(df: pd.DataFrame, final_team_map: dict) -> pd.DataFrame:
        traded_ids = df[df["team"].isin(MULTI_TEAM_CODES)]["player_id"].unique()
        non_traded = df[~df["player_id"].isin(traded_ids)]
        traded_agg = df[
            df["player_id"].isin(traded_ids) & df["team"].isin(MULTI_TEAM_CODES)
        ].copy()
        # Replace the aggregate team code with the player's actual current team
        traded_agg["team"] = traded_agg["player_id"].map(final_team_map)
        return pd.concat([non_traded, traded_agg], ignore_index=True)

    final_team_map = build_final_team_map(ppg)
    ppg = keep_primary(ppg, final_team_map)
    adv = keep_primary(adv, final_team_map)
    return ppg, adv


def load_salaries() -> pd.DataFrame:
    sal = pd.read_csv("salaries.csv")

    # The last column is player_id (mislabeled "-9999" by the export tool)
    sal = sal.rename(columns={"-9999": "player_id", "Tm": "salary_team"})

    sal["salary"] = sal["2025-26"].apply(clean_salary)
    sal = sal[["player_id", "salary"]].dropna(subset=["player_id"])
    sal = sal[sal["salary"].notna()].copy()
    # Salary file lists traded players once per team — deduplicate, keeping first row
    sal = sal.drop_duplicates(subset="player_id", keep="first")
    sal["salary"] = sal["salary"].astype(int)
    sal["salary_millions"] = (sal["salary"] / 1_000_000).round(2)
    return sal


def build_master() -> pd.DataFrame:
    ppg, adv = load_stats()
    sal = load_salaries()

    # Columns to pull from PPG (drop duplicates already in Advanced)
    ppg_cols = [
        "player_id", "player", "team", "pos", "age", "g", "gs",
        "mp_per_game", "pts_per_game", "trb_per_game", "ast_per_game",
        "stl_per_game", "blk_per_game", "tov_per_game",
        "fg_percent", "x3p_percent", "ft_percent",
    ]

    # Columns to pull from Advanced
    adv_cols = [
        "player_id",
        "mp", "per", "ts_percent",
        "ows", "dws", "ws", "ws_48",
        "obpm", "dbpm", "bpm", "vorp",
    ]

    master = (
        ppg[ppg_cols]
        .merge(adv[adv_cols], on="player_id", how="inner")
        .merge(sal, on="player_id", how="left")
    )

    # Normalise team abbreviations: TOT → use primary team from ppg
    # (already handled by keep_primary — TOT rows already have "TOT" as team;
    #  for salary matching the player_id is the reliable key so this is fine)

    # Filter: must have played at least 10 games to be meaningful
    master = master[master["g"] >= 10].copy()

    # Flag players missing salary data
    missing_sal = master["salary"].isna().sum()
    if missing_sal:
        print(f"Warning: {missing_sal} players have no salary data - they will be excluded from cap analysis.")

    # Drop rows with no salary for optimization purposes
    master_with_sal = master.dropna(subset=["salary"]).copy()

    print(f"Total players (>=10 G): {len(master)}")
    print(f"Players with salary:    {len(master_with_sal)}")

    master_with_sal.to_csv("master.csv", index=False)
    print("\nSaved: master.csv")
    return master_with_sal


if __name__ == "__main__":
    df = build_master()
    print("\nSample output:")
    print(df[["player", "team", "pos", "salary_millions", "vorp", "ws", "bpm", "pts_per_game"]].head(20).to_string(index=False))
