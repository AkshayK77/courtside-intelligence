"""
forecast_next_season.py — Project current-season rosters forward one year
and predict next-season wins using the trained win model.

Player projections use an age-based VORP/BPM curve:
  - Under 24: improvement  (+10–15%)
  - 24–26:    approaching peak (+5%)
  - 27–29:    prime/stable   (0%)
  - 30–32:    early decline  (-10%)
  - 33–35:    decline        (-20%)
  - 36+:      late career    (-35%)

Minutes are held constant (no roster-move assumptions).
Saves: next_season_forecast.csv
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import joblib

FEATURES = ["total_vorp", "top_player_vorp", "vorp_concentration", "avg_age", "avg_bpm"]
NEXT_SEASON = 2027

# Age-based multipliers for VORP and BPM
AGE_BREAKPOINTS = [
    (22,  1.15),
    (24,  1.10),
    (27,  1.05),
    (30,  1.00),
    (33,  0.90),
    (36,  0.80),
    (999, 0.65),
]

def vorp_multiplier(age: float) -> float:
    for cutoff, mult in AGE_BREAKPOINTS:
        if age < cutoff:
            return mult
    return 0.65


def project_players(master: pd.DataFrame) -> pd.DataFrame:
    df = master.copy()
    df = df[df["team"] != "TOT"]

    for col in ("vorp", "bpm", "mp", "mp_per_game", "age"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["age_next"] = df["age"] + 1

    # VORP: age curve
    df["vorp_next"] = df.apply(
        lambda r: r["vorp"] * vorp_multiplier(r["age"]), axis=1
    )

    # BPM: same age curve + 10% mean reversion toward 0
    df["bpm_next"] = df.apply(
        lambda r: r["bpm"] * 0.90 * vorp_multiplier(r["age"]), axis=1
    )

    return df


def aggregate_team_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for team, grp in df.groupby("team"):
        total_vorp = grp["vorp_next"].sum()
        top_vorp   = grp["vorp_next"].max()
        avg_age    = grp["age_next"].mean()
        total_mp   = grp["mp"].sum()
        avg_bpm    = (
            (grp["bpm_next"] * grp["mp"]).sum() / total_mp
            if total_mp > 0 else 0.0
        )
        vorp_conc  = top_vorp / max(total_vorp, 1.0)

        rows.append({
            "team":               team,
            "season":             NEXT_SEASON,
            "total_vorp":         round(total_vorp, 2),
            "top_player_vorp":    round(top_vorp, 2),
            "vorp_concentration": round(float(vorp_conc), 4),
            "avg_age":            round(avg_age, 2),
            "avg_bpm":            round(avg_bpm, 4),
        })
    return pd.DataFrame(rows)


def main():
    # Load current roster
    master = pd.read_csv("master.csv")
    print(f"Loaded master.csv — {len(master)} players")

    # Project stats forward one year
    projected = project_players(master)

    # Aggregate to team-level features
    team_features = aggregate_team_features(projected)
    print(f"Projected {len(team_features)} teams to season {NEXT_SEASON}")

    # Load current-season actuals (for comparison)
    try:
        training = pd.read_csv("training_data.csv")
        current = training[training["season"] == training["season"].max()].copy()
        current = current.rename(columns={
            "wins":              "current_wins",
            "total_vorp":       "current_total_vorp",
            "avg_age":          "current_avg_age",
            "avg_bpm":          "current_avg_bpm",
            "team_abbr":        "team",
        })
        team_col = "team" if "team" in current.columns else current.columns[1]
        team_features = team_features.merge(
            current[["team", "current_wins"]],
            on="team", how="left"
        )
    except Exception:
        team_features["current_wins"] = np.nan

    # Load and run the win model
    try:
        model = joblib.load("win_model.pkl")
    except FileNotFoundError:
        print("ERROR: win_model.pkl not found — run win_predictor.py first")
        return

    X = team_features[FEATURES]
    team_features["predicted_wins"] = model.predict(X).round(1)

    # Win change vs current season
    team_features["win_delta"] = (
        team_features["predicted_wins"] - team_features["current_wins"]
    ).round(1)

    # Sort by predicted wins descending
    out = team_features.sort_values("predicted_wins", ascending=False).reset_index(drop=True)
    out.index += 1  # 1-based rank

    print(f"\n{'Rank':<5} {'Team':<6} {'Cur W':>6} {'Proj W':>7} {'Δ':>6}  {'Total VORP':>11}  {'Avg Age':>8}  {'Avg BPM':>8}")
    print("-" * 68)
    for rank, row in out.iterrows():
        cur  = f"{row['current_wins']:.0f}" if not pd.isna(row.get("current_wins")) else "N/A"
        delta = f"{row['win_delta']:+.1f}" if not pd.isna(row.get("win_delta")) else "N/A"
        print(f"{rank:<5} {row['team']:<6} {cur:>6} {row['predicted_wins']:>7.1f} {delta:>6}  "
              f"{row['total_vorp']:>11.2f}  {row['avg_age']:>8.2f}  {row['avg_bpm']:>8.4f}")

    out.to_csv("next_season_forecast.csv", index=False)
    print("\nSaved → next_season_forecast.csv")


if __name__ == "__main__":
    main()
