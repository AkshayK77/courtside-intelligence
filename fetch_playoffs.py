"""
Pulls live 2025-26 playoff AND regular-season per-game stats from the NBA API,
merges them with the existing value_metrics.csv (VORP / overpay analysis),
and writes playoffs.csv — one row per player, combining both worlds.

Run this any time during the playoffs to get updated numbers.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import unicodedata
import pandas as pd
from nba_api.stats.endpoints import leagueleaders

SEASON = "2025-26"

# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize(name: str) -> str:
    """Strip accents and lowercase for fuzzy name matching."""
    nfkd = unicodedata.normalize("NFKD", str(name))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def pull_stats(season_type: str) -> pd.DataFrame:
    print(f"  Fetching {season_type} stats from NBA API...")
    resp = leagueleaders.LeagueLeaders(
        league_id="00",
        per_mode48="PerGame",
        scope="S",
        season=SEASON,
        season_type_all_star=season_type,
        stat_category_abbreviation="PTS",
    )
    df = resp.get_data_frames()[0]
    df["name_key"] = df["PLAYER"].apply(normalize)
    return df


# ── Pull data ─────────────────────────────────────────────────────────────────
print("Pulling live NBA stats...")
rs  = pull_stats("Regular Season")
po  = pull_stats("Playoffs")
print(f"  Regular season: {len(rs)} players")
print(f"  Playoffs:       {len(po)} players")

# ── Rename columns for each season type ──────────────────────────────────────
RS_COLS  = ["name_key", "GP", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV",
            "FG_PCT", "FG3_PCT", "FT_PCT", "EFF"]
PO_COLS  = RS_COLS

rs_slim = rs[RS_COLS].rename(columns={
    "GP": "rs_gp", "MIN": "rs_min", "PTS": "rs_pts", "REB": "rs_reb",
    "AST": "rs_ast", "STL": "rs_stl", "BLK": "rs_blk", "TOV": "rs_tov",
    "FG_PCT": "rs_fg_pct", "FG3_PCT": "rs_fg3_pct", "FT_PCT": "rs_ft_pct",
    "EFF": "rs_eff",
})

po_slim = po[PO_COLS].rename(columns={
    "GP": "po_gp", "MIN": "po_min", "PTS": "po_pts", "REB": "po_reb",
    "AST": "po_ast", "STL": "po_stl", "BLK": "po_blk", "TOV": "po_tov",
    "FG_PCT": "po_fg_pct", "FG3_PCT": "po_fg3_pct", "FT_PCT": "po_ft_pct",
    "EFF": "po_eff",
})

# ── Load existing VORP / value analysis ──────────────────────────────────────
vm = pd.read_csv("value_metrics.csv")
vm["name_key"] = vm["player"].apply(normalize)

# ── Three-way merge ───────────────────────────────────────────────────────────
merged = (
    vm
    .merge(rs_slim, on="name_key", how="left")
    .merge(po_slim, on="name_key", how="left")
)
merged = merged.drop(columns=["name_key"])

# ── Playoff step-up metric ────────────────────────────────────────────────────
# How much better (or worse) is a player's efficiency in the playoffs vs regular season?
# Positive = stepped up; negative = faded.
merged["eff_delta"] = (merged["po_eff"] - merged["rs_eff"]).round(2)

# ── In-playoffs flag ──────────────────────────────────────────────────────────
merged["in_playoffs"] = merged["po_gp"].notna() & (merged["po_gp"] > 0)

# ── Save ──────────────────────────────────────────────────────────────────────
merged.to_csv("playoffs.csv", index=False)
print(f"\nSaved: playoffs.csv  ({len(merged)} players, "
      f"{merged['in_playoffs'].sum()} in the playoffs)")

# ── Quick summary ─────────────────────────────────────────────────────────────
po_players = merged[merged["in_playoffs"]].copy()

print("\n=== TOP 10 PLAYOFF PERFORMERS (by po_eff) ===")
top = po_players.nlargest(10, "po_eff")[
    ["player", "team", "salary_millions", "vorp", "po_gp", "po_pts",
     "po_reb", "po_ast", "po_eff", "eff_delta", "value_tier"]
]
print(top.to_string(index=False))

print("\n=== BIGGEST PLAYOFF STEP-UPS (eff_delta) ===")
stepup = po_players.nlargest(10, "eff_delta")[
    ["player", "team", "salary_millions", "overpay_millions",
     "rs_eff", "po_eff", "eff_delta"]
]
print(stepup.to_string(index=False))

print("\n=== BIGGEST PLAYOFF FADES ===")
fades = po_players.nsmallest(10, "eff_delta")[
    ["player", "team", "salary_millions", "overpay_millions",
     "rs_eff", "po_eff", "eff_delta"]
]
print(fades.to_string(index=False))

print("\n=== BEST OF BOTH WORLDS (underpaid + playoff step-up) ===")
best = po_players[
    (po_players["overpay_millions"] < 0) & (po_players["eff_delta"] > 0)
].nlargest(10, "po_eff")[
    ["player", "team", "salary_millions", "vorp",
     "overpay_millions", "po_eff", "eff_delta"]
]
print(best.to_string(index=False))
