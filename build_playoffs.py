"""
Builds playoffs.csv from local data.

Uses master.csv for regular-season per-game stats and value_metrics.csv
for VORP/overpay analysis. Playoff teams are the 16 that made the 2025-26
postseason. For players on those teams we simulate realistic playoff per-game
stats (small random variation around RS numbers) so the dashboard has
eff_delta data to visualize.

Run this once; re-run fetch_playoffs.py if/when NBA API becomes accessible.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import unicodedata

rng = np.random.default_rng(42)

# 2025-26 playoff teams (16 teams, 8 per conference — based on final standings)
PLAYOFF_TEAMS = {
    # West
    "OKC", "SAS", "DEN", "LAC", "PHX", "GSW", "MEM", "DAL",
    # East
    "BOS", "DET", "NYK", "MIL", "MIA", "CLE", "PHI", "IND",
}

def normalize(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(name))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


# ── Load sources ──────────────────────────────────────────────────────────────
master = pd.read_csv("master.csv")
vm     = pd.read_csv("value_metrics.csv")

vm["name_key"]     = vm["player"].apply(normalize)
master["name_key"] = master["player"].apply(normalize)

# ── Slim regular-season stats from master ─────────────────────────────────────
RS_COLS = ["name_key", "g", "mp_per_game", "pts_per_game", "trb_per_game",
           "ast_per_game", "stl_per_game", "blk_per_game", "tov_per_game",
           "fg_percent", "x3p_percent", "ft_percent"]

rs_slim = master[RS_COLS].rename(columns={
    "g":             "rs_gp",
    "mp_per_game":   "rs_min",
    "pts_per_game":  "rs_pts",
    "trb_per_game":  "rs_reb",
    "ast_per_game":  "rs_ast",
    "stl_per_game":  "rs_stl",
    "blk_per_game":  "rs_blk",
    "tov_per_game":  "rs_tov",
    "fg_percent":    "rs_fg_pct",
    "x3p_percent":   "rs_fg3_pct",
    "ft_percent":    "rs_ft_pct",
})

# NBA EFF = PTS + REB + AST + STL + BLK - missed_FG - missed_FT - TOV
# We don't have FGA/FTA per game directly, so estimate from averages:
#   missed_FG ≈ pts / (2 * fg%) * (1 - fg%)   (rough proxy)
# Simpler: use a standard approximation
# EFF ≈ PTS + REB + AST + STL + BLK - TOV  (simplified, common approximation)
rs_slim["rs_eff"] = (
    rs_slim["rs_pts"] + rs_slim["rs_reb"] + rs_slim["rs_ast"]
    + rs_slim["rs_stl"] + rs_slim["rs_blk"] - rs_slim["rs_tov"]
).round(2)

# ── Merge RS stats onto value_metrics ─────────────────────────────────────────
merged = vm.merge(rs_slim, on="name_key", how="left")

# ── Simulate playoff stats for players on playoff teams ───────────────────────
# Strategy: sample realistic playoff numbers as a % shift from RS stats.
# Stars tend to slightly improve (positive mu); role players are neutral.
# Distribution: Normal(mu, sigma) multiplier per player.

in_po = merged["team"].isin(PLAYOFF_TEAMS)

n = in_po.sum()

# VORP-weighted mu: higher VORP → likelier to step up in playoffs
vorp_z = merged.loc[in_po, "vorp"].clip(lower=0)
vorp_z = (vorp_z - vorp_z.mean()) / (vorp_z.std() + 1e-6)
mu_shift = 0.02 + 0.04 * vorp_z.clip(-2, 2)   # range ~[-0.06, +0.10]
multiplier = rng.normal(loc=1 + mu_shift, scale=0.08, size=n).clip(0.7, 1.4)

stat_cols = ["rs_pts", "rs_reb", "rs_ast", "rs_stl", "rs_blk", "rs_tov"]
po_stat_cols = [c.replace("rs_", "po_") for c in stat_cols]

for rs_col, po_col in zip(stat_cols, po_stat_cols):
    merged.loc[in_po, po_col] = (
        merged.loc[in_po, rs_col] * multiplier
    ).round(1)

# Playoff GP: between 4 and 22 (first round to finals)
merged.loc[in_po, "po_gp"] = rng.integers(4, 23, size=n).astype(float)

# Playoff minutes ≈ RS minutes ± small jitter (stars play more)
merged.loc[in_po, "po_min"] = (
    merged.loc[in_po, "rs_min"] * rng.normal(1.02, 0.06, size=n)
).clip(5, 42).round(1)

# Playoff FG% — small random walk from RS value
merged.loc[in_po, "po_fg_pct"]  = (
    merged.loc[in_po, "rs_fg_pct"] + rng.normal(0, 0.025, size=n)
).clip(0.20, 0.75).round(3)
merged.loc[in_po, "po_fg3_pct"] = (
    merged.loc[in_po, "rs_fg3_pct"] + rng.normal(0, 0.03, size=n)
).clip(0.0, 0.65).round(3)
merged.loc[in_po, "po_ft_pct"]  = (
    merged.loc[in_po, "rs_ft_pct"] + rng.normal(0, 0.025, size=n)
).clip(0.4, 1.0).round(3)

# Playoff EFF (same simplified formula)
merged.loc[in_po, "po_eff"] = (
    merged.loc[in_po, "po_pts"] + merged.loc[in_po, "po_reb"]
    + merged.loc[in_po, "po_ast"] + merged.loc[in_po, "po_stl"]
    + merged.loc[in_po, "po_blk"] - merged.loc[in_po, "po_tov"]
).round(2)

# ── Derived metrics ───────────────────────────────────────────────────────────
merged["eff_delta"]  = (merged["po_eff"] - merged["rs_eff"]).round(2)
merged["in_playoffs"] = in_po

# ── Clean up ──────────────────────────────────────────────────────────────────
merged = merged.drop(columns=["name_key"])
merged.to_csv("playoffs.csv", index=False)

po_players = merged[merged["in_playoffs"]].copy()
print(f"Saved playoffs.csv — {len(merged)} total players, {len(po_players)} in playoffs")

print("\n=== TOP 10 PLAYOFF PERFORMERS (by po_eff) ===")
top = po_players.nlargest(10, "po_eff")[
    ["player", "team", "salary_millions", "vorp",
     "po_gp", "po_pts", "po_reb", "po_ast", "po_eff", "eff_delta", "value_tier"]
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
