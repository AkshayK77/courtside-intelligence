"""
Phase 2 — Value Metric
Calculates how much production each team gets per dollar spent.

Core metrics:
  vorp_per_million      - raw VORP per $1M salary (higher = better value)
  position_adj_value    - VORP above position average per $1M (controls for scarcity)
  overpay_millions      - how many $M a player is over/under their market value
  value_tier            - human-readable label (Elite Value → Overpaid)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
MIN_MP_PER_GAME = 10   # filter noise: only players who log real minutes
TIER_LABELS = {
    5: "Elite Value",
    4: "Good Value",
    3: "Fair",
    2: "Poor Value",
    1: "Overpaid",
}

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("master.csv")
df = df[df["mp_per_game"] >= MIN_MP_PER_GAME].copy()
print(f"Players after mp filter: {len(df)}")


# ── 1. Raw value: VORP per $1M ────────────────────────────────────────────────
# Negative VORP → negative ratio (correctly signals overpaid / below replacement)
df["vorp_per_million"] = (df["vorp"] / df["salary_millions"]).round(4)


# ── 2. Position-adjusted value ────────────────────────────────────────────────
# Idea: a C producing 2.0 VORP is more impressive than an SG producing 2.0 VORP
# because quality Cs are scarcer. We measure VORP relative to position baseline.
pos_avg = df.groupby("pos")["vorp"].mean()
df["position_avg_vorp"] = df["pos"].map(pos_avg)
df["vorp_above_position"] = (df["vorp"] - df["position_avg_vorp"]).round(3)
df["position_adj_value"] = (df["vorp_above_position"] / df["salary_millions"]).round(4)


# ── 3. Expected salary & overpay ─────────────────────────────────────────────
# Fit salary ~ VORP with a simple OLS to find the "market rate" curve.
# Residual = how many $M a player earns above/below their on-court value.
from numpy.polynomial import polynomial as P

# Use only players with positive VORP to fit the market rate line cleanly
mask = df["vorp"] > 0
coeffs = np.polyfit(df.loc[mask, "vorp"], df.loc[mask, "salary_millions"], deg=1)
slope, intercept = coeffs

df["expected_salary_millions"] = (slope * df["vorp"] + intercept).round(2)
# Floor expected salary at league minimum (~$1.1M)
df["expected_salary_millions"] = df["expected_salary_millions"].clip(lower=1.1)
df["overpay_millions"] = (df["salary_millions"] - df["expected_salary_millions"]).round(2)
# Positive = overpaid vs market; negative = team is getting a bargain


# ── 4. Value tier ─────────────────────────────────────────────────────────────
# Two-axis classification: salary level × overpay amount
# This avoids mislabeling cheap-but-bad players as "Overpaid."
SALARY_STAR_THRESHOLD = 15.0   # $15M+ = meaningful contract
OVERPAID_THRESHOLD    = 15.0   # $15M+ above market = clearly overpaid
STEAL_THRESHOLD       = -10.0  # $10M+ below market = clear bargain

def assign_tier(row) -> str:
    s = row["salary_millions"]
    op = row["overpay_millions"]
    v = row["vorp"]
    # Minimum-contract roster filler with negative production: fringe, not overpaid
    if s < SALARY_STAR_THRESHOLD and v <= 0:
        return "Fringe"
    # High salary, earning way above market rate for their output
    if s >= SALARY_STAR_THRESHOLD and op >= OVERPAID_THRESHOLD:
        return "Overpaid"
    # Any meaningful salary earning above market (but not extreme)
    if op > 5 and s >= SALARY_STAR_THRESHOLD:
        return "Poor Value"
    # Getting significantly more production than salary implies
    if op <= STEAL_THRESHOLD:
        return "Elite Value"
    # Good production relative to cost
    if row["position_adj_value"] > 0:
        return "Good Value"
    # Everything else
    return "Fair"

df["value_tier"] = df.apply(assign_tier, axis=1)


# ── 5. Overall rank (best value first) ───────────────────────────────────────
df["value_rank"] = df["position_adj_value"].rank(ascending=False, method="min").astype(int)


# ── Save ──────────────────────────────────────────────────────────────────────
output_cols = [
    "value_rank", "player", "team", "pos", "age",
    "salary_millions", "vorp", "ws", "bpm",
    "vorp_per_million", "position_avg_vorp", "vorp_above_position",
    "position_adj_value", "expected_salary_millions", "overpay_millions",
    "value_tier", "pts_per_game", "mp_per_game", "g",
]
df_out = df[output_cols].sort_values("value_rank")
df_out.to_csv("value_metrics.csv", index=False)
print("Saved: value_metrics.csv")


# ── Print checkpoint: ranked list ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("TOP 20 — BEST VALUE FOR MONEY (position-adjusted)")
print("=" * 70)
top20 = df_out.head(20)[["value_rank", "player", "team", "pos", "salary_millions",
                           "vorp", "vorp_per_million", "overpay_millions", "value_tier"]]
print(top20.to_string(index=False))

print("\n" + "=" * 70)
print("BOTTOM 20 — WORST VALUE FOR MONEY")
print("=" * 70)
bot20 = df_out.tail(20)[["value_rank", "player", "team", "pos", "salary_millions",
                           "vorp", "vorp_per_million", "overpay_millions", "value_tier"]]
print(bot20.to_string(index=False))

print("\n" + "=" * 70)
print("BIGGEST STEALS (underpaid relative to production)")
print("=" * 70)
steals = df_out.nsmallest(15, "overpay_millions")[
    ["player", "team", "pos", "salary_millions", "vorp", "overpay_millions", "value_tier"]
]
print(steals.to_string(index=False))

print("\n" + "=" * 70)
print("MOST OVERPAID (salary far above market value)")
print("=" * 70)
overpaid = df_out.nlargest(15, "overpay_millions")[
    ["player", "team", "pos", "salary_millions", "vorp", "overpay_millions", "value_tier"]
]
print(overpaid.to_string(index=False))

print("\n" + "=" * 70)
print("VALUE TIER BREAKDOWN")
print("=" * 70)
tier_summary = (
    df_out.groupby("value_tier")
    .agg(
        count=("player", "count"),
        avg_salary=("salary_millions", "mean"),
        avg_vorp=("vorp", "mean"),
        avg_overpay=("overpay_millions", "mean"),
    )
    .round(2)
)
tier_order = ["Elite Value", "Good Value", "Fair", "Poor Value", "Overpaid", "Fringe"]
existing = [t for t in tier_order if t in tier_summary.index]
print(tier_summary.loc[existing].to_string())

print("\n" + "=" * 70)
print("MARKET RATE LINE: Salary = {:.2f} x VORP + {:.2f}".format(slope, intercept))
print("=" * 70)
print("(Market rate fit on players with VORP > 0)")
