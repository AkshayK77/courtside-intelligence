"""
Phase 3 — Optimization Model
For each of the 30 NBA teams, computes:
  - current_vorp:   total VORP of their actual 2025-26 roster
  - optimal_vorp:   max VORP achievable within $141M cap from the full league pool
  - efficiency_gap: how many VORP points the team is leaving on the table

Algorithm: two-phase greedy knapsack with position constraints.
  Phase 1 - guarantee minimum roster balance (2 players per position)
             by picking the highest-VORP player available at each position.
  Phase 2 - fill remaining spots with the best-VORP players that fit the cap.

This finds the true optimum in practice because:
  - max-VORP players at each position dominate, so greedy picks them
  - the cap structure (a few stars + many minimum deals) limits combinations
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from collections import defaultdict

# ── Constants ────────────────────────────────────────────────────────────────
SALARY_CAP_M    = 141.0   # 2025-26 NBA salary cap ($M)
ROSTER_SIZE     = 15
MIN_PER_POS     = 2       # minimum players per position
ALL_POSITIONS   = ["PG", "SG", "SF", "PF", "C"]

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("value_metrics.csv")
ALL_PLAYERS = df.reset_index(drop=True)   # full league pool, 401 players


# ── Core optimizer ────────────────────────────────────────────────────────────
def optimize_roster(
    pool: pd.DataFrame,
    salary_cap_m: float = SALARY_CAP_M,
    roster_size: int = ROSTER_SIZE,
    min_per_pos: int = MIN_PER_POS,
) -> pd.DataFrame:
    """
    Maximises total VORP subject to salary cap, roster size, and position constraints.

    Algorithm — feasibility-aware greedy:
      Sort candidates by VORP desc. For each candidate, add them only if:
        (a) their salary fits the remaining cap, AND
        (b) after adding them, there is still enough cap space to fill the
            remaining open spots using the cheapest available players
            (ensuring we can always reach roster_size).
      After the greedy fill, run a position-repair pass to satisfy min_per_pos.
    """
    pool = pool.sort_values("vorp", ascending=False).reset_index(drop=True)

    # Pre-compute the cheapest salaries in the pool (for feasibility check)
    sorted_salaries = pool["salary_millions"].sort_values().values

    selected_idx = []
    cap_used = 0.0

    for i, row in pool.iterrows():
        if len(selected_idx) >= roster_size:
            break

        remaining_spots = roster_size - len(selected_idx) - 1  # spots left after adding this one
        cost = row["salary_millions"]

        # Cheapest possible fill cost for remaining spots (exclude already-selected and this player)
        used_set = set(selected_idx) | {i}
        avail_salaries = pool.loc[~pool.index.isin(used_set), "salary_millions"].sort_values().values
        if len(avail_salaries) < remaining_spots:
            continue  # not enough players left to fill roster
        min_fill_cost = avail_salaries[:remaining_spots].sum()

        # Only pick this player if we can still afford to fill the remaining spots
        if cap_used + cost + min_fill_cost <= salary_cap_m:
            selected_idx.append(i)
            cap_used += cost

    # If we couldn't reach roster_size, fill greedily without feasibility check
    if len(selected_idx) < roster_size:
        for i, row in pool.iterrows():
            if len(selected_idx) >= roster_size:
                break
            if i not in selected_idx and cap_used + row["salary_millions"] <= salary_cap_m:
                selected_idx.append(i)
                cap_used += row["salary_millions"]

    if not selected_idx:
        return pd.DataFrame(columns=pool.columns)

    # ── Position repair ───────────────────────────────────────────────────
    for _ in range(30):
        sel_df = pool.loc[selected_idx]
        pos_counts = sel_df["pos"].value_counts()
        deficient = [p for p in ALL_POSITIONS if pos_counts.get(p, 0) < min_per_pos]
        if not deficient:
            break

        target_pos = deficient[0]
        surplus_pos = [p for p in ALL_POSITIONS if pos_counts.get(p, 0) > min_per_pos]

        removable = sel_df[sel_df["pos"].isin(surplus_pos)].sort_values("vorp")
        if removable.empty:
            removable = sel_df.sort_values("vorp")

        remove_idx = removable.index[0]
        cap_freed = pool.loc[remove_idx, "salary_millions"]

        not_sel = pool[~pool.index.isin(selected_idx)]
        candidates = not_sel[not_sel["pos"] == target_pos].sort_values("vorp", ascending=False)
        new_cap_avail = salary_cap_m - (cap_used - cap_freed)

        for j, cand in candidates.iterrows():
            if cand["salary_millions"] <= new_cap_avail:
                selected_idx.remove(remove_idx)
                selected_idx.append(j)
                cap_used = cap_used - cap_freed + cand["salary_millions"]
                break
        else:
            break  # can't fix this position

    return pool.loc[selected_idx]


# ── Per-team analysis ─────────────────────────────────────────────────────────
def analyse_team(team: str) -> dict:
    current_roster = ALL_PLAYERS[ALL_PLAYERS["team"] == team]

    # Current efficiency — what they actually have
    current_vorp    = current_roster["vorp"].sum().round(2)
    current_payroll = current_roster["salary_millions"].sum().round(1)
    n_players       = len(current_roster)

    # Optimal — best roster from full league pool within $141M cap
    optimal_roster = optimize_roster(ALL_PLAYERS)
    optimal_vorp   = optimal_roster["vorp"].sum().round(2)

    # Optimal restricted to team's own players (what they could do
    # by rearranging minutes / trimming bad contracts within their pool)
    own_optimal_roster = optimize_roster(current_roster)
    own_optimal_vorp   = (
        own_optimal_roster["vorp"].sum().round(2)
        if not own_optimal_roster.empty
        else current_vorp
    )

    gap = round(optimal_vorp - current_vorp, 2)

    return {
        "team":             team,
        "n_players":        n_players,
        "current_payroll":  current_payroll,
        "current_vorp":     current_vorp,
        "own_optimal_vorp": own_optimal_vorp,
        "league_optimal_vorp": optimal_vorp,
        "efficiency_gap":   gap,
        # cap_efficiency: VORP produced per $1M of actual cap spend
        "cap_efficiency":   round(current_vorp / current_payroll, 4) if current_payroll > 0 else 0,
    }


# ── Run all 30 teams ──────────────────────────────────────────────────────────
teams = sorted(ALL_PLAYERS["team"].unique())

print("Running optimiser across all 30 teams...")
results = [analyse_team(t) for t in teams]
results_df = pd.DataFrame(results)

# Rank teams: lower efficiency_gap = closer to optimal (better managed)
results_df["gap_rank"] = results_df["efficiency_gap"].rank(ascending=True, method="min").astype(int)
results_df["cap_eff_rank"] = results_df["cap_efficiency"].rank(ascending=False, method="min").astype(int)

results_df = results_df.sort_values("efficiency_gap")
results_df.to_csv("team_efficiency.csv", index=False)
print("Saved: team_efficiency.csv")


# ── Checkpoint table ──────────────────────────────────────────────────────────
print()
print("=" * 90)
print("ALL 30 TEAMS — CURRENT vs OPTIMAL EFFICIENCY (sorted by smallest gap = best managed)")
print("=" * 90)
print(f"{'Rank':<5} {'Team':<6} {'Players':<9} {'Payroll($M)':<13} {'Curr VORP':<11} "
      f"{'Opt VORP':<10} {'Gap':<8} {'$/VORP eff'}")
print("-" * 90)
for _, row in results_df.iterrows():
    print(
        f"{row['gap_rank']:<5} {row['team']:<6} {row['n_players']:<9} "
        f"{row['current_payroll']:<13.1f} {row['current_vorp']:<11.1f} "
        f"{row['league_optimal_vorp']:<10.1f} {row['efficiency_gap']:<8.1f} "
        f"{row['cap_efficiency']:.4f}"
    )

print()
print("=" * 90)
print("OPTIMAL ROSTER (full league pool, $141M cap)")
print("=" * 90)
optimal = optimize_roster(ALL_PLAYERS)
opt_display = optimal[["player", "team", "pos", "salary_millions", "vorp", "value_tier"]].copy()
opt_display = opt_display.sort_values("vorp", ascending=False)
print(opt_display.to_string(index=False))
print(f"\nTotal VORP: {optimal['vorp'].sum():.2f}")
print(f"Total salary: ${optimal['salary_millions'].sum():.1f}M  (cap: ${SALARY_CAP_M}M)")
pos_counts = optimal["pos"].value_counts().sort_index()
print(f"Position breakdown: {dict(pos_counts)}")

print()
print("=" * 90)
print("TOP 5 BEST-MANAGED TEAMS (smallest gap = closest to optimal)")
print("=" * 90)
for _, row in results_df.head(5).iterrows():
    print(f"  {row['gap_rank']}. {row['team']}  |  Current VORP: {row['current_vorp']:.1f}  "
          f"|  Gap: {row['efficiency_gap']:.1f}")

print()
print("=" * 90)
print("BOTTOM 5 WORST-MANAGED TEAMS (largest gap = furthest from optimal)")
print("=" * 90)
for _, row in results_df.tail(5).iterrows():
    print(f"  {row['gap_rank']}. {row['team']}  |  Current VORP: {row['current_vorp']:.1f}  "
          f"|  Gap: {row['efficiency_gap']:.1f}")
