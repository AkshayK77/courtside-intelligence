"""
Phase 1 checkpoint: verify master.csv looks correct before moving to Phase 2.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

df = pd.read_csv("master.csv")

print("=" * 60)
print("MASTER.CSV — OVERVIEW")
print("=" * 60)
print(f"Rows (players): {len(df)}")
print(f"Columns:        {len(df.columns)}")
print(f"Teams:          {df['team'].nunique()}")
print(f"Positions:      {sorted(df['pos'].unique())}")
print()

print("=" * 60)
print("SALARY RANGE")
print("=" * 60)
print(f"Min salary:  ${df['salary'].min():>12,.0f}  ({df.loc[df['salary'].idxmin(), 'player']})")
print(f"Max salary:  ${df['salary'].max():>12,.0f}  ({df.loc[df['salary'].idxmax(), 'player']})")
print(f"Avg salary:  ${df['salary'].mean():>12,.0f}")
print(f"Median:      ${df['salary'].median():>12,.0f}")
print()

print("=" * 60)
print("TOP 10 BY SALARY")
print("=" * 60)
top_sal = df.nlargest(10, "salary")[["player", "team", "salary_millions", "vorp", "ws"]]
print(top_sal.to_string(index=False))
print()

print("=" * 60)
print("TOP 10 BY VORP")
print("=" * 60)
top_vorp = df.nlargest(10, "vorp")[["player", "team", "salary_millions", "vorp", "ws", "bpm"]]
print(top_vorp.to_string(index=False))
print()

print("=" * 60)
print("TEAM PAYROLL SUMMARY (top 10)")
print("=" * 60)
team_pay = (
    df.groupby("team")["salary"].sum()
    .sort_values(ascending=False)
    .reset_index()
)
team_pay["payroll_millions"] = (team_pay["salary"] / 1e6).round(1)
print(team_pay.head(10).to_string(index=False))
print()

print("=" * 60)
print("MISSING / NULLS CHECK")
print("=" * 60)
nulls = df.isnull().sum()
nulls = nulls[nulls > 0]
if len(nulls):
    print(nulls)
else:
    print("No missing values.")
print()

print("=" * 60)
print("SAMPLE ROWS")
print("=" * 60)
cols = ["player", "team", "pos", "age", "salary_millions", "vorp", "ws", "bpm", "pts_per_game"]
print(df[cols].sample(10, random_state=42).to_string(index=False))
