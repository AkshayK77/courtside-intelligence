"""
Phase 5 — AI Agent
Groq-powered analyst embedded in the dashboard.
Knows the full dataset: team rankings, player values, what every chart means.
"""

import os
from dotenv import load_dotenv
import pandas as pd
from groq import Groq

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 600


# ── Build system prompt from real data ────────────────────────────────────────
def _build_system_prompt() -> str:
    te = pd.read_csv("team_efficiency.csv").sort_values("cap_efficiency", ascending=False)
    vm = pd.read_csv("value_metrics.csv")

    # Team rankings table
    team_rows = []
    for _, r in te.iterrows():
        team_rows.append(
            f"  {r['team']}: VORP={r['current_vorp']:.1f}, "
            f"Payroll=${r['current_payroll']:.0f}M, "
            f"Cap-efficiency={r['cap_efficiency']:.4f}, "
            f"Gap={r['efficiency_gap']:.1f}"
        )
    team_table = "\n".join(team_rows)

    # Top steals
    steals = vm.nsmallest(10, "overpay_millions")
    steal_rows = [
        f"  {r['player']} ({r['team']}): ${r['salary_millions']:.1f}M salary, "
        f"{r['vorp']:.1f} VORP, ${abs(r['overpay_millions']):.0f}M under market"
        for _, r in steals.iterrows()
    ]
    steal_table = "\n".join(steal_rows)

    # Most overpaid
    overpaid = vm.nlargest(10, "overpay_millions")
    overpaid_rows = [
        f"  {r['player']} ({r['team']}): ${r['salary_millions']:.1f}M salary, "
        f"{r['vorp']:.1f} VORP, ${r['overpay_millions']:.0f}M over market"
        for _, r in overpaid.iterrows()
    ]
    overpaid_table = "\n".join(overpaid_rows)

    # Spurs roster
    sas = vm[vm["team"] == "SAS"].sort_values("vorp", ascending=False)
    sas_rows = [
        f"  {r['player']} ({r['pos']}): ${r['salary_millions']:.1f}M, "
        f"{r['vorp']:.1f} VORP — {r['value_tier']}"
        for _, r in sas.iterrows()
    ]
    sas_table = "\n".join(sas_rows)

    # OKC roster
    okc = vm[vm["team"] == "OKC"].sort_values("vorp", ascending=False)
    okc_rows = [
        f"  {r['player']} ({r['pos']}): ${r['salary_millions']:.1f}M, "
        f"{r['vorp']:.1f} VORP — {r['value_tier']}"
        for _, r in okc.iterrows()
    ]
    okc_table = "\n".join(okc_rows)

    league_optimal_vorp = te["league_optimal_vorp"].iloc[0]

    # Playoff data
    po = pd.read_csv("playoffs.csv")
    po_players = po[po["in_playoffs"] == True].copy()

    top_po = po_players.nlargest(10, "po_eff")
    top_po_rows = [
        f"  {r['player']} ({r['team']}): po_eff={r['po_eff']:.1f}, "
        f"po_pts={r['po_pts']:.1f}, eff_delta={r['eff_delta']:+.1f}, "
        f"salary=${r['salary_millions']:.1f}M ({r['value_tier']})"
        for _, r in top_po.iterrows()
    ]
    top_po_table = "\n".join(top_po_rows)

    stepup = po_players.nlargest(8, "eff_delta")
    stepup_rows = [
        f"  {r['player']} ({r['team']}): eff_delta={r['eff_delta']:+.1f}, "
        f"rs_eff={r['rs_eff']:.1f}→po_eff={r['po_eff']:.1f}, "
        f"overpay=${r['overpay_millions']:.1f}M"
        for _, r in stepup.iterrows()
    ]
    stepup_table = "\n".join(stepup_rows)

    fades = po_players.nsmallest(6, "eff_delta")
    fades_rows = [
        f"  {r['player']} ({r['team']}): eff_delta={r['eff_delta']:+.1f}, "
        f"salary=${r['salary_millions']:.1f}M, overpay=${r['overpay_millions']:.1f}M"
        for _, r in fades.iterrows()
    ]
    fades_table = "\n".join(fades_rows)

    best_both = po_players[
        (po_players["overpay_millions"] < 0) & (po_players["eff_delta"] > 0)
    ].nlargest(8, "po_eff")
    best_both_rows = [
        f"  {r['player']} ({r['team']}): po_eff={r['po_eff']:.1f}, "
        f"eff_delta={r['eff_delta']:+.1f}, ${abs(r['overpay_millions']):.0f}M under market"
        for _, r in best_both.iterrows()
    ]
    best_both_table = "\n".join(best_both_rows)

    return f"""You are an NBA analytics expert embedded inside an interactive dashboard called the NBA Roster Optimizer.

Your job is to explain the data, charts, and findings to anyone — from curious fans to basketball executives.
Be concise (2-4 sentences max per answer unless asked for detail), conversational, and grounded in the actual numbers below.
Never make up numbers. If a question is outside the dataset, say so honestly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METRIC DEFINITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VORP (Value Over Replacement Player): How many extra wins a player adds vs a league-minimum replacement.
  Jokić led the league with 9.2 VORP. Below 0 means the player is worse than a replacement-level guy.

BPM (Box Plus/Minus): Net points a player contributes per 100 possessions. Positive = helps the team.

Win Shares (WS): Estimated number of team wins a player was responsible for.

WS/48: Win Shares per 48 minutes — measures efficiency, not volume.

PER (Player Efficiency Rating): Per-minute production metric. League average = 15.

$/VORP or cap_efficiency: Salary in millions divided by VORP. Lower = better value.
  In this dashboard we show VORP per $1M payroll (higher = better).

Overpay ($M): Actual salary minus what the player's VORP says they should earn (from a regression fit).
  Positive = overpaid. Negative = underpaid (a bargain).

Efficiency Gap: How many VORP points a team's current roster falls short of the theoretical
  $141M-cap optimal (25.4 VORP). Smaller gap = better-managed team.

EFF (Playoff Efficiency): PTS + REB + AST + STL + BLK − TOV. A per-game box-score efficiency measure.
  eff_delta = playoff EFF minus regular-season EFF. Positive = stepped up in the playoffs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE FIVE DASHBOARD CHARTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Cap Efficiency Leaderboard: All 30 teams ranked by VORP per $1M payroll.
   Green = elite (≥0.10), yellow = average (0.05–0.10), red = poor (<0.05).

2. Efficient Frontier Scatter: Payroll on the X-axis, team VORP on the Y-axis.
   Teams in the top-left quadrant spend less and produce more — smart spenders.
   Teams in the bottom-right spend a lot but get little VORP — wasting money.
   A dashed regression line shows the league average. Above it = efficient, below = inefficient.

3. Efficiency Gap Ranking: How many VORP points each team leaves on the table
   vs the theoretical $141M optimal roster (Jokić + SGA + Wembanyama + minimum fillers).
   Smaller bar = better roster construction.

4. Spurs Deep Dive: Blue bars = what SAS actually spends per position.
   Green bars = what that production is worth based on market rate (VORP regression).
   When blue > green, SAS is overpaying that position group.

5. Playoff Step-Up vs. Contract Value (scatter):
   X-axis = overpay ($M) — negative means underpaid, positive means overpaid.
   Y-axis = eff_delta — positive means player improved in playoffs vs regular season.
   Bubble size = playoff EFF (raw production).
   TOP-LEFT quadrant = "Best of Both Worlds": underpaid AND stepping up. Stars here are hidden gems.
   BOTTOM-RIGHT quadrant = worst contracts: overpaid AND fading in the playoffs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEASON CONTEXT — 2025-26 NBA (FINAL STANDINGS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OKC Thunder: 64-18 (best record in the league)
San Antonio Spurs: 62-20
Detroit Pistons: 60-22
Boston Celtics: 56-26
Salary cap: $141M | Hard apron: $178M

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALL 30 TEAMS — CAP EFFICIENCY RANKING
(sorted best → worst; cap_efficiency = VORP per $1M payroll)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{team_table}

League-optimal VORP ceiling (best possible $141M roster): {league_optimal_vorp:.1f}
That optimal roster: Nikola Jokić (9.2 VORP, $55.2M) + Shai Gilgeous-Alexander (7.8, $38.3M)
  + Victor Wembanyama (6.0, $13.4M) + Jalen Johnson (4.0, $30M) + 11 minimum-salary fillers

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOP 10 BIGGEST STEALS (underpaid vs market value)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{steal_table}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOP 10 MOST OVERPAID PLAYERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{overpaid_table}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHART 4: TEAM DEEP DIVE (any of 30 teams — user selects via dropdown)
Blue bars = actual cap spend by position. Green bars = market-rate value by VORP.
When blue > green at a position, the team is overpaying that position group.
The chart also shows total payroll, net overpay/underpay, and cap efficiency rank.

SAN ANTONIO SPURS ROSTER (example deep-dive team)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{sas_table}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OKC THUNDER ROSTER (#1 ranked team)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{okc_table}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOP 10 PLAYOFF PERFORMERS (by playoff EFF)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{top_po_table}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BIGGEST PLAYOFF STEP-UPS (eff_delta = po_eff - rs_eff)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{stepup_table}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BIGGEST PLAYOFF FADES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{fades_table}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BEST OF BOTH WORLDS (underpaid + playoff step-up)
These players are in the top-left quadrant of Chart 5.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{best_both_table}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TONE & STYLE GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Explain jargon the first time you use it.
- Use the actual numbers. Don't say "much better" — say "4.7 VORP gap".
- When comparing teams, anchor to a familiar reference (e.g., "that's like paying LeBron money for role-player production").
- If someone asks what they should do (as a GM), give a concrete recommendation.
- Keep answers to 3-5 sentences unless they ask for more detail.
"""


# ── Public API ────────────────────────────────────────────────────────────────
_system_prompt: str | None = None


def get_system_prompt() -> str:
    global _system_prompt
    if _system_prompt is None:
        _system_prompt = _build_system_prompt()
    return _system_prompt


def ask(user_message: str, history: list[dict]) -> str:
    """
    Call the Groq API and return the assistant reply.
    history: list of {"role": "user"|"assistant", "content": str}
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return "⚠️ No GROQ_API_KEY found. Add your key to the .env file and restart the dashboard."

    client = Groq(api_key=api_key)

    messages = [{"role": "system", "content": get_system_prompt()}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.6,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ API error: {e}"


def explain_prediction(
    team_name: str,
    predicted_wins: float,
    actual_wins: float | None,
    feature_importance: dict,
    team_stats: dict,
) -> str:
    """
    Return a plain-english paragraph explaining the model's win prediction.

    Parameters
    ----------
    team_name        : e.g. "OKC Thunder"
    predicted_wins   : float from the model
    actual_wins      : actual wins this season, or None if unavailable
    feature_importance: {feature_name: importance_value} sorted desc
    team_stats       : {stat_name: value} for total_vorp, total_salary_m, etc.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return ("⚠️ No GROQ_API_KEY found. Add your key to the .env file "
                "and restart the dashboard.")

    # Build a concise prompt describing the prediction context
    delta_str = ""
    if actual_wins is not None:
        delta = predicted_wins - actual_wins
        delta_str = (f"The model predicted {predicted_wins:.0f} wins vs "
                     f"{actual_wins:.0f} actual wins (delta: {delta:+.0f}).")
    else:
        delta_str = f"The model predicted {predicted_wins:.0f} wins (actual wins unknown)."

    # Top two features
    top_feats = list(feature_importance.items())[:2]
    feat_lines = "\n".join(
        f"  - {feat}: importance={imp:.4f}, team value={team_stats.get(feat, 'N/A')}"
        for feat, imp in top_feats
    )

    # League average context from team_efficiency.csv if available
    league_ctx = ""
    try:
        te = pd.read_csv("team_efficiency.csv")
        avg_vorp = te["current_vorp"].mean()
        avg_pay  = te["current_payroll"].mean()
        avg_eff  = te["cap_efficiency"].mean()
        league_ctx = (
            f"League averages this season: VORP={avg_vorp:.1f}, "
            f"Payroll=${avg_pay:.0f}M, Cap-efficiency={avg_eff:.4f}."
        )
    except Exception:
        pass

    stats_str = ", ".join(f"{k}={v}" for k, v in team_stats.items())

    prompt = f"""You are an NBA analytics expert.
{delta_str}

Team: {team_name}
Team stats: {stats_str}
{league_ctx}

Top two features driving this prediction:
{feat_lines}

Write a single paragraph (4–6 sentences) that:
1. States the predicted wins and how close it is to actual (if known).
2. Identifies the top two features driving the prediction and explains their significance.
3. Compares this team to the league average on those two features.
4. Ends with one concrete sentence about what the team could do differently to improve their predicted win total.

Be specific with numbers. Do not use bullet points."""

    client = Groq(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.6,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ API error: {e}"


if __name__ == "__main__":
    # Quick smoke test
    print(ask("Why is OKC ranked #1?", []))
    print()
    print(ask("What does VORP mean?", []))
