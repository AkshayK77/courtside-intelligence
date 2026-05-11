# NBA Roster Optimizer — Full Project Context

## What We Are Building

A league-wide NBA roster optimization tool that answers: **"Which NBA teams get the most performance per dollar spent, and how should each team optimally reallocate their cap to maximize wins?"**

This is a portfolio project targeting MBA / Masters in Data Science / Business Analytics applications. It needs to be end-to-end: data pipeline → optimization model → interactive dashboard → embedded AI agent.

---

## The Core Idea (In Plain English)

Imagine you are an NBA GM with $140 million to spend on players. Some GMs are smart with money — they find undervalued players who perform above their salary. Others waste money on overpaid stars who don't move the needle.

This project is a "are you getting your money's worth?" report card for all 30 NBA teams. It tells you which teams spend their money wisely and which don't — and exactly what they should do differently.

---

## Tech Stack

- **Language:** Python
- **Data:** Kaggle dataset (NBA stats from Basketball Reference) — `Player Per Game.csv` and `Advanced.csv`
- **Salary data:** To be sourced (Spotrac was blocked, alternative TBD)
- **Optimization:** PuLP (integer linear programming)
- **Visualization:** Plotly / Dash
- **AI Agent:** Claude API embedded inside the dashboard (claude-sonnet-4-20250514)
- **Portfolio:** GitHub + Medium writeup + Loom walkthrough

---

## Data Sources

- `Player Per Game.csv` — basic stats (points, rebounds, assists, games played, minutes)
- `Advanced.csv` — advanced metrics: VORP, BPM, Win Shares, WS/48, PER
- Salary data — contract values per player in $USD (source TBD, Spotrac was blocked)
- Both CSV files are already downloaded into `C:\nba-optimizer\`
- Data covers the 2025-26 NBA season (final, complete season — OKC Thunder went 64-18, Spurs went 62-20)

---

## Project Structure

```
nba-optimizer/
├── venv/
├── Player Per Game.csv
├── Advanced.csv
├── raw_salaries.csv          (to be created)
├── master.csv                (final merged output of Phase 1)
├── fetch_stats.py            (Phase 1)
├── fetch_advanced.py         (Phase 1)
├── fetch_salaries.py         (Phase 1)
├── explore.py                (Phase 1 - verification)
├── build_master.py           (Phase 1 - merge)
├── value_metric.py           (Phase 2)
├── optimizer.py              (Phase 3)
├── scenarios.py              (Phase 3)
├── dashboard.py              (Phase 4)
└── agent.py                  (Phase 5)
```

---

## Phase Plan

### Phase 1 — Data Pipeline (current phase)
- Load `Player Per Game.csv` and `Advanced.csv` from Kaggle
- Source and load salary data per player
- Merge into one master dataframe: one row per player, columns for team, salary, and all performance metrics
- Export as `master.csv`
- **Checkpoint:** Open `master.csv` and see every NBA player with salary and stats side by side

### Phase 2 — Value Metric
- Calculate `$/VORP` (salary in millions divided by VORP) for every player
- Adjust for position scarcity
- Flag overpaid vs underpaid players
- **Checkpoint:** Ranked list of all players from most to least value-for-money

### Phase 3 — Optimization Model
- Build integer linear program using PuLP for each of the 30 teams
- Objective: maximize total team VORP
- Constraints: salary cap ($141M), roster size (15 players), minimum players per position group, luxury tax hard apron ($178M)
- Run three scenarios per team: current roster, free agency pool, full league pool
- Calculate efficiency gap (current vs optimal) for every team
- **Checkpoint:** Table of all 30 teams with current efficiency, optimal efficiency, and gap

### Phase 4 — Visualization & Dashboard
Four key visuals:
1. **Cap efficiency leaderboard** — all 30 teams ranked by $/VORP with color-coded badges (elite / average / poor)
2. **Efficient frontier scatter** — payroll on X axis, team VORP on Y axis. Teams top-left = smart spenders. Teams bottom-right = wasting money
3. **Efficiency gap ranking** — how much better could each team be without spending more
4. **Spurs deep dive** — current vs optimal cap allocation by position group (blue = current, green = optimal)
- **Checkpoint:** Working Plotly/Dash dashboard openable in a browser

### Phase 5 — AI Agent
- Embed a Claude-powered chat panel inside the Dash dashboard
- Give the agent full context about the data: what VORP means, what the charts show, which teams rank where
- User can ask plain-english questions like "why is OKC ranked #1?" or "what does the scatter chart mean?" and get clear answers grounded in the actual data
- Use `claude-sonnet-4-20250514` via the Anthropic API
- **Checkpoint:** A non-technical person can open the dashboard, look at a chart, ask the agent a question, and understand what they're seeing

### Phase 6 — Portfolio Polish
- Push to GitHub with clean README
- Write Medium post explaining the project in plain english
- Record 2-minute Loom walkthrough of the dashboard

---

## Key Metrics Explained

| Metric | What it means |
|--------|--------------|
| VORP | Value Over Replacement Player — how many wins a player adds vs a replacement-level player |
| BPM | Box Plus/Minus — a player's net point contribution per 100 possessions |
| Win Shares | Estimated number of wins a player contributed |
| WS/48 | Win Shares per 48 minutes — efficiency measure |
| PER | Player Efficiency Rating — overall per-minute production |
| $/VORP | Salary in millions divided by VORP — the core efficiency metric. Lower = better value |

---

## Optimization Model Detail

**Objective:**
```
Maximize: Σ (VORP_i × x_i)  for all players i
```

**Constraints:**
```
Σ salary_i × x_i  ≤  $141M          (salary cap)
Σ x_i = 15                           (roster spots)
Σ x_i ≥ 2 per position group         (roster balance)
total payroll ≤ $178M                 (hard apron)
x_i ∈ {0, 1}                         (binary — on roster or not)
```

This is essentially a knapsack problem — a classic operations research problem.

**Three scenarios per team:**
- Current roster — baseline, how efficient are they right now?
- Free agency pool — if they could sign any available FA, what's optimal?
- Full league pool — theoretical ceiling ignoring trade restrictions

---

## Dashboard AI Agent Detail

The agent knows:
- What every metric means in plain english
- The full league efficiency rankings
- Why specific teams rank where they do (tied to actual data)
- What the efficient frontier chart shows and how to read it
- What the efficiency gap means for each team
- Specific Spurs roster analysis

Example questions the agent should handle:
- "Why is OKC ranked #1?"
- "What does the scatter chart mean?"
- "Why are the Pacers so inefficient?"
- "What should the Wizards do differently?"
- "Explain VORP to me like I'm 5"

---

## Important Notes for Development

1. **NBA.com is blocked for Python scripts** — do not use `nba_api` for live fetching. Use the Kaggle CSV files already on disk.
2. **Basketball Reference returns 403** when scraped with `pd.read_html` directly — must use `requests` with browser headers first, then parse response text.
3. **Virtual environment** is at `C:\nba-optimizer\venv\` — always activate with `venv\Scripts\activate` before running scripts.
4. **Python version:** 3.11 on Windows
5. **Installed packages so far:** nba_api, pandas, requests, beautifulsoup4, lxml, openpyxl, html5lib
6. **Season context:** 2025-26 regular season is complete. OKC Thunder: 64-18. San Antonio Spurs: 62-20. Detroit Pistons: 60-22. Boston Celtics: 56-26.

---

## What Makes This MBA-Level

- End-to-end: data engineering + modeling + visualization + business narrative
- Novel: not a Kaggle tutorial, the problem is self-defined
- Current: uses 2025-26 final season data
- Transferable: same framework applies to any resource allocation problem in business
- The business framing: "Teams face a build-or-buy decision analogous to M&A vs organic growth"
- Executive summary: 1-page memo explaining recommendations as if presenting to a GM
- Sensitivity analysis: what happens if a star player gets injured? How robust is the optimal roster?
