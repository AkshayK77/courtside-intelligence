# Courtside Intelligence

A full-stack NBA roster efficiency analytics system that turns raw player data and contract values into a decision-grade dashboard for front-office style resource allocation. The project spans data ingestion, feature engineering, optimization metrics, and an interactive executive dashboard with an embedded AI analyst.

## What This Delivers
- **League-wide cap efficiency benchmarking** (VORP per $1M) across all 30 teams.
- **Efficient frontier analysis** to surface smart spenders vs. inefficient payrolls.
- **Optimization gap diagnostics** comparing current vs. theoretical roster efficiency.
- **Team deep dive** with position-level salary allocation vs. market value.
- **Playoff step-up analytics** connecting contract value to postseason performance.
- **East vs. West comparative analytics** indexed against league averages.
- **Interactive Dash application** designed for portfolio-grade storytelling.

## Dashboard Highlights
- Cap Efficiency Leaderboard
- Efficient Frontier (Payroll vs. VORP)
- Efficiency Gap Ranking
- Team Deep Dive (all 30 teams)
- East vs. West Conference Comparison
- Playoff Step-Up vs. Contract Value
- Embedded AI Analyst for guided interpretation

## Data Pipeline
1. **Raw stats ingestion** from Basketball Reference exports.
2. **Salary data integration** and normalization.
3. **Master dataset build** combining performance and pay.
4. **Value metric computation** using VORP-based market value modeling.
5. **Team-level aggregation** for efficiency and gap analysis.

## Repository Structure
- `build_master.py` - merges raw stats and salary data into `master.csv`
- `value_metric.py` - computes player value and overpay metrics
- `build_playoffs.py` - generates playoff vs. regular-season deltas
- `optimizer.py` - team efficiency modeling inputs
- `dashboard.py` - Dash app and visual analytics
- `agent.py` - AI analyst helper
- `*.csv` - curated data outputs used by the dashboard

## Quick Start
```bash
# 1) Create and activate a virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# 2) Install dependencies
pip install pandas numpy dash plotly requests beautifulsoup4 lxml

# 3) Run the dashboard
python dashboard.py
```
Open http://127.0.0.1:8050

## Key Metrics
- **VORP**: Value Over Replacement Player
- **Cap Efficiency**: VORP per $1M payroll
- **Efficiency Gap**: delta between current and league-optimal roster VORP
- **Overpay**: salary minus modeled market value

## Notes
- Data is aligned to the 2025-26 NBA season.
- The dashboard is designed for desktop viewing and one-page executive summary layouts.

## License
MIT
