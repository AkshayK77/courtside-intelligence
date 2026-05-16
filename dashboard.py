"""
Phase 4+5+6+7 — NBA Roster Optimizer Dashboard with AI Agent + Win Predictor tab.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output, State, callback, ALL
import plotly.graph_objects as go
import plotly.express as px
import joblib
import agent

# ── Load win model artifacts (optional — tab degrades gracefully if missing) ──
_WIN_MODEL = None
_FEATURE_IMP = None
_TRAINING_DATA = None

try:
    _WIN_MODEL    = joblib.load('win_model.pkl')
    _FEATURE_IMP  = pd.read_csv('feature_importance.csv')
    _TRAINING_DATA = pd.read_csv('training_data.csv')
except Exception:
    pass  # model not yet trained — Win Predictor tab shows instructions

# ── Load data ─────────────────────────────────────────────────────────────────
te = pd.read_csv("team_efficiency.csv")   # 30 teams
vm = pd.read_csv("value_metrics.csv")     # individual players
po = pd.read_csv("playoffs.csv")          # playoff + RS stats merged

# ── Colour palette ────────────────────────────────────────────────────────────
BG_PAGE   = "#0d1117"
BG_CHART  = "#161b22"
BG_CARD   = "#1c2128"
BORDER    = "#30363d"
TEXT      = "#e6edf3"
TEXT_DIM  = "#8b949e"
GREEN     = "#3fb950"
YELLOW    = "#d29922"
RED       = "#f85149"
BLUE      = "#58a6ff"
GOLD      = "#ffd700"

FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"

CHART_LAYOUT = dict(
    paper_bgcolor=BG_CHART,
    plot_bgcolor=BG_CHART,
    font=dict(family=FONT_FAMILY, color=TEXT, size=12),
    title_font=dict(size=14, color=TEXT),
)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1 — Cap Efficiency Leaderboard
# Horizontal bar: VORP per $1M payroll, colour-coded by tier
# ─────────────────────────────────────────────────────────────────────────────
def make_leaderboard():
    df = te.sort_values("cap_efficiency", ascending=True).copy()

    # Colour by performance tier
    def tier_color(v):
        if v >= 0.10:
            return GREEN
        if v >= 0.05:
            return YELLOW
        return RED

    colors = df["cap_efficiency"].apply(tier_color)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["cap_efficiency"],
        y=df["team"],
        orientation="h",
        marker_color=colors,
        text=df["cap_efficiency"].apply(lambda v: f"{v:.4f}"),
        textposition="outside",
        textfont=dict(size=10, color=TEXT),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "VORP per $1M: %{x:.4f}<br>"
            "<extra></extra>"
        ),
    ))

    # Tier legend annotations
    fig.add_annotation(x=0.97, y=0.10, xref="paper", yref="paper",
        text="■ Elite (≥0.10)", font=dict(color=GREEN, size=10),
        showarrow=False, xanchor="right")
    fig.add_annotation(x=0.97, y=0.05, xref="paper", yref="paper",
        text="■ Average (0.05–0.10)", font=dict(color=YELLOW, size=10),
        showarrow=False, xanchor="right")
    fig.add_annotation(x=0.97, y=0.00, xref="paper", yref="paper",
        text="■ Poor (<0.05)", font=dict(color=RED, size=10),
        showarrow=False, xanchor="right")

    fig.update_layout(
        **CHART_LAYOUT,
        title="Cap Efficiency Leaderboard — VORP per $1M Payroll",
        xaxis=dict(
            title="VORP / $1M Payroll",
            gridcolor=BORDER,
            zerolinecolor=BORDER,
            color=TEXT_DIM,
        ),
        yaxis=dict(gridcolor=BORDER, color=TEXT_DIM, tickfont=dict(size=10)),
        height=360,
        margin=dict(l=60, r=120, t=45, b=25),
        bargap=0.25,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart 2 — Efficient Frontier Scatter
# Payroll (X) vs VORP (Y) — top-left = smart spenders
# ─────────────────────────────────────────────────────────────────────────────
def make_frontier():
    df = te.copy()

    # Linear reference line (regression fit)
    coeffs = np.polyfit(df["current_payroll"], df["current_vorp"], 1)
    x_line = np.linspace(df["current_payroll"].min() - 5,
                         df["current_payroll"].max() + 5, 100)
    y_line = coeffs[0] * x_line + coeffs[1]

    # Colour: above regression = green (efficient), below = red (inefficient)
    df["pred_vorp"] = coeffs[0] * df["current_payroll"] + coeffs[1]
    df["residual"] = df["current_vorp"] - df["pred_vorp"]

    def dot_color(r):
        if r > 2:
            return GREEN
        if r > -2:
            return YELLOW
        return RED

    colors = df["residual"].apply(dot_color)
    sizes  = (df["current_vorp"].clip(lower=0) + 5) * 2.5

    fig = go.Figure()

    # Reference line
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines",
        line=dict(color=BORDER, width=1.5, dash="dash"),
        name="League average",
        hoverinfo="skip",
    ))

    # Team dots
    fig.add_trace(go.Scatter(
        x=df["current_payroll"],
        y=df["current_vorp"],
        mode="markers+text",
        marker=dict(color=colors, size=sizes, line=dict(color=BG_CHART, width=1)),
        text=df["team"],
        textposition="top center",
        textfont=dict(size=9, color=TEXT),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Payroll: $%{x:.0f}M<br>"
            "VORP: %{y:.1f}<br>"
            "<extra></extra>"
        ),
        name="Teams",
    ))

    # Quadrant labels
    mid_x = df["current_payroll"].median()
    mid_y = df["current_vorp"].median()
    quadrants = [
        (df["current_payroll"].min(), df["current_vorp"].max() * 0.95,
         "SMART<br>SPENDERS", GREEN),
        (df["current_payroll"].max() * 0.97, df["current_vorp"].max() * 0.95,
         "EXPENSIVE<br>& GOOD", YELLOW),
        (df["current_payroll"].min(), df["current_vorp"].min() + 1,
         "CHEAP<br>& BAD", TEXT_DIM),
        (df["current_payroll"].max() * 0.97, df["current_vorp"].min() + 1,
         "WASTING<br>MONEY", RED),
    ]
    for qx, qy, label, col in quadrants:
        fig.add_annotation(
            x=qx, y=qy, text=label,
            font=dict(size=9, color=col),
            showarrow=False, xanchor="left" if qx < mid_x else "right",
            opacity=0.55,
        )

    fig.update_layout(
        **CHART_LAYOUT,
        title="Efficient Frontier — Payroll vs. Team VORP",
        xaxis=dict(
            title="Total Payroll ($M)",
            gridcolor=BORDER, zerolinecolor=BORDER, color=TEXT_DIM,
        ),
        yaxis=dict(
            title="Total Team VORP",
            gridcolor=BORDER, zerolinecolor=BORDER, color=TEXT_DIM,
        ),
        showlegend=False,
        height=270,
        margin=dict(l=60, r=20, t=45, b=40),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart 3 — Efficiency Gap Ranking
# How many VORP points each team is leaving on the table
# ─────────────────────────────────────────────────────────────────────────────
def make_gap_chart():
    df = te.sort_values("efficiency_gap", ascending=False).copy()

    def gap_color(g):
        if g <= 8:
            return GREEN
        if g <= 15:
            return YELLOW
        return RED

    colors = df["efficiency_gap"].apply(gap_color)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["team"],
        y=df["efficiency_gap"],
        marker_color=colors,
        text=df["efficiency_gap"].apply(lambda v: f"+{v:.1f}"),
        textposition="outside",
        textfont=dict(size=10, color=TEXT),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Current VORP: %{customdata[0]:.1f}<br>"
            "Optimal VORP: %{customdata[1]:.1f}<br>"
            "Gap: %{y:.1f}<br>"
            "<extra></extra>"
        ),
        customdata=df[["current_vorp", "league_optimal_vorp"]].values,
    ))

    # Threshold lines
    fig.add_hline(y=8,  line=dict(color=GREEN,  width=1, dash="dot"), opacity=0.6)
    fig.add_hline(y=15, line=dict(color=YELLOW, width=1, dash="dot"), opacity=0.6)

    fig.update_layout(
        **CHART_LAYOUT,
        title="Efficiency Gap — VORP Points Left on the Table vs. $141M Optimal",
        xaxis=dict(gridcolor=BORDER, color=TEXT_DIM, tickangle=-45,
                   tickfont=dict(size=10)),
        yaxis=dict(
            title="VORP Gap (lower = better managed)",
            gridcolor=BORDER, zerolinecolor=BORDER, color=TEXT_DIM,
        ),
        height=270,
        margin=dict(l=60, r=20, t=45, b=60),
        bargap=0.3,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart 4 — Team Deep Dive (any of the 30 teams)
# Current salary vs market-rate value by position, plus roster table
# ─────────────────────────────────────────────────────────────────────────────
ALL_TEAMS = sorted(vm["team"].unique())

TEAM_NAMES = {
    "ATL": "Atlanta Hawks",     "BOS": "Boston Celtics",   "BRK": "Brooklyn Nets",
    "CHO": "Charlotte Hornets", "CHI": "Chicago Bulls",    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",  "DEN": "Denver Nuggets",   "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers",       "LAL": "LA Lakers",        "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",        "MIL": "Milwaukee Bucks",  "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "OKC Thunder",
    "ORL": "Orlando Magic",     "PHI": "Philadelphia 76ers", "PHO": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",   "UTA": "Utah Jazz",        "WAS": "Washington Wizards",
}

CONFERENCE = {
    "ATL": "East", "BOS": "East", "BRK": "East", "CHO": "East", "CHI": "East",
    "CLE": "East", "DET": "East", "IND": "East", "MIA": "East", "MIL": "East",
    "NYK": "East", "ORL": "East", "PHI": "East", "TOR": "East", "WAS": "East",
    "DAL": "West", "DEN": "West", "GSW": "West", "HOU": "West", "LAC": "West",
    "LAL": "West", "MEM": "West", "MIN": "West", "NOP": "West", "OKC": "West",
    "PHO": "West", "POR": "West", "SAC": "West", "SAS": "West", "UTA": "West",
}

def make_team_deepdive(team_code: str) -> go.Figure:
    roster = vm[vm["team"] == team_code].copy()
    team_label = TEAM_NAMES.get(team_code, team_code)

    pos_order  = ["PG", "SG", "SF", "PF", "C"]
    pos_labels = {"PG": "Point Guard", "SG": "Shooting Guard",
                  "SF": "Small Forward", "PF": "Power Forward", "C": "Center"}

    current = (
        roster.groupby("pos")["salary_millions"]
        .sum()
        .reindex(pos_order, fill_value=0)
    )
    expected = (
        roster.groupby("pos")["expected_salary_millions"]
        .sum()
        .reindex(pos_order, fill_value=0)
    )

    labels = [pos_labels[p] for p in pos_order]

    # Cap summary
    total_sal = roster["salary_millions"].sum()
    total_exp = roster["expected_salary_millions"].sum()
    eff_rank  = ""
    ranked_te = te.sort_values("cap_efficiency", ascending=False).reset_index(drop=True)
    match     = ranked_te[ranked_te["team"] == team_code]
    if not match.empty:
        eff_rank = f" · #{match.index[0] + 1} cap efficiency"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Current Spend",
        x=labels,
        y=current.values,
        marker_color=BLUE,
        opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Current: $%{y:.1f}M<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Market Value (by VORP)",
        x=labels,
        y=expected.values,
        marker_color=GREEN,
        opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Market value: $%{y:.1f}M<extra></extra>",
    ))

    # Annotate mismatches ≥ $4M
    for i, pos in enumerate(pos_order):
        diff = current[pos] - expected[pos]
        if abs(diff) >= 4:
            label_text = f"{'▲' if diff > 0 else '▼'} ${abs(diff):.0f}M"
            color = RED if diff > 0 else GREEN
            fig.add_annotation(
                x=labels[i],
                y=max(current[pos], expected[pos]) + 1.2,
                text=label_text,
                font=dict(size=10, color=color),
                showarrow=False,
            )

    net = total_sal - total_exp
    net_txt = f"  ·  Net overpay: {'▲' if net > 0 else '▼'} ${abs(net):.0f}M"

    fig.update_layout(
        **CHART_LAYOUT,
        title=f"{team_label} — Cap Allocation vs. Market Value by Position"
              f"<br><sub>Total payroll: ${total_sal:.0f}M{net_txt}{eff_rank}</sub>",
        xaxis=dict(gridcolor=BORDER, color=TEXT_DIM),
        yaxis=dict(
            title="Salary ($M)",
            gridcolor=BORDER, zerolinecolor=BORDER, color=TEXT_DIM,
        ),
        barmode="group",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.08,
            xanchor="right", x=1,
            font=dict(color=TEXT),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=270,
        margin=dict(l=60, r=20, t=60, b=35),
        bargap=0.2,
        bargroupgap=0.08,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart 5 — East vs West Conference Comparison
# Index vs league average (100 = league avg)
# ─────────────────────────────────────────────────────────────────────────────
def make_conference_compare():
    df = te.copy()
    df["conference"] = df["team"].map(CONFERENCE).fillna("Unknown")

    metrics = [
        ("Cap Efficiency (VORP/$1M)", "cap_efficiency", True),
        ("Team VORP", "current_vorp", True),
        ("Efficiency Gap (lower is better)", "efficiency_gap", False),
    ]

    conf_avg = df.groupby("conference", as_index=True).mean(numeric_only=True)
    league_avg = df.mean(numeric_only=True)

    def index_value(metric_key, higher_is_better, conf_name):
        conf_val = conf_avg.loc[conf_name, metric_key]
        league_val = league_avg[metric_key]
        if higher_is_better:
            return (conf_val / league_val) * 100
        return (league_val / conf_val) * 100

    x_labels = [m[0] for m in metrics]
    east_vals = [index_value(m[1], m[2], "East") for m in metrics]
    west_vals = [index_value(m[1], m[2], "West") for m in metrics]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="East",
        x=x_labels,
        y=east_vals,
        marker_color=BLUE,
        text=[f"{v:.0f}" for v in east_vals],
        textposition="outside",
        hovertemplate="<b>East</b><br>%{x}: %{y:.1f} index<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="West",
        x=x_labels,
        y=west_vals,
        marker_color=GREEN,
        text=[f"{v:.0f}" for v in west_vals],
        textposition="outside",
        hovertemplate="<b>West</b><br>%{x}: %{y:.1f} index<extra></extra>",
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=(
            "East vs. West — Conference Comparison"
            "<br><sub>Index vs league average (100 = avg; efficiency gap inverted)</sub>"
        ),
        yaxis=dict(
            title="Index (100 = league average)",
            gridcolor=BORDER,
            zerolinecolor=BORDER,
            color=TEXT_DIM,
        ),
        xaxis=dict(gridcolor=BORDER, color=TEXT_DIM),
        barmode="group",
        height=240,
        margin=dict(l=60, r=20, t=60, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(color=TEXT, size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart 5 — Playoff Step-Up vs. Contract Value
# X: overpay_millions  Y: eff_delta  Size: po_eff  Color: value_tier
# Top-left quadrant = underpaid AND stepping up = "Best of Both Worlds"
# ─────────────────────────────────────────────────────────────────────────────
def make_playoff_scatter():
    df = po[po["in_playoffs"]].copy()
    df = df.dropna(subset=["eff_delta", "overpay_millions", "po_eff"])

    TIER_COLORS = {
        "Elite Value": GREEN,
        "Good Value":  BLUE,
        "Fair":        YELLOW,
        "Fringe":      TEXT_DIM,
        "Poor Value":  "#e06c75",
        "Overpaid":    RED,
    }

    fig = go.Figure()

    # Draw quadrant dividers
    fig.add_vline(x=0,  line=dict(color=BORDER, width=1, dash="dash"))
    fig.add_hline(y=0,  line=dict(color=BORDER, width=1, dash="dash"))

    # One trace per tier so the legend is useful
    for tier, color in TIER_COLORS.items():
        sub = df[df["value_tier"] == tier]
        if sub.empty:
            continue
        sizes = (sub["po_eff"].clip(lower=5) * 1.6).clip(upper=40)
        fig.add_trace(go.Scatter(
            x=sub["overpay_millions"],
            y=sub["eff_delta"],
            mode="markers+text",
            name=tier,
            marker=dict(
                size=sizes,
                color=color,
                opacity=0.82,
                line=dict(color=BG_CHART, width=0.8),
            ),
            text=sub.apply(
                lambda r: r["player"].split()[-1] if r["po_eff"] > 28 else "",
                axis=1,
            ),
            textposition="top center",
            textfont=dict(size=9, color=TEXT),
            customdata=sub[["player", "team", "salary_millions",
                            "overpay_millions", "po_pts", "po_eff", "eff_delta"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                "Salary: $%{customdata[2]:.1f}M  ·  Overpay: $%{customdata[3]:.1f}M<br>"
                "Playoff pts: %{customdata[4]:.1f}  ·  Playoff EFF: %{customdata[5]:.1f}<br>"
                "EFF step-up vs RS: %{customdata[6]:+.1f}<br>"
                "<extra></extra>"
            ),
        ))

    # Quadrant labels
    x_min, x_max = df["overpay_millions"].min(), df["overpay_millions"].max()
    y_min, y_max = df["eff_delta"].min(), df["eff_delta"].max()
    pad_x = (x_max - x_min) * 0.04
    pad_y = (y_max - y_min) * 0.04

    quadrant_labels = [
        (x_min + pad_x, y_max - pad_y, "⭐ BEST OF BOTH WORLDS<br>Underpaid + Stepped Up", GREEN,  "left"),
        (x_max - pad_x, y_max - pad_y, "Pricey but Clutch",                                YELLOW, "right"),
        (x_min + pad_x, y_min + pad_y, "Bargain but Fading",                               TEXT_DIM,"left"),
        (x_max - pad_x, y_min + pad_y, "⚠ Overpaid + Fading",                              RED,    "right"),
    ]
    for qx, qy, label, col, anchor in quadrant_labels:
        fig.add_annotation(
            x=qx, y=qy, text=label,
            font=dict(size=9, color=col),
            showarrow=False, xanchor=anchor,
            opacity=0.65,
        )

    fig.update_layout(
        **CHART_LAYOUT,
        title="Playoff Step-Up vs. Contract Value — Who Earns Their Money When It Counts?",
        xaxis=dict(
            title="Overpay vs. Market Rate ($M)  ◀ underpaid · overpaid ▶",
            gridcolor=BORDER, zerolinecolor=BORDER, color=TEXT_DIM,
        ),
        yaxis=dict(
            title="EFF Delta (Playoffs − Regular Season)  ▲ stepped up · faded ▼",
            gridcolor=BORDER, zerolinecolor=BORDER, color=TEXT_DIM,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(color=TEXT, size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=300,
        margin=dict(l=70, r=20, t=60, b=45),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# KPI cards
# ─────────────────────────────────────────────────────────────────────────────
def kpi_card(label, value, sub="", color=TEXT):
    return html.Div([
        html.P(label, style={"margin": "0", "fontSize": "11px",
                             "color": TEXT_DIM, "textTransform": "uppercase",
                             "letterSpacing": "0.8px"}),
        html.H3(value, style={"margin": "4px 0", "fontSize": "22px",
                              "fontWeight": "700", "color": color}),
        html.P(sub, style={"margin": "0", "fontSize": "11px", "color": TEXT_DIM}),
    ], style={
        "backgroundColor": BG_CARD,
        "border": f"1px solid {BORDER}",
        "borderRadius": "8px",
        "padding": "12px 16px",
        "flex": "1",
        "minWidth": "120px",
    })


best_team = te.loc[te["cap_efficiency"].idxmax()]
worst_team = te.loc[te["cap_efficiency"].idxmin()]
most_overpaid = vm.loc[vm["overpay_millions"].idxmax()]
biggest_steal = vm.loc[vm["overpay_millions"].idxmin()]

kpis = html.Div([
    kpi_card("Best Cap Efficiency", best_team["team"],
             f"{best_team['cap_efficiency']:.4f} VORP/$1M", GREEN),
    kpi_card("Worst Cap Efficiency", worst_team["team"],
             f"{worst_team['cap_efficiency']:.4f} VORP/$1M", RED),
    kpi_card("Most Overpaid Player", most_overpaid["player"],
             f"+${most_overpaid['overpay_millions']:.0f}M over market", RED),
    kpi_card("Biggest Steal", biggest_steal["player"],
             f"${abs(biggest_steal['overpay_millions']):.0f}M under market", GREEN),
    kpi_card("League-Optimal VORP Ceiling", f"{te['league_optimal_vorp'].iloc[0]:.1f}",
             "Best 15-man $141M roster", GOLD),
], style={
    "display": "flex",
    "flexWrap": "wrap",
    "gap": "8px",
    "marginBottom": "14px",
})


# ─────────────────────────────────────────────────────────────────────────────
# ASK AI button + per-chart starter questions
# ─────────────────────────────────────────────────────────────────────────────
CHART_QUESTIONS = {
    "leaderboard": "Which teams are getting the most value for their money, and why?",
    "frontier":    "What does the efficient frontier chart tell us about how teams are spending?",
    "gap":         "Which teams have the biggest gap between their current roster and the optimal, and what should they do about it?",
    "deepdive":    "Analyze the team deep dive chart — where is this team overpaying relative to market value, and what should they change?",
    "conference":  "How do the East and West compare on cap efficiency, VORP, and efficiency gap?",
    "playoffs":    "Who are the best bargains in the playoffs — underpaid players who are also stepping up their game?",
}

def _ask_ai_btn(chart_id: str) -> html.Button:
    return html.Button(
        "✦ ASK AI",
        id={"type": "ask-ai-btn", "chart": chart_id},
        n_clicks=0,
        className="ask-ai-tag",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chat helpers
# ─────────────────────────────────────────────────────────────────────────────
def _make_bubble(role: str, text: str) -> html.Div:
    is_user = role == "user"
    return html.Div([
        html.Span("You  " if is_user else "🤖 ", style={
            "color": GOLD if is_user else BLUE, "fontWeight": "600",
        }),
        html.Span(text, style={"color": TEXT, "whiteSpace": "pre-wrap"}),
    ], style={
        "padding": "10px 14px",
        "marginBottom": "8px",
        "backgroundColor": "#1c2a1c" if is_user else "#1a2332",
        "borderRadius": "8px",
        "borderLeft": f"3px solid {GOLD if is_user else BLUE}",
        "fontSize": "13px",
        "lineHeight": "1.6",
    })


def _welcome_bubble() -> html.Div:
    return _make_bubble(
        "assistant",
        "Hi! I'm your NBA analytics analyst. Ask me anything — why a team "
        "ranks where it does, what a metric means, or what a GM should do differently.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Glossary tab
# ─────────────────────────────────────────────────────────────────────────────

GLOSSARY = [
    {
        "term": "VORP — Value Over Replacement Player",
        "abbr": "VORP",
        "color": BLUE,
        "body": (
            "Measures how many extra wins a player adds compared to a "
            "league-minimum replacement-level player, over the full season. "
            "A VORP of 0 means the player is exactly replacement level. "
            "Positive = contributes wins; negative = below replacement. "
            "Nikola Jokić led the league this season at 9.2 VORP."
        ),
        "formula": "VORP = (BPM − (−2.0)) × (MP / (82 × 48)) × team_pace_adj",
        "range": "Typical range: −2 to +10 per season. Elite: >5. Starter: 1–3. Role player: 0–1.",
        "use": "Used in this dashboard to rank cap efficiency, score the optimal roster, and predict team wins.",
    },
    {
        "term": "BPM — Box Plus/Minus",
        "abbr": "BPM",
        "color": BLUE,
        "body": (
            "Estimates how many net points per 100 possessions a player contributes "
            "compared to a league-average player, using only box score stats. "
            "Split into OBPM (offensive) and DBPM (defensive). "
            "League average = 0.0 by definition."
        ),
        "formula": "BPM = OBPM + DBPM  (estimated from PTS, REB, AST, STL, BLK, TOV, FGA, FTA, MP)",
        "range": "Elite: >+6. Good starter: +2 to +4. Average: −1 to +1. Poor: <−2.",
        "use": "Used as a model feature (avg BPM, minutes-weighted) in the Win Predictor.",
    },
    {
        "term": "Win Shares — WS",
        "abbr": "WS",
        "color": GREEN,
        "body": (
            "Estimates the number of team wins a player is responsible for across the season. "
            "Split into Offensive Win Shares (OWS) and Defensive Win Shares (DWS). "
            "The key property: sum of all players' WS on a team ≈ that team's win total. "
            "This is how we derive team wins from player data in the Win Predictor."
        ),
        "formula": "WS = OWS + DWS  (derived from marginal offense/defense vs. league average)",
        "range": "Full season leader: ~15 WS. Solid starter: 4–8 WS. Role player: 1–3 WS.",
        "use": "Win Shares sum per team is the 'actual wins' target in the prediction model.",
    },
    {
        "term": "WS/48 — Win Shares per 48 Minutes",
        "abbr": "WS/48",
        "color": GREEN,
        "body": (
            "The per-minute version of Win Shares — removes the volume bias so you can "
            "compare a bench player who plays 15 minutes to a starter who plays 35. "
            "League average is roughly 0.100."
        ),
        "formula": "WS/48 = WS / (minutes played / 48)",
        "range": "Elite: >0.200. Good: 0.120–0.180. Average: ~0.100. Below average: <0.060.",
        "use": "Shown in the roster table; useful for identifying efficient bench players.",
    },
    {
        "term": "PER — Player Efficiency Rating",
        "abbr": "PER",
        "color": YELLOW,
        "body": (
            "A per-minute summary of a player's statistical accomplishments, "
            "normalized so the league average is always 15.0. "
            "It rewards counting stats (points, rebounds, assists) but "
            "doesn't fully account for defense or shot quality."
        ),
        "formula": "PER = uPER × (15 / lgPER)  where uPER is a weighted sum of per-minute box stats",
        "range": "MVP caliber: >27. All-Star: 20–27. Starter: 15–18. Bench: 10–14. Fringe: <10.",
        "use": "Context metric in the roster table; not a primary model feature due to defensive blind spots.",
    },
    {
        "term": "TS% — True Shooting Percentage",
        "abbr": "TS%",
        "color": YELLOW,
        "body": (
            "Measures shooting efficiency across all three shot types — 2-pointers, "
            "3-pointers, and free throws — on a single scale. "
            "Fixes the flaw in plain FG% which penalizes players who draw fouls or shoot threes."
        ),
        "formula": "TS% = PTS / (2 × (FGA + 0.44 × FTA))",
        "range": "Elite: >62%. Good: 57–62%. Average: ~55%. Poor: <52%.",
        "use": "Used as a context stat. Team-level TS% is a proxy for offensive efficiency.",
    },
    {
        "term": "Cap Efficiency",
        "abbr": "Cap Eff.",
        "color": GREEN,
        "body": (
            "How much VORP a team gets per $1M of payroll. "
            "Higher is better — it means the team is extracting more on-court value "
            "per dollar spent. The primary ranking metric in the Cap Efficiency Leaderboard."
        ),
        "formula": "Cap Efficiency = Team VORP / Total Payroll ($M)",
        "range": "Elite (green): >0.10. Average (yellow): 0.05–0.10. Poor (red): <0.05.",
        "use": "Main ranking metric on the Cap Analytics tab. OKC and SAS lead this season.",
    },
    {
        "term": "Efficiency Gap",
        "abbr": "Gap",
        "color": RED,
        "body": (
            "How many VORP points a team's current roster falls short of the theoretical "
            "best possible $141M roster. The optimal roster is built by greedy selection: "
            "take the highest-VORP players until the cap is full. "
            "A smaller gap = better roster construction relative to the cap."
        ),
        "formula": "Efficiency Gap = League-Optimal VORP (25.4) − Team's Current VORP",
        "range": "Well-managed: <8. Average: 8–15. Poorly constructed: >15.",
        "use": "Shown in the Efficiency Gap chart. Tells GMs how much upside they're leaving on the table.",
    },
    {
        "term": "Overpay ($M)",
        "abbr": "Overpay",
        "color": RED,
        "body": (
            "The difference between a player's actual salary and what their VORP says "
            "they should earn based on a regression of salary vs. VORP across all players. "
            "Positive = overpaid. Negative = underpaid (a bargain). "
            "Max Contracts often produce the biggest overpays when stars decline."
        ),
        "formula": "Overpay = Actual Salary − Predicted Salary(VORP)  [from regression fit]",
        "range": "Most overpaid this season: >$20M over market. Biggest steals: >$15M under.",
        "use": "Drives the playoff scatter chart and the 'Biggest Steal / Most Overpaid' KPI cards.",
    },
    {
        "term": "VORP Concentration",
        "abbr": "Conc.",
        "color": TEXT_DIM,
        "body": (
            "What fraction of a team's total VORP comes from its single best player. "
            "A high concentration means the team is a one-star system — great if healthy, "
            "catastrophic if injured. A low concentration means distributed production "
            "across the roster (more resilient but harder to win a title)."
        ),
        "formula": "VORP Concentration = Top Player VORP / Team Total VORP  (clamped at denominator ≥ 1)",
        "range": "Star-dependent: >0.50. Balanced: 0.25–0.45. Very deep: <0.20.",
        "use": "One of the five features in the Win Predictor model.",
    },
    {
        "term": "EFF — Playoff Efficiency",
        "abbr": "EFF",
        "color": GOLD,
        "body": (
            "A simple per-game box-score efficiency metric used in the Playoff Step-Up chart. "
            "Rewards players who contribute across multiple categories. "
            "eff_delta = playoff EFF minus regular-season EFF — positive means the player "
            "raised their game in the playoffs."
        ),
        "formula": "EFF = PTS + REB + AST + STL + BLK − TOV  (per game)",
        "range": "Elite playoff performer: >35 EFF. Solid contributor: 20–30. Role player: 10–20.",
        "use": "Y-axis of the Playoff Step-Up vs. Contract Value scatter chart.",
    },
]


def glossary_card(entry: dict) -> html.Div:
    return html.Div([
        # Header row: colored abbreviation chip + full term
        html.Div([
            html.Span(entry["abbr"], style={
                "backgroundColor": entry["color"],
                "color": "#0d1117",
                "fontWeight": "800",
                "fontSize": "11px",
                "padding": "3px 9px",
                "borderRadius": "999px",
                "letterSpacing": "0.5px",
                "flexShrink": "0",
            }),
            html.Span(entry["term"], style={
                "fontWeight": "700",
                "fontSize": "14px",
                "color": TEXT,
                "marginLeft": "10px",
            }),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"}),

        # Plain-English explanation
        html.P(entry["body"], style={
            "margin": "0 0 10px",
            "fontSize": "13px",
            "color": TEXT,
            "lineHeight": "1.7",
        }),

        # Formula, range, use — in a subtle sub-row
        html.Div([
            _gloss_sub("Formula", entry["formula"]),
            _gloss_sub("Typical range", entry["range"]),
            _gloss_sub("Used in dashboard", entry["use"]),
        ]),

    ], style={
        "backgroundColor": BG_CARD,
        "border": f"1px solid {BORDER}",
        "borderRadius": "10px",
        "padding": "16px 20px",
        "marginBottom": "10px",
    })


def _gloss_sub(label: str, text: str) -> html.P:
    return html.P([
        html.Span(f"{label}: ", style={
            "color": TEXT_DIM, "fontWeight": "600", "fontSize": "11px",
            "textTransform": "uppercase", "letterSpacing": "0.6px",
        }),
        html.Span(text, style={"color": TEXT_DIM, "fontSize": "12px"}),
    ], style={"margin": "3px 0"})


def make_glossary_tab() -> html.Div:
    return html.Div([
        html.P(
            "Every metric used in this dashboard — plain English definitions, "
            "formulas, typical ranges, and where each one appears.",
            style={"color": TEXT_DIM, "fontSize": "13px",
                   "marginBottom": "18px", "lineHeight": "1.6"},
        ),
        *[glossary_card(e) for e in GLOSSARY],
    ], style={"padding": "4px"})


# ─────────────────────────────────────────────────────────────────────────────
# Win Predictor helpers
# ─────────────────────────────────────────────────────────────────────────────

FEATURES = [
    'total_vorp',
    'top_player_vorp',
    'vorp_concentration',
    'avg_age',
    'avg_bpm',
]

FEATURE_LABELS = {
    'total_vorp':          'Team VORP',
    'top_player_vorp':     'Top Player VORP',
    'vorp_concentration':  'VORP Concentration',
    'avg_age':             'Average Age',
    'avg_bpm':             'Avg BPM (min-weighted)',
}


def get_current_team_features(team_abbr: str) -> dict | None:
    """Aggregate current-season features for one team from master.csv."""
    try:
        master = pd.read_csv('master.csv')
    except FileNotFoundError:
        return None

    roster = master[master['team'] == team_abbr].copy()
    if roster.empty:
        return None

    for col in ('vorp', 'bpm', 'mp', 'age'):
        roster[col] = pd.to_numeric(roster[col], errors='coerce')

    total_vorp = roster['vorp'].sum()
    top_vorp   = roster['vorp'].max()
    avg_age    = roster['age'].mean()
    total_mp   = roster['mp'].sum()
    avg_bpm    = (roster['bpm'] * roster['mp']).sum() / total_mp if total_mp > 0 else 0.0
    vorp_conc  = top_vorp / max(float(total_vorp), 1.0)   # clamp denominator

    return {
        'total_vorp':         round(float(total_vorp), 2),
        'top_player_vorp':    round(float(top_vorp),   2),
        'vorp_concentration': round(float(vorp_conc),  4),
        'avg_age':            round(float(avg_age),    1),
        'avg_bpm':            round(float(avg_bpm),    3),
    }


def get_actual_wins(team_abbr: str) -> float | None:
    """Derive actual wins from WS sum in master.csv (WS ≈ team wins)."""
    try:
        master = pd.read_csv('master.csv')
    except FileNotFoundError:
        return None
    roster = master[master['team'] == team_abbr].copy()
    if roster.empty:
        return None
    ws = pd.to_numeric(roster['ws'], errors='coerce').sum()
    return round(float(ws), 1)


def make_feature_importance_chart(fi_df: pd.DataFrame) -> go.Figure:
    fi_df = fi_df.sort_values('importance', ascending=True)
    labels = [FEATURE_LABELS.get(f, f) for f in fi_df['feature']]

    def bar_color(imp):
        if imp >= 0.25:
            return GREEN
        if imp >= 0.12:
            return BLUE
        return TEXT_DIM

    colors = fi_df['importance'].apply(bar_color)

    fig = go.Figure(go.Bar(
        x=fi_df['importance'],
        y=labels,
        orientation='h',
        marker_color=colors,
        text=fi_df['importance'].apply(lambda v: f'{v:.3f}'),
        textposition='outside',
        textfont=dict(size=10, color=TEXT),
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>',
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title='Feature Importance — Drivers of Win Prediction',
        xaxis=dict(title='Importance', gridcolor=BORDER, zerolinecolor=BORDER, color=TEXT_DIM),
        yaxis=dict(gridcolor=BORDER, color=TEXT_DIM),
        height=260,
        margin=dict(l=160, r=80, t=45, b=25),
        bargap=0.3,
    )
    return fig


def pred_stat_card(label: str, value: str, color=TEXT) -> html.Div:
    return html.Div([
        html.P(label, style={'margin': '0', 'fontSize': '11px', 'color': TEXT_DIM,
                             'textTransform': 'uppercase', 'letterSpacing': '0.8px'}),
        html.H2(value, style={'margin': '6px 0 0', 'fontSize': '32px',
                              'fontWeight': '800', 'color': color, 'lineHeight': '1'}),
    ], style={
        'backgroundColor': BG_CARD,
        'border':         f'1px solid {BORDER}',
        'borderRadius':   '10px',
        'padding':        '16px 20px',
        'flex':           '1',
        'minWidth':       '130px',
        'textAlign':      'center',
    })


def make_win_predictor_tab() -> html.Div:
    """Build the static layout for the Win Predictor tab."""
    model_ready = _WIN_MODEL is not None

    if not model_ready:
        notice = html.Div([
            html.P("⚙️  Win Prediction model not yet trained.", style={
                'color': YELLOW, 'fontWeight': '600', 'fontSize': '14px'}),
            html.P("Run the data pipeline to enable this tab:", style={'color': TEXT_DIM}),
            html.Pre(
                "python fetch_historical.py\n"
                "python build_training_data.py\n"
                "python win_predictor.py",
                style={
                    'backgroundColor': BG_CHART, 'border': f'1px solid {BORDER}',
                    'borderRadius': '6px', 'padding': '12px', 'color': TEXT,
                    'fontSize': '12px', 'fontFamily': 'monospace', 'margin': '8px 0',
                }),
        ], style={'padding': '20px'})
        return html.Div([notice], style={'padding': '12px'})

    all_teams_options = [
        {'label': f"{TEAM_NAMES.get(t, t)} ({t})", 'value': t}
        for t in sorted(TEAM_NAMES.keys())
    ]

    return html.Div([
        # ── Selector row ────────────────────────────────────────────────────
        html.Div([
            html.Span("Team", style={
                'fontWeight': '700', 'fontSize': '13px',
                'color': TEXT_DIM, 'textTransform': 'uppercase', 'letterSpacing': '0.7px',
            }),
            dcc.Dropdown(
                id='wp-team-dropdown',
                options=all_teams_options,
                value='OKC',
                clearable=False,
                className='team-dropdown',
                style={'width': '260px', 'fontSize': '13px'},
            ),
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '14px',
                  'marginBottom': '16px'}),

        # ── Prediction cards + feature importance (side by side) ─────────
        html.Div([
            # Left: prediction cards
            html.Div([
                html.P("Prediction", style={
                    'margin': '0 0 12px', 'fontSize': '11px', 'color': TEXT_DIM,
                    'textTransform': 'uppercase', 'letterSpacing': '0.8px',
                }),
                html.Div([
                    pred_stat_card('Predicted Wins', '—', BLUE),
                    pred_stat_card('Actual Wins',    '—', GREEN),
                    pred_stat_card('Delta',          '—', TEXT_DIM),
                ], id='wp-pred-cards', style={'display': 'flex', 'gap': '10px'}),
            ], style={
                'backgroundColor': BG_CARD, 'border': f'1px solid {BORDER}',
                'borderRadius': '10px', 'padding': '16px',
                'flex': '0 0 360px',
            }),

            # Right: feature importance chart
            html.Div([
                dcc.Graph(
                    id='wp-feature-chart',
                    figure=make_feature_importance_chart(_FEATURE_IMP)
                    if _FEATURE_IMP is not None else go.Figure(),
                    config={'displayModeBar': False},
                ),
            ], style={
                'backgroundColor': BG_CARD, 'border': f'1px solid {BORDER}',
                'borderRadius': '10px', 'padding': '12px', 'flex': '1',
            }),
        ], style={'display': 'flex', 'gap': '12px', 'marginBottom': '12px',
                  'alignItems': 'flex-start'}),

        # ── AI explanation ───────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Span("🤖", style={'fontSize': '14px'}),
                html.Span(" AI Prediction Explanation", style={
                    'fontWeight': '700', 'fontSize': '13px',
                    'color': TEXT, 'marginLeft': '8px',
                }),
            ], style={'marginBottom': '10px'}),
            html.Div(
                "Select a team above to generate an AI explanation.",
                id='wp-explanation',
                style={
                    'color': TEXT_DIM, 'fontSize': '13px',
                    'lineHeight': '1.8', 'whiteSpace': 'pre-wrap',
                },
            ),
        ], style={
            'backgroundColor': BG_CARD, 'border': f'1px solid {BORDER}',
            'borderRadius': '10px', 'padding': '16px',
        }),

    ], style={'padding': '4px'})


# ─────────────────────────────────────────────────────────────────────────────
# App layout
# ─────────────────────────────────────────────────────────────────────────────
app = Dash(__name__, title="NBA Roster Optimizer")

CARD = {
    "backgroundColor": BG_CARD,
    "border": f"1px solid {BORDER}",
    "borderRadius": "10px",
    "padding": "12px",
}

TAB_STYLE = {
    "backgroundColor": BG_CARD,
    "border":          f"1px solid {BORDER}",
    "borderRadius":    "8px 8px 0 0",
    "color":           TEXT_DIM,
    "padding":         "10px 20px",
    "fontFamily":      FONT_FAMILY,
    "fontSize":        "13px",
    "fontWeight":      "600",
}
TAB_SELECTED_STYLE = {
    **TAB_STYLE,
    "backgroundColor": BG_PAGE,
    "color":           TEXT,
    "borderBottom":    f"2px solid {BLUE}",
}

# ── Cap Analytics page content ────────────────────────────────────────────────
cap_analytics_content = html.Div([

    kpis,

    # Row 1: Leaderboard + Frontier + Gap
    html.Div([
        html.Div([
            _ask_ai_btn("leaderboard"),
            dcc.Graph(figure=make_leaderboard(), config={"displayModeBar": False}),
        ], style={**CARD, "flex": "1"}, className="chart-card"),
        html.Div([
            html.Div([
                _ask_ai_btn("frontier"),
                dcc.Graph(figure=make_frontier(), config={"displayModeBar": False}),
            ], style=CARD, className="chart-card"),
            html.Div([
                _ask_ai_btn("gap"),
                dcc.Graph(figure=make_gap_chart(), config={"displayModeBar": False}),
            ], style=CARD, className="chart-card"),
        ], style={"display": "flex", "flexDirection": "column",
                  "gap": "10px", "flex": "1"}),
    ], style={"display": "flex", "gap": "10px", "marginBottom": "10px",
              "alignItems": "flex-start"}),

    # Row 2: Team deep dive
    html.Div([
        html.Div([
            html.Span("Team Deep Dive", style={
                "fontWeight": "700", "fontSize": "13px",
                "color": TEXT_DIM, "textTransform": "uppercase",
                "letterSpacing": "0.7px",
            }),
            dcc.Dropdown(
                id="team-dropdown",
                options=[
                    {"label": f"{TEAM_NAMES.get(t, t)} ({t})", "value": t}
                    for t in ALL_TEAMS
                ],
                value="SAS",
                clearable=False,
                className="team-dropdown",
                style={"width": "240px", "fontSize": "13px"},
            ),
            _ask_ai_btn("deepdive"),
        ], style={"display": "flex", "alignItems": "center",
                  "gap": "14px", "marginBottom": "4px"}),
        dcc.Graph(
            id="team-deepdive-graph",
            figure=make_team_deepdive("SAS"),
            config={"displayModeBar": False},
        ),
    ], style={**CARD, "marginBottom": "10px"}, className="chart-card"),

    # Row 3: East vs West
    html.Div([
        _ask_ai_btn("conference"),
        dcc.Graph(figure=make_conference_compare(), config={"displayModeBar": False}),
    ], style={**CARD, "marginBottom": "10px"}, className="chart-card"),

    # Row 4: Playoff step-up
    html.Div([
        _ask_ai_btn("playoffs"),
        dcc.Graph(figure=make_playoff_scatter(), config={"displayModeBar": False}),
    ], style={**CARD, "marginBottom": "10px"}, className="chart-card"),
])


app.layout = html.Div([

    # ── Header ──────────────────────────────────────────────────────────────
    html.Div([
        html.H1("Courtside Intelligence", style={
            "margin": "0", "fontSize": "24px", "fontWeight": "800",
            "color": TEXT, "letterSpacing": "-0.5px",
        }),
        html.P(
            "League-wide NBA cap efficiency, roster optimization, and win prediction.",
            style={"margin": "4px 0 0", "color": TEXT_DIM, "fontSize": "12px"},
        ),
        html.P("2025-26 NBA Season · Final Standings · Data: Basketball Reference",
               style={"margin": "3px 0 0", "color": TEXT_DIM, "fontSize": "11px"}),
    ], style={
        "marginBottom": "14px",
        "paddingBottom": "12px",
        "borderBottom": f"1px solid {BORDER}",
    }),

    # ── Tabs ─────────────────────────────────────────────────────────────────
    dcc.Tabs(
        id="main-tabs",
        value="cap-analytics",
        style={"marginBottom": "16px"},
        children=[
            dcc.Tab(
                label="📊  Cap Analytics",
                value="cap-analytics",
                style=TAB_STYLE,
                selected_style=TAB_SELECTED_STYLE,
                children=[html.Div(cap_analytics_content, style={"paddingTop": "14px"})],
            ),
            dcc.Tab(
                label="🎯  Win Predictor",
                value="win-predictor",
                style=TAB_STYLE,
                selected_style=TAB_SELECTED_STYLE,
                children=[html.Div(make_win_predictor_tab(), style={"paddingTop": "14px"})],
            ),
            dcc.Tab(
                label="📖  Glossary",
                value="glossary",
                style=TAB_STYLE,
                selected_style=TAB_SELECTED_STYLE,
                children=[html.Div(make_glossary_tab(), style={"paddingTop": "14px"})],
            ),
        ],
    ),

    # ── Hidden stores ─────────────────────────────────────────────────────────
    dcc.Store(id="chat-history", data=[]),
    dcc.Store(id="chat-open",    data=False),

    # ── Floating chat panel ───────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Div([
                html.Span("🤖", style={"fontSize": "16px"}),
                html.Span(" AI Analyst", style={
                    "fontWeight": "700", "fontSize": "14px",
                    "color": TEXT, "marginLeft": "8px",
                }),
                html.Span("Groq · llama-3.3-70b", style={
                    "fontSize": "10px", "color": TEXT_DIM, "marginLeft": "10px",
                }),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Button("✕", id="chat-close", n_clicks=0, style={
                "background": "none", "border": "none", "color": TEXT_DIM,
                "fontSize": "16px", "cursor": "pointer", "padding": "0 4px",
                "lineHeight": "1",
            }),
        ], style={
            "display": "flex", "alignItems": "center",
            "justifyContent": "space-between",
            "padding": "14px 16px 12px",
            "borderBottom": f"1px solid {BORDER}",
            "background": "#161b22",
            "borderRadius": "14px 14px 0 0",
        }),
        html.Div(
            id="chat-messages",
            children=[_welcome_bubble()],
            style={"flex": "1", "overflowY": "auto",
                   "padding": "14px 14px 8px", "minHeight": "0"},
        ),
        html.Div([
            dcc.Input(
                id="chat-input",
                type="text",
                placeholder='Ask anything — "Why is OKC #1?"',
                debounce=False,
                n_submit=0,
            ),
            html.Button("Send", id="chat-send", n_clicks=0),
        ], id="chat-input-row"),
    ], id="chat-panel", className="panel-hidden", style={
        "position": "fixed", "bottom": "100px", "right": "32px",
        "width": "390px", "height": "480px", "zIndex": "1099",
        "display": "flex", "flexDirection": "column",
        "backgroundColor": BG_CARD, "border": f"1px solid {BORDER}",
        "borderRadius": "14px", "boxShadow": "0 12px 48px rgba(0,0,0,0.6)",
        "overflow": "hidden",
        "transition": "opacity 0.2s ease, transform 0.25s cubic-bezier(0.34,1.56,0.64,1)",
        "transformOrigin": "bottom right",
    }),

    # ── Floating action button ────────────────────────────────────────────────
    html.Button([
        html.Span("🤖", style={"fontSize": "20px", "lineHeight": "1", "flexShrink": "0"}),
        html.Span("Ask the AI Analyst", className="fab-label", style={
            "fontSize": "13px", "fontWeight": "700", "whiteSpace": "nowrap",
        }),
    ], id="chat-fab", n_clicks=0, style={
        "position": "fixed", "bottom": "32px", "right": "32px", "zIndex": "1100",
        "display": "flex", "alignItems": "center", "gap": "10px",
        "backgroundColor": BLUE, "color": "#0d1117",
        "border": "none", "borderRadius": "999px", "padding": "14px 18px",
        "cursor": "pointer", "fontFamily": FONT_FAMILY,
        "boxShadow": "0 4px 24px rgba(88,166,255,0.45)",
        "overflow": "hidden", "maxWidth": "56px",
        "transition": "max-width 0.3s cubic-bezier(0.34,1.56,0.64,1), box-shadow 0.2s",
    }),

], style={
    "backgroundColor": BG_PAGE,
    "minHeight": "100vh",
    "padding": "16px 20px",
    "fontFamily": FONT_FAMILY,
    "color": TEXT,
    "boxSizing": "border-box",
})


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

# Toggle panel open/closed via FAB, close button, or ASK AI chart buttons
@callback(
    Output("chat-panel",  "className"),
    Output("chat-open",   "data"),
    Output("chat-input",  "value"),
    Input("chat-fab",     "n_clicks"),
    Input("chat-close",   "n_clicks"),
    Input({"type": "ask-ai-btn", "chart": ALL}, "n_clicks"),
    State("chat-open",    "data"),
    prevent_initial_call=True,
)
def toggle_panel(fab_clicks, close_clicks, ask_clicks, is_open):
    from dash import ctx
    triggered = ctx.triggered_id

    if triggered == "chat-close":
        return "panel-hidden", False, ""

    # One of the ASK AI chart buttons was clicked
    if isinstance(triggered, dict) and triggered.get("type") == "ask-ai-btn":
        chart_id = triggered["chart"]
        question = CHART_QUESTIONS.get(chart_id, "")
        return "panel-visible", True, question

    # FAB toggled
    new_open = not is_open
    return ("panel-visible" if new_open else "panel-hidden"), new_open, ""


# Handle chat messages
@callback(
    Output("chat-messages", "children"),
    Output("chat-history",  "data"),
    Input("chat-send",  "n_clicks"),
    Input("chat-input", "n_submit"),
    State("chat-input",    "value"),
    State("chat-history",  "data"),
    State("chat-messages", "children"),
    prevent_initial_call=True,
)
def handle_chat(n_clicks, n_submit, user_text, history, current_messages):
    if not user_text or not user_text.strip():
        return current_messages, history

    user_text = user_text.strip()
    reply = agent.ask(user_text, history)

    updated_history = history + [
        {"role": "user",      "content": user_text},
        {"role": "assistant", "content": reply},
    ]
    new_bubbles = [
        _make_bubble("user",      user_text),
        _make_bubble("assistant", reply),
    ]
    return current_messages + new_bubbles, updated_history


@callback(
    Output("team-deepdive-graph", "figure"),
    Input("team-dropdown", "value"),
)
def update_team_deepdive(team_code):
    return make_team_deepdive(team_code or "SAS")


# ── Win Predictor callbacks ───────────────────────────────────────────────────

@callback(
    Output("wp-pred-cards",   "children"),
    Output("wp-feature-chart","figure"),
    Output("wp-explanation",  "children"),
    Input("wp-team-dropdown", "value"),
    prevent_initial_call=True,
)
def update_win_predictor(team_abbr):
    if _WIN_MODEL is None or team_abbr is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(**CHART_LAYOUT, height=260)
        return (
            [pred_stat_card('Predicted Wins', '—', BLUE),
             pred_stat_card('Actual Wins',    '—', GREEN),
             pred_stat_card('Delta',          '—', TEXT_DIM)],
            empty_fig,
            "Model not available. Run the training pipeline first.",
        )

    # Compute current-season features for this team
    feats = get_current_team_features(team_abbr)
    if feats is None:
        return (
            [pred_stat_card('Predicted Wins', 'N/A', TEXT_DIM),
             pred_stat_card('Actual Wins',    'N/A', TEXT_DIM),
             pred_stat_card('Delta',          'N/A', TEXT_DIM)],
            make_feature_importance_chart(_FEATURE_IMP),
            f"No data found for {team_abbr} in master.csv.",
        )

    X_input = pd.DataFrame([feats])[FEATURES]
    predicted = float(_WIN_MODEL.predict(X_input)[0])
    predicted_rounded = round(predicted, 1)

    actual = get_actual_wins(team_abbr)
    delta  = round(predicted - actual, 1) if actual is not None else None

    # Cards
    delta_val   = f"{delta:+.0f}" if delta is not None else "N/A"
    delta_color = (GREEN if delta >= 0 else RED) if delta is not None else TEXT_DIM
    actual_val  = f"{actual:.0f}" if actual is not None else "N/A"

    cards = [
        pred_stat_card('Predicted Wins', f"{predicted_rounded:.0f}", BLUE),
        pred_stat_card('Actual Wins',    actual_val,                 GREEN),
        pred_stat_card('Delta',          delta_val,                  delta_color),
    ]

    # Feature importance chart (unchanged; team selection doesn't affect importances)
    fi_fig = make_feature_importance_chart(_FEATURE_IMP)

    # Build feature importance dict for the AI
    fi_dict = dict(zip(_FEATURE_IMP['feature'], _FEATURE_IMP['importance']))

    # AI explanation
    team_name  = TEAM_NAMES.get(team_abbr, team_abbr)
    explanation = agent.explain_prediction(
        team_name=team_name,
        predicted_wins=predicted_rounded,
        actual_wins=actual,
        feature_importance=fi_dict,
        team_stats=feats,
    )

    return cards, fi_fig, explanation


if __name__ == "__main__":
    print("Starting dashboard at http://127.0.0.1:8050")
    app.run(debug=False, port=8050)
