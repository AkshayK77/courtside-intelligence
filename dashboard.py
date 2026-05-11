"""
Phase 4+5+6 — NBA Roster Optimizer Dashboard with AI Agent
Five charts + Groq-powered chat panel.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output, State, callback, ALL
import plotly.graph_objects as go
import plotly.express as px
import agent

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
# App layout
# ─────────────────────────────────────────────────────────────────────────────
app = Dash(__name__, title="NBA Roster Optimizer")

CARD = {
    "backgroundColor": BG_CARD,
    "border": f"1px solid {BORDER}",
    "borderRadius": "10px",
    "padding": "12px",
}

app.layout = html.Div([

    # ── Header ──────────────────────────────────────────────────────────────
    html.Div([
        html.H1("NBA Roster Optimizer", style={
            "margin": "0", "fontSize": "24px", "fontWeight": "800",
            "color": TEXT, "letterSpacing": "-0.5px",
        }),
        html.P(
            "Which teams get the most performance per dollar spent — "
            "and how should they reallocate cap to maximise wins?",
            style={"margin": "4px 0 0", "color": TEXT_DIM, "fontSize": "12px"},
        ),
        html.P("2025-26 NBA Season · Final Standings · Data: Basketball Reference",
               style={"margin": "3px 0 0", "color": TEXT_DIM, "fontSize": "11px"}),
    ], style={
        "marginBottom": "14px",
        "paddingBottom": "12px",
        "borderBottom": f"1px solid {BORDER}",
    }),

    # ── KPI cards ────────────────────────────────────────────────────────────
    kpis,

    # ── Row 1: Leaderboard (left, taller) + Frontier scatter (right) ─────────
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

    # ── Row 2: Team deep dive — any of 30 teams ──────────────────────────────
    html.Div([
        # Header row: label + dropdown + ASK AI
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
                style={
                    "width": "240px",
                    "fontSize": "13px",
                },
            ),
            _ask_ai_btn("deepdive"),
        ], style={
            "display": "flex",
            "alignItems": "center",
            "gap": "14px",
            "marginBottom": "4px",
        }),
        dcc.Graph(
            id="team-deepdive-graph",
            figure=make_team_deepdive("SAS"),
            config={"displayModeBar": False},
        ),
    ], style={**CARD, "marginBottom": "10px"}, className="chart-card"),

        # ── Row 3: East vs West comparison ───────────────────────────────────────
        html.Div([
            _ask_ai_btn("conference"),
            dcc.Graph(figure=make_conference_compare(), config={"displayModeBar": False}),
        ], style={**CARD, "marginBottom": "10px"}, className="chart-card"),

        # ── Row 4: Playoff Step-Up scatter (full width) ───────────────────────────
    html.Div([
        _ask_ai_btn("playoffs"),
        dcc.Graph(figure=make_playoff_scatter(), config={"displayModeBar": False}),
    ], style={**CARD, "marginBottom": "10px"}, className="chart-card"),

    # ── Hidden stores ─────────────────────────────────────────────────────────
    dcc.Store(id="chat-history", data=[]),
    dcc.Store(id="chat-open",    data=False),

    # ── Floating chat panel (hidden by default) ───────────────────────────────
    html.Div([
        # Panel header
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

        # Messages
        html.Div(
            id="chat-messages",
            children=[_welcome_bubble()],
            style={"flex": "1", "overflowY": "auto",
                   "padding": "14px 14px 8px",
                   "minHeight": "0"},
        ),

        # Input row
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
        "position": "fixed",
        "bottom": "100px",
        "right": "32px",
        "width": "390px",
        "height": "480px",
        "zIndex": "1099",
        "display": "flex",
        "flexDirection": "column",
        "backgroundColor": BG_CARD,
        "border": f"1px solid {BORDER}",
        "borderRadius": "14px",
        "boxShadow": "0 12px 48px rgba(0,0,0,0.6)",
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
        "position": "fixed",
        "bottom": "32px",
        "right": "32px",
        "zIndex": "1100",
        "display": "flex",
        "alignItems": "center",
        "gap": "10px",
        "backgroundColor": BLUE,
        "color": "#0d1117",
        "border": "none",
        "borderRadius": "999px",
        "padding": "14px 18px",
        "cursor": "pointer",
        "fontFamily": FONT_FAMILY,
        "boxShadow": "0 4px 24px rgba(88,166,255,0.45)",
        "overflow": "hidden",
        "maxWidth": "56px",
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


if __name__ == "__main__":
    print("Starting dashboard at http://127.0.0.1:8050")
    app.run(debug=False, port=8050)
