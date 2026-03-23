import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os

# ══════════════════════════════════════════════════════════════════════════════
# APP INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

app = dash.Dash(
    __name__,
    title="NSE Undervaluation Dashboard",
    suppress_callback_exceptions=True
)

# ── Colour constants ──────────────────────────────────────────────────────────
FLAG_COLOURS = {
    "Strongly Undervalued":   "#1a5276",
    "Moderately Undervalued": "#27ae60",
    "Weakly Undervalued":     "#f39c12",
    "Overvalued":             "#e74c3c",
}

RISK_COLOURS = {
    "Strong Downside Protection": "#27ae60",
    "Mild Downside Protection":   "#f39c12",
    "Mild Downside Risk":         "#e67e22",
    "High Downside Risk":         "#e74c3c",
    "No Risk Data":               "#95a5a6",
}

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    base = os.path.dirname(os.path.abspath(__file__))

    try:
        companies = pd.read_csv(os.path.join(base, "nse_value_dataset.csv"))
    except FileNotFoundError:
        companies = pd.DataFrame()

    try:
        prices = pd.read_csv(os.path.join(base, "all_price_histories.csv"))
    except FileNotFoundError:
        prices = pd.DataFrame()

    try:
        validation = pd.read_csv(os.path.join(base, "sector_validation_results.csv"))
    except FileNotFoundError:
        validation = pd.DataFrame()

    return companies, prices, validation

companies, prices, validation = load_data()

ALL_SECTORS = sorted(companies["Sector"].unique().tolist()) if not companies.empty else []
ALL_FLAGS   = ["Strongly Undervalued", "Moderately Undervalued",
               "Weakly Undervalued", "Overvalued"]
ALL_TICKERS = sorted(companies["Ticker"].unique().tolist()) if not companies.empty else []

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def kpi_card(title, value, colour="#1f4e79"):
    return html.Div([
        html.P(title, style={
            "margin": "0 0 4px 0", "fontSize": "11px",
            "color": "#888", "textTransform": "uppercase",
            "letterSpacing": "0.05em"
        }),
        html.H3(str(value), style={
            "margin": "0", "fontSize": "22px",
            "fontWeight": "600", "color": colour
        })
    ], style={
        "background": "white", "padding": "16px 20px",
        "borderRadius": "8px", "border": "0.5px solid #e0e0e0",
        "flex": "1", "minWidth": "120px"
    })


def section_header(title):
    return html.H4(title, style={
        "color": "#1f4e79", "margin": "0 0 12px 0",
        "fontSize": "14px", "fontWeight": "600",
        "borderBottom": "2px solid #1a5276",
        "paddingBottom": "6px"
    })


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

sidebar = html.Div([

    html.Div([
        html.H2("NSE", style={
            "color": "#1a5276", "margin": "0",
            "fontSize": "20px", "fontWeight": "700"
        }),
        html.P("Undervaluation Dashboard", style={
            "color": "#666", "margin": "2px 0 0 0",
            "fontSize": "11px"
        }),
    ], style={"marginBottom": "24px"}),

    # Sector filter
    html.Div([
        html.Label("Sectors", style={
            "fontSize": "11px", "fontWeight": "600",
            "color": "#444", "textTransform": "uppercase",
            "letterSpacing": "0.05em", "display": "block",
            "marginBottom": "8px"
        }),
        dcc.Checklist(
            id="sector-filter",
            options=[{"label": s, "value": s} for s in ALL_SECTORS],
            value=ALL_SECTORS,
            style={"fontSize": "12px"},
            labelStyle={"display": "flex", "alignItems": "center",
                        "gap": "6px", "marginBottom": "4px"}
        ),
    ], style={"marginBottom": "20px"}),

    html.Hr(style={"border": "none", "borderTop": "0.5px solid #e0e0e0",
                   "margin": "0 0 20px 0"}),

    # Flag filter
    html.Div([
        html.Label("Valuation Flags", style={
            "fontSize": "11px", "fontWeight": "600",
            "color": "#444", "textTransform": "uppercase",
            "letterSpacing": "0.05em", "display": "block",
            "marginBottom": "8px"
        }),
        dcc.Checklist(
            id="flag-filter",
            options=[
                {"label": html.Span(f, style={"color": FLAG_COLOURS[f]}), "value": f}
                for f in ALL_FLAGS
                if f in companies["Valuation Flag Comprehensive"].unique()
            ],
            value=[f for f in ALL_FLAGS
                   if f in companies["Valuation Flag Comprehensive"].unique()],
            style={"fontSize": "12px"},
            labelStyle={"display": "flex", "alignItems": "center",
                        "gap": "6px", "marginBottom": "4px"}
        ),
    ], style={"marginBottom": "20px"}),

    html.Hr(style={"border": "none", "borderTop": "0.5px solid #e0e0e0",
                   "margin": "0 0 20px 0"}),

    # Risk-free rate slider
    html.Div([
        html.Label("Risk-Free Rate (CBK T-bill)", style={
            "fontSize": "11px", "fontWeight": "600",
            "color": "#444", "textTransform": "uppercase",
            "letterSpacing": "0.05em", "display": "block",
            "marginBottom": "8px"
        }),
        dcc.Slider(
            id="rfr-slider",
            min=5, max=20, step=0.5, value=7.7,
            marks={5: "5%", 10: "10%", 15: "15%", 20: "20%"},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
    ], style={"marginBottom": "20px"}),

    html.Hr(style={"border": "none", "borderTop": "0.5px solid #e0e0e0",
                   "margin": "0 0 20px 0"}),

    # Company selector
    html.Div([
        html.Label("Company Deep Dive", style={
            "fontSize": "11px", "fontWeight": "600",
            "color": "#444", "textTransform": "uppercase",
            "letterSpacing": "0.05em", "display": "block",
            "marginBottom": "8px"
        }),
        dcc.Dropdown(
            id="company-selector",
            options=[{"label": t, "value": t} for t in ALL_TICKERS],
            value=ALL_TICKERS[0] if ALL_TICKERS else None,
            clearable=False,
            style={"fontSize": "12px"}
        ),
    ], style={"marginBottom": "20px"}),

    html.Hr(style={"border": "none", "borderTop": "0.5px solid #e0e0e0",
                   "margin": "0 0 16px 0"}),

    html.P("Data: Synthetic GBM Simulation", style={
        "fontSize": "10px", "color": "#aaa", "textAlign": "center",
        "margin": "0"
    }),
    html.P("NSE Undervaluation Detection System", style={
        "fontSize": "10px", "color": "#aaa", "textAlign": "center",
        "margin": "4px 0 0 0"
    }),

], style={
    "width": "220px", "minWidth": "220px",
    "background": "#f8f9fa", "padding": "24px 16px",
    "borderRight": "0.5px solid #e0e0e0",
    "overflowY": "auto", "height": "100vh",
    "position": "sticky", "top": "0"
})

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ══════════════════════════════════════════════════════════════════════════════

main_panel = html.Div([

    # Header
    html.Div([
        html.Div([
            html.H1("NSE Undervaluation Detection Dashboard", style={
                "margin": "0", "fontSize": "20px",
                "fontWeight": "700", "color": "#1f4e79"
            }),
            html.P(
                "Sector-specific weighted valuation scoring — "
                "Nairobi Securities Exchange",
                style={"margin": "4px 0 0 0", "fontSize": "12px",
                       "color": "#666"}
            ),
        ]),
        html.Span("Synthetic Data", style={
            "background": "#fef3cd", "color": "#856404",
            "padding": "4px 10px", "borderRadius": "4px",
            "fontSize": "11px", "fontWeight": "600"
        })
    ], style={
        "display": "flex", "justifyContent": "space-between",
        "alignItems": "center", "padding": "16px 24px",
        "borderBottom": "0.5px solid #e0e0e0",
        "background": "white"
    }),

    # KPI row
    html.Div(id="kpi-row", style={
        "display": "flex", "gap": "12px",
        "padding": "16px 24px",
        "background": "#f8f9fa",
        "borderBottom": "0.5px solid #e0e0e0"
    }),

    # Tabs
    html.Div([
        dcc.Tabs(
            id="main-tabs",
            value="heatmap",
            children=[
                dcc.Tab(label="Heatmap",        value="heatmap"),
                dcc.Tab(label="Price History",  value="prices"),
                dcc.Tab(label="Fundamentals",   value="fundamentals"),
                dcc.Tab(label="Company Dive",   value="deepdive"),
                dcc.Tab(label="Validation",     value="validation"),
                dcc.Tab(label="Returns",        value="returns"),
            ],
            style={"fontFamily": "sans-serif"},
            colors={"border": "#e0e0e0", "primary": "#1a5276",
                    "background": "#f8f9fa"}
        ),
    ], style={"padding": "0 24px", "background": "white",
              "borderBottom": "0.5px solid #e0e0e0"}),

    # Tab content
    html.Div(id="tab-content", style={"padding": "24px"})

], style={"flex": "1", "overflowY": "auto", "background": "#f0f2f5"})

# ══════════════════════════════════════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

app.layout = html.Div([
    sidebar,
    main_panel
], style={
    "display": "flex", "fontFamily": "Arial, sans-serif",
    "minHeight": "100vh"
})

# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

def filter_companies(selected_sectors, selected_flags):
    if companies.empty:
        return companies
    return companies[
        (companies["Sector"].isin(selected_sectors or ALL_SECTORS)) &
        (companies["Valuation Flag Comprehensive"].isin(
            selected_flags or ALL_FLAGS))
    ]


# ── KPI Row ───────────────────────────────────────────────────────────────────
@app.callback(
    Output("kpi-row", "children"),
    Input("sector-filter", "value"),
    Input("flag-filter", "value")
)
def update_kpis(selected_sectors, selected_flags):
    fc = filter_companies(selected_sectors, selected_flags)
    if fc.empty:
        return []

    strongly = len(fc[fc["Valuation Flag Comprehensive"] == "Strongly Undervalued"])
    overvalued = len(fc[fc["Valuation Flag Comprehensive"] == "Overvalued"])
    avg_score  = fc["Normalised Valuation Score"].mean()

    cum_col = "Cumulative Return" if "Cumulative Return" in fc.columns else None
    avg_ret = fc[cum_col].mean() if cum_col else 0

    return [
        kpi_card("Companies",         len(fc)),
        kpi_card("Strongly UV",        strongly,         "#1a5276"),
        kpi_card("Avg Score",          f"{avg_score:.3f}","#27ae60"),
        kpi_card("Avg Return",         f"{avg_ret:.2%}",  "#f39c12"),
        kpi_card("Overvalued",         overvalued,        "#e74c3c"),
    ]


# ── Tab Content ───────────────────────────────────────────────────────────────
@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
    Input("sector-filter", "value"),
    Input("flag-filter", "value"),
    Input("company-selector", "value"),
    Input("rfr-slider", "value"),
)
def render_tab(tab, selected_sectors, selected_flags,
               selected_company, rfr):
    fc = filter_companies(selected_sectors, selected_flags)

    # ── HEATMAP TAB ───────────────────────────────────────────────────────────
    if tab == "heatmap":
        if fc.empty:
            return html.P("No data available for selected filters.")

        pivot = fc.pivot_table(
            index="Sector", columns="Ticker",
            values="Normalised Valuation Score", aggfunc="mean"
        ).fillna(0)

        fig_heat = px.imshow(
            pivot,
            color_continuous_scale=[
                [0.00, "#e74c3c"],
                [0.25, "#f39c12"],
                [0.50, "#27ae60"],
                [1.00, "#1a5276"],
            ],
            zmin=0, zmax=1,
            title="Normalised Valuation Score — Sector × Company",
            labels={"color": "Score"},
            aspect="auto"
        )
        fig_heat.update_layout(
            height=380, margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="white", plot_bgcolor="white"
        )

        counts = fc["Valuation Flag Comprehensive"].value_counts().reset_index()
        counts.columns = ["Flag", "Count"]

        fig_donut = px.pie(
            counts, values="Count", names="Flag",
            hole=0.45,
            title="Flag Distribution",
            color="Flag",
            color_discrete_map=FLAG_COLOURS
        )
        fig_donut.update_layout(
            height=380, margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="white",
            legend=dict(orientation="v", x=1, y=0.5)
        )

        return html.Div([
            section_header("Valuation Heatmap & Flag Distribution"),
            html.Div([
                html.Div(dcc.Graph(figure=fig_heat),
                         style={"flex": "2"}),
                html.Div(dcc.Graph(figure=fig_donut),
                         style={"flex": "1"}),
            ], style={"display": "flex", "gap": "16px"}),

            # Score distribution table
            html.Div([
                section_header("Score Distribution by Sector"),
                html.Div(
                    fc.groupby(["Sector", "Valuation Flag Comprehensive"])[
                        "Normalised Valuation Score"
                    ].agg(["mean", "count"]).round(3).reset_index()
                    .rename(columns={
                        "mean": "Avg Score", "count": "Companies"
                    }).to_html(index=False, classes="data-table"),
                    style={"overflowX": "auto", "fontSize": "12px"}
                )
            ], style={
                "background": "white", "padding": "16px",
                "borderRadius": "8px", "border": "0.5px solid #e0e0e0",
                "marginTop": "16px"
            })
        ])

    # ── PRICE HISTORY TAB ─────────────────────────────────────────────────────
    elif tab == "prices":
        if prices.empty:
            return html.P("Price history data not available.")

        sector_selector = html.Div([
            html.Label("Select Sector:", style={
                "fontSize": "12px", "marginRight": "8px"
            }),
            dcc.Dropdown(
                id="price-sector-dd",
                options=[{"label": s, "value": s}
                         for s in (selected_sectors or ALL_SECTORS)],
                value=(selected_sectors or ALL_SECTORS)[0],
                clearable=False,
                style={"width": "240px", "fontSize": "12px",
                       "display": "inline-block"}
            )
        ], style={"marginBottom": "16px", "display": "flex",
                  "alignItems": "center"})

        return html.Div([
            section_header("Price History vs Sector Average"),
            sector_selector,
            dcc.Graph(id="price-chart"),
        ])

    # ── FUNDAMENTALS TAB ──────────────────────────────────────────────────────
    elif tab == "fundamentals":
        if fc.empty:
            return html.P("No data available.")

        metrics = ["P/E Ratio", "P/B Ratio", "ROE (%)", "Dividend Yield (%)"]
        available = [m for m in metrics if m in fc.columns]

        controls = html.Div([
            html.Div([
                html.Label("Sector:", style={"fontSize": "12px",
                                             "marginRight": "8px"}),
                dcc.Dropdown(
                    id="fund-sector-dd",
                    options=[{"label": s, "value": s}
                             for s in (selected_sectors or ALL_SECTORS)],
                    value=(selected_sectors or ALL_SECTORS)[0],
                    clearable=False,
                    style={"width": "200px", "fontSize": "12px",
                           "display": "inline-block"}
                )
            ], style={"marginRight": "24px"}),
            html.Div([
                html.Label("Metric:", style={"fontSize": "12px",
                                             "marginRight": "8px"}),
                dcc.RadioItems(
                    id="metric-radio",
                    options=[{"label": m, "value": m} for m in available],
                    value=available[0] if available else None,
                    inline=True,
                    style={"fontSize": "12px"},
                    labelStyle={"marginRight": "16px"}
                )
            ])
        ], style={"display": "flex", "alignItems": "center",
                  "marginBottom": "16px", "flexWrap": "wrap", "gap": "8px"})

        return html.Div([
            section_header("Fundamentals vs Sector Benchmark"),
            controls,
            dcc.Graph(id="fund-chart"),
        ])

    # ── COMPANY DEEP DIVE TAB ─────────────────────────────────────────────────
    elif tab == "deepdive":
        if not selected_company or companies.empty:
            return html.P("Select a company from the sidebar.")

        row = companies[companies["Ticker"] == selected_company]
        if row.empty:
            return html.P(f"No data for {selected_company}.")
        row = row.iloc[0]

        flag   = row.get("Valuation Flag Comprehensive", "N/A")
        score  = row.get("Normalised Valuation Score", 0)
        colour = FLAG_COLOURS.get(flag, "#888")

        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 1], "tickwidth": 1},
                "bar":  {"color": colour, "thickness": 0.3},
                "steps": [
                    {"range": [0.00, 0.25], "color": "#fdecea"},
                    {"range": [0.25, 0.50], "color": "#fef9ec"},
                    {"range": [0.50, 0.75], "color": "#eafaf1"},
                    {"range": [0.75, 1.00], "color": "#d6eaf8"},
                ],
                "threshold": {
                    "line": {"color": colour, "width": 3},
                    "thickness": 0.8, "value": score
                }
            },
            title={"text": "Valuation Score", "font": {"size": 13}}
        ))
        gauge_fig.update_layout(
            height=260, margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor="white"
        )

        ticker_prices = prices[prices["Ticker"] == selected_company] \
            if not prices.empty else pd.DataFrame()
        sector_avg = prices[prices["Sector"] == row.get("Sector", "")]\
            .groupby("Day")["Price"].mean().reset_index() \
            if not prices.empty else pd.DataFrame()

        price_fig = go.Figure()
        if not ticker_prices.empty:
            price_fig.add_trace(go.Scatter(
                x=ticker_prices["Day"], y=ticker_prices["Price"],
                name=selected_company,
                line=dict(color=colour, width=2)
            ))
        if not sector_avg.empty:
            price_fig.add_trace(go.Scatter(
                x=sector_avg["Day"], y=sector_avg["Price"],
                name="Sector Average",
                line=dict(color="#888", width=2, dash="dash")
            ))
        price_fig.update_layout(
            title=f"{selected_company} vs Sector Average",
            xaxis_title="Day", yaxis_title="Price (KES)",
            height=260, margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="white", hovermode="x unified",
            legend=dict(orientation="h", y=-0.2)
        )

        fundamentals_rows = []
        for metric in ["P/E Ratio", "P/B Ratio", "ROE (%)",
                       "Dividend Yield (%)", "Market Cap (KES B)"]:
            if metric in row.index:
                fundamentals_rows.append(
                    html.Tr([
                        html.Td(metric, style={"color": "#666",
                                               "padding": "4px 8px",
                                               "fontSize": "12px"}),
                        html.Td(f"{row[metric]:.2f}",
                                style={"fontWeight": "600",
                                       "padding": "4px 8px",
                                       "fontSize": "12px",
                                       "textAlign": "right"})
                    ])
                )

        return html.Div([
            section_header(f"Company Deep Dive — {selected_company}"),

            html.Div([
                # Left — details
                html.Div([
                    html.Div([
                        html.Div(selected_company, style={
                            "fontSize": "18px", "fontWeight": "700",
                            "color": "#1f4e79"
                        }),
                        html.Div(row.get("Company Name", ""), style={
                            "fontSize": "12px", "color": "#666",
                            "marginTop": "2px"
                        }),
                        html.Div([
                            html.Span(flag, style={
                                "background": colour,
                                "color": "white",
                                "padding": "3px 10px",
                                "borderRadius": "4px",
                                "fontSize": "11px",
                                "fontWeight": "600",
                                "marginTop": "8px",
                                "display": "inline-block"
                            })
                        ]),
                        html.Div(f"Sector: {row.get('Sector', 'N/A')}", style={
                            "fontSize": "12px", "color": "#666",
                            "marginTop": "8px"
                        }),
                    ], style={"marginBottom": "16px"}),

                    html.Table(fundamentals_rows, style={"width": "100%"}),

                    html.Div([
                        html.P("Score Breakdown", style={
                            "fontSize": "11px", "fontWeight": "600",
                            "color": "#444", "margin": "12px 0 4px 0"
                        }),
                        html.Code(
                            row.get("Score Breakdown", "N/A"),
                            style={
                                "fontSize": "10px", "background": "#f8f9fa",
                                "padding": "8px", "borderRadius": "4px",
                                "display": "block", "wordBreak": "break-all",
                                "whiteSpace": "pre-wrap"
                            }
                        )
                    ]) if "Score Breakdown" in row.index else html.Div()

                ], style={
                    "flex": "1", "background": "white",
                    "padding": "16px", "borderRadius": "8px",
                    "border": "0.5px solid #e0e0e0"
                }),

                # Middle — gauge
                html.Div([
                    dcc.Graph(figure=gauge_fig)
                ], style={
                    "flex": "1", "background": "white",
                    "borderRadius": "8px",
                    "border": "0.5px solid #e0e0e0"
                }),

                # Right — price chart
                html.Div([
                    dcc.Graph(figure=price_fig)
                ], style={
                    "flex": "2", "background": "white",
                    "borderRadius": "8px",
                    "border": "0.5px solid #e0e0e0"
                }),

            ], style={"display": "flex", "gap": "16px"})
        ])

    # ── VALIDATION TAB ────────────────────────────────────────────────────────
    elif tab == "validation":
        if validation.empty:
            return html.P("Validation data not available.")

        display_val = validation[
            validation["Sector"].isin(
                (selected_sectors or ALL_SECTORS) + ["── ALL SECTORS ──"]
            )
        ].copy()

        bar_data = display_val[
            display_val["Sector"] != "── ALL SECTORS ──"
        ].copy()

        bar_fig = go.Figure()
        if not bar_data.empty and "Undervalued Mean" in bar_data.columns:
            bar_fig.add_trace(go.Bar(
                name="Undervalued Mean",
                x=bar_data["Sector"],
                y=bar_data["Undervalued Mean"],
                marker_color="#1a5276"
            ))
            if "Overvalued Mean" in bar_data.columns:
                bar_fig.add_trace(go.Bar(
                    name="Overvalued Mean",
                    x=bar_data["Sector"],
                    y=bar_data["Overvalued Mean"],
                    marker_color="#e74c3c"
                ))
        bar_fig.update_layout(
            title="Mean Returns — Undervalued vs Overvalued by Sector",
            barmode="group", height=320,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="white", plot_bgcolor="white",
            legend=dict(orientation="h", y=-0.25),
            yaxis_title="Mean Normalised Return"
        )
        bar_fig.add_hline(y=0, line_dash="dash",
                          line_color="#888", line_width=1)

        display_cols = ["Sector", "Undervalued N", "Overvalued N",
                        "Significance", "Cohen's D Magnitude",
                        "Risk Flag", "Verdict"]
        available_cols = [c for c in display_cols if c in display_val.columns]

        def style_verdict(val):
            if "Validated" in str(val):
                return "color: #27ae60; font-weight: 600"
            if "Contradicted" in str(val):
                return "color: #e74c3c; font-weight: 600"
            if "Insufficient" in str(val):
                return "color: #aaa"
            return ""

        table_rows = []
        for _, r in display_val[available_cols].iterrows():
            cells = []
            for col in available_cols:
                val = r[col]
                style = {}
                if col == "Verdict":
                    if "Validated" in str(val):
                        style = {"color": "#27ae60", "fontWeight": "600"}
                    elif "Contradicted" in str(val):
                        style = {"color": "#e74c3c", "fontWeight": "600"}
                    elif "Insufficient" in str(val):
                        style = {"color": "#aaa"}
                if col == "Risk Flag" and str(val) in RISK_COLOURS:
                    style = {"color": RISK_COLOURS.get(str(val), "#333")}
                cells.append(html.Td(
                    str(val) if not pd.isna(val) else "—",
                    style={"padding": "6px 10px", "fontSize": "11px",
                           "borderBottom": "0.5px solid #f0f0f0",
                           **style}
                ))
            table_rows.append(html.Tr(cells))

        header_row = html.Tr([
            html.Th(c, style={
                "padding": "8px 10px", "fontSize": "11px",
                "background": "#f8f9fa", "fontWeight": "600",
                "color": "#444", "borderBottom": "1px solid #e0e0e0",
                "textAlign": "left"
            }) for c in available_cols
        ])

        return html.Div([
            section_header("Statistical Validation Results"),
            dcc.Graph(figure=bar_fig),
            html.Div([
                html.Table(
                    [html.Thead(header_row), html.Tbody(table_rows)],
                    style={"width": "100%", "borderCollapse": "collapse"}
                )
            ], style={
                "background": "white", "padding": "16px",
                "borderRadius": "8px", "border": "0.5px solid #e0e0e0",
                "marginTop": "16px", "overflowX": "auto"
            })
        ])

    # ── RETURNS TAB ───────────────────────────────────────────────────────────
    elif tab == "returns":
        if fc.empty or "Normalized Return" not in fc.columns:
            return html.P("Returns data not available.")

        flag_order = ["Strongly Undervalued", "Moderately Undervalued",
                      "Weakly Undervalued", "Overvalued"]
        existing   = [f for f in flag_order
                      if f in fc["Valuation Flag Comprehensive"].unique()]

        box_fig = px.box(
            fc,
            x="Valuation Flag Comprehensive",
            y="Normalized Return",
            color="Valuation Flag Comprehensive",
            color_discrete_map=FLAG_COLOURS,
            title="Normalised Returns by Valuation Flag (Z-Score)",
            category_orders={"Valuation Flag Comprehensive": existing},
            points="all"
        )
        box_fig.update_layout(
            height=400, showlegend=False,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis_title="", yaxis_title="Normalised Return (Z-Score)"
        )
        box_fig.add_hline(y=0, line_dash="dash",
                          line_color="#888", line_width=1,
                          annotation_text="Zero return")

        sector_box = px.box(
            fc,
            x="Sector",
            y="Normalized Return",
            color="Valuation Flag Comprehensive",
            color_discrete_map=FLAG_COLOURS,
            title="Returns by Sector and Valuation Flag",
            points=False
        )
        sector_box.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis_title="", yaxis_title="Normalised Return"
        )

        return html.Div([
            section_header("Returns Distribution Analysis"),
            html.Div([
                html.Div(dcc.Graph(figure=box_fig), style={"flex": "1"}),
                html.Div(dcc.Graph(figure=sector_box), style={"flex": "1"}),
            ], style={"display": "flex", "gap": "16px"})
        ])

    return html.Div()


# ── Price chart callback (nested selector) ────────────────────────────────────
@app.callback(
    Output("price-chart", "figure"),
    Input("price-sector-dd", "value"),
    Input("flag-filter", "value"),
    prevent_initial_call=True
)
def update_price_chart(sector, selected_flags):
    if prices.empty or not sector:
        return go.Figure()

    sector_prices   = prices[prices["Sector"] == sector]
    sector_avg      = sector_prices.groupby("Day")["Price"].mean().reset_index()
    sector_companies = companies[
        (companies["Sector"] == sector) &
        (companies["Valuation Flag Comprehensive"].isin(
            selected_flags or ALL_FLAGS))
    ]

    fig = go.Figure()
    for _, r in sector_companies.iterrows():
        ticker_data = sector_prices[sector_prices["Ticker"] == r["Ticker"]]
        flag        = r["Valuation Flag Comprehensive"]
        fig.add_trace(go.Scatter(
            x=ticker_data["Day"], y=ticker_data["Price"],
            name=f"{r['Ticker']} ({flag})",
            line=dict(color=FLAG_COLOURS.get(flag, "#888"), width=1),
            opacity=0.7,
            hovertemplate=(
                f"<b>{r['Ticker']}</b><br>"
                "Day: %{x}<br>Price: KES %{y:.2f}"
                "<extra></extra>"
            )
        ))

    fig.add_trace(go.Scatter(
        x=sector_avg["Day"], y=sector_avg["Price"],
        name="Sector Average",
        line=dict(color="black", width=3, dash="dash"),
        hovertemplate=(
            "<b>Sector Avg</b><br>"
            "Day: %{x}<br>Price: KES %{y:.2f}"
            "<extra></extra>"
        )
    ))
    fig.update_layout(
        title=f"{sector} — 30-Day Price History",
        xaxis_title="Day", yaxis_title="Price (KES)",
        height=450, hovermode="x unified",
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="v", x=1.01, y=1,
                    font=dict(size=10)),
        margin=dict(l=10, r=160, t=40, b=10)
    )
    return fig


# ── Fundamentals chart callback ───────────────────────────────────────────────
@app.callback(
    Output("fund-chart", "figure"),
    Input("fund-sector-dd", "value"),
    Input("metric-radio", "value"),
    Input("flag-filter", "value"),
    prevent_initial_call=True
)
def update_fund_chart(sector, metric, selected_flags):
    if companies.empty or not sector or not metric:
        return go.Figure()

    fc = companies[
        (companies["Sector"] == sector) &
        (companies["Valuation Flag Comprehensive"].isin(
            selected_flags or ALL_FLAGS))
    ].sort_values(metric, ascending=False)

    if fc.empty:
        return go.Figure()

    sector_avg = fc[metric].mean()

    fig = px.bar(
        fc,
        x="Ticker", y=metric,
        color="Valuation Flag Comprehensive",
        color_discrete_map=FLAG_COLOURS,
        title=f"{metric} — {sector} vs Sector Average",
        hover_data=["Company Name", "Normalised Valuation Score"]
        if "Company Name" in fc.columns else ["Normalised Valuation Score"]
    )
    fig.add_hline(
        y=sector_avg, line_dash="dash",
        line_color="#333", line_width=2,
        annotation_text=f"Avg: {sector_avg:.2f}",
        annotation_position="top right"
    )
    fig.update_layout(
        height=380, paper_bgcolor="white",
        plot_bgcolor="white", showlegend=True,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.25, font=dict(size=10))
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * { box-sizing: border-box; }
            body { margin: 0; padding: 0; background: #f0f2f5; }
            .data-table {
                width: 100%; border-collapse: collapse;
                font-size: 12px;
            }
            .data-table th {
                background: #f8f9fa; padding: 6px 10px;
                text-align: left; border-bottom: 1px solid #e0e0e0;
                font-weight: 600; color: #444;
            }
            .data-table td {
                padding: 5px 10px;
                border-bottom: 0.5px solid #f0f0f0;
            }
            .data-table tr:hover { background: #f8f9fa; }
            .tab--selected {
                border-top: 3px solid #1a5276 !important;
                color: #1a5276 !important;
                font-weight: 600 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NSE UNDERVALUATION DASHBOARD")
    print("=" * 60)
    print("Starting dashboard server...")
    print("Open your browser at: http://127.0.0.1:8050")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    app.run(debug=True, port=8050)