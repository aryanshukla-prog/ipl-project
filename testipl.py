import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH    = "ipl.db"
MODEL_PATH = "win_prob_model.pkl"
MATCH_ID   = "1527692"   # ← replace with your match_id
# ─────────────────────────────────────────────────────────────────────────────

conn = sqlite3.connect(DB_PATH)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_match_info(match_id):
    """Return basic match metadata from the matches table."""
    q = "SELECT * FROM matches WHERE id = ?"
    return pd.read_sql(q, conn, params=(match_id,))


def build_chase_states(match_id):
    """
    Reconstruct ball-by-ball chase states and predict win probability
    for the 2nd innings of a given match.
    Returns a DataFrame with one row per ball.
    """
    # --- 1st innings total ---
    q1 = """
        SELECT SUM(runs_total) AS total
        FROM deliveries
        WHERE match_id = ? AND inning = 1
    """
    target_runs = pd.read_sql(q1, conn, params=(match_id,)).iloc[0]["total"] + 1

    # --- 2nd innings ball-by-ball ---
    q2 = """
        SELECT over_num, ball_num, runs_total, is_wicket,
               batting_team, bowling_team
        FROM deliveries
        WHERE match_id = ? AND inning = 2
        ORDER BY over_num, ball_num
    """
    df = pd.read_sql(q2, conn, params=(match_id,))

    if df.empty:
        raise ValueError(f"No 2nd innings data found for match_id={match_id}")

    # cumulative stats
    df["cum_runs"]      = df["runs_total"].cumsum()
    df["cum_wickets"]   = df["is_wicket"].cumsum()
    df["ball_number"]   = range(1, len(df) + 1)
    df["cum_overs"]     = df["ball_number"] / 6

    df["runs_needed"]        = target_runs - df["cum_runs"]
    df["balls_remaining"]    = 120 - df["ball_number"]
    df["wickets_remaining"]  = 10 - df["cum_wickets"]
    df["required_rr"]        = df["runs_needed"] / (df["balls_remaining"] / 6).replace(0, np.nan)
    df["current_rr"]         = df["cum_runs"] / df["cum_overs"].replace(0, np.nan)
    df["rr_diff"]            = df["current_rr"] - df["required_rr"]

    df.fillna(0, inplace=True)
    df["required_rr"] = df["required_rr"].clip(lower=0)

    features = ["cum_runs", "runs_needed", "balls_remaining", "wickets_remaining",
                "required_rr", "current_rr", "rr_diff", "cum_overs"]

    df["win_prob"] = model.predict_proba(df[features])[:, 1]
    df["over_label"] = df["over_num"] + 1   # human-readable over number

    return df, target_runs


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Win Probability Curve
# ═══════════════════════════════════════════════════════════════════════════════

def plot_win_probability_curve(match_id):
    df, target = build_chase_states(match_id)

    batting_team  = df["batting_team"].iloc[0]
    bowling_team  = df["bowling_team"].iloc[0]
    wicket_balls  = df[df["is_wicket"] == 1]

    fig = go.Figure()

    # shaded regions
    fig.add_hrect(y0=0.5, y1=1.0,
                  fillcolor="rgba(0,200,100,0.06)", line_width=0)
    fig.add_hrect(y0=0.0, y1=0.5,
                  fillcolor="rgba(220,50,50,0.06)",  line_width=0)

    # 50 % line
    fig.add_hline(y=0.5, line_dash="dash",
                  line_color="rgba(180,180,180,0.6)", line_width=1)

    # win probability line
    fig.add_trace(go.Scatter(
        x=df["ball_number"],
        y=df["win_prob"],
        mode="lines",
        line=dict(color="#00C8FF", width=2.5),
        name=f"{batting_team} Win Prob",
        hovertemplate=(
            "Over %{customdata[0]}.%{customdata[1]}<br>"
            "Win Prob: %{y:.1%}<br>"
            "Score: %{customdata[2]}/%{customdata[3]}<br>"
            "Need: %{customdata[4]} off %{customdata[5]} balls"
            "<extra></extra>"
        ),
        customdata=df[["over_label", "ball_num",
                        "cum_runs", "cum_wickets",
                        "runs_needed", "balls_remaining"]].values
    ))

    # filled area under curve
    fig.add_trace(go.Scatter(
        x=df["ball_number"], y=df["win_prob"],
        fill="tozeroy",
        fillcolor="rgba(0,200,255,0.08)",
        line=dict(width=0),
        showlegend=False, hoverinfo="skip"
    ))

    # wicket markers
    if not wicket_balls.empty:
        fig.add_trace(go.Scatter(
            x=wicket_balls["ball_number"],
            y=wicket_balls["win_prob"],
            mode="markers",
            marker=dict(symbol="x", size=11,
                        color="#FF4C4C", line=dict(width=2)),
            name="Wicket",
            hovertemplate=(
                "WICKET — Over %{customdata[0]}.%{customdata[1]}<br>"
                "Win Prob dropped to %{y:.1%}"
                "<extra></extra>"
            ),
            customdata=wicket_balls[["over_label", "ball_num"]].values
        ))

    # over boundary lines (every 6 balls)
    for over in range(6, 121, 6):
        fig.add_vline(x=over, line_width=0.4,
                      line_color="rgba(255,255,255,0.1)")

    # phase labels
    for label, x in [("Powerplay", 18), ("Middle Overs", 66), ("Death", 108)]:
        fig.add_annotation(x=x, y=0.97, text=label,
                           showarrow=False, yref="paper",
                           font=dict(size=10, color="rgba(200,200,200,0.5)"))

    fig.update_layout(
        title=dict(
            text=f"<b>Win Probability — {batting_team} vs {bowling_team}</b><br>"
                 f"<sup>Target: {int(target)} runs</sup>",
            font=dict(size=18)
        ),
        xaxis=dict(
            title="Ball Number",
            tickvals=list(range(6, 121, 6)),
            ticktext=[f"Ov {i}" for i in range(1, 21)],
            gridcolor="rgba(255,255,255,0.05)"
        ),
        yaxis=dict(
            title="Win Probability",
            tickformat=".0%",
            range=[0, 1],
            gridcolor="rgba(255,255,255,0.05)"
        ),
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=500
    )

    fig.write_html("win_probability_curve.html")
    fig.show()
    print("✅ Saved: win_probability_curve.html")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Powerplay Wicket Impact Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════

def plot_powerplay_wicket_impact():
    df = pd.read_sql("SELECT * FROM powerplay_leverage", conn)

    # order phases
    phase_order = ["Powerplay (1-6)", "Middle (7-15)", "Death (16-20)"]
    df["phase"] = pd.Categorical(df["phase"], categories=phase_order, ordered=True)
    df = df.sort_values("phase")

    colors = ["#FF6B6B", "#FFA94D", "#69DB7C"]
    bar_colors = colors[:len(df)]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["phase"],
        y=df["avg_wp_drop_pct"],
        marker=dict(
            color=bar_colors,
            line=dict(color="rgba(255,255,255,0.2)", width=1)
        ),
        text=[f"{v:.1f}%" for v in df["avg_wp_drop_pct"]],
        textposition="outside",
        textfont=dict(size=14, color="white"),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Avg Win Prob Drop: %{y:.1f}%<br>"
            "Wickets in phase: %{customdata}"
            "<extra></extra>"
        ),
        customdata=df["wickets"]
    ))

    # annotation: key insight
    fig.add_annotation(
        x="Powerplay (1-6)", y=df[df["phase"] == "Powerplay (1-6)"]["avg_wp_drop_pct"].values[0] + 1.5,
        text="⚡ 2.4× more impactful<br>than a death wicket",
        showarrow=False,
        font=dict(size=11, color="#FFD43B"),
        bgcolor="rgba(0,0,0,0.5)",
        bordercolor="#FFD43B",
        borderwidth=1,
        borderpad=4
    )

    fig.update_layout(
        title=dict(
            text="<b>Powerplay Wicket Impact</b><br>"
                 "<sup>Average win probability drop (%) by match phase</sup>",
            font=dict(size=18)
        ),
        xaxis=dict(title="Match Phase", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(
            title="Avg Win Prob Drop (%)",
            gridcolor="rgba(255,255,255,0.05)",
            range=[0, df["avg_wp_drop_pct"].max() + 4]
        ),
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font=dict(color="white"),
        showlegend=False,
        height=480
    )

    fig.write_html("powerplay_wicket_impact.html")
    fig.show()
    print("✅ Saved: powerplay_wicket_impact.html")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Choke Detector
# ═══════════════════════════════════════════════════════════════════════════════

def plot_choke_detector():
    df = pd.read_sql(
        "SELECT * FROM choke_detector ORDER BY choke_rate_pct DESC",
        conn
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["choke_rate_pct"],
        y=df["batting_team"],
        orientation="h",
        marker=dict(
            color=df["choke_rate_pct"],
            colorscale="RdYlGn_r",
            showscale=True,
            colorbar=dict(title="Choke %", ticksuffix="%")
        ),
        text=[f"{v:.1f}%" for v in df["choke_rate_pct"]],
        textposition="outside",
        textfont=dict(size=11, color="white"),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Choke Rate: %{x:.1f}%<br>"
            "Chokes: %{customdata[0]} / %{customdata[1]} chases"
            "<extra></extra>"
        ),
        customdata=df[["choke_count", "total_chases"]].values
    ))

    fig.update_layout(
        title=dict(
            text="<b>IPL Choke Detector</b><br>"
                 "<sup>Teams that lost despite >80% win probability (chase)</sup>",
            font=dict(size=18)
        ),
        xaxis=dict(
            title="Choke Rate (%)",
            gridcolor="rgba(255,255,255,0.05)",
            range=[0, df["choke_rate_pct"].max() + 5]
        ),
        yaxis=dict(
            title="",
            autorange="reversed",
            gridcolor="rgba(255,255,255,0.05)"
        ),
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font=dict(color="white"),
        height=550,
        margin=dict(l=200)
    )

    fig.write_html("choke_detector.html")
    fig.show()
    print("✅ Saved: choke_detector.html")


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n📊 Chart 1: Win Probability Curve")
    plot_win_probability_curve(MATCH_ID)

    print("\n📊 Chart 2: Powerplay Wicket Impact")
    plot_powerplay_wicket_impact()

    print("\n📊 Chart 3: Choke Detector")
    plot_choke_detector()

    conn.close()
    print("\n✅ All charts generated!")