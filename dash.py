import sqlite3
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import os 
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Win Probability Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #1c1f2e;
        border-radius: 12px;
        padding: 18px 22px;
        border: 1px solid #2e3250;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00c8ff;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #8a8fb5;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #e0e4ff;
        padding: 6px 0 2px 0;
        border-bottom: 2px solid #00c8ff33;
        margin-bottom: 14px;
    }
    div[data-testid="stTabs"] button {
        font-size: 0.95rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

DARK = "#0f1117"
CARD = "#1c1f2e"

# ── Load resources ─────────────────────────────────────────────────────────────
@st.cache_resource
# ── Load resources ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("win_prob_model.pkl", "rb") as f:
        return pickle.load(f)

# Completely independent connection engine to avoid Streamlit multi-threading locks
def get_conn():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "ipl.db")
    return sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
 
@st.cache_data
def load_matches():
    db_connection = get_conn()
    try:
        return pd.read_sql("""
            SELECT match_id, season, date, team1, team2, winner, venue, toss_winner, toss_decision
            FROM matches
            WHERE winner IS NOT NULL
            ORDER BY date DESC
        """, db_connection)
    finally:
        db_connection.close()

@st.cache_data
def load_analysis_tables():
    db_connection = get_conn()
    try:
        swing   = pd.read_sql("SELECT * FROM wicket_swing_index ORDER BY avg_wp_drop_pct DESC", db_connection)
        leverage = pd.read_sql("SELECT * FROM powerplay_leverage", db_connection)
        choke   = pd.read_sql("SELECT * FROM choke_detector ORDER BY choke_rate_pct DESC", db_connection)
        return swing, leverage, choke
    finally:
        db_connection.close()

model   = load_model()
matches = load_matches()
swing_df, leverage_df, choke_df = load_analysis_tables()

FEATURES = ["cum_runs", "runs_needed", "balls_remaining", "wickets_remaining",
            "required_rr", "current_rr", "rr_diff", "cum_overs"]

# ── Win prob builder ───────────────────────────────────────────────────────────
def build_chase_states(match_id):
    db_connection = get_conn()
    
    try:
        # 1. Fetch First Innings Score to establish the target
        q1 = """
            SELECT SUM(runs_total) AS first_innings_total 
            FROM deliveries 
            WHERE match_id = ? AND inning = 1
        """
        target_row = pd.read_sql(q1, db_connection, params=(match_id,))
        
        if target_row.empty or pd.isna(target_row.iloc[0]["first_innings_total"]):
            return None, None
            
        target = int(target_row.iloc[0]["first_innings_total"]) + 1

        # 2. Fetch Second Innings ball-by-ball data
        # Selecting ball_num as both ball_num and ball_number satisfies both math and plots
        q2 = """
            SELECT over_num, 
                   ball_num, 
                   ball_num AS ball_number, 
                   runs_total, is_wicket,
                   extras_type, batting_team, bowling_team, player_out, dismissal_kind
            FROM deliveries 
            WHERE match_id = ? AND inning = 2
            ORDER BY over_num, ball_num
        """
        df = pd.read_sql(q2, db_connection, params=(match_id,))
        
    except Exception as e:
        print(f"Database query error: {e}")
        raise e
    finally:
        try:
            db_connection.close()
        except:
            pass
        
    if df.empty:
        return None, None
        
    # ─── 3. FEATURE ENGINEERING (Executed in the correct mathematical order) ───
    df["is_legal"]   = (~df["extras_type"].isin(["wides", "noballs"])).astype(int)
    df["cum_runs"]   = df["runs_total"].cumsum()
    df["cum_wickets"] = df["is_wicket"].cumsum()
    df["cum_balls"]  = df["is_legal"].cumsum()
    df["cum_overs"]  = df["cum_balls"] / 6
    df["ball_number"] = range(1, len(df) + 1)

    df["runs_needed"]       = target - df["cum_runs"]
    df["balls_remaining"]   = 120 - df["cum_balls"]
    df["wickets_remaining"] = 10 - df["cum_wickets"]
    
    # Safe Run Rate calculations using np.where to prevent Division by Zero errors
    df["required_rr"]       = np.where(
        df["balls_remaining"] > 0,
        df["runs_needed"] / (df["balls_remaining"] / 6), 999)
    df["current_rr"]        = np.where(
        df["cum_overs"] > 0, df["cum_runs"] / df["cum_overs"], 0)
    df["rr_diff"]           = df["current_rr"] - df["required_rr"]

    # Fill numeric feature columns with 0
    numeric_cols = ["cum_runs", "cum_wickets", "cum_balls", "runs_needed",
                    "balls_remaining", "wickets_remaining", "required_rr", "current_rr", "rr_diff"]
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Fill text columns with an empty string
    text_cols = ["extras_type", "player_out", "dismissal_kind"]
    df[text_cols] = df[text_cols].fillna("")

    df["required_rr"] = df["required_rr"].clip(lower=0)
    df["over_label"]  = df["over_num"] + 1

    # ─── 4. ML MODEL PREDICTION ─────────────────────────────────────────
    # Now that all required features are built, your model will run flawlessly
    df["win_prob"] = model.predict_proba(df[FEATURES])[:, 1]
    
    return df, int(target)



    df["is_legal"]   = (~df["extras_type"].isin(["wides", "noballs"])).astype(int)
    df["cum_runs"]   = df["runs_total"].cumsum()
    df["cum_wickets"] = df["is_wicket"].cumsum()
    df["cum_balls"]  = df["is_legal"].cumsum()
    df["cum_overs"]  = df["cum_balls"] / 6
    df["ball_number"] = range(1, len(df) + 1)

    df["runs_needed"]       = target - df["cum_runs"]
    df["balls_remaining"]   = 120 - df["cum_balls"]
    df["wickets_remaining"] = 10 - df["cum_wickets"]
    df["required_rr"]       = np.where(
        df["balls_remaining"] > 0,
        df["runs_needed"] / (df["balls_remaining"] / 6), 999)
    df["current_rr"]        = np.where(
        df["cum_overs"] > 0, df["cum_runs"] / df["cum_overs"], 0)
    df["rr_diff"]           = df["current_rr"] - df["required_rr"]

    # Fill numeric feature columns with 0
    numeric_cols = ["cum_runs", "cum_wickets", "cum_balls", "runs_needed",
                    "balls_remaining", "wickets_remaining", "required_rr", "current_rr", "rr_diff"]
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Fill text columns with an empty string
    text_cols = ["extras_type", "player_out", "dismissal_kind"]
    df[text_cols] = df[text_cols].fillna("")

    df["required_rr"] = df["required_rr"].clip(lower=0)
    df["win_prob"] = model.predict_proba(df[FEATURES])[:, 1]
    df["over_label"]  = df["over_num"] + 1
    return df, int(target)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏏 IPL Dashboard")
    st.markdown("---")

    # model stats
    total_matches = len(matches)
    seasons = sorted(matches["season"].unique())
    st.markdown(f"**Matches in DB:** {total_matches:,}")
    st.markdown(f"**Seasons:** {seasons[0]} – {seasons[-1]}")
    st.markdown(f"**Model accuracy:** 79.1%")
    st.markdown(f"**Chase win rate:** 52.4%")
    st.markdown("---")

    # season filter for match table
    st.markdown("#### Filter Matches")
    selected_season = st.selectbox("Season", ["All"] + list(reversed(seasons)))
    team_list = sorted(set(matches["team1"].tolist() + matches["team2"].tolist()))
    selected_team = st.selectbox("Team", ["All"] + team_list)

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Match Browser",
    "📈 Win Probability",
    "⚡ Wicket Impact",
    "😬 Choke Detector",
    "🏏 Ball-by-Ball"
])

# ───────────────────────────────────────────────────────────────────────────────
# TAB 1 — Match Browser
# ───────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Match Browser</div>', unsafe_allow_html=True)
    st.caption("Browse all matches and copy a Match ID to use in other tabs.")

    filtered = matches.copy()
    if selected_season != "All":
        filtered = filtered[filtered["season"] == selected_season]
    if selected_team != "All":
        filtered = filtered[
            (filtered["team1"] == selected_team) | (filtered["team2"] == selected_team)
        ]

    # summary metrics
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1, c2, c3, c4],
        [len(filtered),
         filtered["team1"].nunique() + filtered["team2"].nunique(),
         filtered["season"].nunique(),
         filtered["winner"].value_counts().index[0] if len(filtered) else "—"],
        ["Matches Shown", "Teams", "Seasons", "Most Wins (filtered)"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    display = filtered[["match_id","date","season","team1","team2","winner","venue"]].copy()
    display["date"] = pd.to_datetime(display["date"]).dt.strftime("%d %b %Y")
    st.dataframe(display, use_container_width=True, height=520,
                 column_config={
                     "match_id": st.column_config.TextColumn("Match ID", width="medium"),
                     "date":     st.column_config.TextColumn("Date"),
                     "season":   st.column_config.TextColumn("Season"),
                     "team1":    st.column_config.TextColumn("Team 1"),
                     "team2":    st.column_config.TextColumn("Team 2"),
                     "winner":   st.column_config.TextColumn("Winner"),
                     "venue":    st.column_config.TextColumn("Venue"),
                 })

# ───────────────────────────────────────────────────────────────────────────────
# TAB 2 — Win Probability Curve
# ───────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Win Probability Curve</div>', unsafe_allow_html=True)
    st.caption("Enter a Match ID from the Match Browser tab to visualize over-by-over win probability.")

    col_input, col_btn = st.columns([3, 1])
    with col_input:
        match_id_input = st.text_input("Match ID", placeholder="e.g. 335982", label_visibility="collapsed")
    with col_btn:
        run_btn = st.button("Generate →", use_container_width=True)

    if run_btn and match_id_input:
        with st.spinner("Computing win probability..."):
            df_chase, target = build_chase_states(match_id_input.strip())

        if df_chase is None:
            st.error("No 2nd innings data found for this Match ID. Check the Match Browser.")
        else:
            # match info
            match_info = matches[matches["match_id"] == match_id_input.strip()]
            if not match_info.empty:
                mi = match_info.iloc[0]
                st.markdown(f"### {mi['team1']} vs {mi['team2']} — {mi['date']}")
                st.markdown(f"**Venue:** {mi['venue']} &nbsp;|&nbsp; **Winner:** {mi['winner']} &nbsp;|&nbsp; **Target:** {target}")

            batting_team = df_chase["batting_team"].iloc[0]
            bowling_team = df_chase["bowling_team"].iloc[0]
            wicket_balls = df_chase[df_chase["is_wicket"] == 1]

            fig = go.Figure()

            fig.add_hrect(y0=0.5, y1=1.0, fillcolor="rgba(0,200,100,0.05)", line_width=0)
            fig.add_hrect(y0=0.0, y1=0.5, fillcolor="rgba(220,50,50,0.05)",  line_width=0)
            fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(180,180,180,0.4)", line_width=1)

            # area fill
            fig.add_trace(go.Scatter(
                x=df_chase["ball_num"], y=df_chase["win_prob"],
                fill="tozeroy", fillcolor="rgba(0,200,255,0.07)",
                line=dict(width=0), showlegend=False, hoverinfo="skip"
            ))

            # main line
            fig.add_trace(go.Scatter(
                x=df_chase["ball_num"], y=df_chase["win_prob"],
                mode="lines", line=dict(color="#00C8FF", width=2.5),
                name=f"{batting_team} Win Prob",
                hovertemplate=(
                    "<b>Over %{customdata[0]}.%{customdata[1]}</b><br>"
                    "Win Prob: <b>%{y:.1%}</b><br>"
                    "Score: %{customdata[2]}/%{customdata[3]}<br>"
                    "Need: %{customdata[4]} off %{customdata[5]} balls"
                    "<extra></extra>"
                ),
                customdata=df_chase[["over_label","ball_num","cum_runs",
                                     "cum_wickets","runs_needed","balls_remaining"]].values
            ))

            # wicket markers
            if not wicket_balls.empty:
                fig.add_trace(go.Scatter(
                    x=wicket_balls["ball_num"], y=wicket_balls["win_prob"],
                    mode="markers",
                    marker=dict(symbol="x", size=12, color="#FF4C4C",
                                line=dict(width=2.5, color="#FF4C4C")),
                    name="Wicket",
                    hovertemplate=(
                        "<b>WICKET</b> — Over %{customdata[0]}.%{customdata[1]}<br>"
                        "%{customdata[2]} — %{customdata[3]}<br>"
                        "Win Prob: %{y:.1%}"
                        "<extra></extra>"
                    ),
                    customdata=wicket_balls[["over_label","ball_num",
                                             "player_out","dismissal_kind"]].values
                ))

            # over lines + phase labels
            for over in range(6, 121, 6):
                fig.add_vline(x=over, line_width=0.3, line_color="rgba(255,255,255,0.08)")
            for label, x in [("Powerplay", 18), ("Middle Overs", 66), ("Death", 108)]:
                fig.add_annotation(x=x, y=0.97, text=label, showarrow=False,
                                   yref="paper", font=dict(size=10, color="rgba(200,200,200,0.4)"))

            fig.update_layout(
                xaxis=dict(title="Ball", tickvals=list(range(6,121,6)),
                           ticktext=[f"Ov {i}" for i in range(1,21)],
                           gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(title="Win Probability", tickformat=".0%",
                           range=[0,1], gridcolor="rgba(255,255,255,0.04)"),
                plot_bgcolor=DARK, paper_bgcolor=DARK,
                font=dict(color="white"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified", height=460, margin=dict(t=30, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)

            # quick stats below chart
            final_wp = df_chase["win_prob"].iloc[-1]
            max_wp   = df_chase["win_prob"].max()
            min_wp   = df_chase["win_prob"].min()
            n_wkts   = int(df_chase["is_wicket"].sum())

            s1, s2, s3, s4 = st.columns(4)
            for col, val, label in zip(
                [s1, s2, s3, s4],
                [f"{max_wp:.0%}", f"{min_wp:.0%}", f"{final_wp:.0%}", str(n_wkts)],
                ["Peak Win Prob", "Lowest Win Prob", "Final Ball WP", "Wickets Fallen"]
            ):
                col.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

    elif run_btn:
        st.warning("Please enter a Match ID.")

# ───────────────────────────────────────────────────────────────────────────────
# TAB 3 — Wicket Impact
# ───────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Wicket Impact Analysis</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("##### Dismissal Type — Avg Win Prob Drop")
        fig_swing = go.Figure(go.Bar(
            x=swing_df["avg_wp_drop_pct"],
            y=swing_df["dismissal_kind"],
            orientation="h",
            marker=dict(
                color=swing_df["avg_wp_drop_pct"],
                colorscale="Blues",
                showscale=False,
                line=dict(color="rgba(255,255,255,0.1)", width=0.5)
            ),
            text=[f"{v:.1f}%" for v in swing_df["avg_wp_drop_pct"]],
            textposition="outside",
            textfont=dict(color="white", size=11),
            hovertemplate="<b>%{y}</b><br>Avg Drop: %{x:.1f}%<br>Count: %{customdata}<extra></extra>",
            customdata=swing_df["n_wickets"]
        ))
        fig_swing.update_layout(
            xaxis=dict(title="Win Prob Drop (%)", gridcolor="rgba(255,255,255,0.05)",
                       range=[0, swing_df["avg_wp_drop_pct"].max() + 3]),
            yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.04)"),
            plot_bgcolor=DARK, paper_bgcolor=DARK,
            font=dict(color="white"), height=380, margin=dict(t=10, l=160)
        )
        st.plotly_chart(fig_swing, use_container_width=True)

    with col_b:
        st.markdown("##### Powerplay Leverage — Phase Impact")
        phase_order = ["Powerplay (1-6)", "Middle (7-15)", "Death (16-20)"]
        lev = leverage_df.copy()
        lev["phase"] = pd.Categorical(lev["phase"], categories=phase_order, ordered=True)
        lev = lev.sort_values("phase")

        fig_lev = go.Figure(go.Bar(
            x=lev["phase"],
            y=lev["avg_wp_drop_pct"],
            marker=dict(color=["#FF6B6B","#FFA94D","#69DB7C"],
                        line=dict(color="rgba(255,255,255,0.15)", width=1)),
            text=[f"{v:.1f}%" for v in lev["avg_wp_drop_pct"]],
            textposition="outside",
            textfont=dict(size=14, color="white"),
            hovertemplate="<b>%{x}</b><br>Avg Drop: %{y:.1f}%<br>Wickets: %{customdata}<extra></extra>",
            customdata=lev["wickets"]
        ))

        powerplay_val = lev[lev["phase"] == "Powerplay (1-6)"]["avg_wp_drop_pct"].values
        death_val     = lev[lev["phase"] == "Death (16-20)"]["avg_wp_drop_pct"].values
        if len(powerplay_val) and len(death_val) and death_val[0] > 0:
            ratio = powerplay_val[0] / death_val[0]
            fig_lev.add_annotation(
                x="Powerplay (1-6)", y=powerplay_val[0] + 1.8,
                text=f"⚡ {ratio:.1f}× more impactful<br>than a death wicket",
                showarrow=False, font=dict(size=10, color="#FFD43B"),
                bgcolor="rgba(0,0,0,0.55)", bordercolor="#FFD43B",
                borderwidth=1, borderpad=4
            )

        fig_lev.update_layout(
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(title="Avg Win Prob Drop (%)", gridcolor="rgba(255,255,255,0.04)",
                       range=[0, lev["avg_wp_drop_pct"].max() + 5]),
            plot_bgcolor=DARK, paper_bgcolor=DARK,
            font=dict(color="white"), height=380,
            showlegend=False, margin=dict(t=10)
        )
        st.plotly_chart(fig_lev, use_container_width=True)

    # raw table
    with st.expander("📊 Raw Data — Wicket Swing Index"):
        st.dataframe(swing_df, use_container_width=True)

# ───────────────────────────────────────────────────────────────────────────────
# TAB 4 — Choke Detector
# ───────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Choke Detector</div>', unsafe_allow_html=True)
    st.caption("Teams that had >80% win probability while chasing and still lost.")

    fig_choke = go.Figure(go.Bar(
        x=choke_df["choke_rate_pct"],
        y=choke_df["batting_team"],
        orientation="h",
        marker=dict(
            color=choke_df["choke_rate_pct"],
            colorscale="RdYlGn_r",
            showscale=True,
            colorbar=dict(title="Choke %", ticksuffix="%", len=0.6)
        ),
        text=[f"{v:.1f}%" for v in choke_df["choke_rate_pct"]],
        textposition="outside",
        textfont=dict(size=11, color="white"),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Choke Rate: %{x:.1f}%<br>"
            "Chokes: %{customdata[0]} / %{customdata[1]} chases"
            "<extra></extra>"
        ),
        customdata=choke_df[["choke_count","total_chases"]].values
    ))

    fig_choke.update_layout(
        xaxis=dict(title="Choke Rate (%)", gridcolor="rgba(255,255,255,0.04)",
                   range=[0, choke_df["choke_rate_pct"].max() + 5]),
        yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.04)"),
        plot_bgcolor=DARK, paper_bgcolor=DARK,
        font=dict(color="white"), height=560,
        margin=dict(l=200, t=20)
    )
    st.plotly_chart(fig_choke, use_container_width=True)

    # metrics row
    worst      = choke_df.iloc[0]
    most_chokes = choke_df.sort_values("choke_count", ascending=False).iloc[0]
    best       = choke_df.iloc[-1]
    avg_choke  = choke_df["choke_rate_pct"].mean()

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1, c2, c3, c4],
        [f"{worst['batting_team']} ({worst['choke_rate_pct']}%)",
         f"{most_chokes['batting_team']} ({int(most_chokes['choke_count'])})",
         f"{best['batting_team']} ({best['choke_rate_pct']}%)",
         f"{avg_choke:.1f}%"],
        ["Highest Choke Rate", "Most Chokes (count)", "Lowest Choke Rate", "Avg Choke Rate"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.2rem">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    with st.expander("📊 Raw Data — Choke Detector"):
        st.dataframe(choke_df, use_container_width=True)

# ───────────────────────────────────────────────────────────────────────────────
# TAB 5 — Ball-by-Ball
# ───────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">Ball-by-Ball Win Probability</div>', unsafe_allow_html=True)
    st.caption("Enter a Match ID to see the full ball-by-ball breakdown with win probability at every delivery.")

    col_i, col_b2 = st.columns([3,1])
    with col_i:
        bb_match_id = st.text_input("Match ID ", placeholder="e.g. 335982", label_visibility="collapsed")
    with col_b2:
        bb_btn = st.button("Load →", use_container_width=True)

    if bb_btn and bb_match_id:
        with st.spinner("Loading ball-by-ball data..."):
            df_bb, tgt = build_chase_states(bb_match_id.strip())

        if df_bb is None:
            st.error("No data found for this Match ID.")
        else:
            match_info = matches[matches["match_id"] == bb_match_id.strip()]
            if not match_info.empty:
                mi = match_info.iloc[0]
                st.markdown(f"### {mi['team1']} vs {mi['team2']} — {mi['date']}  |  Target: {tgt}")

            # search / filter
            over_filter = st.slider("Filter by over", 1, 20, (1, 20))
            df_show = df_bb[
                (df_bb["over_label"] >= over_filter[0]) &
                (df_bb["over_label"] <= over_filter[1])
            ].copy()

            df_show["Win Prob"] = (df_show["win_prob"] * 100).round(1).astype(str) + "%"
            df_show["Over.Ball"] = df_show["over_label"].astype(str) + "." + df_show["ball_num"].astype(str)
            df_show["Wicket"] = df_show["is_wicket"].map({0: "", 1: "🔴"})
            df_show["Dismissal"] = df_show.apply(
                lambda r: f"{r['player_out']} ({r['dismissal_kind']})" if r["is_wicket"] else "", axis=1)

            display_cols = {
                "Over.Ball":        "Over.Ball",
                "batting_team":     "Batting",
                "cum_runs":         "Score",
                "cum_wickets":      "Wkts",
                "runs_needed":      "Need",
                "balls_remaining":  "Balls Left",
                "required_rr":      "Req RR",
                "current_rr":       "Curr RR",
                "Win Prob":         "Win Prob",
                "Wicket":           "W",
                "Dismissal":        "Dismissal",
            }

            out_df = df_show[list(display_cols.keys())].rename(columns=display_cols)
            out_df["Req RR"]  = out_df["Req RR"].round(2)
            out_df["Curr RR"] = out_df["Curr RR"].round(2)

            st.dataframe(
                out_df,
                use_container_width=True,
                height=560,
                column_config={
                    "Win Prob": st.column_config.TextColumn("Win Prob"),
                    "W":        st.column_config.TextColumn("W", width="small"),
                }
            )

            # mini sparkline of just filtered overs
            fig_mini = go.Figure(go.Scatter(
                x=df_show["ball_number"], y=df_show["win_prob"],
                mode="lines+markers",
                line=dict(color="#00C8FF", width=2),
                marker=dict(
                    color=["#FF4C4C" if w else "#00C8FF" for w in df_show["is_wicket"]],
                    size=[10 if w else 5 for w in df_show["is_wicket"]],
                    symbol=["x" if w else "circle" for w in df_show["is_wicket"]]
                ),
                hovertemplate="Over %{customdata}<br>Win Prob: %{y:.1%}<extra></extra>",
                customdata=df_show["Over.Ball"]
            ))
            fig_mini.add_hline(y=0.5, line_dash="dash", line_color="rgba(200,200,200,0.3)")
            fig_mini.update_layout(
                xaxis=dict(title="Ball", gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(title="Win Prob", tickformat=".0%",
                           range=[0,1], gridcolor="rgba(255,255,255,0.04)"),
                plot_bgcolor=DARK, paper_bgcolor=DARK,
                font=dict(color="white"), height=260,
                margin=dict(t=10, b=30)
            )
            st.plotly_chart(fig_mini, use_container_width=True)

    elif bb_btn:
        st.warning("Please enter a Match ID.")
