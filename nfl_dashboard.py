"""
NFL Valuation Dashboard — Streamlit skeleton with fake data
------------------------------------------------------------
Install dependencies:
    pip install streamlit pandas plotly

Run:
    streamlit run nfl_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import random

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="NFL Valuation Dashboard",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# FAKE DATA  — replace with:
#   df = pd.read_csv("player_valuations.csv")
# ─────────────────────────────────────────
random.seed(42)

TEAMS = [
    "SF 49ers", "KC Chiefs", "PHI Eagles", "DAL Cowboys",
    "BUF Bills", "CIN Bengals", "LAR Rams", "GB Packers",
    "MIN Vikings", "BAL Ravens",
]
POSITIONS = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "CB", "S"]
CONFERENCES = {
    "SF 49ers": "NFC", "PHI Eagles": "NFC", "DAL Cowboys": "NFC",
    "LAR Rams": "NFC", "GB Packers": "NFC", "MIN Vikings": "NFC",
    "KC Chiefs": "AFC", "BUF Bills": "AFC", "CIN Bengals": "AFC",
    "BAL Ravens": "AFC",
}

FIRST_NAMES = ["Patrick", "Justin", "Josh", "Lamar", "Joe", "Brock", "Dak",
               "Jalen", "Trevor", "Tua", "Davante", "Tyreek", "Stefon",
               "Justin", "CeeDee", "Deebo", "Ja'Marr", "Cooper", "Travis",
               "Mark", "Darren", "Austin", "Saquon", "Derrick", "Nick",
               "Christian", "Tony", "Aaron", "Micah", "Myles"]
LAST_NAMES  = ["Mahomes", "Jefferson", "Allen", "Jackson", "Burrow", "Purdy",
               "Prescott", "Hurts", "Lawrence", "Tagovailoa", "Adams",
               "Hill", "Diggs", "Jefferson", "Lamb", "Samuel", "Chase",
               "Kupp", "Kelce", "Andrews", "Henry", "Ekeler", "Barkley",
               "Henry", "Chubb", "McCaffrey", "Pollard", "Donald",
               "Parsons", "Garrett"]

def make_fake_data(n=120):
    rows = []
    for i in range(n):
        pos = random.choice(POSITIONS)
        cap = round(random.uniform(0.8, 40.0), 2)
        model_val = round(cap * random.uniform(0.5, 1.8), 2)
        delta = round(model_val - cap, 2)
        if delta > 2:
            verdict = "Undervalued"
        elif delta < -2:
            verdict = "Overvalued"
        else:
            verdict = "Fair"
        rows.append({
            "player_name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            "position": pos,
            "team": random.choice(TEAMS),
            "age": random.randint(22, 36),
            "cap_hit_m": cap,
            "model_value_m": model_val,
            "delta_m": delta,
            "verdict": verdict,
            "confidence": round(random.uniform(60, 99), 1),
            "rushing_yards": random.randint(0, 1800) if pos in ["RB", "QB"] else 0,
            "receiving_yards": random.randint(0, 1600) if pos in ["WR", "TE", "RB"] else 0,
            "touchdowns": random.randint(0, 18),
            "years_remaining": random.randint(1, 5),
            "epa_per_play": round(random.uniform(-0.3, 0.5), 3),
        })
    return pd.DataFrame(rows)

@st.cache_data
def load_data():
    return make_fake_data(120)

df = load_data()

# ─────────────────────────────────────────
# CUSTOM CSS — tighten up Streamlit defaults
# ─────────────────────────────────────────
st.markdown("""
<style>
    /* tighter metric cards */
    [data-testid="metric-container"] {
        background: #1a1e25;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 14px 18px;
    }
    /* hide streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    /* sidebar background */
    [data-testid="stSidebar"] { background: #f0f2f5; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## NFL Valuation")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Overview", "Player Profile", "Team Roster", "Smart Query"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Filter by position**")
    pos_filter = st.multiselect(
        "Position", options=sorted(df["position"].unique()),
        default=[], label_visibility="collapsed",
        placeholder="All positions"
    )

    st.markdown("**Filter by verdict**")
    verdict_filter = st.multiselect(
        "Verdict", options=["Undervalued", "Fair", "Overvalued"],
        default=[], label_visibility="collapsed",
        placeholder="All verdicts"
    )

    st.markdown("---")
    st.caption("(write data details, e.g. last updated")


# helper: apply sidebar filters to any dataframe slice
def apply_filters(data):
    if pos_filter:
        data = data[data["position"].isin(pos_filter)]
    if verdict_filter:
        data = data[data["verdict"].isin(verdict_filter)]
    return data


# ─────────────────────────────────────────
# VERDICT COLOR HELPERS
# ─────────────────────────────────────────
VERDICT_COLOR = {"Undervalued": "#2ecc8a", "Fair": "#5b8ef0", "Overvalued": "#f05c5c"}

def verdict_badge(v):
    color = VERDICT_COLOR.get(v, "#888")
    return f'<span style="background:rgba(0,0,0,0.3);color:{color};border:1px solid {color}55;padding:2px 10px;border-radius:4px;font-size:12px;font-family:monospace;">{v}</span>'


# ═══════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════
if page == "Overview":
    st.title("League snapshot")
    st.caption("Placeholder data")

    filtered = apply_filters(df)

    # ── Metric row ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Undervalued", len(filtered[filtered.verdict == "Undervalued"]), help="Players where model value > cap hit by >$2M")
    c2.metric("Overvalued",  len(filtered[filtered.verdict == "Overvalued"]),  help="Players where cap hit > model value by >$2M")
    c3.metric("Fair value",  len(filtered[filtered.verdict == "Fair"]),         help="Players within $2M of model value")
    c4.metric("Players analyzed", len(filtered))

    st.markdown("---")

    # ── Verdict distribution chart ──
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Verdict breakdown")
        verdict_counts = filtered["verdict"].value_counts().reset_index()
        verdict_counts.columns = ["verdict", "count"]
        fig_pie = px.pie(
            verdict_counts, names="verdict", values="count",
            color="verdict",
            color_discrete_map=VERDICT_COLOR,
            hole=0.55,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#8a90a0", margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15),
            showlegend=True,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("Cap hit vs model value")
        fig_scatter = px.scatter(
            filtered, x="cap_hit_m", y="model_value_m",
            color="verdict", color_discrete_map=VERDICT_COLOR,
            hover_name="player_name",
            hover_data={"position": True, "team": True, "cap_hit_m": ":.1f", "model_value_m": ":.1f"},
            labels={"cap_hit_m": "Cap hit ($M)", "model_value_m": "Model value ($M)"},
            opacity=0.75,
        )
        # add y=x reference line
        max_val = max(filtered["cap_hit_m"].max(), filtered["model_value_m"].max()) + 2
        fig_scatter.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                              line=dict(color="#555c6e", dash="dot", width=1))
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#8a90a0", margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # ── Top tables ──
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Most Undervalued")
        top_under = (filtered[filtered.verdict == "Undervalued"]
                     .sort_values("delta_m", ascending=False)
                     .head(8)[["player_name", "position", "team", "cap_hit_m", "model_value_m", "delta_m"]]
                     .rename(columns={"player_name":"Player","position":"Pos","team":"Team",
                                      "cap_hit_m":"Cap ($M)","model_value_m":"Value ($M)","delta_m":"Δ ($M)"}))
        st.dataframe(top_under, use_container_width=True, hide_index=True)

    with col_b:
        st.subheader("Most Overvalued")
        top_over = (filtered[filtered.verdict == "Overvalued"]
                    .sort_values("delta_m", ascending=True)
                    .head(8)[["player_name", "position", "team", "cap_hit_m", "model_value_m", "delta_m"]]
                    .rename(columns={"player_name":"Player","position":"Pos","team":"Team",
                                     "cap_hit_m":"Cap ($M)","model_value_m":"Value ($M)","delta_m":"Δ ($M)"}))
        st.dataframe(top_over, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════
# PAGE: PLAYER PROFILE
# ═══════════════════════════════════════════
elif page == "Player Profile":
    st.title("Player Profile")

    player_names = sorted(df["player_name"].unique())
    selected = st.selectbox("Search or select a player", player_names)

    row = df[df["player_name"] == selected].iloc[0]
    verdict_color = VERDICT_COLOR.get(row["verdict"], "#888")

    st.markdown("---")

    # ── Header card ──
    c1, c2, c3 = st.columns([0.12, 0.55, 0.33])
    with c1:
        initials = "".join([n[0] for n in row["player_name"].split()][:2])
        st.markdown(
            f'<div style="width:64px;height:64px;border-radius:50%;background:#1a1e25;'
            f'border:1px solid rgba(255,255,255,0.12);display:flex;align-items:center;'
            f'justify-content:center;font-size:20px;font-weight:600;color:#5b8ef0;'
            f'font-family:monospace;">{initials}</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(f"### {row['player_name']}")
        st.caption(f"{row['position']} · {row['team']} · Age {row['age']}")
        st.markdown(verdict_badge(row["verdict"]), unsafe_allow_html=True)
    with c3:
        st.metric("Model confidence", f"{row['confidence']}%")

    st.markdown("---")

    # ── Stats row ──
    st.subheader("Key stats")
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("Cap hit",       f"${row['cap_hit_m']:.1f}M")
    s2.metric("Model value",   f"${row['model_value_m']:.1f}M")
    delta_val = row['delta_m']
    s3.metric("Δ Value",       f"${delta_val:+.1f}M", delta=f"{delta_val:+.1f}M",
              delta_color="normal" if delta_val >= 0 else "inverse")
    s4.metric("Yrs remaining", row["years_remaining"])
    s5.metric("Touchdowns",    row["touchdowns"])
    s6.metric("EPA / play",    f"{row['epa_per_play']:+.3f}")

    st.markdown("---")

    # ── Valuation bar chart ──
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("Valuation Breakdown")
        pos_avg_cap   = round(df[df.position == row["position"]]["cap_hit_m"].mean(), 2)
        pos_avg_value = round(df[df.position == row["position"]]["model_value_m"].mean(), 2)

        fig_bar = go.Figure()
        categories = ["Actual cap hit", "Model value", f"{row['position']} avg cap", f"{row['position']} avg value"]
        values     = [row["cap_hit_m"], row["model_value_m"], pos_avg_cap, pos_avg_value]
        colors     = ["#5b8ef0", "#2ecc8a", "#555c6e", "#555c6e"]

        fig_bar.add_trace(go.Bar(
            x=categories, y=values,
            marker_color=colors,
            text=[f"${v:.1f}M" for v in values],
            textposition="outside",
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#8a90a0", margin=dict(t=30, b=10, l=10, r=10),
            yaxis=dict(showgrid=True, gridcolor="#1a1e25", title="$M"),
            xaxis=dict(showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_r:
        st.subheader("Position Peers")
        peers = (df[(df.position == row["position"]) & (df.player_name != row["player_name"])]
                 .sort_values("delta_m", ascending=False)
                 .head(8)[["player_name", "team", "cap_hit_m", "model_value_m", "verdict"]]
                 .rename(columns={"player_name":"Player","team":"Team",
                                  "cap_hit_m":"Cap ($M)","model_value_m":"Value ($M)","verdict":"Verdict"}))
        st.dataframe(peers, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════
# PAGE: TEAM ROSTER
# ═══════════════════════════════════════════
elif page == "Team Roster":
    st.title("Team Roster")

    col_team, col_filter = st.columns([1, 2])
    with col_team:
        selected_team = st.selectbox("Select team", sorted(df["team"].unique()))
    with col_filter:
        verdict_pick = st.multiselect(
            "Filter by verdict", ["Undervalued", "Fair", "Overvalued"],
            placeholder="All verdicts", label_visibility="visible"
        )

    roster = df[df["team"] == selected_team].copy()
    if verdict_pick:
        roster = roster[roster["verdict"].isin(verdict_pick)]
    if pos_filter:
        roster = roster[roster["position"].isin(pos_filter)]

    # ── Team metrics ──
    st.markdown("---")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Total cap hit",   f"${roster['cap_hit_m'].sum():.1f}M")
    t2.metric("Undervalued",     len(roster[roster.verdict == "Undervalued"]))
    t3.metric("Overvalued",      len(roster[roster.verdict == "Overvalued"]))
    t4.metric("Fair value",      len(roster[roster.verdict == "Fair"]))

    st.markdown("---")

    # ── Roster chart ──
    st.subheader("Roster — cap hit vs model value")
    fig_roster = px.bar(
        roster.sort_values("cap_hit_m", ascending=False).head(20),
        x="player_name", y=["cap_hit_m", "model_value_m"],
        barmode="group",
        color_discrete_map={"cap_hit_m": "#5b8ef0", "model_value_m": "#2ecc8a"},
        labels={"value": "$M", "player_name": "", "variable": ""},
        hover_data={"position": True, "verdict": True},
    )
    fig_roster.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#8a90a0", margin=dict(t=10, b=80, l=10, r=10),
        xaxis_tickangle=-35, legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_roster, use_container_width=True)

    # ── Roster table ──
    st.subheader("Full roster")
    display_cols = ["player_name", "position", "age", "cap_hit_m",
                    "model_value_m", "delta_m", "verdict", "confidence"]
    display = (roster[display_cols]
               .sort_values("delta_m", ascending=False)
               .rename(columns={
                   "player_name":"Player","position":"Pos","age":"Age",
                   "cap_hit_m":"Cap ($M)","model_value_m":"Value ($M)",
                   "delta_m":"Δ ($M)","verdict":"Verdict","confidence":"Confidence %"
               }))
    st.dataframe(display, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════
# PAGE: SMART QUERY
# ═══════════════════════════════════════════
elif page == "Smart Query":
    st.title("Smart query")
    st.caption("Filter the player database with structured controls — wire natural language parsing on top later.")

    st.markdown("---")

    # ── Query controls ──
    q1, q2, q3 = st.columns(3)
    with q1:
        q_verdict  = st.selectbox("Valuation",  ["Any", "Undervalued", "Fair", "Overvalued"])
        q_position = st.selectbox("Position",   ["Any"] + sorted(df["position"].unique().tolist()))
    with q2:
        q_max_cap  = st.slider("Max cap hit ($M)",  min_value=0.5,  max_value=45.0, value=45.0, step=0.5)
        q_min_conf = st.slider("Min confidence (%)", min_value=50, max_value=99, value=60)
    with q3:
        q_conf_val = st.selectbox("Conference", ["Any", "AFC", "NFC"])
        q_max_age  = st.slider("Max age", min_value=22, max_value=40, value=40)

    # ── Example query chips ──
    st.markdown("**Quick filters**")
    chip_cols = st.columns(4)
    PRESETS = [
        ("Undervalued RBs under $6M",  {"verdict":"Undervalued","position":"RB","max_cap":6.0}),
        ("Overvalued WRs",             {"verdict":"Overvalued", "position":"WR"}),
        ("Best value QBs",             {"verdict":"Undervalued","position":"QB"}),
        ("Cheap TEs on rookie deals",  {"verdict":"Undervalued","position":"TE","max_cap":5.0}),
    ]
    preset_selected = None
    for i, (label, _) in enumerate(PRESETS):
        if chip_cols[i].button(label, use_container_width=True):
            preset_selected = PRESETS[i][1]

    # apply preset if clicked
    if preset_selected:
        q_verdict  = preset_selected.get("verdict", q_verdict)
        q_position = preset_selected.get("position", q_position)
        q_max_cap  = preset_selected.get("max_cap", q_max_cap)

    # ── Run query ──
    results = df.copy()
    if q_verdict  != "Any":    results = results[results["verdict"]  == q_verdict]
    if q_position != "Any":    results = results[results["position"] == q_position]
    results = results[results["cap_hit_m"]  <= q_max_cap]
    results = results[results["confidence"] >= q_min_conf]
    results = results[results["age"]        <= q_max_age]
    if q_conf_val != "Any":
        results = results[results["team"].map(CONFERENCES) == q_conf_val]

    st.markdown("---")
    st.subheader(f"Results — {len(results)} players")

    if len(results) == 0:
        st.info("No players match the current filters.")
    else:
        # summary metrics
        r1, r2, r3 = st.columns(3)
        r1.metric("Avg cap hit",     f"${results['cap_hit_m'].mean():.1f}M")
        r2.metric("Avg model value", f"${results['model_value_m'].mean():.1f}M")
        r3.metric("Avg Δ value",     f"${results['delta_m'].mean():+.1f}M")

        # results table
        display_cols = ["player_name","position","team","age","cap_hit_m",
                        "model_value_m","delta_m","verdict","confidence"]
        display = (results[display_cols]
                   .sort_values("delta_m", ascending=False)
                   .rename(columns={
                       "player_name":"Player","position":"Pos","team":"Team","age":"Age",
                       "cap_hit_m":"Cap ($M)","model_value_m":"Value ($M)",
                       "delta_m":"Δ ($M)","verdict":"Verdict","confidence":"Conf %"
                   }))
        st.dataframe(display, use_container_width=True, hide_index=True)

        # scatter of results
        if len(results) > 1:
            fig_q = px.scatter(
                results, x="cap_hit_m", y="model_value_m",
                color="verdict", color_discrete_map=VERDICT_COLOR,
                hover_name="player_name",
                hover_data={"position":True,"team":True},
                size="confidence", size_max=18,
                labels={"cap_hit_m":"Cap hit ($M)","model_value_m":"Model value ($M)"},
            )
            max_v = max(results["cap_hit_m"].max(), results["model_value_m"].max()) + 2
            fig_q.add_shape(type="line", x0=0, y0=0, x1=max_v, y1=max_v,
                            line=dict(color="#555c6e", dash="dot", width=1))
            fig_q.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#8a90a0", margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_q, use_container_width=True)


