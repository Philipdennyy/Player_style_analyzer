"""
app.py
======
Streamlit entry point for the Football Player Style Analyzer.

Run with:
    streamlit run app.py

HOW IT WORKS
  1. On first run, data_loader.load_events() downloads La Liga 2020/21 from
     StatsBomb (~2 min) and caches it locally.  All later runs load instantly.
  2. Streamlit caches the loaded DataFrame with @st.cache_data so switching
     players doesn't reload data.
  3. User picks a player from the dropdown (sidebar).
  4. App computes 3 metrics, assigns a role, then renders 3 visualisations.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))   # ensure local imports resolve

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader     import load_events, get_player_list, get_player_events, get_player_minutes
from visuals         import plot_heatmap, plot_passmap, plot_shotmap
from role_classifier import compute_metrics, classify_role, HIGH_PASSES_P90, HIGH_SHOTS_P90, ADV_AVG_X

# ── Page config — MUST be the first Streamlit call ────────────────────────────
st.set_page_config(
    page_title="⚽ Player Style Analyzer",
    page_icon="⚽",
    layout="wide",
)

# ── Dark theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0d1117 0%, #0f1923 100%);
    min-height: 100vh;
}
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}

/* Role card */
.role-card {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}
.role-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #58a6ff, #3fb950);
}
.role-title { font-size: 1.8rem; font-weight: 800; color: #58a6ff; margin: 0; }
.role-sub   { color: #8b949e; font-size: 0.85rem; margin-top: 0.2rem; }
.bullet {
    color: #e2e8f0; font-size: 0.9rem;
    padding: 0.35rem 0.75rem; margin-top: 0.45rem;
    background: rgba(88,166,255,0.08);
    border-left: 3px solid #58a6ff;
    border-radius: 0 6px 6px 0;
}

/* Metric pills in the header row */
.pill {
    display: inline-block;
    background: rgba(63,185,80,0.12);
    border: 1px solid rgba(63,185,80,0.3);
    border-radius: 16px;
    padding: 0.15rem 0.75rem;
    font-size: 0.8rem; color: #3fb950;
    margin: 0.15rem;
}

/* Section headers */
.sec-head {
    font-size: 1.05rem; font-weight: 700;
    color: #e2e8f0; margin: 1.5rem 0 0.6rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #30363d;
}

/* Global overrides */
h1, h2, h3 { color: #e2e8f0 !important; }
label { color: #8b949e !important; font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Load & cache data ─────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    """
    Load all La Liga 2020/21 events and build the player list.
    Streamlit caches the return value — this function runs only once per session.
    """
    events  = load_events()
    players = get_player_list(events)
    return events, players


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(players):
    """Render sidebar controls; return the selected player name."""
    st.sidebar.markdown("## ⚽ Player Style Analyzer")
    st.sidebar.caption("La Liga 2020/21 · StatsBomb Open Data")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Select a player**")

    # Optional search box to filter the dropdown
    search = st.sidebar.text_input(
        "Search", placeholder="e.g. Messi, Busquets, De Jong…",
        label_visibility="collapsed",
    )
    filtered = [p for p in players if search.lower() in p.lower()] if search else players

    if not filtered:
        st.sidebar.warning("No match — showing all players.")
        filtered = players

    # Default to Messi if present; otherwise first player alphabetically
    default_idx = next((i for i, p in enumerate(filtered) if "Messi" in p), 0)
    selected = st.sidebar.selectbox(
        "Player", filtered, index=default_idx,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""

""".format(p=HIGH_PASSES_P90, s=HIGH_SHOTS_P90, x=ADV_AVG_X))

    return selected


# ── Role emoji & description ──────────────────────────────────────────────────

ROLE_META = {
    "Playmaker": {
        "emoji": "🎯",
        "desc": "High ball-circulation volume, operates through midfield, rarely shoots.",
    },
    "Attacker/Finisher": {
        "emoji": "⚡",
        "desc": "Frequent shots from advanced positions — the primary scoring threat.",
    },
    "Deep/Defensive": {
        "emoji": "🛡️",
        "desc": "Low shooting output, sits deep — defensive anchor or holding midfielder.",
    },
}


# ── Main page ─────────────────────────────────────────────────────────────────

def main():

    # Load data (shows spinner on very first run only)
    with st.spinner("⚽ Loading StatsBomb data… (first run ~2 min, cached after that)"):
        try:
            events, players = load_data()
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            st.info("Check your internet connection. Data is fetched from StatsBomb's GitHub.")
            st.stop()

    if not players:
        st.error("No players found. Check data_loader.py configuration.")
        st.stop()

    # Sidebar → selected player name
    player = render_sidebar(players)

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <h1 style="font-size:2rem; margin-bottom:0; color:#e2e8f0;">
        ⚽ {player}
    </h1>
    <p style="color:#8b949e; margin-top:0.2rem; margin-bottom:0.5rem;">
        La Liga 2020/21 · StatsBomb Open Data · Rule-based Role Classification
    </p>
    <hr style="border-color:#21262d; margin:0.5rem 0 1rem 0;">
    """, unsafe_allow_html=True)

    # ── Compute metrics ───────────────────────────────────────────────────────
    player_events  = get_player_events(events, player)
    minutes        = get_player_minutes(events, player)
    metrics        = compute_metrics(player_events, minutes)
    role, evidence = classify_role(metrics)

    # Summary pill row (raw metric values at a glance)
    st.markdown(f"""
    <div style="margin-bottom:1rem;">
        <span class="pill">⏱ {minutes:.0f} min played</span>
        <span class="pill">📋 {metrics['passes_p90']} passes/90</span>
        <span class="pill">🎯 {metrics['shots_p90']} shots/90</span>
        <span class="pill">📍 avg x = {metrics['avg_x']}/120</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Role card ─────────────────────────────────────────────────────────────
    meta    = ROLE_META.get(role, {"emoji": "⚽", "desc": ""})
    bullets = "".join(f'<div class="bullet">✦ {e}</div>' for e in evidence)

    st.markdown(f"""
    <div class="role-card">
        <div style="display:flex; align-items:center; gap:1rem;">
            <div style="font-size:3rem;">{meta['emoji']}</div>
            <div>
                <p class="role-title">{role}</p>
                <p class="role-sub">{meta['desc']}</p>
            </div>
        </div>
        <div style="margin-top:0.9rem;">
            <p style="color:#6e7681; font-size:0.75rem; margin-bottom:0.2rem;">
                KEY EVIDENCE
            </p>
            {bullets}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 3 Visualisations ──────────────────────────────────────────────────────
    st.markdown('<p class="sec-head">📊 Visualisations</p>', unsafe_allow_html=True)

    # Row 1: Heatmap  |  Pass Map  (side by side)
    col_heat, col_pass = st.columns(2)

    with col_heat:
        st.markdown("**1 · Touch Heatmap** — where this player operates on the pitch")
        with st.spinner("Drawing heatmap…"):
            try:
                fig = plot_heatmap(player_events, player)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as e:
                st.error(f"Heatmap error: {e}")

    with col_pass:
        st.markdown("**2 · Pass Map** — 🟢 complete · 🔴 incomplete")
        with st.spinner("Drawing pass map…"):
            try:
                fig = plot_passmap(player_events, player)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as e:
                st.error(f"Pass map error: {e}")

    # Row 2: Shot Map (full width — defenders have very few shots, half-width looks odd)
    st.markdown("**3 · Shot Map** — bubble size = xG · colour = outcome")
    with st.spinner("Drawing shot map…"):
        try:
            # Centre the shot map — doesn't need to fill the full wide layout
            _, shot_col, _ = st.columns([1, 2, 1])
            with shot_col:
                fig = plot_shotmap(player_events, player)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        except Exception as e:
            st.error(f"Shot map error: {e}")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <hr style="border-color:#21262d; margin-top:2rem;">
    <p style="text-align:center; color:#484f58; font-size:0.78rem;">
        Data: StatsBomb Open Data (free licence) ·
        Built with Streamlit, mplsoccer, pandas, matplotlib
    </p>
    """, unsafe_allow_html=True)


# Streamlit runs this file as a script — the guard below ensures main() always runs
if __name__ == "__main__" or True:
    main()
