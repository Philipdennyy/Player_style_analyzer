"""
visuals.py
==========
All three pitch visualisations in a single file.

Each function:
  - Takes the player's event DataFrame (pre-filtered and with x/y parsed)
    and the player's name string.
  - Returns a matplotlib Figure ready to hand to st.pyplot() in app.py.

Design: consistent dark pitch (#0d1117 background, blue lines) across all
three charts — gives the portfolio app a cohesive, professional look.

INTERVIEW TIP: each function is self-contained and can be explained in
30 seconds. The only external library is mplsoccer, which handles the
pitch drawing — the rest is standard matplotlib.
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend required inside Streamlit
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mplsoccer import Pitch, VerticalPitch

# ── Shared colour palette (dark theme) ────────────────────────────────────────
PITCH_BG   = "#0d1117"   # near-black background
LINE_COLOR = "#58a6ff"   # steel-blue pitch lines


# ─────────────────────────────────────────────────────────────────────────────
# 1. HEATMAP  — touch density across the pitch
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(player_events, player_name: str):
    """
    Draw a kernel-density-estimation (KDE) heatmap of where the player
    was on the ball most often during the season.

    HOW IT WORKS
    - 'touches' = every event with a valid x, y location.
    - mplsoccer's kdeplot() fits a 2-D Gaussian kernel to those points,
      producing a smooth density surface instead of cluttered dots.
    - The 'hot' colormap makes high-density zones glow orange/white against
      the dark background — immediately readable at a glance.

    WHAT IT REVEALS
    - A midfielder shows density in the centre circle.
    - A striker shows density near the penalty area.
    - A winger shows density along the flanks.
    """
    touches = player_events.dropna(subset=["x", "y"])

    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color=PITCH_BG,
        line_color=LINE_COLOR,
        linewidth=1.5,
    )
    fig, ax = pitch.draw(figsize=(10, 7))
    fig.patch.set_facecolor(PITCH_BG)

    if len(touches) < 10:
        # Guard: not enough points for a meaningful KDE
        ax.text(60, 40, "Not enough touch data",
                color="white", ha="center", va="center", fontsize=14)
    else:
        pitch.kdeplot(
            touches["x"], touches["y"],
            ax=ax,
            cmap="hot",       # warm colours for high density
            fill=True,        # shade under the density curve
            alpha=0.75,       # slight transparency so pitch lines show through
            levels=30,        # number of contour bands
            bw_adjust=0.85,   # smoothing bandwidth (lower = sharper peaks)
            zorder=2,
        )

    ax.set_title(
        f"{player_name} — Touch Heatmap",
        color="white", fontsize=15, fontweight="bold", pad=12,
    )
    ax.annotate(
        f"n = {len(touches):,} touches  ·  StatsBomb La Liga 2020/21",
        xy=(0.5, -0.02), xycoords="axes fraction",
        ha="center", fontsize=9, color="#8b949e",
    )
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. PASS MAP  — arrows coloured by complete / incomplete
# ─────────────────────────────────────────────────────────────────────────────

def plot_passmap(player_events, player_name: str):
    """
    Draw every pass attempt as an arrow from start → end location.

    Colour encodes outcome only (no progressive-pass logic):
      Green arrow  = pass completed (ball reached the intended target)
      Red arrow    = pass incomplete (intercepted, missed, out of play)

    STATSBOMB CONVENTION
    In StatsBomb data, a completed pass has NaN in the 'pass_outcome' column.
    Only failed passes are labelled (e.g. 'Incomplete', 'Out', 'Offside').

    We cap display at 600 arrows (sampled randomly) so the chart stays
    readable for high-volume passers like Busquets.
    """
    passes = player_events[player_events["type"] == "Pass"].copy()
    passes = passes.dropna(subset=["x", "y", "end_x", "end_y"])

    # Identify which passes were completed (NaN outcome = complete)
    if "pass_outcome" in passes.columns:
        passes["complete"] = passes["pass_outcome"].isna()
    else:
        passes["complete"] = True   # safe fallback if column is absent

    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color=PITCH_BG,
        line_color=LINE_COLOR,
        linewidth=1.5,
    )
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.patch.set_facecolor(PITCH_BG)

    if passes.empty:
        ax.text(60, 40, "No pass data",
                color="white", ha="center", va="center", fontsize=14)
        plt.tight_layout()
        return fig

    # Sample for readability (keep all if under the cap)
    display = passes.sample(n=min(600, len(passes)), random_state=42)
    complete   = display[display["complete"]]
    incomplete = display[~display["complete"]]

    # Green arrows — completed passes
    if not complete.empty:
        pitch.arrows(
            complete["x"], complete["y"],
            complete["end_x"], complete["end_y"],
            ax=ax,
            color="#2ecc71",   # green
            width=1.0,
            headwidth=3, headlength=2,
            alpha=0.40,
            zorder=2,
        )

    # Red arrows — incomplete passes
    if not incomplete.empty:
        pitch.arrows(
            incomplete["x"], incomplete["y"],
            incomplete["end_x"], incomplete["end_y"],
            ax=ax,
            color="#e74c3c",   # red
            width=0.8,
            headwidth=2.5, headlength=2,
            alpha=0.65,
            zorder=3,
        )

    # Stats for subtitle
    comp_rate = passes["complete"].mean() * 100

    ax.set_title(
        f"{player_name} — Pass Map",
        color="white", fontsize=15, fontweight="bold", pad=12,
    )
    ax.annotate(
        f"Total: {len(passes):,} passes  ·  Completion: {comp_rate:.0f}%  "
        f"({'showing sample of 600' if len(passes) > 600 else 'all shown'})",
        xy=(0.5, 1.02), xycoords="axes fraction",
        ha="center", fontsize=9, color="#e2e8f0",
    )

    # Legend
    legend_items = [
        Line2D([0], [0], color="#2ecc71", linewidth=2, label="Complete"),
        Line2D([0], [0], color="#e74c3c", linewidth=2, label="Incomplete"),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper left",
        facecolor="#161b22", edgecolor="#30363d",
        labelcolor="white", fontsize=9,
    )
    ax.annotate("StatsBomb La Liga 2020/21",
        xy=(0.5, -0.02), xycoords="axes fraction",
        ha="center", fontsize=9, color="#8b949e",
    )
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. SHOT MAP  — locations sized by xG, coloured by outcome
# ─────────────────────────────────────────────────────────────────────────────

def plot_shotmap(player_events, player_name: str):
    """
    Draw all shot attempts on a vertical half-pitch.

    Bubble SIZE   proportional to xG (expected goals from StatsBomb's model).
                  Bigger bubble = the model rated it a better scoring chance.
    Bubble COLOUR encodes outcome:
                  Gold    = Goal
                  Blue    = Saved
                  Orange  = Blocked
                  Grey    = Off target / wayward

    WHAT IT REVEALS
    - A pure finisher (Benzema) shows many large bubbles in the box.
    - A deep midfielder (Busquets) shows almost no bubbles at all.
    - Shot quality vs. quantity: lots of small bubbles = many speculative efforts.
    """
    shots = player_events[player_events["type"] == "Shot"].copy()
    shots = shots.dropna(subset=["x", "y"])

    # Find the xG column — StatsBomb names it 'shot_statsbomb_xg'
    xg_col = next(
        (c for c in shots.columns if "statsbomb_xg" in c.lower() or c == "shot_statsbomb_xg"),
        None,
    )
    shots["xg"] = (
        shots[xg_col].fillna(0.05).clip(0.01, 1.0) if xg_col else 0.05
    )

    # Map outcome string → fill colour
    def _color(outcome):
        if not isinstance(outcome, str):
            return "#868e96"
        if "Goal"    in outcome: return "#FFD700"   # gold
        if "Saved"   in outcome: return "#4dabf7"   # blue
        if "Blocked" in outcome: return "#ff922b"   # orange
        return "#868e96"                             # grey (off target / wayward)

    outcome_col = next(
        (c for c in shots.columns if "shot_outcome" in c.lower()), None
    )
    shots["color"] = (
        shots[outcome_col].astype(str).apply(_color) if outcome_col else "#868e96"
    )

    # Vertical half-pitch — shows only the attacking 60 metres
    pitch = VerticalPitch(
        pitch_type="statsbomb",
        half=True,
        pitch_color=PITCH_BG,
        line_color=LINE_COLOR,
        linewidth=1.5,
        pad_top=5,
    )
    fig, ax = pitch.draw(figsize=(8, 8))
    fig.patch.set_facecolor(PITCH_BG)

    if shots.empty:
        ax.text(40, 95, "No shot data",
                color="white", ha="center", va="center", fontsize=14)
    else:
        # pitch.scatter() handles StatsBomb → VerticalPitch coordinate rotation
        pitch.scatter(
            shots["x"], shots["y"],
            s=shots["xg"] * 2000 + 60,   # scale: xG=0.05→ ~160, xG=0.50→ ~1060
            c=shots["color"],
            edgecolors="#ffffff",
            linewidths=0.6,
            alpha=0.85,
            zorder=4,
            ax=ax,
        )

    # Summary stats
    total_xg = shots["xg"].sum() if not shots.empty else 0
    goals     = (shots["color"] == "#FFD700").sum() if not shots.empty else 0

    ax.set_title(
        f"{player_name} — Shot Map",
        color="white", fontsize=15, fontweight="bold", pad=12,
    )
    ax.annotate(
        f"Shots: {len(shots)}  ·  Total xG: {total_xg:.2f}  ·  Goals: {goals}",
        xy=(0.5, 1.02), xycoords="axes fraction",
        ha="center", fontsize=10, color="#e2e8f0",
    )

    # Legend — outcome colours
    legend_items = [
        Line2D([0],[0], marker="o", color="none", markerfacecolor="#FFD700",
               markeredgecolor="white", markersize=10, label="Goal"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor="#4dabf7",
               markeredgecolor="white", markersize=10, label="Saved"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor="#ff922b",
               markeredgecolor="white", markersize=10, label="Blocked"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor="#868e96",
               markeredgecolor="white", markersize=10, label="Off Target"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower left",
        facecolor="#161b22", edgecolor="#30363d",
        labelcolor="white", fontsize=8,
    )
    ax.annotate(
        "Bubble size = xG  ·  StatsBomb La Liga 2020/21",
        xy=(0.5, -0.02), xycoords="axes fraction",
        ha="center", fontsize=9, color="#8b949e",
    )
    plt.tight_layout()
    return fig
