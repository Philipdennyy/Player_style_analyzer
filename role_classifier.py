"""
role_classifier.py
==================
Simple, fully rule-based player role classification.

This module computes exactly 3 metrics from a player's event data and
applies fixed thresholds to assign one of 3 roles. Every decision can be
explained in plain English — no percentile tables, no machine learning,
no black boxes.

─────────────────────────────────────────────────────────────────────────
THE 3 METRICS
─────────────────────────────────────────────────────────────────────────
  passes_p90   Completed passes per 90 minutes.
               A high value means the player is heavily involved in
               build-up play and ball circulation.

  shots_p90    Shots per 90 minutes.
               A high value signals a finishing threat / attacking role.

  avg_x        Mean x-coordinate across all touches (StatsBomb scale 0–120,
               where 0 = own goal line, 120 = opponent goal line).
               High avg_x → player operates in advanced areas.
               Low avg_x  → player sits deep (defensive role).

─────────────────────────────────────────────────────────────────────────
THE 3 ROLES & FIXED THRESHOLDS
─────────────────────────────────────────────────────────────────────────
  Playmaker          passes_p90 ≥ 45  AND  shots_p90 < 3
                     High ball circulation, low shooting → creative midfielder.

  Attacker/Finisher  shots_p90 ≥ 3   AND  avg_x ≥ 75
                     High shot volume, advanced position → striker / 10.

  Deep/Defensive     everything else
                     Low shots, deeper average position → holding mid / defender.

THRESHOLD RATIONALE (easy to explain to an interviewer)
  45 passes/90   ≈ 1 completed pass every 2 minutes — genuinely high involvement.
  3 shots/90     ≈ 1 shot per 30 min — the rough threshold between
                 "contributes shots" and "rarely shoots".
  avg_x ≥ 75     ≈ the final third of the pitch (60–120 is the attacking half;
  avg_x ≥ 72     ≈ the final third of the pitch (60–120 is the attacking half;
                 72 = 60% up the pitch, clearly advanced).
"""

# ── Thresholds ─────────────────────────────────────────────────────────────────
HIGH_PASSES_P90 = 35.0   # passes per 90 that qualify as "high volume"
HIGH_SHOTS_P90  = 2.0    # shots per 90 that qualify as "high frequency"
ADV_AVG_X       = 72.0    # mean x-position (out of 120) that qualifies as "advanced"
# 72/120 = 60% up the pitch — clearly in the attacking half (halfway line is at 60)


def compute_metrics(player_events, minutes: float) -> dict:
    """
    Compute the 3 classification metrics for a single player.

    Parameters
    ----------
    player_events : pd.DataFrame
        This player's events, with 'x' column already parsed from location.
    minutes : float
        Estimated minutes played (from data_loader.get_player_minutes).

    Returns
    -------
    dict  { 'passes_p90': float, 'shots_p90': float, 'avg_x': float }
    """
    # per90 = the multiplier that converts a raw count into a per-90 rate.
    # e.g. if a player played 900 minutes and made 400 passes:
    #   per90 = 90/900 = 0.1  →  passes_p90 = 400 * 0.1 = 40
    per90 = 90.0 / max(minutes, 1.0)

    # ── Metric 1: completed passes per 90 ────────────────────────────────────
    passes = player_events[player_events["type"] == "Pass"]
    if "pass_outcome" in passes.columns:
        # StatsBomb: NaN in 'pass_outcome' means the pass was completed
        completed = passes[passes["pass_outcome"].isna()]
    else:
        completed = passes   # column absent → assume all complete

    passes_p90 = round(len(completed) * per90, 1)

    # ── Metric 2: shots per 90 ────────────────────────────────────────────────
    shots     = player_events[player_events["type"] == "Shot"]
    shots_p90 = round(len(shots) * per90, 2)

    # ── Metric 3: average x-position ─────────────────────────────────────────
    touches = player_events.dropna(subset=["x"])
    avg_x   = round(float(touches["x"].mean()), 1) if not touches.empty else 60.0

    return {
        "passes_p90": passes_p90,
        "shots_p90":  shots_p90,
        "avg_x":      avg_x,
    }


def classify_role(metrics: dict) -> tuple:
    """
    Apply fixed threshold rules to assign one of 3 roles.

    Rules are checked in priority order:
      1. Playmaker         — if high passes AND low shots
      2. Attacker/Finisher — if high shots AND advanced position
      3. Deep/Defensive    — default (catches everyone else)

    Parameters
    ----------
    metrics : dict  returned by compute_metrics()

    Returns
    -------
    role     : str        one of the 3 role names
    evidence : list[str]  exactly 2 plain-English bullet strings
    """
    p = metrics["passes_p90"]
    s = metrics["shots_p90"]
    x = metrics["avg_x"]

    # ── Rule 1: Playmaker ─────────────────────────────────────────────────────
    if p >= HIGH_PASSES_P90 and s <= HIGH_SHOTS_P90 and 60 <= x < ADV_AVG_X:
        role = "Playmaker"
        evidence = [
            f"{p:.0f} completed passes per 90 — high ball-circulation volume",
            f"Average touch position x={x:.0f}/120 — mid-to-deep on the pitch",
        ]

    # ── Rule 2: Attacker / Finisher ───────────────────────────────────────────
    elif s >= HIGH_SHOTS_P90 and x >= ADV_AVG_X:
        role = "Attacker/Finisher"
        evidence = [
            f"{s:.1f} shots per 90 — frequent finishing attempts",
            f"Average touch position x={x:.0f}/120 — operates in the final third",
        ]

    # ── Rule 3: Deep / Defensive (default) ───────────────────────────────────
    elif x < 60:
        role = "Deep/Defensive"
        evidence = [
            f"{s:.1f} shots per 90 — low attacking output",
            f"Average touch position x={x:.0f}/120 — deep or defensive area",
        ]
    else:
          role = "Playmaker"
          evidence = [
            f"{p:.0f} completed passes per 90 — high ball-circulation volume",
            f"Average touch position x={x:.0f}/120 — mid-to-deep on the pitch",
        ]

    return role, evidence
