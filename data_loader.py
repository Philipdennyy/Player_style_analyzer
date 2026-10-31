"""
data_loader.py
==============
Fetches La Liga 2020/21 event data from StatsBomb's free open-data repository,
caches it locally so the app never re-downloads, and provides simple helpers
to filter and parse events for a single player.

StatsBomb pitch coordinate system (used throughout this project):
  x: 0 (own goal line)  →  120 (opponent goal line)
  y: 0 (left touchline) →   80 (right touchline)
"""

import os
import pickle

import pandas as pd
from statsbombpy import sb

# ── Cache location ─────────────────────────────────────────────────────────────
# All files live inside a 'cache/' subfolder next to this script.
CACHE_DIR  = os.path.join(os.path.dirname(__file__), "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "events.pkl")

# ── One competition, one season ────────────────────────────────────────────────
# La Liga 2020/21 — StatsBomb open data covers all Barcelona matches
# for this season (38 games), giving Barcelona players a full season
# of data. Well-known players: Messi, Busquets, De Jong, Jordi Alba,
# Pedri, Griezmann, Dest, Ter Stegen, Lenglet.
COMPETITION_ID = 11   # La Liga
SEASON_ID      = 90   # 2020/21

# Minimum minutes a player must have played to appear in the dropdown.
# 1000 minutes ≈ 11 full matches — enough data for reliable per-90 metrics.
# Players below this threshold (fringe squad / brief sub appearances) are hidden.
MIN_MINUTES = 1000


def load_events() -> pd.DataFrame:
    """
    Load all match events for La Liga 2020/21.

    First call  → downloads from StatsBomb (~2 minutes), saves to cache/events.pkl.
    All later calls → loads the pickle in seconds, no network needed.

    Returns
    -------
    pd.DataFrame  (one row per event, all matches in the season combined)
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # If we've already cached the data, return it immediately
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    # First run: fetch the list of matches, then every match's events
    print("Downloading La Liga 2020/21 from StatsBomb (one-time, ~2 minutes)…")
    matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)

    frames = []
    for match_id in matches["match_id"]:
        events = sb.events(match_id=match_id)
        events["match_id"] = match_id   # keep the match ID so we can group by it later
        frames.append(events)

    all_events = pd.concat(frames, ignore_index=True)

    # Persist locally — next run skips all of the above
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(all_events, f)

    print(f"Saved {len(all_events):,} events → {CACHE_FILE}")
    return all_events


def get_player_list(events: pd.DataFrame) -> list:
    """
    Return a sorted list of players who have played at least MIN_MINUTES
    across the season.

    Method:
      1. For each (player, match) pair, take the maximum minute seen.
         This estimates how long the player was on the pitch in that match.
      2. Sum those per-match maxima across all matches → total season minutes.
      3. Keep only players at or above MIN_MINUTES.

    This is the same logic used in get_player_minutes(), applied to every
    player at once using groupby (no slow per-player loop).
    """
    if "player" not in events.columns or "minute" not in events.columns:
        return []

    # Step 1+2: max minute per (player, match), then sum across matches
    player_minutes = (
        events.dropna(subset=["player"])
        .groupby(["player", "match_id"])["minute"]
        .max()
        .groupby(level="player")
        .sum()
    )

    # Step 3: keep players at or above the threshold
    qualified = player_minutes[player_minutes >= MIN_MINUTES].index.tolist()
    return sorted(qualified)


def get_player_events(events: pd.DataFrame, player_name: str) -> pd.DataFrame:
    """
    Filter the full events table to a single player and expand the location
    columns from lists into plain float columns.

    StatsBomb stores locations as Python lists: [x, y].
    We split them into 'x', 'y', 'end_x', 'end_y' so the viz functions can
    use them directly without any extra parsing.

    Parameters
    ----------
    events      : full events DataFrame (all players)
    player_name : exact name string as it appears in the 'player' column

    Returns
    -------
    pd.DataFrame  (this player's events only, with x/y columns added)
    """
    df = events[events["player"] == player_name].copy()

    # Expand start location → x, y
    if "location" in df.columns:
        df["x"] = df["location"].apply(
            lambda v: float(v[0]) if isinstance(v, list) and len(v) >= 2 else None
        )
        df["y"] = df["location"].apply(
            lambda v: float(v[1]) if isinstance(v, list) and len(v) >= 2 else None
        )

    # Expand pass end location → end_x, end_y  (needed for pass map arrows)
    if "pass_end_location" in df.columns:
        df["end_x"] = df["pass_end_location"].apply(
            lambda v: float(v[0]) if isinstance(v, list) and len(v) >= 2 else None
        )
        df["end_y"] = df["pass_end_location"].apply(
            lambda v: float(v[1]) if isinstance(v, list) and len(v) >= 2 else None
        )

    return df


def get_player_minutes(events: pd.DataFrame, player_name: str) -> float:
    """
    Estimate total minutes played by this player across the season.

    Method: take the maximum 'minute' value seen for the player in each match,
    then sum those maxima across all their matches.
    This is a simple, explainable approximation — good enough for per-90 stats.

    Returns at least 1.0 to prevent division-by-zero in the classifier.
    """
    player_events = events[events["player"] == player_name]

    if player_events.empty or "minute" not in player_events.columns:
        return 90.0   # safe fallback

    # max minute per match → total across season
    minutes = player_events.groupby("match_id")["minute"].max().sum()
    return max(float(minutes), 1.0)
