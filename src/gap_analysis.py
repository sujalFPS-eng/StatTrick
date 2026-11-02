"""Gap analysis: find a club's weakest position-unit metrics and the players who fix them.

Fixes vs. the version that lived inside the app:
  - Every unit used to be scored on all ten outfield style features, so a goalkeeping unit
    could be flagged for "Tackles / 90" (and a keeper with 0.00 tackles recommended for it).
    Each unit is now scored only on the metrics that describe its job (GAP_METRICS).
  - Goalkeepers are judged on shot-stopping (PSxG net / 90) against the average
    top-five-league keeper. The tactical archetypes were built from outfield style
    and say nothing about how well a keeper stops shots.
  - Unit profiles are computed from the player table here rather than read from the stored
    club_profiles table, which only holds outfield features and only profiles a unit with
    three or more players on 400+ minutes. That rule left 90 of 97 clubs (in the October
    data) without a goalkeeping profile at all; a goalkeeping unit is usually one keeper.
  - The app returned only the single largest gap. find_gaps returns every positive gap,
    ranked, so the app can show one per unit (or the top few within one unit).
  - Candidates are z-scored on the gap metric against every player in their position
    group, not just against the other candidates, and must be above that group's average.
    Otherwise a thin pool made a below-average player look like a fix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import POS_GROUP

GROUP_TO_POS = {"GK": ["GK"], "DEF": ["CB", "FULLBACK"], "MID": ["CM", "CDM", "CAM"], "ATT": ["ST", "WINGER"]}
GROUP_NAMES = {"GK": "Goalkeeping", "DEF": "Defence", "MID": "Midfield", "ATT": "Attack"}

# Metrics that describe each unit's job. Anything not listed is never reported as a gap.
GAP_METRICS = {
    "GK": ["psxg_net_per90"],      # progressive passes are ~0.04 / 90 for keepers: no signal
    "DEF": ["tkl_per90", "int_per90", "clr_per90", "blk_per90", "aer_won_per90", "prgp_per90", "prgc_per90"],
    "MID": ["prgp_per90", "prgc_per90", "xag_per90", "xg_per90", "tkl_per90", "int_per90"],
    "ATT": ["xg_per90", "xag_per90", "sh_per90", "prgc_per90", "prgp_per90"],
}

# Groups judged against the average player in the group (ideal z = 0), not an archetype.
ARCHETYPE_FREE_GROUPS = {"GK"}

MIN_UNIT_MINUTES = 400
MIN_UNIT_PLAYERS = {"GK": 1, "DEF": 2, "MID": 2, "ATT": 2}
MIN_GAP_SD = 0.0               # a gap is reported only if the unit is below its target


def add_pos_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pos_group"] = out["pos_clean"].astype(str).str.upper().map(POS_GROUP)
    return out


def unit_profiles(players: pd.DataFrame) -> pd.DataFrame:
    """Minutes-weighted per-90 of every club's position units (one row per squad x group)."""
    df = add_pos_group(players)
    df = df[pd.to_numeric(df["min"], errors="coerce") >= MIN_UNIT_MINUTES].dropna(subset=["pos_group"])
    feats = sorted({f for fs in GAP_METRICS.values() for f in fs if f in df.columns})
    records = []
    for (squad, group), g in df.groupby(["squad", "pos_group"]):
        if len(g) < MIN_UNIT_PLAYERS.get(group, 2):
            continue
        mins = pd.to_numeric(g["min"], errors="coerce").fillna(0.0)
        row = {"squad": squad, "pos_group": group, "n_players": len(g)}
        for f in GAP_METRICS.get(group, []):
            if f not in feats:
                continue
            v = pd.to_numeric(g[f], errors="coerce")
            w = mins[v.notna()]
            row[f] = float((v.dropna() * w).sum() / w.sum()) if w.sum() > 0 else np.nan
        records.append(row)
    return pd.DataFrame(records)


def _archetype(club_profiles: pd.DataFrame | None, squad: str, group: str) -> str | None:
    if club_profiles is None or club_profiles.empty or "tactical_archetype" not in club_profiles.columns:
        return None
    cp = club_profiles
    rows = cp[cp["squad"].astype(str).str.lower() == squad.lower()]
    if "pos_group" in cp.columns:
        rows = rows[rows["pos_group"] == group]
    return str(rows["tactical_archetype"].iloc[0]) if not rows.empty else None


def find_gaps(players: pd.DataFrame, club: str, groups: list[str] | None = None,
              club_profiles: pd.DataFrame | None = None,
              archetype_profiles: dict | None = None) -> pd.DataFrame:
    """Every unit metric where `club` falls short of its target, largest first.

    Target = the club's archetype blueprint for that unit (z units vs. the same unit at
    other clubs) when one exists, else 0 (the average unit). Columns: pos_group, metric,
    gap (SD below target), club_z, target_z, club_value, peer_mean, archetype.
    Rows with gap <= MIN_GAP_SD are dropped; the caller decides what to show if empty.
    """
    units = unit_profiles(players)
    cols = ["pos_group", "metric", "gap", "club_z", "target_z", "club_value", "peer_mean", "archetype"]
    if units.empty:
        return pd.DataFrame(columns=cols)
    mine = units[units["squad"].astype(str).str.lower() == club.lower()]
    if groups:
        mine = mine[mine["pos_group"].isin(groups)]
    out = []
    for _, row in mine.iterrows():
        group = row["pos_group"]
        archetype = None if group in ARCHETYPE_FREE_GROUPS else _archetype(club_profiles, club, group)
        ideal = (archetype_profiles or {}).get(archetype, {}) if archetype else {}
        peers = units[units["pos_group"] == group]
        for f in GAP_METRICS.get(group, []):
            if f not in row.index or pd.isna(row[f]):
                continue
            col = pd.to_numeric(peers[f], errors="coerce").dropna()
            if len(col) < 5:
                continue
            sd = col.std()
            z = (row[f] - col.mean()) / (sd if sd and sd > 0 else 1.0)
            target = float(ideal.get(f, 0.0))
            out.append({"pos_group": group, "metric": f, "gap": float(target - z), "club_z": float(z),
                        "target_z": target, "club_value": float(row[f]), "peer_mean": float(col.mean()),
                        "archetype": archetype})
    res = pd.DataFrame(out, columns=cols)
    return res.sort_values("gap", ascending=False).reset_index(drop=True)


def pick_gaps(gaps: pd.DataFrame, per_group: int, limit: int) -> pd.DataFrame:
    """Top `per_group` positive gaps in each unit, then the `limit` largest overall."""
    pos = gaps[gaps["gap"] > MIN_GAP_SD]
    if pos.empty:
        return pos
    return (pos.sort_values("gap", ascending=False).groupby("pos_group", sort=False).head(per_group)
               .head(limit).reset_index(drop=True))


def rank_candidates(players: pd.DataFrame, metric: str, group: str, exclude_club: str,
                    max_age: float = 28, min_minutes: float = 900, only_below_band: bool = False,
                    recruitable: set | None = None, n: int = 5, gk_age_bonus: float = 4) -> pd.DataFrame:
    """Players in `group` who are above their group's average at `metric` and priced below
    their performance value, ranked by metric strength (x1.5) plus value edge.
    Keepers peak later, so they get `gk_age_bonus` extra years on `max_age`."""
    if group == "GK":
        max_age = max_age + gk_age_bonus
    df = add_pos_group(players)
    group_pool = df[(df["pos_group"] == group) & (pd.to_numeric(df["min"], errors="coerce") >= min_minutes)]
    vals = pd.to_numeric(group_pool.get(metric), errors="coerce") if metric in group_pool.columns else None
    if vals is None or vals.dropna().empty:
        return group_pool.iloc[0:0]
    mean, sd = vals.mean(), vals.std()
    sd = sd if sd and sd > 0 else 1.0

    c = group_pool.copy()
    c["metric_z"] = ((pd.to_numeric(c[metric], errors="coerce") - mean) / sd).clip(-2.5, 2.5)
    c = c[(c["squad"].astype(str).str.lower() != exclude_club.lower())
          & (pd.to_numeric(c["age_clean"], errors="coerce") <= max_age)
          & (pd.to_numeric(c["surplus_pct"], errors="coerce") > 0)
          & (c["metric_z"] > 0)]
    if "match_method" in c.columns:
        c = c[c["match_method"] == "exact"]
    if recruitable is not None:
        c = c[c["player"].isin(recruitable)]
    if only_below_band and "outside_band" in c.columns:
        c = c[c["outside_band"] == 1]
    if c.empty:
        return c
    value_col = "edge_z" if "edge_z" in c.columns else "surplus_pct"
    v = pd.to_numeric(c[value_col], errors="coerce")
    v_sd = v.std() if v.std() and v.std() > 0 else 1.0
    c["value_z"] = ((v - v.mean()) / v_sd).clip(-2.5, 2.5).fillna(0.0)
    c["gap_score"] = 1.5 * c["metric_z"] + c["value_z"]
    return c.sort_values("gap_score", ascending=False).head(n)
