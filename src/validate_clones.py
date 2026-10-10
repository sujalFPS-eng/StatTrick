"""Out-of-sample check for the clone engine.

The question a clone has to answer is "who plays like this player?". If the engine is any good, a
player's profile built from EARLIER seasons should find his own LATEST season among its closest
matches - the player is the best possible clone of himself. This script:

  1. builds every player's time-decayed profile from the seasons before a holdout season, and a
     second profile from the holdout season alone,
  2. scores each earlier profile against every holdout profile with the real engine,
  3. reports how often the player's own holdout season ranks 1st / top 5 / top 10, and how much
     his top-10 clone list overlaps between the two periods,

once with minutes shrinkage and once without, so the effect of the change is measured, not assumed.

Usage:
    python src/validate_clones.py                 # holdout = last completed season with full event data
    python src/validate_clones.py --holdout 2024  # holdout = 2024-25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common import DATA_DIR, current_season_start, season_label, today, write_json
from src.similarity_engine import PlayerSimilarityEngine

MIN_PAST_MINUTES = 1500        # same bar as the live clone pool
MIN_HOLDOUT_MINUTES = 900      # ten full matches in the holdout season
TOP_K = 10


def _positions() -> pd.Series:
    """pkey -> pos_clean, from the live table if reachable, else the local CSV export."""
    try:
        from src.database import read_table
        master = read_table("players_master")
    except Exception as exc:                                    # no DB here: fall back to the CSV
        print(f"(players_master not readable: {exc}; using {DATA_DIR / 'master_scouting_db.csv'})")
        master = pd.read_csv(DATA_DIR / "master_scouting_db.csv", low_memory=False)
    master.columns = [c.lower() for c in master.columns]
    return master.dropna(subset=["pkey", "pos_clean"]).drop_duplicates("pkey").set_index("pkey")["pos_clean"]


def default_holdout(ps: pd.DataFrame) -> int:
    """Last completed season with the full event feed. A season without progressive/defensive
    stats would test mostly imputed values (FBref lost its Opta feed after 2024-25)."""
    done = ps[ps["season_start"] < current_season_start()]
    coverage = done.groupby("season_start")["prgp_per90"].apply(lambda s: s.notna().mean())
    full = coverage[coverage >= 0.5]
    return int(full.index.max() if len(full) else coverage.index.max())


def build_pool(ps: pd.DataFrame, holdout: int) -> pd.DataFrame:
    from src.valuation_model import aggregate_players

    past = aggregate_players(ps[ps["season_start"] < holdout])
    future = aggregate_players(ps[ps["season_start"] == holdout])
    past = past[past["min"] >= MIN_PAST_MINUTES]
    future = future[future["min"] >= MIN_HOLDOUT_MINUTES]
    both = sorted(set(past["pkey"]) & set(future["pkey"]))

    pos = _positions()
    both = [k for k in both if k in pos.index]
    rows = []
    for tag, frame in (("past", past), ("future", future)):
        f = frame[frame["pkey"].isin(both)].copy()
        f["pos_clean"] = f["pkey"].map(pos)
        f["period"] = tag
        f["player"] = f["pkey"] + "|" + tag                     # unique names for the engine
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def evaluate(pool: pd.DataFrame, shrink: bool) -> dict:
    eng = PlayerSimilarityEngine(df=pool, shrink=shrink)
    period = eng.df["period"].to_numpy()
    pkey = eng.df["pkey"].to_numpy()
    fut_idx = np.flatnonzero(period == "future")
    past_idx = np.flatnonzero(period == "past")
    fut_row = {pkey[j]: j for j in fut_idx}

    ranks, overlaps = [], []
    for i in past_idx:
        score, _, _ = eng._scored(i)
        own = fut_row[pkey[i]]
        ranks.append(int((score[fut_idx] > score[own]).sum()) + 1)

        # does the player keep the same clones from one period to the next?
        past_others = past_idx[past_idx != i]
        clones_then = set(pkey[past_others[np.argsort(-score[past_others])[:TOP_K]]])
        score_now, _, _ = eng._scored(own)
        fut_others = fut_idx[fut_idx != own]
        clones_now = set(pkey[fut_others[np.argsort(-score_now[fut_others])[:TOP_K]]])
        overlaps.append(len(clones_then & clones_now) / TOP_K)

    ranks = np.array(ranks)
    return {"players": len(ranks), "top1": float((ranks == 1).mean()), "top5": float((ranks <= 5).mean()),
            "top10": float((ranks <= TOP_K).mean()), "median_rank": float(np.median(ranks)),
            "clone_overlap@10": float(np.mean(overlaps))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--holdout", type=int, default=None,
                    help="start year of the holdout season (default: last completed season with event data)")
    args = ap.parse_args()

    from src.valuation_model import build_player_seasons
    ps = build_player_seasons()
    if args.holdout is None:
        args.holdout = default_holdout(ps)
    pool = build_pool(ps, args.holdout)
    n = int((pool["period"] == "past").sum())
    print(f"\nHoldout season {season_label(args.holdout)}: {n:,} players with >= {MIN_PAST_MINUTES:,} earlier "
          f"minutes and >= {MIN_HOLDOUT_MINUTES} holdout minutes\n")
    with_s, without_s = evaluate(pool, shrink=True), evaluate(pool, shrink=False)
    # the Methodology tab in the app reads this file
    write_json(DATA_DIR / "clone_validation.json", {
        "holdout": season_label(args.holdout), "players": n, "run_on": str(today()),
        "with_shrinkage": {k: round(v, 4) for k, v in with_s.items()},
        "without_shrinkage": {k: round(v, 4) for k, v in without_s.items()},
    })
    res = pd.DataFrame({"with shrinkage": with_s, "without shrinkage": without_s}).T
    for c in ("top1", "top5", "top10", "clone_overlap@10"):
        res[c] = (res[c] * 100).map("{:.1f}%".format)
    print(res.to_string())
    print(f"\ntop-k = share of players whose own {season_label(args.holdout)} season is among the k closest "
          f"of {n:,} holdout profiles to his earlier profile (chance for top 10: {TOP_K / max(n, 1):.1%}).")


if __name__ == "__main__":
    main()
