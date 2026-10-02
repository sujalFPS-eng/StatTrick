"""StatTrick valuation pipeline (patched for PostgreSQL).

What changed vs. the original:
  #1  every player-season is collapsed to ONE row (key = accent-free name + birth year +
      season + club); richer sources fill gaps in poorer ones instead of being added twice.
  #2  explicit per-source column maps (no more global alias dict that missed
      'expected goals', wrong 'blocks', 'won', 'psxg+/-' ...); cleaned_* files now
      actually contribute minutes (their 'Avg Mins per Match' column is total minutes).
  #3  missing data stays NaN (XGBoost handles NaN natively); a has_advanced flag is exported.
  #4  Transfermarkt match is blocked on birth year, prefers active players, is accent-proof,
      and every unmatched >=1500-minute player is written to data/unmatched_players.csv.
  #5  players who have not played in the last two seasons are excluded (not recruitable).
  #6  predictions are OUT-OF-FOLD for every player (no in-sample flattery); metrics are honest.
  #7  surplus is also exported in % / log terms (surplus_pct, surplus_log).
  #8  no hard-coded years, contract years from today, age from date of birth,
      league labels canonicalised (5 dummies, not 10).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor

from src.common import (
    DATA_DIR, canon_club, canon_league, canon_season, current_season_start, make_pkey,
    name_key, season_start, season_weight, today, write_json,
)
from src.common import infer_live_season_start, read_json, season_label
from src.database import replace_tables
from src.market_forecast import build_market_forecast

MIN_CAREER_MINUTES = 1500      # deduplicated minutes across all seasons
MIN_MARKET_VALUE_M = 0.25      # was 0.5 - less truncation of the target
RECENT_SEASONS = 2             # must have played in one of the last N seasons
TM_ACTIVE_LOOKBACK = 2         # TM 'last_season' >= current_start - 2
CONFORMAL_TIERS = 5            # predicted-value tiers, each calibrated to 70% coverage
FUZZY_CUTOFF = 88              # after accent stripping + birth-year blocking this is safe

RAW_STATS = ["gls", "ast", "xg", "xag", "sh", "sot", "prgc", "prgp",
             "tkl", "int", "clr", "blk", "aer_won", "saves", "psxg_net"]
PER90_COLS = [f"{s}_per90" for s in RAW_STATS]
ADVANCED = ["xg", "xag", "prgc", "prgp"]
ID_COLS = ["player", "born", "pos", "squad", "league", "age", "min"]

# ---------------------------------------------------------------------------- sources
# Column maps are applied AFTER lower-casing + stripping. One explicit map per source.
CLEANED_MAP = {
    "avg mins per match": "min",        # despite the name this column holds TOTAL minutes
    "goals": "gls", "assists": "ast", "expected goals": "xg",
    "progressive carries": "prgc", "progressive passes": "prgp",
    "total shots": "sh", "tackles attempted": "tkl", "interceptions": "int",
    "clearances": "clr", "saves": "saves",
}
MAP_2425 = {                              # plain 'blocks' (passing table) is dropped first
    "blocks_stats_defense": "blk", "won": "aer_won", "psxg+/-": "psxg_net",
}

def _yyzz_season(stem: str) -> str | None:
    """'fbref_2425' / 'fbref_live_2627' -> '2024-2025' / '2026-2027'."""
    digits = stem.rsplit("_", 1)[-1]
    if len(digits) == 4 and digits.isdigit() and (int(digits[:2]) + 1) % 100 == int(digits[2:]):
        return f"20{digits[:2]}-20{digits[2:]}"
    return None


def discover_sources() -> list:
    """Sources are found on disk, not hard-coded by year, so a new season needs no code change:
      fbref_YYZZ.csv       full-season exports (rank 0, richest)
      cleaned_*.csv        older cleaned seasons (rank 1)
      fbref_live_YYZZ.csv  weekly snapshots written by ingest_fbref.py, one per season (rank 1);
                           at season rollover the finished season's last snapshot stays put
      live_fbref_data.csv  the latest scrape (rank 1); season inferred from the table itself
      fbref_historical_stats.csv  basic multi-season backfill (rank 2)"""
    # tuple = (path, tag, rank (lower wins), fixed season (None = read from file), column map, drop)
    out = []
    for p in sorted(DATA_DIR.glob("fbref_[0-9][0-9][0-9][0-9].csv")):
        season = _yyzz_season(p.stem)
        if season:
            out.append((p.name, p.stem, 0, season, MAP_2425, ["blocks"]))
    for p in sorted(DATA_DIR.glob("cleaned_*.csv")):
        out.append((p.name, "cleaned", 1, None, CLEANED_MAP, []))
    for p in sorted(DATA_DIR.glob("fbref_live_[0-9][0-9][0-9][0-9].csv")):
        season = _yyzz_season(p.stem)
        if season:
            out.append((p.name, "live_snapshot", 1, season, {}, []))
    out.append(("live_fbref_data.csv", "live", 1, "LIVE", {}, []))
    out.append(("fbref_historical_stats.csv", "historical", 2, None, {}, []))
    return out


def _read(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, low_memory=False)
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df.loc[:, ~df.columns.duplicated()].copy()      # keep first of any duplicate header
    except pd.errors.EmptyDataError:
        return None


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")


def load_source(fname, tag, rank, season, colmap, drop) -> pd.DataFrame | None:
    path = DATA_DIR / fname
    if not path.exists():
        print(f"  ! missing {fname} (skipped)")
        return None
    
    df = _read(path)
    if df is None:
        print(f"  ! empty file {fname} (skipped)")
        return None
        
    df = df.drop(columns=[c for c in drop if c in df.columns])
    df = df.rename(columns=colmap)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    if "comp" in df.columns and "league" not in df.columns:
        df = df.rename(columns={"comp": "league"})

    # derived stats for the cleaned_* files
    if tag == "cleaned":
        if {"shots blocked", "passes blocked"} <= set(df.columns):
            df["blk"] = _num(df["shots blocked"]) + _num(df["passes blocked"])
        # 'total shots' has already been renamed to 'sh' by CLEANED_MAP above
        if {"sh", "% shots on target"} <= set(df.columns):
            df["sot"] = (_num(df["sh"]) * _num(df["% shots on target"]) / 100).round()

    if season == "LIVE":
        # label from the table itself, not just the calendar (see infer_live_season_start)
        df["season"] = season_label(infer_live_season_start(_num(df["min"]).max() if "min" in df.columns else None))
    elif season:
        df["season"] = season
    df["src"], df["rank"] = tag, rank

    for c in ID_COLS + RAW_STATS:
        if c not in df.columns:
            df[c] = np.nan
    keep = ID_COLS + RAW_STATS + ["season", "src", "rank"]
    out = df[keep].copy()
    print(f"  + {fname:32s} {len(out):6d} rows  [{tag}]")
    return out


def attach_understat(df: pd.DataFrame) -> pd.DataFrame:
    """Fill xG / xA from Understat wherever FBref has none (2025-26 onward, after FBref lost its
    Opta feed; also xA for the older 'cleaned' seasons if those Understat files are present).
    FBref's own values are never overwritten.

    Matching, per season: exact accent-free name first; otherwise a name that contains / is
    contained in the other, at the SAME club, with minutes that agree. Understat reports a player
    who moved mid-season once, so its per-90 rate is applied to each FBref stint's minutes."""
    files = sorted(DATA_DIR.glob("understat_[0-9][0-9][0-9][0-9].csv"))
    df["xg_src"] = np.where(df["xg"].notna(), "fbref", None)
    if not files:
        print("  (no understat_YYZZ.csv files - xG stays as published by FBref)")
        return df
    filled = 0
    for path in files:
        season = _yyzz_season(path.stem)
        u = pd.read_csv(path)
        if season is None or not {"player_name", "team_title", "time", "xG", "xA"} <= set(u.columns):
            print(f"  ! {path.name}: unexpected layout, skipped"); continue
        u = u[pd.to_numeric(u["time"], errors="coerce") > 0].reset_index(drop=True)
        u["nk"] = u["player_name"].map(name_key)
        u["teams"] = u["team_title"].astype(str).map(lambda t: {canon_club(x.strip()) for x in t.split(",")})
        by_name = u.groupby("nk").indices
        used: set = set()
        idx = df.index[df["season"] == season]
        groups = df.loc[idx].groupby("pkey", sort=False)
        pending = []
        for pkey, g in groups:                                   # pass 1: exact names
            nk, squads, mins = name_key(g["player"].iloc[0]), set(g["squad"]), g["min"].sum()
            cand = [i for i in by_name.get(nk, []) if i not in used]
            if len(cand) > 1:                                    # namesakes: same club, then closest minutes
                same = [i for i in cand if u.at[i, "teams"] & squads]
                cand = sorted(same or cand, key=lambda i: abs(u.at[i, "time"] - mins))[:1]
            if cand:
                used.add(cand[0]); pending.append((g.index, cand[0]))
            else:
                pending.append((g.index, None))
        for k, (rows, hit) in enumerate(pending):                # pass 2: partial names at the same club
            if hit is not None:
                continue
            g = df.loc[rows]
            nk, squads, mins = name_key(g["player"].iloc[0]), set(g["squad"]), g["min"].sum()
            best, best_score = None, 0
            for i in u.index[u["teams"].map(lambda t: bool(t & squads))]:
                if i in used or abs(u.at[i, "time"] - mins) > max(120, 0.2 * mins):
                    continue
                score = fuzz.token_set_ratio(nk, u.at[i, "nk"])
                if score >= 85 and score > best_score:
                    best, best_score = i, score
            if best is not None:
                used.add(best); pending[k] = (rows, best)
        matched = 0
        for rows, hit in pending:
            if hit is None:
                continue
            matched += 1
            rate_xg, rate_xa = u.at[hit, "xG"] / u.at[hit, "time"], u.at[hit, "xA"] / u.at[hit, "time"]
            need_xg, need_xa = df.loc[rows, "xg"].isna(), df.loc[rows, "xag"].isna()
            df.loc[rows[need_xg], "xg"] = rate_xg * df.loc[rows[need_xg], "min"]
            df.loc[rows[need_xg], "xg_src"] = "understat"
            df.loc[rows[need_xa], "xag"] = rate_xa * df.loc[rows[need_xa], "min"]
            filled += int(need_xg.sum())
        n = len(pending)
        mins_all = df.loc[idx, "min"].sum()
        mins_hit = sum(df.loc[rows, "min"].sum() for rows, hit in pending if hit is not None)
        print(f"  + {path.name:32s} matched {matched:,}/{n:,} players ({mins_hit / max(mins_all, 1):.0%} of minutes) [{season}]")
    print(f"Understat filled xG for {filled:,} player-season rows")
    return df


def build_player_seasons() -> pd.DataFrame:
    print("Loading sources...")
    frames = [f for f in (load_source(*s) for s in discover_sources()) if f is not None]
    df = pd.concat(frames, ignore_index=True)

    df["season"] = df["season"].map(canon_season)
    df["league"] = df["league"].map(canon_league)
    df["squad"] = df["squad"].map(canon_club)
    df["min"] = _num(df["min"])
    df["born"] = pd.to_numeric(df["born"], errors="coerce")
    df["age"] = df["age"].astype(str).str.split("-").str[0]
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["pos"] = df["pos"].astype(str).replace("nan", "")
    for c in RAW_STATS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["player"].notna() & (df["player"] != "Player") & df["season"].notna()]
    df = df[df["min"] > 0].copy()
    df["pkey"] = make_pkey(df["player"], df["born"])

    before = len(df)
    df = (df.sort_values("rank", kind="stable")
            .groupby(["pkey", "season", "squad"], as_index=False, sort=False).first())
    print(f"Collapsed {before:,} source rows -> {len(df):,} player-season rows "
          f"({before - len(df):,} duplicates removed)")

    # sanity: same player+season at two clubs with identical minutes = probably a spelling variant
    twins = df.groupby(["pkey", "season"]).filter(lambda g: len(g) > 1 and g["min"].nunique() < len(g))
    print(f"Possible residual club-spelling duplicates: {len(twins)} rows")

    df = attach_understat(df)

    m90 = df["min"] / 90.0
    for s in RAW_STATS:
        df[f"{s}_per90"] = df[s] / m90            # NaN stays NaN
    df["season_start"] = df["season"].map(season_start)
    return df


# ---------------------------------------------------------------------------- aggregate
def aggregate_players(ps: pd.DataFrame) -> pd.DataFrame:
    ps = ps.copy()
    ps["w90"] = (ps["min"] / 90.0) * ps["season"].map(season_weight)

    agg = {"min": ("min", "sum"), "n_seasons": ("season", "nunique"),
           "last_season_start": ("season_start", "max")}
    for c in PER90_COLS:
        valid = ps[c].notna()
        ps[f"{c}__num"] = np.where(valid, ps[c] * ps["w90"], 0.0)
        ps[f"{c}__den"] = np.where(valid, ps["w90"], 0.0)
        agg[f"{c}__num"] = (f"{c}__num", "sum")
        agg[f"{c}__den"] = (f"{c}__den", "sum")
    ps["_adv"] = ps[[f"{a}_per90" for a in ADVANCED]].notna().all(axis=1)
    ps["_adv_season"] = np.where(ps["_adv"], ps["season_start"], np.nan)
    agg["adv_last_season_start"] = ("_adv_season", "max")
    # how much of the career the tracking data actually covers (xG as the marker stat)
    # (progressive passes mark the full event feed; xG alone can now come from Understat)
    ps["_xg_min"] = np.where(ps["prgp_per90"].notna(), ps["min"], 0.0)
    ps["_xg_season"] = np.where(ps["xg_per90"].notna(), ps["season_start"], np.nan)
    agg["xg_last_season_start"] = ("_xg_season", "max")
    agg["_xg_min"] = ("_xg_min", "sum")

    out = ps.groupby("pkey").agg(**agg).reset_index()
    for c in PER90_COLS:
        den = out.pop(f"{c}__den")
        num = out.pop(f"{c}__num")
        out[c] = np.where(den > 0, num / den.replace(0, np.nan), np.nan)   # NaN, not 0
    out["has_advanced"] = out[[f"{a}_per90" for a in ADVANCED]].notna().all(axis=1).astype(int)
    out["adv_minutes_share"] = (out.pop("_xg_min") / out["min"]).round(3)
    out["adv_seasons_ago"] = current_season_start() - out["adv_last_season_start"]      # NaN = never
    out["xg_last_season"] = out.pop("xg_last_season_start").map(
        lambda y: f"{int(y)}-{int(y) + 1}" if pd.notna(y) else None)

    # identity from the most recent season (largest stint if he moved mid-season)
    latest = (ps.sort_values(["season_start", "min"], ascending=False)
                .drop_duplicates("pkey")[["pkey", "player", "squad", "league", "age", "born"]])
    _last = ps[ps["season_start"] == ps.groupby("pkey")["season_start"].transform("max")]
    latest_squads = _last.groupby("pkey")["squad"].agg(list).rename("latest_squads")
    latest_leagues = _last.groupby("pkey")["league"].agg(list).rename("latest_leagues")
    recent = ps[ps["season_start"] >= ps.groupby("pkey")["season_start"].transform("max") - 1].copy()
    recent["ncomma"] = recent["pos"].str.count(",")
    posfull = (recent.sort_values("ncomma", ascending=False)
                     .drop_duplicates("pkey")[["pkey", "pos"]].rename(columns={"pos": "pos_full"}))
    career_squads = ps.groupby("pkey")["squad"].agg(lambda s: sorted(set(s))).rename("career_squads")
    out = (out.merge(latest, on="pkey").merge(posfull, on="pkey", how="left")
              .merge(latest_squads, on="pkey", how="left")
              .merge(latest_leagues, on="pkey", how="left")
              .merge(career_squads, on="pkey", how="left"))
    out["last_season"] = out["last_season_start"].map(lambda y: f"{int(y)}-{int(y) + 1}")
    out["adv_last_season"] = out["adv_last_season_start"].map(
        lambda y: f"{int(y)}-{int(y) + 1}" if pd.notna(y) else None)
    return out


# ---------------------------------------------------------------------------- Transfermarkt
def _prep_tm(active_only: bool = True) -> pd.DataFrame:
    tm = pd.read_csv(DATA_DIR / "players.csv", low_memory=False)
    tm.columns = [c.strip().lower() for c in tm.columns]
    if active_only:
        tm = tm[tm["market_value_in_eur"].notna()].copy()
        tm = tm[tm["last_season"] >= current_season_start() - TM_ACTIVE_LOOKBACK].copy()  # active only
    else:                                   # historical identities, for the forecast training set
        tm = tm.copy()
        tm["market_value_in_eur"] = tm["market_value_in_eur"].fillna(0)
    tm["nkey"] = tm["name"].map(name_key)
    tm["dob"] = pd.to_datetime(tm["date_of_birth"], errors="coerce")
    tm["dob_year"] = tm["dob"].dt.year
    tm["market_value_m"] = tm["market_value_in_eur"] / 1e6
    tm["contract_exp"] = pd.to_datetime(tm["contract_expiration_date"], errors="coerce")
    return tm



def _surname_match(nk: str, born, squad: str, by_year: dict) -> pd.DataFrame:
    """Last-resort stage for nickname / shortened-name variants ('Andy' vs 'Andrew Robertson').
    Requires an EXACT birth year, one name's surname (last token) contained in the other, and either a compatible
    first name (same first two letters) or a matching club."""
    toks = nk.split()
    if pd.isna(born) or len(toks) < 2 or int(born) not in by_year:
        return by_year.get(-1, pd.DataFrame())
    pool = by_year[int(born)]
    club_key = name_key(squad)
    keep = []
    for idx, cand_key, club in zip(pool.index, pool["nkey"], pool["current_club_name"]):
        ct = cand_key.split()
        if len(ct) < 2 or not (toks[-1] in ct[1:] or ct[-1] in toks[1:]):
            continue                              # the shared token must be a SURNAME (last token)
        first_ok = toks[0][:2] == ct[0][:2]
        club_ok = fuzz.WRatio(club_key, name_key(club)) >= 85
        if first_ok or club_ok:
            keep.append(idx)
    return pool.loc[keep] if keep else pool.iloc[0:0]

# FBref display name + birth year -> Transfermarkt name, for players the two sites call
# different things. Checked before any fuzzy logic.
NAME_ALIASES = {
    "savio|2004": "savinho",
}
CLUB_CORROBORATION = 85


def _club_ok(squads, tm_club) -> bool:
    """True if the Transfermarkt club is one this player has actually appeared for."""
    if not isinstance(tm_club, str) or not isinstance(squads, (list, tuple)):
        return False
    ck = name_key(canon_club(tm_club))
    raw = name_key(tm_club)
    return any(max(fuzz.WRatio(name_key(s), ck), fuzz.WRatio(name_key(s), raw)) >= CLUB_CORROBORATION
               for s in squads if s)


def _corroborated(pool: pd.DataFrame, born, squads) -> pd.DataFrame:
    """A non-exact name match is only accepted with an EXACT birth year AND a club the player
    has appeared for. Without this, WRatio's substring score (exactly 90) matched
    'Sávio' -> 'Noah Saviolo', 'Lucas' -> 'Lucas Bernadou', 'Igor' -> 'Grigoris Kastanos'."""
    if pd.isna(born) or pool.empty:
        return pool.iloc[0:0]
    keep = (pool["dob_year"] == int(born)) & pool["current_club_name"].map(lambda c: _club_ok(squads, c))
    return pool[keep]


def match_transfermarkt(players: pd.DataFrame, tm: pd.DataFrame) -> pd.DataFrame:
    by_year = {int(y): g for y, g in tm.groupby("dob_year")}
    rows = []
    for r in players.itertuples(index=False):
        nk, born = name_key(r.player), r.born
        squads = getattr(r, "career_squads", None)
        if pd.notna(born):
            nk = NAME_ALIASES.get(f"{nk}|{int(born)}", nk)
        if pd.notna(born):
            parts = [by_year[y] for y in (int(born) - 1, int(born), int(born) + 1) if y in by_year]
            cands, cutoff = (pd.concat(parts) if parts else tm.iloc[0:0]), FUZZY_CUTOFF
        else:                                     # no birth year -> exact name only
            cands, cutoff = tm, 100
        if cands.empty:
            rows.append((r.pkey, None, 0.0, "none")); continue

        exact = cands[cands["nkey"] == nk]
        if not exact.empty:
            pool, score, method = exact, 100.0, "exact"
        else:
            hits = process.extract(nk, cands["nkey"], scorer=fuzz.WRatio, score_cutoff=cutoff, limit=6)
            pool = cands.iloc[0:0]
            if hits:
                top = max(h[1] for h in hits)
                pool = _corroborated(cands.loc[[h[2] for h in hits]], born, squads)
                score, method = float(top), "fuzzy"
            if pool.empty:
                pool = _corroborated(_surname_match(nk, born, r.squad, by_year), born, squads)
                score, method = 80.0, "surname"
            if pool.empty:                         # nothing corroborated -> unmatched, not guessed
                rows.append((r.pkey, None, 0.0, "none")); continue

        if len(pool) > 1:                          # namesakes: year -> club -> recency -> value
            club_key = name_key(r.squad)
            pool = pool.assign(
                _dy=(pool["dob_year"] - born).abs() if pd.notna(born) else 0,
                _club=pool["current_club_name"].map(lambda c: fuzz.WRatio(club_key, name_key(c))),
            ).sort_values(["_dy", "_club", "last_season", "market_value_m"],
                          ascending=[True, False, False, False])
        rows.append((r.pkey, pool.index[0], score, method))
    m = pd.DataFrame(rows, columns=["pkey", "tm_idx", "match_score", "match_method"])
    m = m.dropna(subset=["tm_idx"]).astype({"tm_idx": int})
    info = tm.loc[m["tm_idx"], ["player_id", "name", "current_club_name", "dob", "market_value_m",
                                "contract_exp", "sub_position", "international_caps"]].reset_index(drop=True)
    info.columns = ["tm_player_id", "tm_match_name", "tm_club", "dob", "actual_value_m", "contract_exp",
                    "tm_sub_position", "international_caps"]
    return pd.concat([m.drop(columns="tm_idx").reset_index(drop=True), info], axis=1)


# ---------------------------------------------------------------------------- roles
# Transfermarkt's scouted position is the primary role label. FBref only publishes GK/DF/MF/FW,
# and inferring the role from per-90 thresholds put Declan Rice and Lamine Yamal at CAM and
# Trent Alexander-Arnold at CB. The heuristic below is now only a fallback.
TM_ROLE = {
    "Goalkeeper": "GK", "Centre-Back": "CB", "Left-Back": "FULLBACK", "Right-Back": "FULLBACK",
    "Defensive Midfield": "CDM", "Central Midfield": "CM", "Attacking Midfield": "CAM",
    "Left Winger": "WINGER", "Right Winger": "WINGER", "Left Midfield": "WINGER",
    "Right Midfield": "WINGER", "Centre-Forward": "ST", "Second Striker": "ST",
}


def assign_role(row: pd.Series) -> str:
    return TM_ROLE.get(row.get("tm_sub_position")) or classify_role(row)


def classify_role(row: pd.Series) -> str:
    """FBref 'Pos' only has GK/DF/MF/FW combinations (no LB/RW tags), so use the first
    token as the primary role and the second as a hint; NaN-safe when advanced data is missing."""
    pos = str(row.get("pos_full", "")).upper().replace(" ", "")
    toks = [t[:2] for t in pos.split(",") if t]
    primary, secondary = (toks + ["", ""])[:2]
    g = lambda k: row.get(k) if pd.notna(row.get(k)) else np.nan
    prgc, prgp, clr, sh = g("prgc_per90"), g("prgp_per90"), g("clr_per90"), g("sh_per90")
    xag, gls, ast = g("xag_per90"), g("gls_per90"), g("ast_per90")
    tkl, itc = g("tkl_per90"), g("int_per90")

    if primary == "GK":
        return "GK"
    if primary == "DF":
        if secondary == "MF":
            return "FULLBACK"
        if pd.notna(prgc) and prgc >= 1.8 and (pd.isna(clr) or clr < 2.2):
            return "FULLBACK"
        return "CB"
    if primary == "FW":
        if pd.notna(prgc) and prgc >= 2.3:
            return "WINGER"
        if secondary == "MF" and pd.notna(prgp) and prgp >= 4.0:
            return "CAM"
        if secondary == "MF" and not row.get("has_advanced", 0):
            return "WINGER"
        return "ST"
    # midfield
    if (pd.notna(sh) and sh >= 1.6) or (pd.notna(xag) and xag >= 0.14) or \
       (np.nansum([gls, ast]) >= 0.28 and pd.notna(gls)):
        return "CAM"
    if pd.notna(itc):
        defensive = (tkl + itc) if pd.notna(tkl) else itc * 2.4
        if defensive >= 2.8 and (pd.isna(sh) or sh < 1.2):
            return "CDM"
    return "CM"


# ---------------------------------------------------------------------------- model
# "Why this value": every feature belongs to one plain-language factor.
WHY_GROUPS = ["age", "output", "experience", "league", "position", "coverage"]      # performance model only


def _why_group(col: str) -> str:
    if col == "age_clean": return "age"
    if col == "contract_years_left": return "contract"
    if col == "min": return "experience"
    if col in ("club_goals_pm", "club_caps"): return "club"
    if col == "international_caps": return "international"
    if col.startswith("league_"): return "league"
    if col.startswith("pos_clean_"): return "position"
    if col in ("has_advanced", "adv_minutes_share", "adv_seasons_ago"): return "coverage"
    return "output"                                   # every per-90 stat


def _oof_with_contribs(X: pd.DataFrame, y: np.ndarray, cv):
    """Out-of-fold predictions PLUS out-of-fold SHAP contributions: each player's explanation
    comes from the same fold model that priced him, so the factors sum exactly to his estimate."""
    import xgboost as xgb
    oof = np.zeros(len(y))
    contrib = np.zeros((len(y), X.shape[1] + 1))       # last column = baseline
    for tr, te in cv.split(X):
        model = _xgb().fit(X.iloc[tr], y[tr])
        oof[te] = model.predict(X.iloc[te])
        contrib[te] = model.get_booster().predict(xgb.DMatrix(X.iloc[te]), pred_contribs=True)
    groups = [_why_group(c) for c in X.columns]
    out = {g: contrib[:, [i for i, gg in enumerate(groups) if gg == g]].sum(axis=1) for g in WHY_GROUPS}
    out["base"] = contrib[:, -1]
    return oof, out


def quality_gate(new: dict, prev: dict | None) -> list[str]:
    """Checks run BEFORE anything is uploaded. A non-empty list blocks the swap, so a bad scrape or a
    broken join can never replace good production tables. STATTRICK_SKIP_GATE=1 overrides."""
    fails = []
    if new["players"] < 1500:
        fails.append(f"only {new['players']:,} players (< 1,500)")
    if new["oof_r2_log"] < 0.60:
        fails.append(f"performance model log R2 {new['oof_r2_log']} (< 0.60)")
    if not 0.62 <= new["band_coverage_15_85"] <= 0.78:
        fails.append(f"fair-band coverage {new['band_coverage_15_85']:.0%} (target 70%, allowed 62-78%)")
    unmatched_share = new["unmatched_players"] / max(new["players"] + new["unmatched_players"], 1)
    if unmatched_share > 0.05:
        fails.append(f"{unmatched_share:.0%} of players unmatched to Transfermarkt (> 5%)")
    if prev and prev.get("players"):
        if new["players"] < 0.85 * prev["players"]:
            fails.append(f"player count fell {prev['players']:,} -> {new['players']:,} (more than 15%)")
        if prev.get("oof_r2_log") and new["oof_r2_log"] < prev["oof_r2_log"] - 0.08:
            fails.append(f"log R2 fell {prev['oof_r2_log']} -> {new['oof_r2_log']} (more than 0.08)")
    return fails


def _append_history(metrics: dict) -> None:
    """One row per run, so drift is visible over time (data/metrics_history.csv)."""
    fc = metrics.get("forecast") or {}
    row = {k: metrics.get(k) for k in ["built_on", "players", "unmatched_players", "oof_r2_log", "oof_r2_eur",
                                       "oof_mae_eur_m", "band_coverage_15_85", "tm_club_mismatch",
                                       "xg_coverage_current_season"]}
    row["forecast_r2_of_change"] = fc.get("r2_of_change")
    row["forecast_within_25pct"] = fc.get("within_25pct")
    path = DATA_DIR / "metrics_history.csv"
    hist = pd.read_csv(path) if path.exists() else pd.DataFrame()
    hist = pd.concat([hist[hist.get("built_on") != row["built_on"]] if len(hist) else hist,
                      pd.DataFrame([row])], ignore_index=True)
    hist.to_csv(path, index=False)


def _xgb(**kw):
    return XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=5, subsample=0.8,
                        colsample_bytree=0.8, random_state=42, n_jobs=-1, **kw)


def build_valuation_model():
    ps = build_player_seasons()

    # ---- timeline export (per-season, NaNs intact, slim schema)
    tl_cols = ["pkey", "player", "born", "pos", "squad", "league", "season", "min", "age"] + RAW_STATS + PER90_COLS
    
    # (uploaded once, together with players_master, at the end of the run)

    players = aggregate_players(ps)
    print(f"\nUnique players: {len(players):,}")
    cur = current_season_start()
    players_all = players[players["min"] >= MIN_CAREER_MINUTES].copy()      # incl. players who have left
    players = players[(players["min"] >= MIN_CAREER_MINUTES)
                      & (players["last_season_start"] >= cur - (RECENT_SEASONS - 1))].copy()
    print(f"After >= {MIN_CAREER_MINUTES} min and played in last {RECENT_SEASONS} seasons: {len(players):,}")

    print("Matching Transfermarkt (birth-year blocked)...")
    tm = _prep_tm()
    match = match_transfermarkt(players, tm)
    df = players.merge(match, on="pkey", how="left")

    unmatched = df[df["actual_value_m"].isna()][["player", "squad", "league", "born", "min", "last_season"]]
    # Keeping unmatched as local CSV since this acts purely as a debug log for the modeler
    unmatched.sort_values("min", ascending=False).to_csv(DATA_DIR / "unmatched_players.csv", index=False)
    print(f"  matched {df['actual_value_m'].notna().sum():,} | unmatched {len(unmatched):,} "
          f"(listed in data/unmatched_players.csv) | fuzzy {(df['match_method'] == 'fuzzy').sum()}")

    df = df.dropna(subset=["actual_value_m"])
    df = df[df["actual_value_m"] > MIN_MARKET_VALUE_M].copy()

    # club: if he moved mid-season prefer the stint that matches Transfermarkt's current club
    def pick_squad(r):
        sq = r["latest_squads"]
        if isinstance(sq, list) and len(sq) > 1 and isinstance(r["tm_club"], str):
            i = max(range(len(sq)), key=lambda k: fuzz.WRatio(name_key(sq[k]), name_key(r["tm_club"])))
            lg = r["latest_leagues"]
            # the league must come from the SAME stint as the squad
            return pd.Series([sq[i], lg[i] if isinstance(lg, list) and len(lg) == len(sq) else r["league"]])
        return pd.Series([r["squad"], r["league"]])
    df[["squad", "league"]] = df.apply(pick_squad, axis=1).values

    # ---- age / contract relative to *today*, not a hard-coded year
    t = pd.Timestamp(today())
    age_dob = (t - df["dob"]).dt.days / 365.25
    df["age_clean"] = age_dob.fillna(today().year - df["born"]).fillna(df["age"]).round(1)
    df["contract_years_left"] = ((df["contract_exp"] - t).dt.days / 365.25).clip(lower=0, upper=7)  # NaN kept
    # Transfermarkt still lists a club he no longer plays for -> its value and contract describe
    # the pre-move player. Flag it and blank the contract rather than model the old deal.
    df["tm_club_mismatch"] = [int(not _club_ok([sq], tc)) for sq, tc in zip(df["squad"], df["tm_club"])]
    df.loc[df["tm_club_mismatch"] == 1, "contract_years_left"] = np.nan

    df["pos_clean"] = df.apply(assign_role, axis=1)
    df["pos_source"] = np.where(df["tm_sub_position"].isin(list(TM_ROLE)), "transfermarkt", "heuristic")
    df["raw_pos"] = df["pos_full"]

    # unique display names (two different 'Aaron Ramsey's must not collide in the app)
    dup = df["player"].duplicated(keep=False)
    df["fb_name"] = df["player"]
    df.loc[dup, "player"] = df.loc[dup, "player"] + " (" + df.loc[dup, "squad"] + ")"

    # ---- context features
    # Club level, measured WITHOUT any Transfermarkt price so the model stays an independent
    # opinion rather than an echo of the market it is judged against:
    #   club_goals_pm - the club's goals per match over the last two seasons (FBref)
    #   club_caps     - mean international caps of his team-mates (leave-one-out)
    # Together these match the accuracy of "team-mates' market value" (log R2 0.859 either way).
    _rec = ps[ps["season_start"] >= cur - 1].groupby("squad").agg(_g=("gls", "sum"), _m=("min", "sum"))
    _goals_pm = (_rec["_g"] / (_rec["_m"] / 990.0)).where(_rec["_m"] >= 9900)      # >= 10 matches of minutes
    df["club_goals_pm"] = df["squad"].map(_goals_pm).round(3)
    df["international_caps"] = pd.to_numeric(df["international_caps"], errors="coerce")
    _caps = df["international_caps"].fillna(0)
    _n = _caps.groupby(df["squad"]).transform("size")
    df["club_caps"] = np.where(_n > 1, (_caps.groupby(df["squad"]).transform("sum") - _caps)
                               / (_n - 1).clip(lower=1), np.nan).round(2)
    # How many players of a similar age the market values anywhere near this highly. A low count
    # means the model has almost nothing to learn from (e.g. a 33-year-old valued at EUR 60M).
    _age, _val = df["age_clean"].to_numpy(float), df["actual_value_m"].to_numpy(float)
    df["n_comparables"] = [int(((np.abs(_age - a) <= 2) & (_val >= 0.5 * v)).sum()) - 1
                           for a, v in zip(_age, _val)]

   # ---- design matrix (NaN preserved for XGBoost)
    league_d = pd.get_dummies(df["league"], prefix="league").astype(int)
    pos_d = pd.get_dummies(df["pos_clean"], prefix="pos_clean").astype(int)
    
    # TWO SEPARATE QUESTIONS, TWO SEPARATE FEATURE SETS
    #   PERFORMANCE value  - what is this output worth? On-pitch output, minutes, age, position,
    #                        league. Nothing about who he plays for, his status or his deal.
    #                        This is predicted_value_m and drives the screener / edge.
    #   MARKET-SIDE factors - club level, international caps, contract length. They move the price
    #                        without saying how good the player is. The full model that adds them
    #                        is exported as market_profile_value_m; the difference between the two
    #                        is the status_premium_m shown beside the performance value.
    market_side = ["contract_years_left", "club_goals_pm", "club_caps", "international_caps"]
    # Model B is for players without the full event feed; it still gets xG / xA (Understat has them)
    basic_per90 = [f"{s}_per90" for s in RAW_STATS if s not in ("prgc", "prgp")]

    # Model A: advanced event data available
    feats_adv = ["age_clean", "min", "has_advanced", "adv_minutes_share", "adv_seasons_ago"] + PER90_COLS
    X_adv = pd.concat([df[feats_adv].apply(pd.to_numeric, errors="coerce"), league_d, pos_d], axis=1)
    # Model B: basic output only (no event data)
    feats_basic = ["age_clean", "min"] + basic_per90
    X_basic = pd.concat([df[feats_basic].apply(pd.to_numeric, errors="coerce"), league_d, pos_d], axis=1)
    # the same two models with the market-side factors added
    _mkt = df[market_side].apply(pd.to_numeric, errors="coerce")
    X_adv_full, X_basic_full = pd.concat([X_adv, _mkt], axis=1), pd.concat([X_basic, _mkt], axis=1)
    
    y_raw = df["actual_value_m"].values
    y_log = np.log1p(y_raw)

    # ---- OUT-OF-FOLD predictions: Dual-Model Architecture
    print(f"Training XGBoost Dual-Model Architecture on {len(df):,} players...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    def _train_and_conformalize(X_matrix, name):
        print(f"  -> Training Model {name} ({X_matrix.shape[1]} features)...")
        oof_log, why = _oof_with_contribs(X_matrix, y_log, cv)
        try:
            lo_log = cross_val_predict(_xgb(objective="reg:quantileerror", quantile_alpha=0.15), X_matrix, y_log, cv=cv)
            hi_log = cross_val_predict(_xgb(objective="reg:quantileerror", quantile_alpha=0.85), X_matrix, y_log, cv=cv)
        except Exception as exc:
            res = y_log - oof_log
            lo_log, hi_log = oof_log + np.quantile(res, 0.15), oof_log + np.quantile(res, 0.85)
        
        # Conformalise PER PREDICTED-VALUE TIER (Mondrian). One global correction hit 70% overall
        # but only ~47% for the most valuable fifth, where the band matters most.
        lo_log, hi_log = np.minimum(lo_log, oof_log), np.maximum(hi_log, oof_log)
        score = np.maximum(lo_log - y_log, y_log - hi_log)
        tier = pd.qcut(pd.Series(oof_log).rank(method="first"), CONFORMAL_TIERS, labels=False).values
        q = np.zeros_like(score)
        for t_ in np.unique(tier):
            s_ = score[tier == t_]
            q[tier == t_] = np.quantile(s_, min(1.0, 0.70 * (1 + 1 / len(s_))))
        # clamp AFTER shifting so a negative q can never push a bound across the estimate
        return oof_log, np.minimum(lo_log - q, oof_log), np.maximum(hi_log + q, oof_log), why
        
    oof_log_a, lo_log_a, hi_log_a, why_a = _train_and_conformalize(X_adv, "A (Advanced/Historical)")
    oof_log_b, lo_log_b, hi_log_b, why_b = _train_and_conformalize(X_basic, "B (Basic/Live)")
    
    # Splice Predictions: Route to Model B if advanced stats are missing
    has_adv = df["has_advanced"].astype(bool).values
    oof_log = np.where(has_adv, oof_log_a, oof_log_b)
    lo_log = np.where(has_adv, lo_log_a, lo_log_b)
    hi_log = np.where(has_adv, hi_log_a, hi_log_b)
    # log-space contribution of each factor to this player's estimate (same routing as the price)
    for g in WHY_GROUPS + ["base"]:
        df[f"why_{g}"] = np.round(np.where(has_adv, why_a[g], why_b[g]), 4)

    # market-profile estimate (performance + club level, caps, contract) and the premium it implies
    print("  -> Training market-profile models (performance + club, caps, contract)...")
    full_a, _ = _oof_with_contribs(X_adv_full, y_log, cv)
    full_b, _ = _oof_with_contribs(X_basic_full, y_log, cv)
    full_log = np.where(has_adv, full_a, full_b)
    df["market_profile_value_m"] = np.round(np.expm1(full_log), 2)
    df["status_premium_m"] = np.round(np.expm1(full_log) - np.expm1(oof_log), 2)

    # ---- MARKET FORECAST (separate model; openly uses Transfermarkt's own value history)
    # Identities: current players keep the match made above; players who have since left the
    # top-5 leagues are matched against the full Transfermarkt list so that the forecast is
    # trained on everyone who was playing at the time, not only on survivors.
    forecast_metrics = {"available": False, "reason": "not run"}
    value_history = None
    try:
        gone = players_all[~players_all["pkey"].isin(df["pkey"])]
        gone_match = match_transfermarkt(gone, _prep_tm(active_only=False)) if len(gone) else pd.DataFrame()
        ids = pd.concat([df[["pkey", "tm_player_id", "dob"]],
                         gone_match[["pkey", "tm_player_id", "dob"]] if len(gone_match) else None],
                        ignore_index=True)
        forecast, forecast_metrics, value_history = build_market_forecast(ps, ids)
        if forecast is not None:
            df = df.merge(forecast, on="pkey", how="left")
            value_history = value_history[value_history["pkey"].isin(set(df["pkey"]))]
        else:
            print(f"Market forecast skipped: {forecast_metrics.get('reason')}")
    except Exception as exc:                                    # noqa: BLE001 - never block the main pipeline
        forecast_metrics = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
        print(f"WARNING: market forecast failed and was skipped ({forecast_metrics['reason']})")

    pred = np.expm1(oof_log)
    df["predicted_value_m"] = np.round(pred, 2)
    df["pred_value_low_m"] = np.round(np.clip(np.expm1(lo_log), 0, None), 2)
    df["pred_value_high_m"] = np.round(np.expm1(hi_log), 2)
    df["surplus_value_m"] = np.round(pred - y_raw, 2)
    df["surplus_log"] = np.round(oof_log - y_log, 4)                       # log-space edge
    df["surplus_pct"] = np.round(100 * np.expm1(oof_log - y_log), 1)      # (1+pred)/(1+actual) - 1

    # predicted_value_m is the conditional MEDIAN (expm1 of a log-space mean). For euro totals use
    # the Duan-smeared mean; summing medians understated the pool by ~16%.
    smear = float(np.mean(np.exp(y_log - oof_log)))
    df["predicted_mean_m"] = np.round(np.exp(oof_log) * smear - 1, 2)
    # Raw surplus is regression to the mean (corr with log price = -sqrt(1 - R2)). edge_z measures
    # the gap in units of the player's own band half-width; outside_band is the actionable flag.
    half = np.maximum((hi_log - lo_log) / 2.0, 1e-6)
    df["edge_z"] = np.round((oof_log - y_log) / half, 3)
    df["outside_band"] = np.where(y_raw < df["pred_value_low_m"], 1,
                                  np.where(y_raw > df["pred_value_high_m"], -1, 0))   # 1 = market below band

    r2, mae = r2_score(y_raw, pred), mean_absolute_error(y_raw, pred)
    r2_log = r2_score(y_log, oof_log)
    cover = float(((y_raw >= df["pred_value_low_m"]) & (y_raw <= df["pred_value_high_m"])).mean())
    print("\n--- Out-of-fold model evaluation (honest, every row) ---")
    print(f"R2 (EUR): {r2:.3f} | R2 (log): {r2_log:.3f} | MAE: EUR {mae:.2f}M | 15-85 band coverage: {cover:.0%}")

   # ---- export (keeps the columns the app already expects + the new ones)
    out = pd.concat([df.reset_index(drop=True), league_d.reset_index(drop=True),
                     pos_d.reset_index(drop=True)], axis=1)
                     
    # RESTORE: Ensure the raw text column for the UI isn't lost among the dummies
    if "league" not in out.columns and "league" in df.columns:
        out["league"] = df["league"].values
        
    out = out.drop(columns=["latest_squads", "latest_leagues", "career_squads", "contract_exp", "last_season_start", "adv_last_season_start",
                            "tm_player_id"], errors="ignore")
    
    cur_rows = ps[ps["season_start"] == ps["season_start"].max()]
    metrics = {
        "built_on": str(today()), "season_in_progress": f"{cur}-{cur + 1}",
        "players": int(len(out)), "unmatched_players": int(len(unmatched)),
        "oof_r2_eur": round(float(r2), 3), "oof_r2_log": round(float(r2_log), 3),
        "market_profile_r2_log": round(float(r2_score(y_log, full_log)), 3),
        "oof_mae_eur_m": round(float(mae), 2), "band_coverage_15_85": round(cover, 3),
        "players_without_advanced_stats": int((out["has_advanced"] == 0).sum()),
        "tm_club_mismatch": int(out["tm_club_mismatch"].sum()),
        "non_exact_tm_matches": int((out["match_method"] != "exact").sum()),
        "xg_coverage_current_season": round(float(cur_rows["xg"].notna().mean()), 3),
        "smearing_factor": round(smear, 4),
        "forecast": forecast_metrics,
    }

    # ---- quality gate: nothing reaches the database unless these pass
    fails = quality_gate(metrics, read_json(DATA_DIR / "model_metrics.json", default=None))
    if metrics["xg_coverage_current_season"] < 0.5:
        print(f"WARNING: xG covers only {metrics['xg_coverage_current_season']:.0%} of this season's players "
              "(has ingest_understat.py run?)")
    if fails and os.getenv("STATTRICK_SKIP_GATE") != "1":
        sys.exit("QUALITY GATE FAILED - production tables were NOT changed:\n  - " + "\n  - ".join(fails)
                 + "\n(set STATTRICK_SKIP_GATE=1 to upload anyway)")
    if fails:
        print("WARNING: quality gate failed but STATTRICK_SKIP_GATE=1 is set:\n  - " + "\n  - ".join(fails))

    # all tables go live together, or none does
    print("Uploading players_master, player_timeline and player_value_history to Postgres...")
    tables = {"players_master": out, "player_timeline": ps[ps["min"] >= 90][tl_cols]}
    if value_history is not None and len(value_history):
        tables["player_value_history"] = value_history
    replace_tables(tables)

    write_json(DATA_DIR / "model_metrics.json", metrics)
    _append_history(metrics)
    print(f"Uploaded players_master to Postgres ({len(out):,} players); quality gate passed")

if __name__ == "__main__":
    build_valuation_model()